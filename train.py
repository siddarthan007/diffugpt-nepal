import inspect
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import DiffusionGPT, ModelArgs

# =====================================================================
# Nepali masked-diffusion training (~50M params) — RTX 5090 32GB / 94GB RAM
# Reads pre-tokenised Nepali data produced OFF-GPU by the Phase-A pipeline:
#   data_dir/train.bin , data_dir/val.bin  (uint16 token stream, nanoGPT-style)
#   data_dir/nepali_bpe_16k.model          (SentencePiece model, for sample logs)
# =====================================================================

# --- paths ---
out_dir = "out"
data_dir = "data"
tokenizer_path = os.path.join(data_dir, "nepali_bpe_16k.model")

# --- optimisation config ---
micro_batch_size = 64  # 64 * 512 = 32,768 tokens / micro-step (calibrate to VRAM)
gradient_accumulation_steps = (
    8  # effective batch = 8 * 32,768 = 262,144 tokens / update
)
eval_interval = 500
eval_iters = 50
log_interval = 20
max_iters = 40000  # WSD; wallclock-bound (decay auto-triggers near the end)
learning_rate = 3e-4  # peak LR (LLaDA/MDLM band for ~50M)
min_lr = 3e-5
warmup_iters = 500
decay_frac = 0.2  # WSD: last 20% of steps decay peak -> min
weight_decay = 0.1
grad_clip = 1.0
t_eps = 1e-3  # clamp on continuous time to bound the 1/t loss weight
compile_model = True

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device == "cuda" else "cpu"
amp_dtype = torch.bfloat16  # native BF16 Tensor Core execution
ctx = torch.autocast(
    device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda")
)

torch.manual_seed(1337)
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

os.makedirs(out_dir, exist_ok=True)

# --- memmap data pipeline (uint16 token stream lives on disk, paged by the OS) ---
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
print(f"Data: train {len(train_data):,} tokens | val {len(val_data):,} tokens")

args = ModelArgs(device=device)
mask_id = args.vocab_size

# --- optional SentencePiece decoder for readable sample logs ---
decode = None
if os.path.exists(tokenizer_path):
    try:
        import sentencepiece as spm

        _sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        decode = lambda ids: _sp.decode([int(i) for i in ids if int(i) < mask_id])
    except Exception as e:
        print(f"(SentencePiece decode unavailable: {e})")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block_size, (micro_batch_size,))
    x_0 = torch.stack(
        [torch.from_numpy(d[i : i + args.block_size].astype(np.int64)) for i in ix]
    )

    # Continuous-time low-discrepancy (stratified/antithetic) sampling on (0, 1]
    u = torch.rand(1).item()
    t = (u + torch.arange(micro_batch_size).float() / micro_batch_size) % 1.0
    t = t.clamp(min=t_eps, max=1.0)

    if device_type == "cuda":
        x_0 = x_0.pin_memory().to(device, non_blocking=True)
        t = t.pin_memory().to(device, non_blocking=True)
    else:
        x_0, t = x_0.to(device), t.to(device)
    return x_0, t


print("Initializing DiffusionGPT...")
model = DiffusionGPT(args).to(device)
n_params = sum(p.numel() for p in model.parameters())
n_params_non_emb = n_params - model.token_embedding.weight.numel()
print(
    f"  params: {n_params / 1e6:.1f}M total | {n_params_non_emb / 1e6:.1f}M non-embedding"
)

if compile_model:
    # NOTE: use *-no-cudagraphs. Plain "max-autotune" captures CUDA graphs, which
    # break with our tied embedding (token_embedding.weight is lm_head.weight -> used
    # twice) + grad-accum + interleaved eval/sample: "accessing tensor output of
    # CUDAGraphs that has been overwritten". This keeps Triton autotuning, drops graphs.
    print("Compiling model (max-autotune-no-cudagraphs, ~2-3 min first time)...")
    model = torch.compile(model, mode="max-autotune-no-cudagraphs")

use_fused = "fused" in inspect.getfullargspec(torch.optim.AdamW).args
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    weight_decay=weight_decay,
    fused=use_fused,
)


def get_lr(it):
    # Warmup-Stable-Decay (WSD): linear warmup -> flat peak -> linear decay over the final decay_frac.
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    decay_start = int((1.0 - decay_frac) * max_iters)
    if it < decay_start:
        return learning_rate
    if it >= max_iters:
        return min_lr
    r = (it - decay_start) / max(1, (max_iters - decay_start))
    return learning_rate - r * (learning_rate - min_lr)


def compute_masked_loss(x_0, t):
    # Absorbing forward process: mask each token independently with prob = t (linear schedule).
    B, T = x_0.shape
    mask = torch.rand(B, T, device=x_0.device) < t.unsqueeze(1)
    x_t = torch.where(mask, mask_id, x_0)

    with ctx:
        logits = model(x_t, t)
        ce = F.cross_entropy(
            logits.view(-1, args.vocab_size + 1), x_0.view(-1), reduction="none"
        ).view(B, T)

    # MDLM / LLaDA estimator: per sequence  (1/t) * (1/T) * sum_{masked} CE .  Averaged over the batch.
    ce_masked_sum = (ce * mask.float()).sum(dim=1)
    loss = ((1.0 / t) * ce_masked_sum / T).mean()
    return loss


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_0, t = get_batch(split)
            losses[k] = compute_masked_loss(x_0, t).item()
        out[split] = losses.mean().item()
    model.train()
    return out


# --- training loop ---
best_val_loss = float("inf")
t0 = time.time()
x_0, t = get_batch("train")

print("Starting training...")
for it in range(max_iters):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g["lr"] = lr

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(
            f"\n---> step {it:05d}: train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}"
        )
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            raw = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                {"model": raw.state_dict(), "args": args, "step": it},
                os.path.join(out_dir, "ckpt.pt"),
            )
            print("     [checkpoint saved]")
        if decode is not None:
            ids = model.sample(
                batch_size=1, seq_len=120, temperature=0.8, top_k=15, num_steps=64
            )
            print(f"     sample: {decode(ids[0].tolist())!r}\n")

    for micro_step in range(gradient_accumulation_steps):
        loss = compute_masked_loss(x_0, t) / gradient_accumulation_steps
        loss.backward()
        x_0, t = get_batch("train")

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if it % log_interval == 0 and it > 0:
        dt = time.time() - t0
        ms_per_iter = dt * 1000 / log_interval
        tok_per_sec = (
            micro_batch_size * gradient_accumulation_steps * args.block_size
        ) / (ms_per_iter / 1000)
        print(
            f"iter {it:05d}: loss {loss.item() * gradient_accumulation_steps:.4f} | {ms_per_iter:.1f} ms/iter | ~{tok_per_sec:,.0f} tok/sec"
        )
        t0 = time.time()
