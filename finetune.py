"""Instruction-tuning (SFT) for chat-style behavior — LLaDA recipe on masked diffusion.

Starts from the pretrained checkpoint and fine-tunes on Nepali instruction/response
pairs from Aya (CohereLabs/aya_dataset, Apache-2.0, human-written, commercial-OK).
Only the RESPONSE tokens are masked and supervised; the prompt is kept clean (the same
way it will be given as an anchor at inference). Saves out/ckpt_sft.pt.

Run (pod):  python finetune.py
            python finetune.py --extra saillab/alpaca-nepali-cleaned   # +52k (CC-BY-NC, non-commercial!)
"""
import argparse
import inspect
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import DiffusionGPT
import prompt_format as PF


def load_pretrained(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.device = device
    model = DiffusionGPT(args).to(device)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    return model, args


def _aya_nepali():
    """Yield (instruction, answer) from Aya's Nepali subset (tries both org ids)."""
    from datasets import load_dataset
    for repo in ("CohereLabs/aya_dataset", "CohereForAI/aya_dataset"):
        try:
            ds = load_dataset(repo, split="train")
        except Exception as e:
            print(f"  ({repo} failed: {e})")
            continue
        n = 0
        for r in ds:
            lang = str(r.get("language", "")).lower()
            code = str(r.get("language_code", "")).lower()
            if lang == "nepali" or code in ("nep", "npi", "ne"):
                inp, tgt = r.get("inputs"), r.get("targets")
                if inp and tgt:
                    yield inp, tgt
                    n += 1
        print(f"  Aya ({repo}): {n} Nepali pairs")
        if n > 0:
            return


def _extra_dataset(repo):
    """Optional extra instruction data (e.g. alpaca-nepali — CC-BY-NC, non-commercial)."""
    from datasets import load_dataset
    ds = load_dataset(repo, split="train")
    n = 0
    for r in ds:
        instr = r.get("instruction") or r.get("inputs") or ""
        inp = r.get("input", "")
        out = r.get("output") or r.get("targets") or ""
        if instr and out:
            yield (instr + ("\n" + inp if inp else "")), out
            n += 1
    print(f"  extra ({repo}): {n} pairs")


def build_examples(sp, eod_id, max_len, extra=None):
    pairs = list(_aya_nepali())
    if extra:
        pairs += list(_extra_dataset(extra))
    ex = []
    skipped = 0
    for instr, ans in pairs:
        full, mask = PF.build_example_ids(sp, instr, ans, eod_id)
        if len(full) > max_len:
            full, mask = full[:max_len], mask[:max_len]
        if not any(mask):      # answer truncated away -> useless
            skipped += 1
            continue
        pad = max_len - len(full)
        full = full + [eod_id] * pad
        mask = mask + [False] * pad
        ex.append((full, mask))
    print(f"  usable SFT examples: {len(ex)} (skipped {skipped})")
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="out/ckpt_best.pt")
    ap.add_argument("--out", default="out/ckpt_sft.pt")
    ap.add_argument("--tok", default="data/nepali_bpe_16k.model")
    ap.add_argument("--extra", default=None, help="optional extra HF instruction dataset (may be non-commercial)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--t_eps", type=float, default=1e-3)
    ap.add_argument("--compile", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tok)
    eod_id = sp.piece_to_id("<eod>")

    print(f"Loading pretrained {args.base} ...")
    model, margs = load_pretrained(args.base, device)
    mask_id = model.mask_token_id
    n_out = margs.vocab_size + 1
    assert args.max_len <= margs.block_size

    print("Building SFT set from Aya (Nepali) ...")
    examples = build_examples(sp, eod_id, args.max_len, extra=args.extra)
    if len(examples) < 16:
        print("!! Very few Nepali SFT examples — behavior tuning will be weak. "
              "Consider --extra saillab/alpaca-nepali-cleaned (non-commercial).")
    X = torch.tensor([e[0] for e in examples], dtype=torch.long)
    M = torch.tensor([e[1] for e in examples], dtype=torch.bool)

    if args.compile:
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    use_fused = "fused" in inspect.getfullargspec(torch.optim.AdamW).args
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1, fused=use_fused)
    ctx = torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                         dtype=torch.bfloat16, enabled=(device == "cuda"))

    n = len(examples)
    steps_per_epoch = max(1, n // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, total_steps // 20)
    rng = random.Random(1337)

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        r = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 * args.lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - 0.1 * args.lr)

    def sft_loss(xb, mb):
        B, L = xb.shape
        t = torch.rand(B, device=device).clamp(min=args.t_eps, max=1.0)
        # mask ONLY response tokens, each with prob t
        mask_flags = (torch.rand(B, L, device=device) < t.unsqueeze(1)) & mb
        x_t = torch.where(mask_flags, mask_id, xb)
        with ctx:
            logits = model(x_t, t)
            ce = F.cross_entropy(logits.view(-1, n_out), xb.view(-1), reduction="none").view(B, L)
        resp = mb.float().sum(dim=1).clamp(min=1)          # per-example answer length
        per_ex = (1.0 / t) * (ce * mask_flags.float()).sum(dim=1) / resp
        return per_ex.mean()

    print(f"SFT: {n} examples | {args.epochs} epochs | {total_steps} steps")
    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for s in range(steps_per_epoch):
            idx = order[s * args.batch:(s + 1) * args.batch]
            xb = X[idx].to(device)
            mb = M[idx].to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = sft_loss(xb, mb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        # sample a test answer at each epoch
        model.eval()
        pid = PF.build_prompt_ids(sp, "नेपाली साहित्यको बारेमा छोटो वर्णन गर्नुहोस्।")
        ids = model.sample(batch_size=1, seq_len=args.max_len, temperature=1.0,
                           top_p=0.92, remask_noise=0.5, num_steps=128,
                           device=device, prompt_ids=pid)
        ans = PF.strip_response(sp.decode([t for t in ids[0].tolist() if t < mask_id]))
        model.train()
        print(f"[epoch {epoch+1}/{args.epochs}] loss {loss.item():.4f} | "
              f"{time.time()-t0:.0f}s\n   Q: नेपाली साहित्यको बारेमा...\n   A: {ans[:200]!r}")

    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": raw.state_dict(), "args": margs, "step": step,
                "sft": True, "template": PF.PROMPT_TEMPLATE}, args.out)
    print(f"\nSaved SFT model -> {args.out}")
    print("Serve it: python serve/server.py  (auto-prefers ckpt_sft.pt -> chat mode)")


if __name__ == "__main__":
    main()
