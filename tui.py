import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from model import DiffusionGPT, ModelArgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="out/ckpt.pt")
    parser.add_argument("--tokenizer", type=str, default="data/nepali_bpe_16k.model")
    parser.add_argument("--len", type=int, default=120, help="total length of tokens to decode")
    parser.add_argument("--steps", type=int, default=128, help="number of denoising steps")
    parser.add_argument("--temp", type=float, default=1.0, help="creativity temperature")
    parser.add_argument("--top_k", type=int, default=0, help="top-k pool (0 = off, use top_p)")
    parser.add_argument("--top_p", type=float, default=0.92, help="nucleus sampling threshold")
    parser.add_argument("--noise", type=float, default=0.5, help="remask gumbel noise (anti-repetition)")
    parser.add_argument("--delay", type=float, default=0.05, help="animation delay per step")
    parser.add_argument("--prompt", type=str, default="", help="prefix prompt to continue")
    # Gibberish restoration / denoising mode
    parser.add_argument("--gibberish", type=str, default="", help="corrupted text to denoise and restore")
    parser.add_argument("--noise_level", type=float, default=0.45, help="fraction of gibberish to mask")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.system("cls" if os.name == "nt" else "clear")
    print("\033[1;33m[+] INITIALIZING NEPALI DIFFUSION DENOISING CORE...\033[0m")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cli_args.tokenizer)

    ckpt = torch.load(cli_args.ckpt, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.device = device

    model = DiffusionGPT(args).to(device)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()

    mask_id = model.mask_token_id
    num_steps = cli_args.steps
    decode = lambda l: sp.decode([int(t) for t in l if int(t) < mask_id])
    def piece(i):
        p = sp.id_to_piece(int(i))
        if p == "<eod>":
            return " "
        if p.startswith("<0x") and p.endswith(">"):  # byte-fallback token -> hide
            return ""
        return p.replace("▁", " ")

    is_gibberish_mode = len(cli_args.gibberish) > 0

    if is_gibberish_mode:
        # --- GIBBERISH RESTORATION / DENOISING MODE ---
        raw_ids = sp.encode(cli_args.gibberish)
        seq_len = len(raw_ids)
        x_0_noisy = torch.tensor(raw_ids, dtype=torch.long, device=device).unsqueeze(0)
        mask_matrix = torch.rand_like(x_0_noisy.float()) < cli_args.noise_level
        x_t = torch.where(mask_matrix, mask_id, x_0_noisy)
        is_anchor = ~mask_matrix  # surviving anchors treated as context
        start_step = max(1, int(num_steps * cli_args.noise_level))
    else:
        # --- STANDARD GENERATION MODE ---
        seq_len = cli_args.len
        prompt_str = cli_args.prompt.replace("\\n", "\n")
        prompt_ids = sp.encode(prompt_str) if prompt_str else []
        prompt_len = min(len(prompt_ids), seq_len)

        x_t = torch.full((1, seq_len), mask_id, dtype=torch.long, device=device)
        is_anchor = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
        if prompt_len > 0:
            x_t[0, :prompt_len] = torch.tensor(prompt_ids[:prompt_len], dtype=torch.long, device=device)
            is_anchor[0, :prompt_len] = True
        start_step = num_steps

    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        for t_val in reversed(range(1, start_step + 1)):
            frac = t_val / num_steps
            t = torch.full((1,), frac, dtype=torch.float32, device=device)

            with torch.no_grad():
                logits = model(x_t, t) / cli_args.temp
                logits[..., mask_id] = -float("Inf")
                logits = model._filter_logits(logits, cli_args.top_k or None, cli_args.top_p)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs.view(-1, probs.size(-1)))
                x_0_pred = dist.sample().view(1, seq_len)

            currently_masked = x_t == mask_id
            chosen_probs = torch.gather(probs, 2, x_0_pred.unsqueeze(-1)).squeeze(-1)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(chosen_probs) + 1e-9) + 1e-9)
            confidence = chosen_probs + (cli_args.noise * gumbel_noise)
            confidence = confidence.masked_fill(~currently_masked, -float("inf"))

            order = torch.argsort(confidence, dim=1, descending=True)
            num_masked_per_row = currently_masked.sum(dim=1)

            if is_gibberish_mode:
                effective_ratio = (t_val - 1) / start_step * cli_args.noise_level
            else:
                effective_ratio = model.get_mask_prob((t_val - 1) / num_steps)
            num_to_keep_masked = 0 if t_val == 1 else int(math.ceil(effective_ratio * seq_len))
            num_to_unmask = (num_masked_per_row - num_to_keep_masked).clamp(min=0)

            rank = torch.arange(seq_len, device=device).unsqueeze(0)
            unmask_this_step = rank < num_to_unmask.unsqueeze(1)
            unmask_positions = torch.zeros(1, seq_len, dtype=torch.bool, device=device)
            unmask_positions.scatter_(1, order, unmask_this_step)

            x_next = torch.where(unmask_positions, x_0_pred, x_t)
            newly_unmasked = currently_masked & (x_next != mask_id)
            x_t = x_next

            # --- TUI RENDER ENGINE ---
            sys.stdout.write("\033[H")
            vram = torch.cuda.memory_allocated(device) / (1024 ** 2) if device == "cuda" else 0
            mode_title = "DENOISING NEPALI TEXT" if is_gibberish_mode else "GENERATION IN PROGRESS"

            sys.stdout.write("\033[1;32m" + "=" * 72 + "\033[0m\n")
            sys.stdout.write(f"\033[1;32m  DIFFUSION STATUS: {mode_title:<47}\033[0m\n")
            sys.stdout.write(
                f"\033[1;32m  STEP: \033[1;37m{t_val:02d}/{start_step:02d}\033[1;32m   |   MASK: "
                f"\033[1;37m{effective_ratio * 100:5.1f}%\033[1;32m   |   VRAM: \033[1;37m{vram:6.1f}MB/32GB\033[0m\n"
            )
            sys.stdout.write("\033[1;32m" + "=" * 72 + "\033[0m\n\n")

            for i in range(seq_len):
                char_idx = x_t[0, i].item()
                is_new = newly_unmasked[0, i].item()
                anchor_here = is_anchor[0, i].item()

                if char_idx == mask_id:
                    sys.stdout.write("\033[2;37m░\033[0m")  # dim mask box
                    continue

                display = piece(char_idx).replace("\n", "↵").replace("\t", "→")
                if anchor_here and is_gibberish_mode:
                    sys.stdout.write(f"\033[0;35m{display}\033[0m")   # magenta surviving anchors
                elif anchor_here:
                    sys.stdout.write(f"\033[0;33m{display}\033[0m")   # yellow prompt
                elif is_new:
                    sys.stdout.write(f"\033[1;7;36m{display}\033[0m") # cyan flash for new decode
                else:
                    sys.stdout.write(f"\033[0;32m{display}\033[0m")   # green stabilized text

            sys.stdout.write("\n\n")
            sys.stdout.flush()
            time.sleep(cli_args.delay)

        final_text = decode(x_t[0].tolist())
        sys.stdout.write("\n\033[1;33m[✓] DECODED TRANSMISSION:\033[0m\n")
        sys.stdout.write(f"\033[0;37m{final_text}\033[0m\n")

    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
