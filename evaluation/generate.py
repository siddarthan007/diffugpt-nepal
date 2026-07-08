"""Load the trained model + tokenizer, draw unconditional samples, and load a
human-written Nepali reference set (for MAUVE / gen-PPL comparison)."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import DiffusionGPT  # noqa: E402


def load_model(ckpt_path, tok_path, device=None):
    import sentencepiece as spm
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor(model_file=tok_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    args.device = device
    model = DiffusionGPT(args).to(device)
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval()
    return model, sp, device


@torch.no_grad()
def generate_samples(model, sp, n, seq_len=120, steps=96, temperature=1.0, top_k=50,
                     top_p=None, remask_noise=0.1, device="cuda", batch_size=16):
    mask_id = model.mask_token_id
    out = []
    done = 0
    while done < n:
        b = min(batch_size, n - done)
        ids = model.sample(batch_size=b, seq_len=seq_len, temperature=temperature,
                           top_k=top_k, top_p=top_p, remask_noise=remask_noise,
                           num_steps=steps, device=device)
        for row in ids.tolist():
            out.append(sp.decode([t for t in row if t < mask_id]).strip())
        done += b
        print(f"  generated {done}/{n}", end="\r", flush=True)
    print()
    return out


def load_references(path, n, min_chars=40):
    """Read up to n human docs from a held-out file (1 doc/line)."""
    refs = []
    if not os.path.exists(path):
        return refs
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if len(s) >= min_chars:
                refs.append(s)
            if len(refs) >= n:
                break
    return refs


def build_corpus_vocab(path, min_count=5, max_lines=400000):
    """Known-word lexicon from the training corpus (words with freq >= min_count).
    Used by devanagari.word_rate as a zero-dependency 'real word' dictionary."""
    from collections import Counter
    from evaluation.devanagari import _strip_word
    c = Counter()
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            for w in line.split():
                sw = _strip_word(w)
                if sw:
                    c[sw] += 1
    return {w for w, n in c.items() if n >= min_count}
