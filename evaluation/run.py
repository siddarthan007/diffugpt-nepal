"""Phase-D evaluation runner. Generates samples from a checkpoint, then reports
Devanagari validity + dictionary word-rate + diversity + gen-PPL + MAUVE, honestly
(gen-PPL is always shown next to diversity and the human-reference gen-PPL band).

Run (on the pod, after training):
  python -m evaluation.run --ckpt out/ckpt_best.pt --n 300
  python -m evaluation.run --ckpt out/ckpt.pt --n 100 --quick   # fast, skips MAUVE
"""
import argparse
import json
import os

from evaluation import devanagari, diversity, generate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/ckpt_best.pt")
    ap.add_argument("--tok", default="data/nepali_bpe_16k.model")
    ap.add_argument("--ref", default="data/literature_test.txt")
    ap.add_argument("--corpus", default="data/nepali_clean.txt", help="lexicon source for word-rate")
    ap.add_argument("--n", type=int, default=300, help="number of samples to generate")
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--scorer", default="Sakonii/distilgpt2-nepali")
    ap.add_argument("--quick", action="store_true", help="skip gen-PPL + MAUVE (offline metrics only)")
    ap.add_argument("--out", default="out/eval_report.json")
    args = ap.parse_args()

    print(f"Loading model from {args.ckpt} ...")
    model, sp, device = generate.load_model(args.ckpt, args.tok)
    step = None
    try:
        import torch
        step = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("step")
    except Exception:
        pass

    print(f"Generating {args.n} samples (seq_len={args.seq_len}, steps={args.steps}, "
          f"temp={args.temp}, top_k={args.top_k}) ...")
    gen = generate.generate_samples(model, sp, args.n, seq_len=args.seq_len, steps=args.steps,
                                    temperature=args.temp, top_k=args.top_k, device=device)
    refs = generate.load_references(args.ref, args.n)
    print(f"  references loaded: {len(refs)}")

    report = {"ckpt": args.ckpt, "step": step, "n_samples": len(gen),
              "gen_config": {"seq_len": args.seq_len, "steps": args.steps,
                             "temp": args.temp, "top_k": args.top_k}}

    # --- offline: Devanagari validity + word-rate + diversity ---
    report["devanagari"] = devanagari.report(gen)
    vocab = generate.build_corpus_vocab(args.corpus)
    if vocab:
        report["word_rate"] = devanagari.word_rate(gen, vocab)
        report["word_rate"]["lexicon_size"] = len(vocab)
    report["diversity"] = diversity.report(gen)
    if refs:  # human reference baseline for the same offline metrics
        report["reference_devanagari"] = devanagari.report(refs)
        report["reference_diversity"] = diversity.report(refs)

    # --- model-based: gen-PPL + MAUVE ---
    if not args.quick:
        from evaluation import scorers
        print("Scoring gen-PPL (Sakonii/distilgpt2-nepali) ...")
        report["gen_ppl"] = scorers.gen_perplexity(gen, model_name=args.scorer, device=device)
        if refs:
            report["gen_ppl_reference"] = scorers.gen_perplexity(refs, model_name=args.scorer, device=device)
        if refs:
            print("Scoring MAUVE ...")
            report["mauve"] = scorers.mauve_score(gen, refs, device=device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    _print_report(report, gen)
    print(f"\nSaved -> {args.out}")


def _print_report(r, gen):
    d = r.get("devanagari", {})
    div = r.get("diversity", {})
    print("\n" + "=" * 62)
    print(f" NEPALI DIFFUSION — EVAL  (step {r.get('step')}, n={r.get('n_samples')})")
    print("=" * 62)
    print(" Devanagari validity")
    print(f"   well-formed .............. {d.get('pct_wellformed', 0):6.1f} %   (target >99)")
    print(f"   NFC-stable ............... {d.get('pct_nfc_stable', 0):6.1f} %")
    print(f"   devanagari ratio ......... {d.get('mean_devanagari_ratio', 0):6.3f}")
    print(f"   foreign-letter ratio ..... {d.get('mean_foreign_ratio', 0):6.3f}")
    print(f"   orphan combining /1k ..... {d.get('orphan_combining_per_1k', 0):6.2f}")
    if "word_rate" in r:
        wr = r["word_rate"]
        print(f"   dict word-rate ........... {wr.get('word_rate_pct', 0):6.1f} %   "
              f"(lexicon {wr.get('lexicon_size', 0):,}, {wr.get('words_checked', 0):,} words)")
    print(" Diversity  (higher distinct = better; higher rep = degenerate)")
    print(f"   distinct-1/2/3 ........... {div.get('distinct_1', 0):.3f} / {div.get('distinct_2', 0):.3f} / {div.get('distinct_3', 0):.3f}")
    print(f"   rep-3 / rep-4 ............ {div.get('rep_3', 0):.3f} / {div.get('rep_4', 0):.3f}")
    if div.get("self_bleu") is not None:
        print(f"   self-BLEU ................ {div.get('self_bleu'):.1f}   (lower = more diverse)")
    if "reference_diversity" in r:
        rd = r["reference_diversity"]
        print(f"   [human ref distinct-2 .... {rd.get('distinct_2', 0):.3f}]")
    if "gen_ppl" in r:
        print(" Fluency (gen-PPL under Sakonii/distilgpt2-nepali; read WITH diversity)")
        print(f"   gen-PPL (model) .......... {r.get('gen_ppl'):8.1f}")
        if "gen_ppl_reference" in r:
            print(f"   gen-PPL (human ref) ...... {r.get('gen_ppl_reference'):8.1f}   <- target band")
    if "mauve" in r:
        m = r["mauve"]
        if m.get("mauve") is not None:
            print(f" MAUVE (vs human) ........... {m['mauve']:.4f}   (1.0 = indistinguishable, feat={m.get('featurizer')})")
        else:
            print(f" MAUVE ..................... n/a ({m.get('error')})")
    print("=" * 62)
    print(" sample:")
    for s in gen[:3]:
        print(f"   • {s[:110]}")
    print("=" * 62)


if __name__ == "__main__":
    main()
