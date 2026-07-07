"""Phase A.4 — train the Nepali SentencePiece BPE tokenizer on data/nepali_clean.txt.

Choices (grounded in arXiv 2512.14585 ppl-21.8 recipe + 2404.18071 tokenizer study):
  - BPE, monolingual Nepali, vocab 16,384 (embedding-param-budget choice for ~50M)
  - byte_fallback -> no OOV; <eod> control symbol = document separator used by prepare.py
  - normalization_rule_name=identity (clean.py already did NFC + IndicNLP)
Docs are chunked to <=MAX_TOK_LINE_CHARS so long articles are not skipped by
SentencePiece's max_sentence_length.

Optional fertility sweep:  python train_tokenizer.py --sweep 16384 24576 32000
"""
import argparse
import os

import sentencepiece as spm

import config as C

CHUNKS = os.path.join(C.DATA, "_tok_train_chunks.txt")


def build_chunks():
    """Split merged docs into <=MAX_TOK_LINE_CHARS pieces (on Nepali danda where possible)."""
    n = 0
    with open(C.MERGED, encoding="utf-8") as fin, open(CHUNKS, "w", encoding="utf-8") as fout:
        for line in fin:
            doc = line.rstrip("\n")
            while len(doc) > C.MAX_TOK_LINE_CHARS:
                cut = doc.rfind("।", 0, C.MAX_TOK_LINE_CHARS)
                if cut <= 0:
                    cut = doc.rfind(" ", 0, C.MAX_TOK_LINE_CHARS)
                if cut <= 0:
                    cut = C.MAX_TOK_LINE_CHARS
                fout.write(doc[:cut + 1].strip() + "\n")
                doc = doc[cut + 1:].strip()
                n += 1
            if doc:
                fout.write(doc + "\n")
                n += 1
    print(f"tokenizer training lines: {n:,} -> {CHUNKS}")


def train(vocab_size, prefix):
    spm.SentencePieceTrainer.train(
        input=CHUNKS,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="identity",
        control_symbols=["<eod>"],
        unk_id=0,
        input_sentence_size=8_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 16,
    )
    print(f"[vocab={vocab_size}] wrote {prefix}.model")


def fertility(model_path, sample_lines=20000):
    sp = spm.SentencePieceProcessor(model_file=model_path)
    toks = words = 0
    with open(C.LIT_TEST if os.path.exists(C.LIT_TEST) else CHUNKS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            line = line.strip()
            words += line.count(" ") + 1
            toks += len(sp.encode(line))
    return toks / max(1, words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, nargs="*", default=None,
                    help="vocab sizes to compare on held-out fertility (tokens/word)")
    args = ap.parse_args()

    build_chunks()

    if args.sweep:
        print("\n=== fertility sweep (lower tokens/word = better compression) ===")
        for v in args.sweep:
            pfx = os.path.join(C.DATA, f"_sweep_{v}")
            train(v, pfx)
            print(f"  vocab {v:6d} -> fertility {fertility(pfx + '.model'):.3f} tokens/word")
        print("Pick a knee; set VOCAB_SIZE in config.py + ModelArgs, then rerun without --sweep.")
        return

    train(C.VOCAB_SIZE, C.TOK_PREFIX)
    print(f"held-out fertility: {fertility(C.TOK_MODEL):.3f} tokens/word")


if __name__ == "__main__":
    main()
