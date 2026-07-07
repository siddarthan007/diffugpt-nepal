"""Phase A.5 — tokenize the corpus into uint16 token streams for training.

Writes:
  data/train.bin , data/val.bin   (from data/nepali_clean.txt, 99/1 split)
  data/test.bin                    (from held-out data/literature_test.txt)
  data/meta.json                   (vocab size, mask id, token counts)

Docs are separated by the <eod> control token so the model can learn boundaries.
Tokens are uint16 (vocab 16k < 65535); MASK (id==vocab_size) never appears here.
"""
import json
import os

import numpy as np
import sentencepiece as spm

import config as C


def load_sp():
    sp = spm.SentencePieceProcessor(model_file=C.TOK_MODEL)
    eod = sp.piece_to_id("<eod>")
    return sp, eod


def tokenize_file(sp, eod, path):
    arrs = []
    if not os.path.exists(path):
        return np.zeros(0, dtype=np.uint16)
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            ids = sp.encode(line)
            ids.append(eod)
            arrs.append(np.asarray(ids, dtype=np.uint16))
            if (i + 1) % 100000 == 0:
                print(f"  tokenized {i+1:,} docs", flush=True)
    if not arrs:
        return np.zeros(0, dtype=np.uint16)
    return np.concatenate(arrs)


def main():
    sp, eod = load_sp()
    vocab = sp.get_piece_size()
    assert vocab == C.VOCAB_SIZE, f"tokenizer vocab {vocab} != config {C.VOCAB_SIZE}"

    print("Tokenizing merged corpus ...")
    stream = tokenize_file(sp, eod, C.MERGED)
    n = len(stream)
    n_val = int(n * C.VAL_FRACTION)
    train, val = stream[:n - n_val], stream[n - n_val:]
    train.tofile(os.path.join(C.DATA, "train.bin"))
    val.tofile(os.path.join(C.DATA, "val.bin"))

    print("Tokenizing held-out literature test set ...")
    test = tokenize_file(sp, eod, C.LIT_TEST)
    test.tofile(os.path.join(C.DATA, "test.bin"))

    meta = {
        "vocab_size": C.VOCAB_SIZE,
        "mask_token_id": C.VOCAB_SIZE,
        "eod_id": int(eod),
        "train_tokens": int(len(train)),
        "val_tokens": int(len(val)),
        "test_tokens": int(len(test)),
        "tokenizer": os.path.basename(C.TOK_MODEL),
    }
    with open(C.META, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Phase A complete ===")
    print(f"  train.bin  {len(train):,} tokens")
    print(f"  val.bin    {len(val):,} tokens")
    print(f"  test.bin   {len(test):,} tokens (held-out literature)")
    print(f"  meta.json  {meta}")
    print("Now run:  python train.py")


if __name__ == "__main__":
    main()
