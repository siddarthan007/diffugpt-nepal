"""Phase A.3 — cross-source MinHash+LSH fuzzy dedup, hold out a literature test
set, then oversample-mix into data/nepali_clean.txt.

News corpora reprint articles across sites (this is why OSCAR-ne was dropped by the
Nepali GPT paper), so document-level fuzzy dedup @0.8 Jaccard is the single biggest
quality lever. Literature/wiki docs are up-weighted by repetition (config weights).
"""
import os
import random

from datasketch import MinHash, MinHashLSH

import config as C

random.seed(C.SEED)


def shingles(doc, k=C.SHINGLE_K):
    words = doc.split()
    if len(words) < k:
        return {doc}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def minhash(doc):
    m = MinHash(num_perm=C.MINHASH_PERM)
    for s in shingles(doc):
        m.update(s.encode("utf-8"))
    return m


def main():
    lsh = MinHashLSH(threshold=C.MINHASH_THRESHOLD, num_perm=C.MINHASH_PERM)
    kept = []            # (source, doc, weight)
    heldout = []         # literature test docs
    uid = 0
    stats = {}

    for name, spec in C.SOURCES.items():
        path = os.path.join(C.CLEAN, f"{name}.txt")
        if not os.path.exists(path):
            continue
        w = spec["weight"]
        hold = spec.get("holdout", 0.0)
        k = d = h = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                doc = line.rstrip("\n")
                if not doc:
                    continue
                m = minhash(doc)
                if lsh.query(m):        # near-duplicate already present
                    d += 1
                    continue
                lsh.insert(str(uid), m)
                uid += 1
                if hold > 0 and random.random() < hold:
                    heldout.append(doc)
                    h += 1
                else:
                    kept.append((name, doc, w))
                    k += 1
        stats[name] = {"kept": k, "dup_dropped": d, "heldout": h}
        print(f"[{name}] unique {k:,} | dup-dropped {d:,} | heldout {h:,}")

    # oversample-mix (repeat each doc `weight` times) and shuffle
    mixed = []
    for name, doc, w in kept:
        mixed.extend([doc] * w)
    random.shuffle(mixed)

    with open(C.MERGED, "w", encoding="utf-8") as f:
        for doc in mixed:
            f.write(doc + "\n")
    with open(C.LIT_TEST, "w", encoding="utf-8") as f:
        for doc in heldout:
            f.write(doc + "\n")

    total_chars = sum(len(d) for d in mixed)
    total_words = sum(d.count(" ") + 1 for d in mixed)
    print(f"\nMerged corpus: {len(mixed):,} docs (post-oversample) | "
          f"{total_chars/1e6:.0f}M chars | ~{total_words/1e6:.0f}M words -> {C.MERGED}")
    print(f"Est. tokens @ ~1.7 fertility: ~{total_words*1.7/1e6:.0f}M token-passes/epoch")
    print(f"Held-out literature test: {len(heldout):,} docs -> {C.LIT_TEST}")

    # per-source share of the mixed corpus (by repeated docs)
    from collections import Counter
    share = Counter()
    for name, doc, w in kept:
        share[name] += w
    tot = sum(share.values())
    print("Mix shares (by doc count):", {k: f"{v/tot*100:.0f}%" for k, v in share.items()})


if __name__ == "__main__":
    main()
