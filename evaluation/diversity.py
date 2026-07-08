"""Text diversity metrics — the mandatory companion to gen-PPL.

gen-PPL is trivially gamed by low-entropy repetitive text (a model that outputs the
same fluent phrase forever scores great). Always pair it with diversity + a repetition
measure (Holtzman et al. 2020; Shaib et al. 2024). Word-level (whitespace) n-grams.
"""
from collections import Counter


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts, n):
    """Corpus-level distinct-n: unique n-grams / total n-grams across all samples."""
    total = 0
    uniq = set()
    for t in texts:
        toks = t.split()
        gs = _ngrams(toks, n)
        total += len(gs)
        uniq.update(gs)
    return (len(uniq) / total) if total else 0.0


def rep_n(texts, n):
    """Mean per-sample repetition: 1 - (unique n-grams / total n-grams). Higher = more
    repetitive (degenerate). rep_n -> 1.0 means the sample loops the same n-gram."""
    vals = []
    for t in texts:
        toks = t.split()
        gs = _ngrams(toks, n)
        if not gs:
            continue
        vals.append(1.0 - len(set(gs)) / len(gs))
    return (sum(vals) / len(vals)) if vals else 0.0


def self_bleu(texts, sample=60):
    """Mean BLEU of each sample vs the rest — high self-BLEU = samples are near-copies
    (low diversity). Uses sacrebleu if available; returns None otherwise."""
    try:
        from sacrebleu.metrics import BLEU
    except Exception:
        return None
    if len(texts) < 3:
        return None
    texts = texts[:sample]
    bleu = BLEU(effective_order=True)
    scores = []
    for i, hyp in enumerate(texts):
        refs = [texts[j] for j in range(len(texts)) if j != i]
        try:
            scores.append(bleu.sentence_score(hyp, refs).score)
        except Exception:
            continue
    return (sum(scores) / len(scores)) if scores else None


def report(texts):
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "distinct_4": distinct_n(texts, 4),
        "rep_3": rep_n(texts, 3),
        "rep_4": rep_n(texts, 4),
        "self_bleu": self_bleu(texts),
    }
