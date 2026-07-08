"""Intrinsic Devanagari validity — no GPU, no model. The cheapest honest signal:
is the model even emitting well-formed Nepali script?

Checks per sample:
  - foreign-char ratio (non-Devanagari letters leaking in)
  - orphan combining marks (a matra/virama/anusvara not attached to a base) — the
    classic sign of a model that hasn't learned Devanagari cluster structure
  - NFC stability
A sample is "well-formed" if it has no orphan combining marks, foreign-letter ratio
below a threshold, and is NFC-stable.
"""
import unicodedata

# --- Devanagari codepoint classes (U+0900..U+097F) ---
_INDEP_VOWEL = set(range(0x0904, 0x0915)) | {0x0960, 0x0961, 0x0972, 0x0973, 0x0974,
                                             0x0975, 0x0976, 0x0977}
_CONSONANT = set(range(0x0915, 0x093A)) | set(range(0x0958, 0x0960)) | set(range(0x0978, 0x0980))
_OM = {0x0950}
_DIGIT = set(range(0x0966, 0x0970))
_DANDA = {0x0964, 0x0965, 0x0970}                     # । ॥ ॰
# combining / dependent (must attach to a preceding base or combining):
_COMBINING = (set(range(0x0900, 0x0904))              # anusvara/visarga/candrabindu
              | {0x093A, 0x093B, 0x093C}              # signs + nukta
              | set(range(0x093E, 0x094D))            # vowel signs (matras)
              | {0x094D}                              # virama
              | set(range(0x094E, 0x0950))
              | set(range(0x0951, 0x0958))            # accents/svaras
              | set(range(0x0962, 0x0964)))           # vocalic matras
_BASE = _INDEP_VOWEL | _CONSONANT | _OM
_DEVANAGARI = _BASE | _COMBINING | _DIGIT | _DANDA
_ALLOWED_PUNCT = set(" \n\t.,!?;:\"'()[]{}-–—…%/‍‌") | set("0123456789")

FOREIGN_RATIO_MAX = 0.05   # >5% foreign letters => not well-formed Nepali
MIN_DEVANAGARI_RATIO = 0.5  # must be majority Devanagari (rejects space/underscore/foreign junk)


def _is_foreign_letter(ch):
    o = ord(ch)
    if o in _DEVANAGARI or ch in _ALLOWED_PUNCT:
        return False
    return ch.isalpha()  # Latin/other scripts leaking in


def analyze(text):
    n = len(text)
    if n == 0:
        return {"chars": 0, "devanagari_ratio": 0.0, "foreign_ratio": 0.0,
                "orphan_combining": 0, "nfc_stable": True, "wellformed": False}
    deva = foreign = orphan = 0
    prev_attachable = False  # prev char was a base or combining (so a matra may follow)
    for ch in text:
        o = ord(ch)
        if o in _DEVANAGARI:
            deva += 1
        if o in _COMBINING:
            if not prev_attachable:
                orphan += 1
        if _is_foreign_letter(ch):
            foreign += 1
        prev_attachable = (o in _BASE) or (o in _COMBINING)
    nonspace = sum(1 for c in text if not c.isspace()) or 1
    foreign_ratio = foreign / nonspace
    nfc = unicodedata.normalize("NFC", text) == text
    deva_ratio = deva / nonspace
    wellformed = (orphan == 0) and (foreign_ratio <= FOREIGN_RATIO_MAX) and nfc \
        and (deva_ratio >= MIN_DEVANAGARI_RATIO)
    return {
        "chars": n,
        "devanagari_ratio": deva / nonspace,
        "foreign_ratio": foreign_ratio,
        "orphan_combining": orphan,
        "nfc_stable": nfc,
        "wellformed": wellformed,
    }


def report(texts):
    if not texts:
        return {}
    a = [analyze(t) for t in texts]
    total_chars = sum(x["chars"] for x in a) or 1
    return {
        "n_samples": len(texts),
        "pct_wellformed": 100.0 * sum(x["wellformed"] for x in a) / len(a),
        "mean_devanagari_ratio": sum(x["devanagari_ratio"] for x in a) / len(a),
        "mean_foreign_ratio": sum(x["foreign_ratio"] for x in a) / len(a),
        "orphan_combining_per_1k": 1000.0 * sum(x["orphan_combining"] for x in a) / total_chars,
        "pct_nfc_stable": 100.0 * sum(x["nfc_stable"] for x in a) / len(a),
    }


# --- dictionary / vocabulary word rate ---
def _strip_word(w):
    return "".join(c for c in w if ord(c) in _BASE or ord(c) in _COMBINING)


def word_rate(texts, vocab):
    """Fraction of Devanagari words that appear in `vocab` (a set of known words).
    vocab is typically built from the training corpus (words with freq >= K)."""
    total = hit = 0
    for t in texts:
        for raw in t.split():
            w = _strip_word(raw)
            if not w:
                continue
            total += 1
            if w in vocab:
                hit += 1
    return {"word_rate_pct": (100.0 * hit / total) if total else 0.0, "words_checked": total}
