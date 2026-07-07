"""Phase A.2 — clean + normalize each raw source into data/clean/{source}.txt.

Pipeline per doc (mirrors arXiv 2512.14585 + CulturaX):
  HTML/URL/email strip -> Unicode NFC -> IndicNLP DevanagariNormalizer
  -> whitespace normalize -> Devanagari-ratio filter -> length filter
  -> exact-line dedup within the source.

Legacy Preeti/Kantipur font garble (ASCII-encoded "Devanagari") is Latin text,
so it is removed automatically by the Devanagari-ratio filter.
"""
import html
import os
import re
import unicodedata

import config as C

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_WS = re.compile(r"\s+")
_DEVA = re.compile(r"[ऀ-ॿ]")

# --- IndicNLP Devanagari normalizer (graceful fallback to NFC-only) -----------
try:
    from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
    _norm = IndicNormalizerFactory().get_normalizer("ne")
    def indic_normalize(t):
        return _norm.normalize(t)
except Exception:
    try:
        from indicnlp.normalize.indic_normalize import DevanagariNormalizer
        _dn = DevanagariNormalizer()
        def indic_normalize(t):
            return _dn.normalize(t)
    except Exception:
        print("(indic-nlp-library unavailable — NFC-only normalization)")
        def indic_normalize(t):
            return t


def devanagari_ratio(s):
    non_space = [c for c in s if not c.isspace()]
    if not non_space:
        return 0.0
    return len(_DEVA.findall(s)) / len(non_space)


def clean_doc(doc):
    doc = html.unescape(doc)
    doc = _TAG.sub(" ", doc)
    doc = _URL.sub(" ", doc)
    doc = _EMAIL.sub(" ", doc)
    doc = unicodedata.normalize("NFC", doc)
    doc = indic_normalize(doc)
    doc = _WS.sub(" ", doc).strip()
    return doc


def clean_source(name, spec):
    src = os.path.join(C.RAW, f"{name}.txt")
    if not os.path.exists(src):
        print(f"[{name}] no raw file -> skip")
        return
    dst = os.path.join(C.CLEAN, f"{name}.txt")
    seen = set()
    kept = dropped = 0
    thr = spec["min_dev_ratio"]
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            doc = clean_doc(line)
            if len(doc) < C.MIN_DOC_CHARS:
                dropped += 1
                continue
            if devanagari_ratio(doc) < thr:
                dropped += 1
                continue
            h = hash(doc)
            if h in seen:
                dropped += 1
                continue
            seen.add(h)
            fout.write(doc + "\n")
            kept += 1
    print(f"[{name}] kept {kept:,} | dropped {dropped:,} -> {dst}")


def main():
    for name, spec in C.SOURCES.items():
        clean_source(name, spec)


if __name__ == "__main__":
    main()
