"""Phase A.1 — download raw Nepali text to data/raw/{source}.txt (one doc per line).

Streams from HuggingFace with a per-source character budget so nothing has to fit
in RAM. Each source is independent: a failure (gated/offline/renamed) is logged and
skipped, not fatal. Re-running skips sources whose raw file already exists.
"""
import os
import sys

import config as C


def _write_docs(out_path, doc_iter, max_chars, label):
    written = 0
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for doc in doc_iter:
            if not doc:
                continue
            doc = " ".join(doc.split())  # collapse internal whitespace -> 1 doc per line
            if not doc:
                continue
            f.write(doc + "\n")
            written += len(doc)
            n += 1
            if n % 20000 == 0:
                print(f"  [{label}] {n:,} docs / {written/1e6:.1f}M chars", flush=True)
            if written >= max_chars:
                break
    print(f"  [{label}] DONE: {n:,} docs / {written/1e6:.1f}M chars -> {out_path}", flush=True)
    return n, written


def _hf_iter(kw, text_col, max_chars):
    from datasets import load_dataset
    ds = load_dataset(streaming=True, **kw)
    col = text_col
    for i, ex in enumerate(ds):
        if col is None:  # auto-detect first string column
            col = next((k for k, v in ex.items() if isinstance(v, str) and len(v) > 20), None)
            if col is None:
                continue
        yield ex.get(col, "")


def _url_iter(url, max_chars):
    import requests
    txt = requests.get(url, timeout=60).text
    # split a poem file into stanza-ish docs on blank lines
    buf = []
    for line in txt.splitlines():
        if line.strip():
            buf.append(line.strip())
        elif buf:
            yield " ".join(buf)
            buf = []
    if buf:
        yield " ".join(buf)


def download_source(name, spec):
    out_path = os.path.join(C.RAW, f"{name}.txt")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[{name}] already downloaded -> skip", flush=True)
        return
    kind, kw = spec["loader"]
    print(f"[{name}] downloading ({kind}) budget={spec['max_chars']/1e6:.0f}M chars ...", flush=True)
    try:
        if kind == "hf":
            it = _hf_iter(kw, spec["text_col"], spec["max_chars"])
        elif kind == "url":
            it = _url_iter(kw["url"], spec["max_chars"])
        else:
            raise ValueError(f"unknown loader {kind}")
        _write_docs(out_path, it, spec["max_chars"], name)
    except Exception as e:
        print(f"[{name}] SKIPPED — {type(e).__name__}: {e}", flush=True)
        if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
            os.remove(out_path)


def main():
    only = sys.argv[1:] or list(C.SOURCES.keys())
    for name in only:
        download_source(name, C.SOURCES[name])
    print("\nDownload phase complete. Present raw files:")
    for fn in sorted(os.listdir(C.RAW)):
        p = os.path.join(C.RAW, fn)
        print(f"  {fn:20s} {os.path.getsize(p)/1e6:8.1f} MB")


if __name__ == "__main__":
    main()
