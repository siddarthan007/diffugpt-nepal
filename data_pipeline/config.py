"""Shared config + paths for the Phase-A Nepali data pipeline (all off-GPU)."""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data")
RAW = os.path.join(DATA, "raw")
CLEAN = os.path.join(DATA, "clean")

MERGED = os.path.join(
    DATA, "nepali_clean.txt"
)  # deduped + mixed training corpus (1 doc/line)
LIT_TEST = os.path.join(
    DATA, "literature_test.txt"
)  # held-out clean literature (never trained on)
TOK_PREFIX = os.path.join(DATA, "nepali_bpe_16k")  # -> .model / .vocab
TOK_MODEL = TOK_PREFIX + ".model"
META = os.path.join(DATA, "meta.json")

VOCAB_SIZE = 16384  # real tokens; MASK lives at id==VOCAB_SIZE (outside SentencePiece)
VAL_FRACTION = 0.01

# --- source registry ---------------------------------------------------------
# max_chars: raw download budget per source (tune down for a faster first pass).
# min_dev_ratio: fraction of non-space chars that must be Devanagari to keep a doc.
# weight: oversampling factor when mixing (literature up-weighted).
# holdout: fraction diverted to the held-out literature test set.
SOURCES = {
    "iriisnepal": {
        "loader": ("hf", {"path": "IRIISNEPAL/Nepali-Text-Corpus", "split": "train"}),
        "text_col": "Article",
        "max_chars": 750_000_000,
        "min_dev_ratio": 0.35,
        "weight": 1,
        "holdout": 0.01,
    },
    "wikipedia": {
        "loader": (
            "hf",
            {"path": "wikimedia/wikipedia", "name": "20231101.ne", "split": "train"},
        ),
        "text_col": "text",
        "max_chars": 150_000_000,
        "min_dev_ratio": 0.35,
        "weight": 2,
        "holdout": 0.02,
    },
    "indiccorp": {
        # Nepali is a single plain-text file data/ne.txt (verified via HF tree API),
        # loaded through the generic text builder — NOT a data_dir config.
        "loader": (
            "hf",
            {
                "path": "ai4bharat/IndicCorpV2",
                "data_files": "data/ne.txt",
                "split": "train",
            },
        ),
        "text_col": "text",
        "max_chars": 100_000_000,
        "min_dev_ratio": 0.35,
        "weight": 1,
        "holdout": 0.01,
    },
    "textbooks": {
        "loader": (
            "hf",
            {"path": "dineshkarki/nepali-textbooks-corpus", "split": "train"},
        ),
        "text_col": "text",
        "max_chars": 60_000_000,
        "min_dev_ratio": 0.20,  # lighter: literary/poetic line breaks
        "weight": 4,
        "holdout": 0.10,
    },
    "devkota": {
        "loader": (
            "url",
            {
                "url": "https://raw.githubusercontent.com/devkotasawal1/Poem-Generator/master/lspd.txt"
            },
        ),
        "text_col": None,
        "max_chars": 10_000_000,
        "min_dev_ratio": 0.20,
        "weight": 4,  # public-domain literature, up-weight hard
        "holdout": 0.15,
    },
}

MINHASH_PERM = 64
MINHASH_THRESHOLD = 0.80
SHINGLE_K = 5
MIN_DOC_CHARS = 60  # drop docs shorter than this after cleaning
MAX_TOK_LINE_CHARS = 2000  # chunk size when feeding docs to the SentencePiece trainer
SEED = 1337

for _d in (DATA, RAW, CLEAN):
    os.makedirs(_d, exist_ok=True)
