#!/usr/bin/env bash
# Phase A — build the Nepali corpus + tokenizer + token bins. All off-GPU.
# Run from the repo root:  bash data_pipeline/run_phase_a.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== A.1 download =="        && python download.py
echo "== A.2 clean =="           && python clean.py
echo "== A.3 dedup + mix =="     && python dedup_mix.py
echo "== A.4 train tokenizer ==" && python train_tokenizer.py
echo "== A.5 prepare bins =="    && python prepare.py

echo
echo "Phase A done. Artifacts in data/: nepali_bpe_16k.model, train.bin, val.bin, test.bin, meta.json"
echo "Next (on the RTX 5090):  python train.py"
