# diffugpt-nepal

A small **masked discrete-diffusion** language model specialized for **Nepali language & literature** (Devanagari), designed to train on a single **RTX 5090 (32GB)** within a **$10 / ~10 GPU-hour** budget — plus a ChatGPT-style web page with a Hollywood-style denoising visualization.

Method = absorbing-state masked diffusion (MDLM / MaskGIT family): the training loss is a `1/t`-weighted average of masked-token cross-entropy, sampling is confidence-based iterative unmasking. See the design rationale + citations in the project notes.

## Model (~50M params)
`vocab=16384, n_embd=512, n_layer=12, n_head=8, block_size=512`, RoPE + RMSNorm + SwiGLU, bidirectional attention, continuous-time diffusion `t∈(0,1]`, dedicated `[MASK]` at id `vocab_size`.

## Layout
```
model.py                 DiffusionGPT + sample_stream() generator (feeds TUI + web UI)
train.py                 continuous-time MDLM training (reads data/*.bin)
tui.py                   terminal Hollywood denoising visualization
data_pipeline/           Phase-A off-GPU corpus + tokenizer build
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# Phase A — build corpus + tokenizer + token bins (off-GPU, free)
bash data_pipeline/run_phase_a.sh          # -> data/{nepali_bpe_16k.model,train.bin,val.bin,test.bin}
# optional: pick vocab by fertility first
#   python data_pipeline/train_tokenizer.py --sweep 16384 24576 32000

# Phase C — train (RTX 5090)
python train.py                            # -> out/ckpt.pt

# visualize the denoising in the terminal
python tui.py --prompt "नेपाली साहित्य" --steps 64
python tui.py --gibberish "<corrupted nepali text>" --noise_level 0.45
```

## Data sources (all verified genuinely Nepali; licenses in `data_pipeline/config.py`)
- **IRIISNEPAL/Nepali-Text-Corpus** (MIT) — primary, clean edited prose
- **wikimedia/wikipedia `20231101.ne`** (CC-BY-SA) — encyclopedic
- **ai4bharat/IndicCorpV2 `npi_Deva`** (CC0) — breadth
- **Devkota poems** (public-domain author) + **nepali-textbooks-corpus** — literature, up-weighted

Gated sources need `huggingface-cli login`; any failing source is skipped, not fatal.

## Web UI

```bash
bash run.sh                # picks best ckpt, prints URL, starts server (Ctrl-C to stop)
bash run.sh --tmux         # detached (survives SSH drop); attach: tmux attach -t nepali
bash run.sh --port 7860 --install
# or directly:
python serve/server.py     # http://localhost:8000
```
On RunPod, `run.sh` prints the `https://<POD_ID>-<PORT>.proxy.runpod.net` URL — expose that port as HTTP.
FastAPI + WebSocket streams each denoising step to the browser. Auto-detects mode:
**LIVE** if `out/ckpt.pt` + tokenizer exist, else **DEMO** (synthetic denoise over sample
Nepali text — the UI works before training). Editorial paper aesthetic, dark denoising
stage with per-akshara reveal animation + a correctly-shaped reading panel below.

## Evaluation

```bash
python -m evaluation.run --ckpt out/ckpt_best.pt --n 300        # full suite
python -m evaluation.run --ckpt out/ckpt.pt --n 100 --quick     # offline metrics only (no GPU scorers)
```
Reports, honestly (gen-PPL never shown alone):
- **Devanagari validity** — % well-formed script, NFC-stable, orphan-combining rate (target >99%)
- **Dictionary word-rate** — % generated words in the corpus lexicon (zero-dependency)
- **Diversity** — distinct-1/2/3, rep-3/4, self-BLEU (catches degenerate repetition that gen-PPL rewards)
- **gen-PPL** — samples scored by frozen AR `Sakonii/distilgpt2-nepali`, shown next to the human-reference band
- **MAUVE** — distribution similarity to held-out human Nepali, MuRIL-featurized (1.0 = indistinguishable)

## Chat (optional instruction-tuning)

```bash
python finetune.py                       # SFT on Aya Nepali (Apache-2.0) -> out/ckpt_sft.pt
python finetune.py --extra saillab/alpaca-nepali-cleaned   # +52k pairs (CC-BY-NC, non-commercial)
python serve/server.py                   # auto-prefers ckpt_sft.pt -> CHAT mode
```
LLaDA recipe: only the **response** tokens are masked/supervised; the prompt stays clean.
At inference the prompt is given as anchor tokens and the model denoises the answer. Serving
auto-detects the SFT checkpoint, wraps input in the `### निर्देशन: / ### उत्तर:` template, and
shows only the answer.

## Status
- [x] Model + training + TUI (from-scratch masked diffusion, ~50M)
- [x] Phase-A data/tokenizer pipeline
- [x] Trained 40k · resume · WSD · spot-safe checkpointing
- [x] Phase-D eval suite (validity · diversity · gen-PPL · MAUVE)
- [x] Sampler tuned (nucleus + remask-noise) → near-human diversity, no retrain
- [x] FastAPI + WebSocket serving + web denoising viz (conjoined Devanagari)
- [x] Optional SFT (Aya) for chat-style behavior
- [ ] `python serve/server.py` on pod → LIVE / CHAT
