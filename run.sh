#!/usr/bin/env bash
# One-command launcher for the Nepali diffusion web app.
#   bash run.sh                 # foreground, port 8000
#   bash run.sh --port 7860     # custom port
#   bash run.sh --tmux          # detached in tmux session 'nepali' (survives SSH drop)
#   bash run.sh --install       # pip install -r requirements.txt first
# Env: PORT, PYTHON override defaults.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8000}"
PY="${PYTHON:-python}"
TMUX_MODE=0
INSTALL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --tmux|-d) TMUX_MODE=1; shift ;;
    --install) INSTALL=1; shift ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p out
[ "$INSTALL" = 1 ] && $PY -m pip install -r requirements.txt

# report which checkpoint the server will pick (it auto-selects the same order)
CKPT=""
for c in out/ckpt_sft.pt out/ckpt_best.pt out/ckpt.pt; do
  [ -f "$c" ] && { CKPT="$c"; break; }
done
if [ -z "$CKPT" ]; then
  echo "!! no checkpoint in out/  -> DEMO mode"
else
  case "$CKPT" in
    *ckpt_sft.pt) echo ">> checkpoint: $CKPT  (LIVE + CHAT)" ;;
    *)            echo ">> checkpoint: $CKPT  (LIVE)" ;;
  esac
fi
[ -f data/nepali_bpe_16k.model ] || echo "!! tokenizer data/nepali_bpe_16k.model missing -> DEMO mode"

echo ">> port: $PORT"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo ">> RunPod URL: https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net  (expose $PORT as HTTP in the pod)"
else
  echo ">> local URL: http://localhost:${PORT}"
fi

CMD="PORT=$PORT $PY serve/server.py"
if [ "$TMUX_MODE" = 1 ]; then
  command -v tmux >/dev/null || { echo "tmux not found (apt-get install -y tmux)"; exit 1; }
  tmux kill-session -t nepali 2>/dev/null || true
  tmux new -d -s nepali "$CMD 2>&1 | tee -a out/serve.log"
  echo ">> started in tmux 'nepali'.  attach: tmux attach -t nepali  |  logs: tail -f out/serve.log"
else
  echo ">> Ctrl-C to stop"
  eval "$CMD"
fi
