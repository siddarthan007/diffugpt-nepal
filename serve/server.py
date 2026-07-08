"""Phase D — FastAPI + WebSocket server for the Nepali diffusion model.

Streams each denoising step to the browser so the front-end can animate the reveal.
Runs in one of two modes, auto-detected at startup:
  LIVE  — out/ckpt.pt + data/nepali_bpe_16k.model present -> real model.sample_stream
  DEMO  — otherwise -> synthetic denoising over sample Nepali text (UI works pre-training)

Run:  python serve/server.py      (then open http://localhost:8000)
"""
import asyncio
import math
import os
import random
import threading
import unicodedata

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
STATIC = os.path.join(HERE, "static")
TOK = os.path.join(REPO, "data", "nepali_bpe_16k.model")
# prefer the instruction-tuned model, then best-val, then latest
CKPT_CANDIDATES = [os.path.join(REPO, "out", f) for f in ("ckpt_sft.pt", "ckpt_best.pt", "ckpt.pt")]
CKPT = next((c for c in CKPT_CANDIDATES if os.path.exists(c)), CKPT_CANDIDATES[-1])

# --- try to bring up the real model; fall back to DEMO cleanly ---------------
MODEL = None
SP = None
DEVICE = "cpu"
MASK_ID = None
CHAT = False   # True when serving an instruction-tuned checkpoint

try:
    import sys
    sys.path.insert(0, REPO)          # so `import model` / `import prompt_format` resolve
    import torch  # noqa
    import sentencepiece as spm
    import prompt_format as PF
    if os.path.exists(TOK):
        SP = spm.SentencePieceProcessor(model_file=TOK)
    if os.path.exists(CKPT) and SP is not None:
        from model import DiffusionGPT
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
        args = ckpt["args"]; args.device = DEVICE
        MODEL = DiffusionGPT(args).to(DEVICE)
        sd = ckpt["model"]
        for k in list(sd.keys()):
            if k.startswith("_orig_mod."):
                sd[k[len("_orig_mod."):]] = sd.pop(k)
        MODEL.load_state_dict(sd)
        MODEL.eval()
        MASK_ID = MODEL.mask_token_id
        CHAT = bool(ckpt.get("sft", False))
except Exception as e:  # noqa
    print(f"(model/tokenizer not fully available -> DEMO mode: {type(e).__name__}: {e})")

MODE = "live" if MODEL is not None else "demo"
print(f"[server] mode = {MODE.upper()}{' + CHAT' if CHAT else ''}  ckpt={os.path.basename(CKPT)}  device={DEVICE}")

# --- Devanagari helpers ------------------------------------------------------
_COMBINING = set(range(0x0900, 0x0904)) | set(range(0x093A, 0x0950)) | \
             set(range(0x0951, 0x0958)) | set(range(0x0962, 0x0964)) | {0x200C, 0x200D}
DEMO_TEXTS = [
    "नेपाली साहित्यको इतिहास लामो, गहिरो र समृद्ध छ। कविता, कथा र निबन्धले हाम्रो सभ्यता बोल्छ।",
    "हिमालको काखमा बसेको यो देश गीत, लय र भाकाले भरिएको छ। हरेक भाषामा एउटा कथा लुकेको हुन्छ।",
    "यो जिन्दगी खै के हो खै के? कहिले हाँसो, कहिले आँसु, कहिले साँझको उदास बत्ती।",
    "शब्दहरू पानीजस्तै बग्छन्, अर्थहरू ढुङ्गाजस्तै रहन्छन्। लेखकले समयलाई कागजमा बाँध्छ।",
]
SCRAMBLE = list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह अआइईउऊएऐओऔ")


def aksharas(text):
    out, cur = [], ""
    for ch in text:
        if ord(ch) in _COMBINING and cur:
            cur += ch
        else:
            if cur:
                out.append(cur)
            cur = ch
    if cur:
        out.append(cur)
    return out


def cosine_keep(frac):
    return math.cos((1.0 - frac) * (math.pi / 2.0))


# --- frame producers ---------------------------------------------------------
def frames_live(params):
    import torch
    prompt = params.get("prompt", "").strip()
    # CHAT: wrap the user message in the instruction template so the denoised region
    # is the ANSWER (prompt tokens are anchors, exactly as during SFT).
    if CHAT and prompt:
        prompt_ids = PF.build_prompt_ids(SP, prompt)
    else:
        prompt_ids = SP.encode(prompt) if prompt else []
    seq_len = int(params.get("length", 120))
    steps = int(params.get("steps", 64))

    def piece(i):
        p = SP.id_to_piece(int(i))
        if p == "<eod>":
            return " "
        if p.startswith("<0x") and p.endswith(">"):   # byte-fallback (e.g. stray ASCII) -> hide
            return ""
        return p.replace("▁", " ")

    # Tuned decoding (eval sweep winner "B"): pure nucleus + high remask noise beats
    # the repetition collapse. top_k disabled in favor of top_p.
    with torch.no_grad():
        for fr in MODEL.sample_stream(batch_size=1, seq_len=seq_len, num_steps=steps,
                                      temperature=float(params.get("temperature", 1.0)),
                                      top_k=None, top_p=0.92, remask_noise=0.5,
                                      device=DEVICE, prompt_ids=prompt_ids):
            x = fr["x_t"][0].tolist()
            newly = fr["newly"][0].tolist()
            anch = fr["anchor"][0].tolist()
            conf = fr["confidence"][0].tolist()
            toks = []
            for i, idv in enumerate(x):
                if idv == MASK_ID:
                    toks.append({"t": "", "s": "mask", "c": 0.0})
                else:
                    s = "anchor" if anch[i] else ("new" if newly[i] else "revealed")
                    toks.append({"t": piece(idv), "s": s, "c": round(float(conf[i]), 3)})
            if CHAT:  # answer = the generated (non-anchor) region — position-based, robust
                text = SP.decode([idv for i, idv in enumerate(x) if not anch[i] and idv < MASK_ID])
            else:
                text = SP.decode([idv for idv in x if idv < MASK_ID])
            mask_pct = 100.0 * sum(1 for idv in x if idv == MASK_ID) / max(1, len(x))
            yield {"type": "frame", "step": fr["step"], "total": fr["total"],
                   "mask_pct": round(mask_pct, 1), "tokens": toks, "text": text}
    yield {"type": "done"}


def frames_demo(params):
    prompt = params.get("prompt", "").strip()
    body = random.choice(DEMO_TEXTS)
    text = (prompt + " " + body).strip() if prompt else body
    toks = aksharas(text)
    n = len(toks)
    steps = max(8, min(int(params.get("steps", 48)), 96))
    n_prompt = len(aksharas(prompt)) if prompt else 0

    revealed = [False] * n
    for i in range(n_prompt):
        revealed[i] = True
    order = list(range(n_prompt, n))
    random.Random(1337).shuffle(order)
    conf = [0.0] * n

    for step in range(steps, 0, -1):
        keep = cosine_keep((step - 1) / steps)
        target_masked = 0 if step == 1 else int(math.ceil(keep * (n - n_prompt)))
        want_revealed = (n - n_prompt) - target_masked
        newly = set()
        already = sum(1 for i in range(n_prompt, n) if revealed[i])
        need = max(0, want_revealed - already)
        for _ in range(need):
            if not order:
                break
            idx = order.pop(0)
            revealed[idx] = True
            newly.add(idx)
            conf[idx] = round(random.uniform(0.55, 0.99), 3)
        out = []
        for i in range(n):
            if not revealed[i]:
                out.append({"t": "", "s": "mask", "c": 0.0})
            elif i < n_prompt:
                out.append({"t": toks[i], "s": "anchor", "c": 1.0})
            else:
                out.append({"t": toks[i], "s": "new" if i in newly else "revealed", "c": conf[i]})
        cur_text = "".join(toks[i] for i in range(n) if revealed[i])
        mask_pct = 100.0 * sum(1 for r in revealed if not r) / max(1, n)
        yield {"type": "frame", "step": step, "total": steps,
               "mask_pct": round(mask_pct, 1), "tokens": out, "text": cur_text}
    yield {"type": "done"}


# --- app ---------------------------------------------------------------------
app = FastAPI()
_gpu_lock = asyncio.Lock()  # one generation at a time (single GPU)


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/api/mode")
async def mode():
    return {"mode": MODE, "device": DEVICE, "chat": CHAT}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            params = await websocket.receive_json()
            delay = float(params.get("delay", 0.06))
            async with _gpu_lock:
                loop = asyncio.get_event_loop()
                q: asyncio.Queue = asyncio.Queue()

                def worker():
                    producer = frames_live if MODE == "live" else frames_demo
                    try:
                        for frame in producer(params):
                            loop.call_soon_threadsafe(q.put_nowait, frame)
                    except Exception as e:  # noqa
                        loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "msg": str(e)})
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, None)

                threading.Thread(target=worker, daemon=True).start()
                await websocket.send_json({"type": "start", "mode": MODE})
                while True:
                    frame = await q.get()
                    if frame is None:
                        break
                    await websocket.send_json(frame)
                    if frame.get("type") == "frame":
                        await asyncio.sleep(delay)
    except WebSocketDisconnect:
        return


app.mount("/", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
