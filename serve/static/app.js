// Nepali Diffusion — front-end denoising visualization
const $ = (id) => document.getElementById(id);
const SCRAMBLE = [..."कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह अआइईउऊएऐओऔ"];

const el = {
  prompt: $("prompt"), length: $("length"), steps: $("steps"), temp: $("temp"), speed: $("speed"),
  run: $("run"), grid: $("grid"), stage: $("stage"), reading: $("reading"), outPanel: $("output-panel"),
  hudStep: $("hud-step"), hudMask: $("hud-mask"), hudState: $("hud-state"), progress: $("progress"),
  badge: $("mode-badge"),
};
const SPEEDS = [0.14, 0.09, 0.06, 0.035, 0.015];
const SPEED_LABEL = ["सुस्त", "मन्द", "सामान्य", "छिटो", "तीव्र"];

let maskNodes = [];   // live <span.m> nodes to scramble
let busy = false;

// live slider labels
const bind = (input, out, fmt = (v) => v) => {
  const upd = () => { $(out).textContent = fmt(input.value); };
  input.addEventListener("input", upd); upd();
};
bind(el.length, "length-val");
bind(el.steps, "steps-val");
bind(el.temp, "temp-val", (v) => Number(v).toFixed(2));
bind(el.speed, "speed-val", (v) => SPEED_LABEL[v]);

// mode badge
fetch("/api/mode").then((r) => r.json()).then((d) => {
  el.badge.textContent = d.chat ? "chat" : (d.mode === "live" ? "live model" : "demo");
  el.badge.classList.add(d.chat ? "live" : d.mode);
  if (d.chat) el.prompt.placeholder = "प्रश्न सोध्नुहोस्…";  // "ask a question"
}).catch(() => { el.badge.textContent = "offline"; });

// ---- mask scramble loop (only touches masked spans) ----
let lastScramble = 0;
function scramble(ts) {
  if (ts - lastScramble > 70) {
    lastScramble = ts;
    for (const n of maskNodes) {
      n.textContent = SCRAMBLE[(Math.random() * SCRAMBLE.length) | 0];
    }
  }
  requestAnimationFrame(scramble);
}
requestAnimationFrame(scramble);

// ---- render the stage as flowing text ----
// Consecutive revealed/anchor tokens are concatenated into ONE run span so the
// browser shapes Devanagari across token boundaries (ligatures/matras join).
// Each masked position is its own span (placeholder; no ligature crosses it).
function renderStage(tokens) {
  const frag = document.createDocumentFragment();
  const masks = [];
  let i = 0;
  const n = tokens.length;
  while (i < n) {
    if (tokens[i].s === "mask") {
      const s = document.createElement("span");
      s.className = "m";
      s.textContent = SCRAMBLE[(Math.random() * SCRAMBLE.length) | 0];
      frag.appendChild(s);
      masks.push(s);
      i++;
    } else {
      let text = "", grew = false, allAnchor = true;
      while (i < n && tokens[i].s !== "mask") {
        text += tokens[i].t;
        if (tokens[i].s === "new") grew = true;
        if (tokens[i].s !== "anchor") allAnchor = false;
        i++;
      }
      const s = document.createElement("span");
      s.className = "r" + (allAnchor ? " a" : "") + (grew ? " grew" : "");
      s.textContent = text;
      frag.appendChild(s);
    }
  }
  el.grid.replaceChildren(frag);
  maskNodes = masks;
}

function applyFrame(f) {
  renderStage(f.tokens);
  const done = f.total - f.step + 1;
  el.hudStep.textContent = `${done}/${f.total}`;
  el.hudMask.textContent = `${f.mask_pct}%`;
  el.progress.style.width = `${100 * done / f.total}%`;
  if (f.text) el.reading.textContent = f.text;
}

// ---- run ----
function run() {
  if (busy) return;
  busy = true;
  el.run.disabled = true;
  el.stage.classList.add("busy");
  el.outPanel.hidden = false;
  el.reading.textContent = "";
  el.hudState.textContent = "denoising";

  const length = +el.length.value;
  renderStage(Array.from({ length }, () => ({ s: "mask", t: "" })));

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => ws.send(JSON.stringify({
    prompt: el.prompt.value,
    length,
    steps: +el.steps.value,
    temperature: +el.temp.value,
    top_k: 20,
    delay: SPEEDS[+el.speed.value],
  }));
  ws.onmessage = (ev) => {
    const f = JSON.parse(ev.data);
    if (f.type === "frame") applyFrame(f);
    else if (f.type === "done") { el.hudState.textContent = "complete"; finish(ws); }
    else if (f.type === "error") { el.hudState.textContent = "error"; console.error(f.msg); finish(ws); }
  };
  ws.onclose = () => finish(ws);
  ws.onerror = () => { el.hudState.textContent = "connection error"; finish(ws); };
}

function finish(ws) {
  if (!busy) return;
  busy = false;
  el.run.disabled = false;
  try { ws.close(); } catch (e) {}
}

el.run.addEventListener("click", run);
el.prompt.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
