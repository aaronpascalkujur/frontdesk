"use strict";

/* Frontdesk web frontend.
 *
 * Draws the turn loop and pushes button presses back. All audio work stays in
 * Python; this file never touches a microphone. */

const body = document.body;
const canvas = document.getElementById("orb");
const ctx = canvas.getContext("2d");
const stateLabel = document.getElementById("state-label");
const hintEl = document.getElementById("hint");
const streamEl = document.getElementById("stream");
const emptyEl = document.getElementById("empty");
const connEl = document.getElementById("conn");
const agentsEl = document.getElementById("agents");
const learnedEl = document.getElementById("learned");

/* Mirrors the palette in style.css. Canvas cannot read CSS custom properties,
   so the two lists have to be kept in step by hand. */
const COLOURS = {
  idle:         [91, 107, 134],
  listening:    [34, 211, 238],
  transcribing: [251, 191, 36],
  thinking:     [167, 139, 250],
  speaking:     [251, 113, 133],
  offline:      [63, 70, 83],
};

const HINTS = {
  idle:         "Press space, or click the orb, to talk",
  listening:    "Listening — press space again when you are done",
  transcribing: "Working out what you said…",
  thinking:     "Deciding whether this needs an agent…",
  speaking:     "Speaking",
  offline:      "Frontdesk is not running",
};

/* Matches MIN_CACHE_CONFIDENCE in chat.py. Only used to flag a transcript in
   the feed as one that will not be learned from. */
const SHAKY_BELOW = -0.75;

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let state = "idle";
let level = 0;        // smoothed, what we draw
let target = 0;       // most recent reading from Python
const turns = new Map();   // turn id -> { msg, card }

/* ------------------------------------------------------------------- orb */

let size = 420;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (!rect.width) return;
  size = rect.width;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function blob(cx, cy, radius, wobble, t, lobes, phase) {
  ctx.beginPath();
  for (let a = 0; a <= Math.PI * 2 + 0.05; a += 0.045) {
    const r =
      radius +
      wobble *
        (Math.sin(a * lobes + t * 1.15 + phase) * 0.62 +
          Math.sin(a * (lobes + 3) - t * 0.75 + phase * 1.7) * 0.38);
    const x = cx + Math.cos(a) * r;
    const y = cy + Math.sin(a) * r;
    a === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
}

function draw(now) {
  requestAnimationFrame(draw);
  const t = now / 1000;

  /* Levels arrive only while audio is flowing. Decaying the target every frame
     means the orb settles instead of freezing at the last value when a stream
     stops mid-utterance. */
  target *= 0.90;
  level += (target - level) * 0.22;

  const [r, g, b] = COLOURS[state] || COLOURS.idle;
  const rgb = (alpha) => `rgba(${r}, ${g}, ${b}, ${alpha})`;

  ctx.clearRect(0, 0, size, size);
  const cx = size / 2;
  const cy = size / 2;

  const breath = reduceMotion ? 0 : Math.sin(t * 1.25) * 0.5 + 0.5;
  const busy = state === "thinking" || state === "transcribing";

  /* Audio states are driven by real amplitude; the silent states have no signal
     to follow, so they animate on the clock instead. */
  let energy = 0;
  if (state === "listening" || state === "speaking") {
    energy = Math.min(1, Math.sqrt(level) * 2.6);
  } else if (busy) {
    energy = 0.18 + breath * 0.16;
  } else if (state === "idle") {
    energy = 0.05 + breath * 0.07;
  }
  if (reduceMotion) energy = Math.min(energy, 0.25);

  const base = size * 0.15 * (1 + energy * 0.3);
  /* Kept well under a quarter of the radius. A polar curve whose wobble rivals
     its radius folds back through itself, which reads as flower petals rather
     than a blob. */
  const wobble = base * (0.05 + energy * 0.17);

  /* The outer stop has to land inside the canvas. Any colour still visible at
     the edge gets cut off square by the element bounds. */
  const glow = ctx.createRadialGradient(cx, cy, base * 0.2, cx, cy, size * 0.47);
  glow.addColorStop(0, rgb(0.3 + energy * 0.22));
  glow.addColorStop(0.35, rgb(0.07));
  glow.addColorStop(0.7, rgb(0.015));
  glow.addColorStop(1, rgb(0));
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, size, size);

  /* Every layer shares a lobe count and drifts only in phase, so the halo
     nests around the core like ripples. Mixing lobe counts makes the outlines
     cross one another, which looks like petals rather than one body. */
  const lobes = 3;
  for (let i = 3; i >= 1; i--) {
    blob(cx, cy, base * (1 + i * 0.13), wobble * (1 + i * 0.3), t, lobes, i * 0.7);
    ctx.fillStyle = rgb(0.05 + (3 - i) * 0.03);
    ctx.fill();
  }

  blob(cx, cy, base, wobble * 0.7, t, lobes, 0);
  ctx.fillStyle = rgb(0.92);
  ctx.fill();

  if (busy && !reduceMotion) {
    ctx.beginPath();
    ctx.arc(cx, cy, base * 1.8, t * 2.4, t * 2.4 + Math.PI * 0.55);
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = rgb(0.8);
    ctx.lineCap = "round";
    ctx.stroke();
  }
}

/* ------------------------------------------------------------------ feed */

function addNode(node) {
  if (emptyEl && emptyEl.parentNode) emptyEl.remove();
  streamEl.appendChild(node);
  // Only chase the bottom if the user has not scrolled up to read something.
  const nearBottom =
    streamEl.scrollHeight - streamEl.scrollTop - streamEl.clientHeight < 180;
  if (nearBottom) streamEl.scrollTop = streamEl.scrollHeight;
  return node;
}

function message(who, text, classes = "") {
  const el = document.createElement("div");
  el.className = `msg ${classes}`.trim();
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const content = document.createElement("span");
  content.textContent = text;
  el.append(label, content);
  return addNode(el);
}

function tag(el, text) {
  if (!el) return;
  const span = document.createElement("span");
  span.className = "tag";
  span.textContent = text;
  el.querySelector(".who").appendChild(span);
}

function seconds(ms) {
  return `${(ms / 1000).toFixed(1)}s`;
}

function startCard(turn, text) {
  const el = document.createElement("div");
  el.className = "card running";
  el.innerHTML =
    '<div class="top"><span class="agent">dispatching…</span>' +
    '<span class="elapsed" data-start="' + Date.now() + '">0.0s</span></div>' +
    '<div class="task"></div>';
  el.querySelector(".task").textContent = text;
  const entry = turns.get(turn) || {};
  entry.card = el;
  turns.set(turn, entry);
  return addNode(el);
}

function finishCard(data) {
  const entry = turns.get(data.turn) || {};
  const el = entry.card || startCard(data.turn, "");
  el.classList.remove("running");
  el.querySelector(".agent").textContent = data.agent || "agent";
  el.querySelector(".elapsed").textContent = `${data.elapsed}s`;
  el.querySelector(".elapsed").removeAttribute("data-start");
  if (data.text) {
    const bodyEl = document.createElement("div");
    bodyEl.className = "body";
    bodyEl.textContent = data.text;
    el.appendChild(bodyEl);
  }
  if (data.file) {
    const fileEl = document.createElement("div");
    fileEl.className = "file";
    fileEl.textContent = data.file;
    el.appendChild(fileEl);
  }
}

// Running agents show a live clock, so a multi-minute task never looks stuck.
setInterval(() => {
  document.querySelectorAll(".elapsed[data-start]").forEach((el) => {
    el.textContent = seconds(Date.now() - Number(el.dataset.start));
  });
}, 100);

/* ---------------------------------------------------------------- events */

function setState(value) {
  state = value;
  body.dataset.state = value;
  stateLabel.textContent = value;
  hintEl.textContent = HINTS[value] || "";
}

function handle(data) {
  switch (data.kind) {
    case "state":
      setState(data.value);
      break;

    case "level":
      target = Math.max(target, data.value || 0);
      break;

    case "agents":
      if (data.names && data.names.length) {
        agentsEl.textContent = `${data.names.length} agents · ${data.names.join(", ")}`;
        agentsEl.hidden = false;
      }
      if (data.learned) {
        learnedEl.textContent = `${data.learned} learned ${data.learned === 1 ? "reply" : "replies"}`;
        learnedEl.hidden = false;
      }
      break;

    case "heard": {
      if (!data.text) break;
      const shaky = data.confidence !== null && data.confidence < SHAKY_BELOW;
      const el = message("you", data.text, shaky ? "user shaky" : "user");
      if (shaky) tag(el, "low confidence · not learned");
      turns.set(data.turn, { ...(turns.get(data.turn) || {}), msg: el });
      break;
    }

    case "route": {
      const entry = turns.get(data.turn);
      if (entry && entry.msg) {
        const how = data.route === "task" ? "to an agent" : data.tier;
        tag(entry.msg, `${how} · ${data.ms}ms`);
      }
      break;
    }

    case "say":
      message("frontdesk", data.text, data.spoken === false ? "desk error" : "desk");
      break;

    case "task":
      startCard(data.turn, data.text);
      break;

    case "result":
      finishCard(data);
      break;

    case "error":
      message("problem", data.message, "desk error");
      break;
  }
}

/* --------------------------------------------------------------- transport */

function connect() {
  const source = new EventSource("/events");

  source.onopen = () => {
    connEl.dataset.live = "true";
    connEl.textContent = "live";
    /* The server replays recent history on every connect, so the feed is wiped
       first — otherwise a reconnect duplicates everything already on screen. */
    streamEl.innerHTML = "";
    turns.clear();
  };

  source.onmessage = (e) => {
    try {
      handle(JSON.parse(e.data));
    } catch (err) {
      /* One malformed frame should not take down the stream. */
    }
  };

  source.onerror = () => {
    connEl.dataset.live = "false";
    connEl.textContent = "reconnecting";
    // EventSource retries on its own; nothing to do but show it.
  };
}

function press() {
  if (state === "offline") return;
  fetch("/press", { method: "POST" }).catch(() => {});
}

document.getElementById("talk").addEventListener("click", press);

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat) {
    e.preventDefault();
    press();
  }
});

window.addEventListener("resize", resize);
resize();
setState("idle");
connect();
requestAnimationFrame(draw);
