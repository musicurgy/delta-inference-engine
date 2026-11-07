// serve/web/everyday.js - everyday use, on top of app.js (its $, esc, fmt, toast, store, headers, settings, ext ...):
//   the loading screen (GET /health's model_state and load steps), Pause / Resume (POST /unload and /load),
//   "answer ready" notifications (opt-in), chat modes with the expected wait, and the "Connect an app" card.
"use strict";

// small icons in the sprite's style (24 px grid, 1.5 px strokes, a faint fill)
const P4_ICONS = {
  pause: '<rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" opacity=".12"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" opacity=".12"/><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>',
  play: '<path d="M7 5.5v13a1 1 0 0 0 1.5.86l11-6.5a1 1 0 0 0 0-1.72l-11-6.5A1 1 0 0 0 7 5.5Z" fill="currentColor" opacity=".12"/><path d="M7 5.5v13a1 1 0 0 0 1.5.86l11-6.5a1 1 0 0 0 0-1.72l-11-6.5A1 1 0 0 0 7 5.5Z"/>',
  list: '<path d="M8 6h12M8 12h12M8 18h12"/><circle cx="4" cy="6" r="1"/><circle cx="4" cy="12" r="1"/><circle cx="4" cy="18" r="1"/>',
  pin: '<path d="M9 4h6l-1 6 3 3v2H7v-2l3-3-1-6Z" fill="currentColor" opacity=".12"/><path d="M9 4h6l-1 6 3 3v2H7v-2l3-3-1-6ZM12 15v5"/>',
  pencil: '<path d="M4 20h4L19 9l-4-4L4 16v4Z" fill="currentColor" opacity=".12"/><path d="M4 20h4L19 9l-4-4L4 16v4ZM13.5 6.5l4 4"/>',
  search: '<circle cx="11" cy="11" r="6"/><path d="m20 20-4.5-4.5"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5M5 20h14"/>',
  plug: '<path d="M9 3v5M15 3v5M6 8h12v3a6 6 0 0 1-12 0V8ZM12 17v4"/>',
};
function p4icon(name, cls = "st-icon") {
  return `<svg class="${cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" ` +
         `stroke-linejoin="round" aria-hidden="true">${P4_ICONS[name]}</svg>`;
}
function secs(s) {
  if (s == null) return "";
  s = Math.max(0, Math.round(s));
  return s < 90 ? `${s} s` : `${Math.round(s / 60)} min`;
}

// ------------------------------------------------------------------ model state: loading screen, Pause / Resume
// GET /health (no key needed) says "loaded", "loading" (with the start's step and an estimate from the last start),
// "paused" (unloaded: Resume, or any request, loads it) or "failed".  Polled every second while it changes, every 3 s
// otherwise; the Monitor's /metrics poll goes on as before.
let modelState = null, loadInfo = null, healthFails = 0, pauseBusy = false, lastLoadedAt = 0;
async function pollModel() {
  let delay = 3000;
  try {
    const h = await (await fetch("health", {cache: "no-store"})).json();
    healthFails = 0;
    const was = modelState;
    modelState = h.model_state || (h.loaded ? "loaded" : "paused");   // an older server: no model_state
    loadInfo = h.load || null;
    if (modelState === "loading" || pauseBusy) delay = 1000;
    if (was && was !== "loaded" && modelState === "loaded") {
      lastLoadedAt = Date.now();
      if (was === "loading") toast("success", "DeepSeek is ready", "", 2500);
    }
    if (was === "loaded" && modelState === "paused" && loadInfo && loadInfo.reason) {
      toast("info", "DeepSeek paused by itself", loadInfo.reason.replace(/^Paused by itself: /, ""), 8000);
    }
  } catch (e) {                                   // the server is restarting, stopped, or a failed start ended it
    healthFails += 1;
    if (modelState === "failed") loadInfo = {...(loadInfo || {}), gone: true};
    else if (healthFails >= 3) modelState = "gone";
    delay = 2000;
  }
  paintModelState();
  setTimeout(pollModel, delay);
}

function paintModelState() {
  const card = $("load-card"), btn = $("pause-btn");
  const st = modelState, li = loadInfo || {};
  // the card above the chat
  if (st === "loading") {
    const pct = li.last_s ? Math.min(98, (100 * li.elapsed_s) / li.last_s) : (100 * (li.index - 1)) / Math.max(1, li.steps);
    const eta = li.eta_s != null ? (li.eta_s > 0 ? `About ${secs(li.eta_s)} left (the last start took ${secs(li.last_s)}).`
                                                 : `Taking longer than last time (${secs(li.last_s)}); still working.`)
                                 : "The first start takes 1-3 minutes.";
    card.innerHTML = `<div class="load-card__head"><span class="card-title">Starting DeepSeek</span>` +
      `<span class="muted small">step ${fmt(li.index)} of ${fmt(li.steps)} · ${secs(li.elapsed_s)}</span></div>` +
      `<div class="load-card__step"></div><div class="st-progress"><div class="st-progress__bar" style="width:${pct.toFixed(0)}%"></div></div>` +
      `<div class="muted small">${esc(eta)} Your PC can be slow for a minute or two while the model is read; that is normal. ` +
      `You can type your question now: it is answered as soon as the model is ready.</div>`;
    card.querySelector(".load-card__step").textContent = li.text || "Loading";
    card.dataset.state = "loading";
    card.hidden = false;
  } else if (st === "failed") {
    card.innerHTML = `<div class="load-card__head"><span class="card-title">DeepSeek did not start</span></div>` +
      `<div class="msg-error"></div><div class="muted small">The DeepSeek window says more. ` +
      `${li.gone ? "The server has stopped; start DeepSeek again. " : ""}For a bug report, run ` +
      `<b>Start DeepSeek (debug).bat</b>: it packs everything needed into one zip.</div>`;
    card.querySelector(".msg-error").textContent = li.error || "The engine stopped while it was starting.";
    card.dataset.state = "failed";
    card.hidden = false;
  } else if (st === "paused" && (!messages.length || li.reason)) {
    card.innerHTML = `<div class="load-card__head"><span class="card-title">DeepSeek is paused</span>` +
      `<button class="st-btn st-btn--primary" type="button" data-resume>${p4icon("play")}Resume</button></div>` +
      (li.reason ? `<div class="load-card__step"></div>` : "") +
      `<div class="muted small">The RAM and the graphics card are free for other programs. Resume, or just send a ` +
      `message: it loads first${li.last_s ? ` (about ${secs(li.last_s)})` : " (about a minute)"}.</div>`;
    if (li.reason) card.querySelector(".load-card__step").textContent = li.reason;
    card.dataset.state = "paused";
    card.hidden = false;
  } else if (st === "gone") {
    card.innerHTML = `<div class="load-card__head"><span class="card-title">The server is not reachable</span></div>` +
      `<div class="muted small">Is the DeepSeek window still open? This page reconnects by itself when it is back.</div>`;
    card.dataset.state = "gone";
    card.hidden = false;
  } else {
    card.hidden = true;
  }
  if (!card.hidden && !messages.length) $("chat-empty").hidden = true;
  else if (card.hidden) $("chat-empty").hidden = messages.length > 0;
  // the header button
  btn.hidden = !st || st === "gone" || st === "failed";
  btn.disabled = st === "loading" || !!pauseBusy;
  let html;
  if (st === "loading" || pauseBusy) {
    html = `${p4icon("play")}<span>${pauseBusy === "pausing" ? "Pausing…" : "Loading…"}</span>`;
  } else if (st === "paused") {
    html = `${p4icon("play")}<span>Resume</span>`;
    btn.title = "Load DeepSeek again" + (li.last_s ? ` (about ${secs(li.last_s)})` : " (about a minute)");
  } else {
    html = `${p4icon("pause")}<span>Pause</span>`;
    btn.title = "Pause DeepSeek: frees its RAM and the graphics card (for games or other programs). Resume takes about a minute.";
  }
  if (btn.dataset.html !== html) { btn.innerHTML = html; btn.dataset.html = html; }
  // the header pill and the composer hint
  if (st === "loading") setPill("reading", `Loading model · ${fmt(li.index)}/${fmt(li.steps)}`);
  else if (st === "paused" && !busy) setPill("idle", "Paused");
  else if (st === "failed") setPill("error", "Start failed");
  if (!busy) $("composer-hint").textContent = st === "loading" ? "Sent when the model is ready" :
    st === "paused" ? "Sending loads the model first" : "Shift+Enter: new line";
}
ext.afterRender.push(() => { if (modelState && modelState !== "loaded") paintModelState(); });

async function pauseModel() {
  if (busy) { toast("warn", "Still writing", "Stop the answer first, or wait for it."); return; }
  pauseBusy = "pausing";
  paintModelState();
  try {
    const r = await fetch("unload", {method: "POST", headers: headers(true), body: "{}"});
    const b = await r.json().catch(() => ({}));
    if (r.status === 409) toast("warn", "Not paused", "Another app's request is running. Try again when it has finished.", 6000);
    else if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    else if (b.status === "unsupported") toast("info", "Cannot pause", "This server's engine cannot be unloaded.");
    else toast("success", "Paused", "The RAM and the graphics card are free. Resume (or a new message) loads it again.", 5000);
  } catch (e) {
    toast("error", "Not paused", String(e.message || e), 6000);
  }
  pauseBusy = false;
  modelState = null;
  pollNow();
}
async function resumeModel() {
  pauseBusy = "resuming";
  modelState = "loading";
  loadInfo = {index: 1, steps: 5, elapsed_s: 0, text: "Starting", last_s: (loadInfo || {}).last_s};
  paintModelState();
  setTimeout(pollNow, 500);
  try {
    const r = await fetch("load", {method: "POST", headers: headers(true), body: "{}"});   // returns once it is loaded
    if (!r.ok) {
      const b = await r.json().catch(() => ({}));
      throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    }
  } catch (e) {
    toast("error", "Not loaded", String(e.message || e), 8000);
  }
  pauseBusy = false;
  pollNow();
}
function pollNow() { fetch("health", {cache: "no-store"}).then((r) => r.json()).then((h) => {
  modelState = h.model_state || (h.loaded ? "loaded" : "paused"); loadInfo = h.load || null; paintModelState();
}).catch(() => {}); }
$("pause-btn").addEventListener("click", () => { if (modelState === "paused") resumeModel(); else pauseModel(); });
$("load-card").addEventListener("click", (e) => { if (e.target.closest("[data-resume]")) resumeModel(); });

// ------------------------------------------------------------------ "answer ready" notifications (opt-in)
// Only while this tab is in the background.  Text-free: the notification says that an answer is ready and how long it
// is, not what it says.  Browsers allow notifications on http://127.0.0.1 / localhost (and https); on a plain
// http://192.168.x.x address they refuse them, so the switch says so.
let notifyOn = !!store.get("notify", false);
let titleBase = document.title, titleTimer = null;
function notifySupported() { return "Notification" in window && window.isSecureContext; }
function paintNotify() {
  $("notify-toggle").setAttribute("aria-checked", String(notifyOn && notifySupported() && Notification.permission === "granted"));
  if (!notifySupported()) $("notify-note").textContent = "this browser allows it only on http://127.0.0.1 (this PC) or https";
  else if (Notification.permission === "denied") $("notify-note").textContent = "blocked for this page in the browser's site settings";
}
$("notify-toggle").addEventListener("click", async () => {
  if (!notifySupported()) { toast("info", "Not on this address", "Browsers show notifications on http://127.0.0.1 (this PC) or https only.", 6000); return; }
  if (notifyOn && Notification.permission === "granted") { notifyOn = false; }
  else {
    const p = Notification.permission === "default" ? await Notification.requestPermission() : Notification.permission;
    notifyOn = p === "granted";
    if (!notifyOn) toast("warn", "Not allowed", "The browser blocked notifications for this page.", 6000);
  }
  store.set("notify", notifyOn);
  paintNotify();
});
function chime() {
  try {
    const a = new (window.AudioContext || window.webkitAudioContext)();
    [[880, 0], [1320, 0.12]].forEach(([f, t]) => {
      const o = a.createOscillator(), g = a.createGain();
      o.frequency.value = f; o.type = "sine";
      g.gain.setValueAtTime(0.0001, a.currentTime + t);
      g.gain.exponentialRampToValueAtTime(0.08, a.currentTime + t + 0.02);
      g.gain.exponentialRampToValueAtTime(0.0001, a.currentTime + t + 0.35);
      o.connect(g).connect(a.destination); o.start(a.currentTime + t); o.stop(a.currentTime + t + 0.4);
    });
    setTimeout(() => a.close(), 1000);
  } catch (e) { /* no audio: the notification is enough */ }
}
ext.afterAnswer.push((m, t) => {
  if (!document.hidden || m.stopped) return;
  document.title = `(done) ${titleBase}`;
  if (!(notifyOn && notifySupported() && Notification.permission === "granted")) return;
  const took = (performance.now() - t.sentAt) / 1000;
  const body = m.error ? "The request failed." : `${t.tokens ? `${fmt(t.tokens)} tokens, ` : ""}${secs(took)}`;
  try {
    const n = new Notification(m.error ? "DeepSeek: no answer" : "DeepSeek: your answer is ready", {body, tag: "delta-answer"});
    n.onclick = () => { window.focus(); showTab("chat"); n.close(); };
    chime();
  } catch (e) { /* some browsers need a service worker for notifications: the tab title still says it */ }
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) document.title = titleBase; });
paintNotify();

// ------------------------------------------------------------------ modes: Quick / Think / Code / Write
// Named presets of the Sampling drawer's own settings (thinking level and temperature) plus, for Code and Write, a
// short instruction sent as the system message.  The wait shown under them is this browser's own: the median time
// from send to the first word of the answer over the last answers in that mode.
const MODES = [
  {id: "quick", name: "Quick", thinking: "none", temperature: 0.6, tip: "No thinking: answers at once. Weaker on hard questions.",
   wait: "answers right away"},
  {id: "think", name: "Think", thinking: "high", temperature: 0.6, tip: "Thinks first (the default): best on hard questions.",
   wait: "usually 30-90 s of thinking first"},
  {id: "code", name: "Code", thinking: "medium", temperature: 0.3, tip: "Code: thinks a little, focused answers with code.",
   wait: "usually 10-40 s first", system: "You are an expert programmer. Answer with correct, complete, runnable code and short explanations. Ask when something essential is missing."},
  {id: "write", name: "Write", thinking: "low", temperature: 0.8, tip: "Writing: a short think, then natural prose.",
   wait: "usually a few seconds first", system: "You are a skilled writer. Write clear, natural prose in the user's language and tone; no filler."},
];
function currentMode() {
  // the chosen mode while the drawer still holds its settings (changing them there makes it "custom"); none chosen
  // yet: the mode the settings match (the defaults are Think's)
  const id = store.get("mode", null);
  const fits = (m) => m && m.thinking === settings.thinking && +m.temperature === +settings.temperature;
  const chosen = MODES.find((x) => x.id === id);
  return fits(chosen) ? chosen : id ? null : MODES.find(fits) || null;
}
function modeWait(m) {
  const w = (store.get("waits", {})[m.id] || []).slice(-8);
  if (w.length < 2) return m.wait;
  const s = [...w].sort((a, b) => a - b)[Math.floor(w.length / 2)];
  return s < 3 ? "your recent answers: started at once" : `your recent answers: about ${secs(s)} before the answer`;
}
function paintModes() {
  const cur = currentMode();
  // presets are starting points (not measured as the best for each kind of task): Sampling changes them
  $("modes").innerHTML = `<span class="mode-label muted small" title="Presets of the Sampling settings - starting points; ` +
    `change them in Sampling">Starting point:</span>` +
    MODES.map((m) => `<button type="button" role="radio" class="mode-chip" data-mode="${m.id}" ` +
    `aria-checked="${cur && cur.id === m.id}" title="${esc(m.tip)} A starting point: adjust it in Sampling.">${esc(m.name)}</button>`).join("") +
    `<span class="mode-wait muted small">${cur ? esc(`${cur.name}: ${modeWait(cur)}`) : "Custom settings (Sampling)"}</span>`;
}
$("modes").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mode]");
  if (!b) return;
  const m = MODES.find((x) => x.id === b.dataset.mode);
  settings = {...settings, thinking: m.thinking, temperature: m.temperature};
  store.set("sampling", settings);
  store.set("mode", m.id);
  paintModes();
  $("input").focus();
});
ext.beforeSend.push((body) => {
  const m = currentMode();
  if (m && m.system && !body.messages.some((x) => x.role === "system")) body.messages.unshift({role: "system", content: m.system});
});
ext.afterAnswer.push((msg, t) => {
  const m = currentMode();
  if (!m || msg.error || msg.stopped || !t.answerAt) return;
  const waits = store.get("waits", {});
  waits[m.id] = [...(waits[m.id] || []), Math.round((t.answerAt - t.sentAt) / 100) / 10].slice(-8);
  store.set("waits", waits);
  paintModes();
});
// the drawer's Apply may change thinking / temperature: the chips follow
$("s-apply").addEventListener("click", () => setTimeout(paintModes, 0));
paintModes();

// ------------------------------------------------------------------ "Connect an app" (Setup tab)
// The settings each app needs, with Copy buttons, and a Test that sends a 1-token request the way an app would.
const APPS = {
  claude: {name: "Claude Code", how: "In the Command Prompt you start it from:",
           text: (b, k, m) => `set ANTHROPIC_BASE_URL=${b}\nset ANTHROPIC_AUTH_TOKEN=${k}\nset ANTHROPIC_MODEL=${m}\nset CLAUDE_CODE_MAX_CONTEXT_TOKENS=1000000\nclaude`,
           note: "Without CLAUDE_CODE_MAX_CONTEXT_TOKENS, Claude Code assumes 200K tokens and compacts there."},
  opencode: {name: "OpenCode", how: "In opencode.jsonc (in your project, or ~/.config/opencode/):",
             text: (b, k, m) => JSON.stringify({$schema: "https://opencode.ai/config.json", providers: {delta: {
               name: "Delta (local)", package: "@opencode/ai/providers/openai-compatible",
               settings: {baseURL: `${b}/v1`, ...(k !== "local" ? {apiKey: k} : {})},
               models: {[m]: {name: "DeepSeek V4.1 Flash", limit: {context: 1000000, output: 32768}}}}}}, null, 2)},
  openai: {name: "Cline, Aider, Continue, Open WebUI, other OpenAI apps", how: "In the app's provider settings (OpenAI-compatible):",
           text: (b, k, m) => `Base URL:        ${b}/v1\nAPI key:         ${k}\nModel:           ${m}\nContext window:  1000000`},
  anthropic: {name: "Other Anthropic apps", how: "In the app's settings (Anthropic API):",
              text: (b, k, m) => `Base URL:  ${b}\nAPI key:   ${k}\nModel:     ${m}`},
  curl: {name: "Test from a terminal (curl)", how: "In the Command Prompt:",
         text: (b, k, m) => `curl ${b}/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer ${k}" -d "{\\"model\\": \\"${m}\\", \\"messages\\": [{\\"role\\": \\"user\\", \\"content\\": \\"Hi\\"}], \\"max_tokens\\": 20}"`},
};
let connectApp = store.get("connect_app", "claude");
function paintConnect() {
  const el = $("connect");
  if (!el) return;
  const base = location.origin, model = health.model || "deepseek";
  const key = health.api_key ? "<your API key>" : "local";
  const a = APPS[connectApp] || APPS.claude;
  const text = a.text(base, key, model);
  el.innerHTML = `<div class="connect__head"><span class="card-title">${p4icon("plug", "st-icon st-icon--sm")} Connect an app</span>` +
    `<select class="st-input" id="connect-app" aria-label="App">${Object.entries(APPS).map(([id, x]) =>
      `<option value="${id}"${id === connectApp ? " selected" : ""}>${esc(x.name)}</option>`).join("")}</select></div>` +
    `<div class="muted small">${esc(a.how)}</div>` +
    `<div class="st-code"><div class="st-code__head"><span>${esc(a.name)}</span><button class="st-btn st-btn--icon" type="button" ` +
    `data-copy="${esc(text)}" aria-label="Copy">${icon("copy")}</button></div><pre><code>${esc(text)}</code></pre></div>` +
    (a.note ? `<div class="muted small">${esc(a.note)}</div>` : "") +
    `<div class="muted small">${health.api_key
      ? "This server needs an API key: use the one in the server's config (\"api_key\" in ds41-windows.json). It is the same key this page asks for under Settings."
      : "No API key needed on this PC: where an app insists on one, any text works (\"local\")."} Any model name works; this one is the server's.</div>` +
    `<div class="cfg-actions"><button class="st-btn st-btn--secondary" type="button" id="connect-test">Test the connection</button>` +
    `<span class="muted small" id="connect-msg" role="status"></span></div>`;
  $("connect-app").onchange = () => { connectApp = $("connect-app").value; store.set("connect_app", connectApp); paintConnect(); };
  $("connect-test").onclick = testConnection;
}
async function testConnection() {
  const msg = $("connect-msg");
  msg.textContent = modelState === "paused" ? "Loading the model first (about a minute) …" : "Asking for one word …";
  const t0 = performance.now();
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify({
      model: health.model, messages: [{role: "user", content: "Say OK."}], max_tokens: 1, reasoning_effort: "none"})});
    const b = await r.json().catch(() => ({}));
    if (r.status === 401) throw new Error("the server wants an API key (Settings below)");
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    msg.textContent = `Works: the server answered in ${fmt((performance.now() - t0) / 1000, 1)} s. Apps use the same address.`;
  } catch (e) {
    msg.textContent = `Did not work: ${e.message || e}`;
  }
}

// ------------------------------------------------------------------ "This PC's health" (Setup tab)
// What the launcher's Windows check found at this start (windows_tips.py: power plan, GPU scheduling, large pages,
// RAM speed, page file), each with how to fix it.  Read only: the page changes nothing on the PC.
ext.afterSetup.push((c) => {
  let card = $("health-card");
  if (!Array.isArray(c.health)) { if (card) card.hidden = true; return; }
  if (!card) {
    card = document.createElement("div");
    card.className = "st-card health-card";
    card.id = "health-card";
    $("setup-card").after(card);
  }
  card.hidden = false;
  card.innerHTML = `<div class="req-head"><span class="card-title">This PC's health</span>` +
    `<span class="muted small">checked at this start</span></div>` +
    (c.health.length ? `<p class="muted small">Optional settings that make DeepSeek faster or steadier (README.md, ` +
      `"Windows settings for speed"). They need administrator rights, the BIOS or a restart, so the app does not change them itself.</p>` +
      `<ul class="health-list">${c.health.map((x) => `<li><span class="st-badge st-badge--queued">Tip</span>` +
        `<div><div class="health-text">${esc(x.text[0].toUpperCase() + x.text.slice(1))}</div>` +
        `<div class="muted small">${esc(x.how || "")}</div></div></li>`).join("")}</ul>`
      : `<p class="muted small">${icon("check", "st-icon st-icon--sm")} All the checked Windows settings are as fast as they go.</p>`);
});

// ------------------------------------------------------------------ "Use from your phone" and auto-pause (Setup tab)
// The switch makes the launcher listen on this network with an API key (it makes one if there is none; the server
// refuses to listen beyond this PC without one) and restarts.  The QR code holds the address with the key after "#":
// a phone that opens it keeps the key in its browser, and the part after "#" never reaches any server.
let phoneShowKey = false;
const isThisPc = ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
async function postSetup(body) {
  const r = await fetch("setup", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
  const b = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
  return b;
}
async function waitForRestart(msgEl) {
  msgEl.textContent = "Restarting with the new setting (about a minute); this page comes back by itself.";
  await new Promise((d) => setTimeout(d, 3000));
  for (let i = 0; i < 300; i++) {
    try { if ((await fetch("health", {cache: "no-store"})).ok) { location.reload(); return; } } catch (e) { /* restarting */ }
    await new Promise((d) => setTimeout(d, 2000));
  }
  msgEl.textContent = "It has not come back. Look at the DeepSeek window: it says why.";
}
function paintPhone(c) {
  const ph = c.phone;
  let card = $("phone-card");
  if (!ph) { if (card) card.hidden = true; return; }
  if (!card) {
    card = document.createElement("div");
    card.className = "st-card phone-card";
    card.id = "phone-card";
    ($("health-card") || $("setup-card")).after(card);
  }
  card.hidden = false;
  const urls = (ph.addresses || []).map((ip) => `http://${ip}:${ph.port}/`);
  const link = urls.length && ph.key ? `${urls[0]}#key=${encodeURIComponent(ph.key)}` : "";
  const fw = ph.firewall === true ? "Windows Firewall lets other devices in (rule \"Delta DeepSeek " + ph.port + "\")."
    : ph.firewall === false ? "Windows Firewall still blocks other devices." : "";
  card.innerHTML = `<div class="req-head"><span class="card-title">Use from your phone</span>` +
    `<span class="muted small">${ph.on ? "on: this network can reach it, with the key" : "off: this PC only"}</span></div>` +
    `<div class="toggle-row"><span>Reachable from other devices on this network<br><span class="muted small">` +
    `${ph.own_config ? "set in ds41-windows.json (\"host\" and \"api_key\"): change it there" :
      "makes an API key, listens on this network (port " + ph.port + ") and restarts DeepSeek"}</span></span>` +
    `<button class="st-toggle" role="switch" id="phone-toggle" aria-checked="${ph.on}" aria-label="Reachable from other devices"` +
    `${ph.own_config ? " disabled" : ""}></button></div>` +
    (ph.on ? `<div class="phone-grid"><div class="phone-qr" id="phone-qr">` +
        (link ? `<button class="st-btn st-btn--secondary" type="button" id="phone-qr-btn">Show the QR code</button>` : "") + `</div>` +
      `<div class="phone-facts"><dl class="facts">` +
        urls.map((u) => `<dt>Address</dt><dd><code>${esc(u)}</code><button class="st-btn st-btn--icon" data-copy="${esc(u)}" aria-label="Copy">${icon("copy")}</button></dd>`).join("") +
        `<dt>API key</dt><dd><code>${esc(phoneShowKey ? ph.key : "•".repeat(12))}</code>` +
        `<button class="st-btn st-btn--icon" type="button" id="phone-key-show" title="${phoneShowKey ? "Hide" : "Show"}" aria-label="Show the key">${p4icon("search")}</button>` +
        `<button class="st-btn st-btn--icon" data-copy="${esc(ph.key)}" aria-label="Copy the key">${icon("copy")}</button></dd></dl>` +
        `<p class="muted small">Scan the code with the phone's camera: the page opens with the key already in it. Other ` +
        `apps use the address + <code>v1</code> and the key. Anyone with the key can use the model: keep it like a password.</p></div></div>` +
      `<div class="phone-fw"><span class="muted small">${esc(fw)}</span>` +
        (isThisPc && ph.firewall !== true ? `<button class="st-btn st-btn--secondary" type="button" data-fw="add">Allow through Windows Firewall</button>` : "") +
        (isThisPc && ph.firewall === true ? `<button class="st-btn st-btn--secondary" type="button" data-fw="remove">Remove the rule</button>` : "") +
      `</div><details class="muted small"><summary>What the firewall button does</summary><p>Windows asks you first (an ` +
        `administrator prompt). It adds this rule for private networks only:</p><pre><code>${esc(ph.firewall_add)}</code></pre>` +
        `<p>To undo it: <code>${esc(ph.firewall_remove)}</code></p></details>`
      : "") +
    `<details class="muted small"><summary>Outside your home, and as an app on the phone</summary>` +
      `<p>Away from home: install Tailscale (free for personal use) on this PC and the phone; the phone then reaches this ` +
      `PC privately, as if it were at home. Never use Tailscale Funnel: it makes the page public.</p>` +
      `<p>As an app: open the page on the phone, then the browser's menu > "Add to Home Screen" (iPhone: Share > ` +
      `"Add to Home Screen"). On a plain http:// address it opens as a shortcut; notifications need https or this PC.</p></details>` +
    `<span class="muted small" id="phone-msg" role="status"></span>`;
  const qrBtn = $("phone-qr-btn");
  if (qrBtn) qrBtn.onclick = () => { $("phone-qr").innerHTML = QR.qrSvg(link, "M") || "The address is too long for a QR code."; };
  const show = $("phone-key-show");
  if (show) show.onclick = () => { phoneShowKey = !phoneShowKey; paintPhone(c); };
  for (const b of card.querySelectorAll("[data-fw]")) b.onclick = async () => {
    $("phone-msg").textContent = "Windows asks for permission now (look for the prompt on this PC) …";
    try {
      const r = await postSetup({firewall: b.dataset.fw});
      ph.firewall = r.firewall;
      paintPhone(c);
      $("phone-msg").textContent = r.firewall ? "Allowed." : b.dataset.fw === "add" ? "Not added (cancelled or refused)." : "Removed.";
    } catch (e) { $("phone-msg").textContent = `Not changed: ${e.message || e}`; }
  };
  $("phone-toggle").onclick = async () => {
    const on = !ph.on;
    if (!confirm(on ? "DeepSeek restarts (about a minute) and then answers devices on this network that have the API key. " +
                      "This page keeps the key. Continue?"
                    : "DeepSeek restarts (about a minute) and then answers this PC only. Continue?")) return;
    try {
      const r = await postSetup({phone: on});
      if (r.api_key) { store.set("apikey", r.api_key); $("api-key").value = r.api_key; }
      $("phone-toggle").disabled = true;
      waitForRestart($("phone-msg"));
    } catch (e) { toast("error", "Not changed", String(e.message || e), 6000); }
  };
}
function paintAutoPause(c) {
  if (c.auto_pause_mib == null) return;
  let row = $("autopause-row");
  if (!row) {
    row = document.createElement("div");
    row.className = "autopause";
    row.id = "autopause-row";
    $("setup-apply").closest(".cfg-actions").before(row);
  }
  const mib = c.auto_pause_mib;
  row.innerHTML = `<div class="toggle-row"><span>Pause by itself when a game needs the graphics card<br><span class="muted small">` +
    `while DeepSeek is idle: when another program takes ${mib ? fmt(mib / 1024, 1) : "1"} GB or more of its memory, or keeps ` +
    `it busy; the next message loads it again</span></span>` +
    `<button class="st-toggle" role="switch" id="autopause-toggle" aria-checked="${mib > 0}" aria-label="Pause by itself for games"></button></div>`;
  $("autopause-toggle").onclick = async () => {
    try {
      const r = await postSetup({auto_pause_mib: mib > 0 ? 0 : 1024});
      c.auto_pause_mib = r.auto_pause_mib;
      paintAutoPause(c);
      toast("success", r.auto_pause_mib ? "Auto-pause on" : "Auto-pause off", r.auto_pause_mib ? "It pauses when a game needs the card." : "");
    } catch (e) { toast("error", "Not changed", String(e.message || e), 6000); }
  };
}
ext.afterSetup.push((c) => { paintPhone(c); paintAutoPause(c); });

// ------------------------------------------------------------------ start
$("history-btn").innerHTML = p4icon("list");
loadHealth().then(paintConnect);
pollModel();
