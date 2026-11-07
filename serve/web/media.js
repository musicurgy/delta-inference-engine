// serve/web/media.js - documents, dictation and read-aloud in the chat.  Loaded after app.js and uses its globals
// ($, toast, headers, health, attachments, renderAttachments, messages, kfmt, fmt, store).
//  * Documents (PDF, Word, Excel, PowerPoint): read by the server (POST v1/extract) into text with a header; the chip
//    shows the token count and the time the model needs to read it the first time, before anything is sent.
//  * Dictation: the microphone button.  Hold it and talk (push-to-talk), or click to start and click again to stop.
//    The browser records 16 kHz WAV; the server's speech recognition writes the text into the message box.
//  * Read aloud: a speaker button under every answer.  The answer is spoken sentence by sentence (the next sentence is
//    made while the current one plays); code blocks are skipped.
//  * Talk mode: the Talk button - a hands-free conversation with barge-in (see "talk mode" below).
"use strict";

const media = (() => {
  const svg = (body) => `<svg class="st-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" ` +
    `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
  const ICON = {
    mic: svg('<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21M9 21h6"/>'),
    speaker: svg('<path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z"/><path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11"/>'),
    stop: svg('<rect x="6.5" y="6.5" width="11" height="11" rx="2"/>'),
    wait: svg('<path d="M12 3a9 9 0 1 0 9 9" class="media-spin"/>'),
  };
  let voice = {asr: false, tts: false};

  async function loadVoice() {
    try {
      const h = await (await fetch("health")).json();
      voice = h.voice || voice;
    } catch (e) { setTimeout(loadVoice, 3000); return; }
    micButton();
    talkButton();
    if (voice.tts) voiceField();
    document.querySelectorAll("[data-msg-speak]").forEach((b) => {
      const el = b.closest(".st-msg");
      if (el) speakButton(b, messages[+el.dataset.i], false);
    });
  }

  // the read-aloud voice and speed, in the Sampling drawer (kept in this browser)
  const VOICES = {"American English": "af_heart af_bella af_nicole af_sarah af_nova af_sky af_alloy af_aoede af_jessica " +
    "af_kore af_river am_michael am_fenrir am_puck am_echo am_eric am_liam am_onyx am_adam am_santa",
    "British English": "bf_emma bf_isabella bf_alice bf_lily bm_george bm_fable bm_lewis bm_daniel",
    "Spanish": "ef_dora em_alex em_santa", "French": "ff_siwis", "Italian": "if_sara im_nicola",
    "Portuguese (Brazil)": "pf_dora pm_alex pm_santa", "Hindi": "hf_alpha hf_beta hm_omega hm_psi",
    "Japanese": "jf_alpha jf_gongitsune jf_nezumi jf_tebukuro jm_kumo",
    "Chinese": "zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi zm_yunjian zm_yunxi zm_yunxia zm_yunyang"};
  function voiceField() {
    const body = document.querySelector("#drawer .st-drawer__body");
    if (!body || $("s-voice")) return;
    const f = document.createElement("div");
    f.className = "st-field";
    const cur = store.get("voice", "af_heart"), speed = +store.get("voice_speed", 1);
    f.innerHTML = `<label class="st-label" for="s-voice">Read-aloud voice <output id="o-voice-speed"></output></label>` +
      `<select class="st-input" id="s-voice">${Object.entries(VOICES).map(([lang, v]) => `<optgroup label="${lang}">` +
        v.split(" ").map((n) => `<option value="${n}"${n === cur ? " selected" : ""}>${n.slice(3)} (${n[1] === "f" ? "female" : "male"})</option>`).join("") +
        "</optgroup>").join("")}</select>` +
      `<input class="st-slider" type="range" id="s-voice-speed" min="0.6" max="1.6" step="0.05" value="${speed}" aria-label="Speaking speed">`;
    body.appendChild(f);
    const out = () => { $("o-voice-speed").textContent = `${fmt(+$("s-voice-speed").value, 2)}x`; };
    out();
    $("s-voice").onchange = () => store.set("voice", $("s-voice").value);
    $("s-voice-speed").oninput = () => { store.set("voice_speed", +$("s-voice-speed").value); out(); };
  }

  // ---------------------------------------------------------------- documents
  const DOC_EXT = /\.(pdf|docx|docm|xlsx|xlsm|pptx|pptm|doc|xls|ppt)$/i;
  const DOC_TYPE = /^application\/(pdf|msword|vnd\.ms-(excel|powerpoint)|vnd\.openxmlformats-officedocument\.)/;
  const MAX_DOC = 64e6;
  const READ_TOK_S = 650;                       // first-read speed if the server gives none (README: 600-740 tok/s)

  const isDocument = (f) => DOC_EXT.test(f.name || "") || DOC_TYPE.test(f.type || "");
  const readUrl = (f) => new Promise((ok, bad) => {
    const r = new FileReader();
    r.onload = () => ok(r.result);
    r.onerror = () => bad(r.error || new Error("the file could not be read"));
    r.readAsDataURL(f);
  });
  const secs = (s) => (s < 1 ? "under 1 s" : s < 90 ? `~${fmt(Math.ceil(s))} s` : `~${fmt(s / 60, 1)} min`);

  async function addDocument(f) {
    if (f.size > MAX_DOC) { toast("warn", "Document too large", `${f.name} is over 64 MB.`); return; }
    const a = {kind: "file", name: f.name, pending: true};
    attachments.push(a);
    renderAttachments();
    try {
      const data = await readUrl(f);
      const r = await fetch("v1/extract", {method: "POST", headers: headers(true),
                                           body: JSON.stringify({name: f.name, data, media_type: f.type})});
      let j = {};
      try { j = await r.json(); } catch (e) { /* not JSON */ }
      if (r.status === 401) throw new Error("this server needs an API key: add it under About > Settings");
      if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
      if (j.error) throw new Error(j.error);
      Object.assign(a, {pending: false, text: j.text,
                        doc: {tokens: j.tokens, seconds: j.read_seconds, type: j.type, pages: j.pages, note: j.note}});
      if (j.note) toast("info", f.name, j.note, 7000);
    } catch (e) {
      const i = attachments.indexOf(a);
      if (i >= 0) attachments.splice(i, 1);
      toast("warn", "Could not read the document", `${f.name}: ${e.message || e}`, 7000);
    }
    renderAttachments();
  }

  // tokens of an attachment: the server's count for a document, an estimate for a text file
  const tokensOf = (a) => (a.doc ? a.doc.tokens : a.kind === "file" && a.text ? Math.ceil(a.text.length / 3.6) : 0);

  function chipInfo(chip, a) {
    const s = document.createElement("span");
    s.className = "media-chip-info";
    if (a.pending) {
      s.textContent = "reading…";
    } else if (a.doc) {
      const unit = a.doc.type === "Excel" ? "sheet" : a.doc.type === "PowerPoint" ? "slide" : "page";
      const pages = a.doc.pages != null ? `, ${fmt(a.doc.pages)} ${unit}${a.doc.pages === 1 ? "" : "s"}` : "";
      s.textContent = `${kfmt(a.doc.tokens)} tokens`;
      chip.title = `${a.doc.type}${pages}: ${fmt(a.doc.tokens)} tokens, read in ${secs(a.doc.seconds)} the first time` +
                   (a.doc.note ? `\n${a.doc.note}` : "");
    } else {
      return;
    }
    chip.appendChild(s);
  }

  function attachmentSummary(box, list) {
    const docs = list.filter((a) => a.doc);
    const total = list.reduce((n, a) => n + tokensOf(a), 0);
    if (!docs.length || !total) return;
    const rate = docs.reduce((r, a) => (a.doc.seconds > 0 ? a.doc.tokens / a.doc.seconds : r), READ_TOK_S);
    const s = document.createElement("span");
    s.className = "media-att-sum";
    s.textContent = `${kfmt(total)} tokens · first read ${secs(total / rate)}, follow-ups reuse it`;
    if (health.max_context && total > health.max_context) {
      s.classList.add("media-att-sum--over");
      s.textContent = `${kfmt(total)} tokens: more than the ${ctxfmt(health.max_context)} context`;
    }
    box.appendChild(s);
  }

  // ---------------------------------------------------------------- dictation
  const MAX_REC_S = 300;
  let rec = null;                 // {stream, ctx, proc, chunks, rate, started, timer}
  let micBusy = false, downAt = 0, startedByDown = false, ignoreClick = false;

  function micButton() {
    let b = $("mic-btn");
    if (!b) {
      b = document.createElement("button");
      b.type = "button";
      b.id = "mic-btn";
      b.className = "st-btn st-btn--icon";
      b.setAttribute("aria-label", "Dictate");
      $("attach-btn").after(b);
      b.addEventListener("pointerdown", onDown);
      b.addEventListener("pointerup", onUp);
      b.addEventListener("click", onClick);
    }
    b.innerHTML = micBusy ? ICON.wait : rec ? ICON.stop : ICON.mic;
    b.classList.toggle("media-rec", !!rec);
    b.title = rec ? "Listening: release (or click) to stop, Esc to cancel"
      : voice.asr ? "Dictate: hold to talk, or click to start and again to stop"
      : "Dictation is off (Enable Voice.bat sets it up)";
  }
  function onDown(e) {
    if (e.button !== 0 || rec || micBusy) return;
    downAt = performance.now();
    startedByDown = true;
    start();
  }
  function onUp() {
    if (rec && startedByDown && performance.now() - downAt > 450) { ignoreClick = true; startedByDown = false; stop(); }
  }
  function onClick() {
    if (ignoreClick) { ignoreClick = false; return; }
    if (startedByDown) { startedByDown = false; return; }           // a short tap: keep listening until the next click
    if (rec) stop(); else if (!micBusy) start();                         // the keyboard (Enter / Space on the button)
  }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && rec) { e.preventDefault(); stop(true); } });

  async function start() {
    if (!voice.asr) {
      startedByDown = false;
      toast("info", "Dictation is off", "Run Enable Voice.bat in the DeepSeek folder, then start DeepSeek again.", 6000);
      return;
    }
    if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      startedByDown = false;
      toast("warn", "No microphone on this address", "Browsers allow the microphone only on https:// or on this PC " +
            "(http://127.0.0.1 or localhost). Open the page that way, or put HTTPS in front of the server.", 9000);
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true,
                                                                  noiseSuppression: true, autoGainControl: true}});
    } catch (e) {
      startedByDown = false;
      toast("warn", "No microphone", e.name === "NotAllowedError" ? "The browser was not allowed to use it." : String(e.message || e));
      return;
    }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const chunks = [];
    proc.onaudioprocess = (ev) => chunks.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
    src.connect(proc);
    proc.connect(ctx.destination);
    rec = {stream, ctx, proc, src, chunks, rate: ctx.sampleRate, started: performance.now(),
           timer: setTimeout(() => stop(), MAX_REC_S * 1000)};
    micButton();
  }

  async function stop(cancel = false) {
    const r = rec;
    if (!r) return;
    rec = null;
    clearTimeout(r.timer);
    try { r.src.disconnect(); r.proc.disconnect(); } catch (e) { /* already */ }
    r.stream.getTracks().forEach((t) => t.stop());
    r.ctx.close();
    micButton();
    const n = r.chunks.reduce((k, c) => k + c.length, 0);
    if (cancel || n / r.rate < 0.3) return;
    const pcm = new Float32Array(n);
    let o = 0;
    for (const c of r.chunks) { pcm.set(c, o); o += c.length; }
    micBusy = true;
    micButton();
    const slow = setTimeout(() => toast("info", "Starting speech recognition", "The first dictation loads the model (a few seconds)."), 2500);
    try {
      const text = await transcribe(wav16k(pcm, r.rate));
      if (text) insertText(text);
      else toast("info", "Nothing heard", "No words were recognized.");
    } catch (e) {
      toast("warn", "Dictation failed", e.message || String(e), 7000);
    } finally {
      clearTimeout(slow);
      micBusy = false;
      micButton();
    }
  }

  // Float32 at the recording rate -> a 16 kHz 16-bit mono WAV (averaging the samples each output sample covers)
  function wav16k(x, rate) {
    const ratio = rate / 16000, n = Math.floor(x.length / ratio);
    const out = new DataView(new ArrayBuffer(44 + n * 2));
    const str = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
    str(0, "RIFF"); out.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt ");
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
    out.setUint32(24, 16000, true); out.setUint32(28, 32000, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
    str(36, "data"); out.setUint32(40, n * 2, true);
    for (let i = 0; i < n; i++) {
      const a = Math.floor(i * ratio), b = Math.max(a + 1, Math.floor((i + 1) * ratio));
      let s = 0;
      for (let k = a; k < b && k < x.length; k++) s += x[k];
      const v = Math.max(-1, Math.min(1, s / (b - a)));
      out.setInt16(44 + i * 2, v < 0 ? v * 32768 : v * 32767, true);
    }
    return new Uint8Array(out.buffer);
  }
  function b64(bytes) {
    let s = "";
    for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 32768));
    return btoa(s);
  }
  async function transcribe(wav) {
    const r = await fetch("v1/audio/transcriptions", {method: "POST", headers: headers(true),
                                                      body: JSON.stringify({file: "data:audio/wav;base64," + b64(wav)})});
    let j = {};
    try { j = await r.json(); } catch (e) { /* not JSON */ }
    if (r.status === 401) throw new Error("this server needs an API key: add it under About > Settings");
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    return (j.text || "").trim();
  }
  function insertText(text) {
    const t = $("input");
    const a = t.selectionStart ?? t.value.length, b = t.selectionEnd ?? t.value.length;
    const before = t.value.slice(0, a), after = t.value.slice(b);
    const pre = before && !/\s$/.test(before) ? " " : "", post = after && !/^\s/.test(after) ? " " : "";
    t.value = before + pre + text + post + after;
    const at = (before + pre + text).length;
    t.focus();
    t.setSelectionRange(at, at);
    t.dispatchEvent(new Event("input"));
  }

  // ---------------------------------------------------------------- read aloud
  let playing = null;             // {m, pl}: the answer being read aloud

  function speakButton(b, m, streaming) {
    if (!b) return;
    b.hidden = streaming || !m || !m.text || !voice.tts;
    const on = playing && playing.m === m;
    b.innerHTML = on ? ICON.stop : ICON.speaker;
    b.title = on ? "Stop reading" : "Read aloud";
    b.setAttribute("aria-label", on ? "Stop reading the answer" : "Read the answer aloud");
  }
  function refreshSpeakers() {
    document.querySelectorAll("#chat [data-msg-speak]").forEach((b) => {
      const el = b.closest(".st-msg");
      if (el) speakButton(b, messages[+el.dataset.i], false);
    });
  }

  // what is said: the answer without code blocks, tables, Markdown marks and bare links; a heading ends as a sentence
  function speakable(md) {
    return md.replace(/```[\s\S]*?(```|$)/g, "\n")
      .replace(/^\s*\|.*$/gm, "")
      .replace(/<[^>]+>/g, " ")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/https?:\/\/\S+/g, " ")
      .replace(/`([^`]*)`/g, "$1")
      .replace(/^\s{0,3}#{1,6}\s+(.*?)[.:!?]?\s*$/gm, "$1.")
      .replace(/^\s{0,3}(>|[-*+]|\d+[.)])\s+/gm, "")
      .replace(/^\s*:?-{3,}.*$/gm, "")
      .replace(/(\*\*|__|\*|_|~~)(\S[\s\S]*?)\1/g, "$2")
      .replace(/[ \t]+/g, " ").trim();
  }
  // sentences, joined into pieces of up to ~240 characters (short pieces first, so the first sound comes soon)
  function pieces(text) {
    const sent = text.split(/(?<=[.!?。！？…])\s+|\n+/).map((s) => s.trim()).filter(Boolean);
    const out = [];
    for (const s of sent) {
      const parts = s.length > 400 ? s.split(/(?<=[,;:])\s+/) : [s];
      for (const p of parts) {
        const last = out[out.length - 1];
        const cap = out.length < 2 ? 120 : 240;
        // a line without an end mark (a list item) gets one, so the voice pauses there
        if (last && last.length + p.length < cap) out[out.length - 1] = `${last}${/[.!?。！？…,;:]$/.test(last) ? "" : "."} ${p}`;
        else out.push(p);
      }
    }
    return out;
  }

  // A queue of text pieces spoken in order: each piece is made while the one before plays.  push() adds, end() says
  // no more will come, stop() ends it at once; .done resolves when everything is said (or it was stopped).
  function player(onError) {
    const ctl = new AbortController();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const queue = [];
    let ended = false, wake = null, src = null;
    const fetchPart = async (text) => {
      const r = await fetch("v1/audio/speech", {method: "POST", headers: headers(true), signal: ctl.signal,
        body: JSON.stringify({input: text, voice: store.get("voice", "af_heart"), speed: +store.get("voice_speed", 1),
                              response_format: "wav"})});
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try { msg = (await r.json()).error.message || msg; } catch (e) { /* not JSON */ }
        throw new Error(msg);
      }
      return ctx.decodeAudioData(await r.arrayBuffer());
    };
    const p = {
      ctl,
      push(text) {
        if (!text || !text.trim() || ctl.signal.aborted) return;
        const job = fetchPart(text.trim());
        job.catch(() => {});                                // an error shows when it is awaited
        queue.push(job);
        if (wake) { wake(); wake = null; }
      },
      end() { ended = true; if (wake) { wake(); wake = null; } },
      stop() {
        ctl.abort();
        try { if (src) src.stop(); } catch (e) { /* not started */ }
        if (wake) { wake(); wake = null; }
      },
    };
    p.done = (async () => {
      const slow = setTimeout(() => toast("info", "Starting read-aloud", "The first time loads the voice (a few seconds)."), 3000);
      try {
        for (;;) {
          if (ctl.signal.aborted) return;
          if (!queue.length) {
            if (ended) return;
            await new Promise((ok) => { wake = ok; });
            continue;
          }
          const buf = await queue.shift();
          clearTimeout(slow);
          if (ctl.signal.aborted) return;
          await new Promise((done) => {
            src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.onended = done;
            ctl.signal.addEventListener("abort", done);
            src.start();
          });
        }
      } catch (e) {
        if (e.name !== "AbortError" && !ctl.signal.aborted && onError) onError(e);
      } finally {
        clearTimeout(slow);
        ctx.close();
      }
    })();
    return p;
  }
  const readAloudFailed = (e) => toast("warn", "Read-aloud failed", e.message || String(e), 7000);

  async function speak(m) {
    if (playing) { const same = playing.m === m; stopSpeaking(); if (same) return; }
    const parts = pieces(speakable(m.text || ""));
    if (!parts.length) return;
    const pl = player(readAloudFailed);
    playing = {m, pl};
    refreshSpeakers();
    parts.forEach((t) => pl.push(t));
    pl.end();
    await pl.done;
    if (playing && playing.pl === pl) { playing = null; refreshSpeakers(); }
  }
  function stopSpeaking() {
    const p = playing;
    if (!p) return;
    playing = null;
    p.pl.stop();
    refreshSpeakers();
  }
  $("chat").addEventListener("click", (e) => {
    const b = e.target.closest("[data-msg-speak]");
    if (!b) return;
    const el = b.closest(".st-msg");
    if (el && messages[+el.dataset.i]) speak(messages[+el.dataset.i]);
  });

  // ---------------------------------------------------------------- talk mode (hands-free)
  // The Talk button: speak, pause, and the question is sent by itself; the answer is read aloud sentence by sentence as
  // it is written (thinking off for these turns, so the first words come soon).  Talking while it answers stops the
  // answer and the reading and listens again (barge-in).  End of speech: an energy detector with a noise floor that
  // follows the room (~0.7 s of quiet ends a turn); the browser's echo cancellation keeps the answer's own sound out,
  // and while it speaks a louder, longer sound is needed to interrupt.  Headphones work best.
  const QUIET_END_MS = 700, MIN_SPEECH_MS = 350, BARGE_MS = 450, PREROLL = 3;
  let talk = null;

  function talkButton() {
    let b = $("talk-btn");
    if (!b) {
      b = document.createElement("button");
      b.type = "button";
      b.id = "talk-btn";
      b.className = "st-btn st-btn--icon";
      b.innerHTML = svg('<path d="M4 10v4M8 7v10M12 4v16M16 8v8M20 11v2"/>');
      b.onclick = () => (talk ? talkStop() : talkStart());
      $("mic-btn").after(b);
    }
    b.hidden = !(voice.asr && voice.tts);
    b.setAttribute("aria-pressed", String(!!talk));
    b.dataset.state = talk ? talk.state : "";
    const say = {listening: "listening", hearing: "hearing you", thinking: "working on it", speaking: "answering"};
    b.title = talk ? `Talk mode: ${say[talk.state] || talk.state} (click to end)` : "Talk: a hands-free conversation (click to start)";
    b.setAttribute("aria-label", b.title);
  }
  function setState(s) { if (talk) { talk.state = s; talkButton(); } }

  async function talkStart() {
    if (!window.isSecureContext || !navigator.mediaDevices) { start(); return; }   // start() says why there is no mic
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true,
                                                                  noiseSuppression: true, autoGainControl: true}});
    } catch (e) { toast("warn", "No microphone", String(e.message || e)); return; }
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    talk = {stream, ctx, src, proc, rate: ctx.sampleRate, state: "listening", noise: 0.01, pre: [], frames: [],
            speechMs: 0, quietMs: 0, loudMs: 0, pl: null, turn: 0};
    proc.onaudioprocess = (ev) => onFrame(new Float32Array(ev.inputBuffer.getChannelData(0)));
    src.connect(proc);
    proc.connect(ctx.destination);
    talkButton();
    toast("info", "Talk mode", "Speak; a short pause sends it. Talk over the answer to interrupt. Headphones work best.", 6000);
  }
  function talkStop() {
    const t = talk;
    if (!t) return;
    talk = null;
    if (t.pl) t.pl.stop();
    try { t.src.disconnect(); t.proc.disconnect(); } catch (e) { /* already */ }
    t.stream.getTracks().forEach((x) => x.stop());
    t.ctx.close();
    talkButton();
  }

  function onFrame(x) {
    const t = talk;
    if (!t) return;
    let e = 0;
    for (let i = 0; i < x.length; i++) e += x[i] * x[i];
    const rms = Math.sqrt(e / x.length), ms = (x.length / t.rate) * 1000;
    const answering = t.state === "thinking" || t.state === "speaking";
    const voiced = rms > Math.max(0.015, t.noise * 3);
    const loud = answering ? rms > Math.max(0.04, t.noise * 6) : voiced;
    if (!voiced && t.state !== "hearing") t.noise = t.noise * 0.95 + rms * 0.05;   // the room's level, while quiet
    if (answering) {                                                            // barge-in: a sustained loud sound
      t.loudMs = loud ? t.loudMs + ms : Math.max(0, t.loudMs - ms / 2);      // a dip between words is not the end
      t.pre.push(x);
      if (t.pre.length > PREROLL + 8) t.pre.shift();
      if (t.loudMs >= BARGE_MS) interrupt();
      return;
    }
    if (t.state === "listening") {
      t.pre.push(x);
      if (t.pre.length > PREROLL) t.pre.shift();
      if (loud) { t.frames = [...t.pre]; t.speechMs = ms; t.quietMs = 0; setState("hearing"); }
      return;
    }
    if (t.state === "hearing") {
      t.frames.push(x);
      if (loud) { t.speechMs += ms; t.quietMs = 0; } else t.quietMs += ms;
      if (t.quietMs >= QUIET_END_MS) {
        const frames = t.frames;
        t.frames = []; t.pre = [];
        if (t.speechMs < MIN_SPEECH_MS) { setState("listening"); return; }      // a click or a cough
        utterance(frames);
      }
    }
  }

  function interrupt() {
    const t = talk;
    if (!t) return;
    if (t.pl) { t.pl.stop(); t.pl = null; }
    if (busy) busy.controller.abort();                 // app.js: the answer being written
    t.turn++;
    t.frames = [...t.pre];
    t.pre = [];
    t.speechMs = t.loudMs;
    t.quietMs = 0;
    t.loudMs = 0;
    setState("hearing");
  }

  async function utterance(frames) {
    const t = talk;
    const turn = ++t.turn;
    setState("thinking");
    const n = frames.reduce((k, f) => k + f.length, 0), pcm = new Float32Array(n);
    let o = 0;
    for (const f of frames) { pcm.set(f, o); o += f.length; }
    let text = "";
    try { text = await transcribe(wav16k(pcm, t.rate)); } catch (e) { toast("warn", "Dictation failed", e.message || String(e)); }
    window.dispatchEvent(new CustomEvent("strata-talk", {detail: {text, seconds: n / t.rate, busy: !!busy}}));
    if (!talk || talk.turn !== turn) return;            // talk mode ended, or the user spoke again meanwhile
    if (!text || busy) { setState("listening"); return; }
    $("input").value = text;
    const prev = settings.thinking;
    settings.thinking = "none";                          // short spoken turns: no thinking (send() reads it at once)
    const sending = send();
    settings.thinking = prev;
    const m = messages[messages.length - 1];
    const pl = player(readAloudFailed);
    t.pl = pl;
    let said = 0;
    const feed = (final) => {                           // the finished sentences of what is written so far
      const sp = speakable(m.text || "");
      let cut = final ? sp.length : -1;
      if (!final) {
        const re = /[.!?。！？…:;](?=\s)|\n/g;
        let k;
        while ((k = re.exec(sp)) !== null) if (k.index + 1 > said) cut = k.index + 1;
      }
      if (cut > said) { pl.push(sp.slice(said, cut)); said = cut; if (talk && talk.turn === turn) setState("speaking"); }
    };
    const tick = setInterval(() => feed(false), 200);
    await sending;
    clearInterval(tick);
    if (!talk || talk.turn !== turn) { pl.stop(); return; }
    feed(true);
    pl.end();
    await pl.done;
    if (talk && talk.turn === turn) { t.pl = null; setState("listening"); }
  }

  micButton();
  talkButton();
  loadVoice();
  return {isDocument, addDocument, chipInfo, attachmentSummary, speakButton, speak, stopSpeaking,
          get voice() { return voice; }, speakable, pieces, wav16k};
})();
window.media = media;
