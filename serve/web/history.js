// serve/web/history.js - the chat list, on top of app.js (messages, settings, renderChat, toast, store, esc, ext ...)
// and everyday.js (p4icon).  Chats are saved in this browser's IndexedDB only when "Save chats in this browser" is on
// - off by default, as before: then every chat lives in the tab only.  With it on: search, pins, rename, delete,
// export / import (JSON), and a "temporary chat" that is not saved.  Nothing goes to the server.
"use strict";

const hist = {db: null, chats: [], current: null, temporary: false, query: "", editing: null};

function hdbOpen() {
  return new Promise((resolve, reject) => {
    let r;
    try { r = indexedDB.open("delta-chats", 1); } catch (e) { reject(e); return; }
    r.onupgradeneeded = () => r.result.createObjectStore("chats", {keyPath: "id"});
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
    r.onblocked = () => reject(new Error("blocked"));
  });
}
function hdb(mode, fn) {
  return new Promise((resolve, reject) => {
    const tx = hist.db.transaction("chats", mode);
    const req = fn(tx.objectStore("chats"));
    tx.oncomplete = () => resolve(req ? req.result : undefined);
    tx.onerror = () => reject(tx.error);
    tx.onabort = () => reject(tx.error);
  });
}
const hput = (c) => hdb("readwrite", (s) => s.put(c)).catch((e) => toast("error", "Chat not saved", String(e.message || e), 6000));
const hdel = (id) => hdb("readwrite", (s) => s.delete(id));

function newId() { return `c${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`; }
function titleOf(msgs) {
  const first = msgs.find((m) => m.role === "user" && (m.text || "").trim());
  const t = first ? first.text.trim().replace(/\s+/g, " ") : (msgs.find((m) => m.role === "user") ? "Files and pictures" : "New chat");
  return t.length > 60 ? `${t.slice(0, 57)}…` : t;
}
function copyMessages(msgs) { return JSON.parse(JSON.stringify(msgs)); }

// ------------------------------------------------------------------ saving (app.js calls ext.save after each answer)
function histSave() {
  if (!settings.keep || hist.temporary || !messages.length || busy) return;
  let c = hist.chats.find((x) => x.id === hist.current);
  if (!c) {
    c = {id: newId(), created: Date.now(), pinned: false, named: false};
    hist.chats.push(c);
    hist.current = c.id;
  }
  c.messages = copyMessages(messages);
  c.updated = Date.now();
  if (!c.named) c.title = titleOf(messages);
  hput(c);
  paintHistory();
}
function histNew(temporary = false) {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return true; }
  if (!settings.keep && !temporary) return false;          // no history: app.js clears the chat (with Undo), as before
  histSave();
  messages = [];
  hist.current = null;
  hist.temporary = temporary;
  renderChat();
  paintHistory();
  $("input").focus();
  return true;
}
function histOpen(id) {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  const c = hist.chats.find((x) => x.id === id);
  if (!c) return;
  if (!hist.temporary) histSave();
  else if (messages.length && !confirm("This temporary chat is not saved. Leave it?")) return;
  messages = copyMessages(c.messages || []);
  hist.current = c.id;
  hist.temporary = false;
  renderChat();
  paintHistory();
  if (matchMedia("(max-width: 900px)").matches) histShow(false);
}
function histDelete(id) {
  const i = hist.chats.findIndex((x) => x.id === id);
  if (i < 0) return;
  const [c] = hist.chats.splice(i, 1);
  hdel(id);
  if (hist.current === id) { messages = []; hist.current = null; renderChat(); }
  paintHistory();
  toast("info", "Chat deleted", c.title, 6000, {label: "Undo", run: () => { hist.chats.push(c); hput(c); paintHistory(); }});
}
function histRename(id, title) {
  const c = hist.chats.find((x) => x.id === id);
  if (!c) return;
  title = (title || "").trim();
  if (title) { c.title = title.slice(0, 120); c.named = true; } else { c.named = false; c.title = titleOf(c.messages || []); }
  hput(c);
}
function histPin(id) {
  const c = hist.chats.find((x) => x.id === id);
  if (!c) return;
  c.pinned = !c.pinned;
  hput(c);
  paintHistory();
}

// "Save chats in this browser" switched (here or in Sampling)
function histKeep(on) {
  try { localStorage.removeItem("strata.chat"); } catch (e) { /* ignore */ }   // the one-chat copy of before
  if (on) { histSave(); paintHistory(); return; }
  const n = hist.chats.length;
  if (n && confirm(`Chats are no longer saved. Also delete the ${n} chat${n > 1 ? "s" : ""} saved in this browser?\n\n` +
                   "OK: delete them now.  Cancel: keep them (you can delete them later here).")) {
    hdb("readwrite", (s) => s.clear()).then(() => { hist.chats = []; hist.current = null; paintHistory(); });
  }
  paintHistory();
}
function setKeep(on) {
  settings = {...settings, keep: on};
  store.set("sampling", settings);
  histKeep(on);
}

// ------------------------------------------------------------------ export / import
function histExport() {
  if (!hist.chats.length) { toast("info", "No saved chats yet"); return; }
  const data = {format: "delta-chats", version: 1, exported: new Date().toISOString(), chats: hist.chats};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], {type: "application/json"}));
  a.download = `delta-chats-${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
async function histImport(file) {
  if (!settings.keep) { toast("info", "Save chats is off", "Turn on \"Save chats in this browser\" to import chats.", 6000); return; }
  let data;
  try { data = JSON.parse(await file.text()); } catch (e) { toast("error", "Not imported", "That file is not a chats export (JSON).", 6000); return; }
  const list = Array.isArray(data) ? data : Array.isArray(data.chats) ? data.chats : null;
  if (!list) { toast("error", "Not imported", "No chats in that file.", 6000); return; }
  let n = 0;
  for (const x of list) {
    if (!x || !Array.isArray(x.messages)) continue;
    const msgs = x.messages.filter((m) => m && (m.role === "user" || m.role === "assistant"))
      .map((m) => ({...m, text: String(m.text || ""), time: +m.time || Date.now()}));
    if (!msgs.length) continue;
    const have = hist.chats.find((c) => c.id === x.id);
    if (have && JSON.stringify(have.messages) === JSON.stringify(msgs)) continue;   // already here
    const c = {id: !x.id || have ? newId() : String(x.id), title: String(x.title || titleOf(msgs)).slice(0, 120), named: !!x.named,
               pinned: !!x.pinned, created: +x.created || Date.now(), updated: +x.updated || Date.now(), messages: msgs};
    hist.chats.push(c);
    await hput(c);
    n += 1;
  }
  paintHistory();
  toast(n ? "success" : "info", n ? `Imported ${n} chat${n > 1 ? "s" : ""}` : "Nothing new to import");
}

// ------------------------------------------------------------------ the sidebar
function dayGroup(t) {
  const d = new Date(t), now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = (day(now) - day(d)) / 86400000;
  return diff < 1 ? "Today" : diff < 2 ? "Yesterday" : diff < 7 ? "Previous 7 days" : diff < 30 ? "Previous 30 days" : "Older";
}
function matches(c, q) {
  if (!q) return true;
  if ((c.title || "").toLowerCase().includes(q)) return true;
  return (c.messages || []).some((m) => (m.text || "").toLowerCase().includes(q));
}
function paintHistory() {
  const el = $("history");
  const q = hist.query.toLowerCase();
  const shown = hist.chats.filter((c) => matches(c, q)).sort((a, b) => (b.pinned - a.pinned) || (b.updated - a.updated));
  const groups = [];
  for (const c of shown) {
    const g = c.pinned ? "Pinned" : dayGroup(c.updated);
    if (!groups.length || groups[groups.length - 1][0] !== g) groups.push([g, []]);
    groups[groups.length - 1][1].push(c);
  }
  const item = (c) => `<div class="h-item${c.id === hist.current ? " is-current" : ""}" data-id="${esc(c.id)}">` +
    (hist.editing === c.id ? `<input class="st-input h-rename" value="${esc(c.title)}" aria-label="Chat name" maxlength="120">`
      : `<button type="button" class="h-open" title="${esc(c.title)}">${esc(c.title)}</button>`) +
    `<span class="h-actions"><button type="button" class="st-btn st-btn--icon" data-act="pin" title="${c.pinned ? "Unpin" : "Pin"}" ` +
    `aria-label="${c.pinned ? "Unpin" : "Pin"}"${c.pinned ? ' aria-pressed="true"' : ""}>${p4icon("pin")}</button>` +
    `<button type="button" class="st-btn st-btn--icon" data-act="rename" title="Rename" aria-label="Rename">${p4icon("pencil")}</button>` +
    `<button type="button" class="st-btn st-btn--icon" data-act="delete" title="Delete" aria-label="Delete">${icon("trash")}</button></span></div>`;
  let list;
  if (!hist.db) list = `<p class="muted small h-note">This browser cannot keep chats for this page (private window or blocked site data).</p>`;
  else if (!settings.keep) list = `<p class="muted small h-note">Chats are not saved: each one lives in this tab only and is gone when you ` +
    `close it (the private default).${hist.chats.length ? ` ${hist.chats.length} chat${hist.chats.length > 1 ? "s" : ""} saved earlier ${hist.chats.length > 1 ? "are" : "is"} still here: turn saving on to see ${hist.chats.length > 1 ? "them" : "it"}.` : ""}</p>`;
  else if (!shown.length) list = `<p class="muted small h-note">${q ? "No chat matches." : "No saved chats yet. They appear here after the first answer."}</p>`;
  else list = groups.map(([g, cs]) => `<div class="h-group">${esc(g)}</div>` + cs.map(item).join("")).join("");
  const keepOn = !!settings.keep && !!hist.db;
  el.innerHTML = `<div class="history__head"><span class="card-title">Chats</span><span class="spacer"></span>` +
    `<button type="button" class="st-btn st-btn--icon" data-act="new" title="New chat" aria-label="New chat">${icon("new-chat")}</button>` +
    `<button type="button" class="st-btn st-btn--icon h-close" data-act="close" title="Close" aria-label="Close the chat list">${icon("chevron")}</button></div>` +
    (keepOn ? `<label class="history__search">${p4icon("search", "st-icon st-icon--sm")}<input type="search" id="h-search" ` +
      `placeholder="Search chats" value="${esc(hist.query)}" aria-label="Search chats"></label>` : "") +
    (hist.temporary ? `<div class="h-temp">${icon("info", "st-icon st-icon--sm")}<span>Temporary chat: not saved.</span>` +
      `<button type="button" class="st-btn st-btn--secondary" data-act="keep-this">Save it</button></div>` : "") +
    `<div class="history__list">${list}</div>` +
    `<div class="history__foot"><div class="toggle-row"><span>Save chats in this browser<br><span class="muted small">` +
    `${keepOn ? "kept on this PC only, until you delete them" : "off: nothing is written to disk"}</span></span>` +
    `<button class="st-toggle" role="switch" data-act="keep" aria-checked="${keepOn}" aria-label="Save chats in this browser"${hist.db ? "" : " disabled"}></button></div>` +
    (keepOn ? `<div class="h-buttons"><button type="button" class="st-btn st-btn--secondary" data-act="temp" title="A chat that is not saved">New temporary chat</button>` +
      `<button type="button" class="st-btn st-btn--icon" data-act="export" title="Export all chats (JSON)" aria-label="Export all chats">${icon("download")}</button>` +
      `<button type="button" class="st-btn st-btn--icon" data-act="import" title="Import chats (JSON)" aria-label="Import chats">${p4icon("upload")}</button>` +
      `<input type="file" id="h-file" accept=".json,application/json" hidden></div>` : "") + `</div>`;
  const search = $("h-search");
  if (search) {
    search.oninput = () => { hist.query = search.value; paintHistory(); const s2 = $("h-search"); s2.focus(); s2.setSelectionRange(s2.value.length, s2.value.length); };
  }
  const ren = el.querySelector(".h-rename");
  if (ren) {
    ren.focus(); ren.select();
    const finish = (save) => { if (hist.editing === null) return; const id = hist.editing; hist.editing = null; if (save) histRename(id, ren.value); paintHistory(); };
    ren.onkeydown = (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); };
    ren.onblur = () => finish(true);
  }
  const f = $("h-file");
  if (f) f.onchange = () => { if (f.files[0]) histImport(f.files[0]); f.value = ""; };
}
function histShow(open) {
  $("history").dataset.open = String(open);
  store.set("history_open", open);
  $("history-btn").setAttribute("aria-expanded", String(open));
}
$("history").addEventListener("click", (e) => {
  const b = e.target.closest("[data-act], .h-open");
  if (!b) return;
  const row = b.closest(".h-item");
  const id = row ? row.dataset.id : null;
  const act = b.classList.contains("h-open") ? "open" : b.dataset.act;
  if (act === "open") histOpen(id);
  else if (act === "pin") histPin(id);
  else if (act === "rename") { hist.editing = id; paintHistory(); }
  else if (act === "delete") histDelete(id);
  else if (act === "new") histNew(false) || $("new-btn").click();
  else if (act === "temp") histNew(true);
  else if (act === "keep-this") { hist.temporary = false; histSave(); paintHistory(); }
  else if (act === "keep") setKeep(!settings.keep);
  else if (act === "export") histExport();
  else if (act === "import") $("h-file").click();
  else if (act === "close") histShow(false);
});
$("history-btn").addEventListener("click", () => histShow($("history").dataset.open !== "true"));

// ------------------------------------------------------------------ start
(async () => {
  try {
    hist.db = await hdbOpen();
    hist.chats = (await hdb("readonly", (s) => s.getAll())) || [];
  } catch (e) {
    hist.db = null;                                   // no IndexedDB: app.js's one-chat localStorage copy goes on
    paintHistory();
    return;
  }
  // the chat app.js restored from its one-chat localStorage copy (before 0.94) becomes the first saved chat
  let legacy = null;
  try { legacy = JSON.parse(localStorage.getItem("strata.chat") || "null"); } catch (e) { /* ignore */ }
  if (settings.keep && Array.isArray(legacy) && legacy.length) {
    const c = {id: newId(), title: titleOf(legacy), named: false, pinned: false, created: legacy[0].time || Date.now(),
               updated: legacy[legacy.length - 1].time || Date.now(), messages: legacy};
    hist.chats.push(c);
    await hput(c);
    hist.current = c.id;
  }
  try { if (settings.keep) localStorage.removeItem("strata.chat"); } catch (e) { /* ignore */ }
  ext.save = histSave;
  ext.newChat = () => histNew(false);
  ext.keepChanged = histKeep;
  // with saving on, the page opens on the last chat, as the one-chat copy did
  if (settings.keep && !hist.current && !messages.length && hist.chats.length) {
    const last = [...hist.chats].sort((a, b) => b.updated - a.updated)[0];
    messages = copyMessages(last.messages || []);
    hist.current = last.id;
    renderChat();
  }
  paintHistory();
})();
histShow(store.get("history_open", !matchMedia("(max-width: 900px)").matches));
paintHistory();
