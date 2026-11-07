"""What the model is doing about being loaded, for the web page's loading screen and its Pause / Resume button.

The start narrator (server.narrate_start) reads the engine's log and says each step in the server window; it also
calls step() here, so GET /health can tell the page the same steps.  Service.ensure_loaded() calls begin() / done() /
failed() around an engine start, and done() remembers how long the start took (the run config's
"<config>.last-start.json"), so the next start can show "ready in about 40 s (last start: 58 s)".

GET /health gains two keys (no API key needed: no prompt text, only the state):
  "model_state": "loaded" | "loading" | "paused" (unloaded: the next request or /load loads it) | "failed"
  "load": {"step": key, "text": words, "index": n, "steps": N, "elapsed_s": s, "last_s": s, "eta_s": s,
           "error": text (failed only)}
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

# the start's steps, in order, as the narrator finds them in the engine's log (narrate_start's keys)
STEPS = [
    ("weights", "Reading the model's weights"),
    ("arena", "Loading the experts into RAM - the PC can be slow or stop responding for 1-3 minutes; this is normal"),
    ("loaded", "Experts loaded"),
    ("cache", "Filling the graphics card's expert cache"),
    ("up", "Almost ready"),
]
_KEYS = [k for k, _ in STEPS]

_lock = threading.Lock()
_state = {"active": False, "t0": None, "step": None, "detail": None, "error": None, "last_s": None}
last_start_path: str | None = None          # set by main() from the run config: <config>.last-start.json


def _read_last() -> float | None:
    if not last_start_path:
        return None
    try:
        v = json.loads(Path(last_start_path).read_text(encoding="utf-8")).get("seconds")
        return float(v) if v and float(v) > 0 else None
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def begin() -> None:
    """an engine start begins (a lazy first load, a Resume, a request after a pause, a restart after a crash)"""
    with _lock:
        _state.update(active=True, t0=time.time(), step=None, detail=None, error=None, last_s=_read_last())


def step(key: str, detail: str | None = None) -> None:
    """the narrator found this step in the engine's log (unknown keys are ignored)"""
    if key not in _KEYS:
        return
    with _lock:
        if not _state["active"]:            # a start not begun here (an engine restarted by other code): still shown
            _state.update(active=True, t0=time.time(), error=None, last_s=_read_last())
        _state.update(step=key, detail=detail)


def done() -> None:
    """the engine said READY: remember how long it took (the next start's estimate)"""
    with _lock:
        took = time.time() - _state["t0"] if _state["t0"] else None
        _state.update(active=False, step=None, detail=None, error=None)
    if took and last_start_path:
        try:
            Path(last_start_path).write_text(json.dumps({"seconds": round(took, 1), "at": int(time.time())}),
                                             encoding="utf-8")
        except OSError:
            pass


def failed(error: str) -> None:
    with _lock:
        _state.update(active=False, error=str(error)[:2000])


def clear_error() -> None:
    with _lock:
        _state["error"] = None


def health(svc) -> dict:
    """the two keys GET /health adds"""
    with _lock:
        st = dict(_state)
    engine = getattr(svc, "engine", None)
    loaded = svc.loaded()
    starting = bool(getattr(engine, "starting", False))
    if loaded and not st["active"]:
        return {"model_state": "loaded"}
    now = time.time()
    if st["active"] or starting:
        idx = _KEYS.index(st["step"]) if st["step"] in _KEYS else 0
        elapsed = now - st["t0"] if st["t0"] else 0.0
        last = st["last_s"]
        text = dict(STEPS)[st["step"] or "weights"]
        if st["detail"]:
            text += f" ({st['detail']})"
        return {"model_state": "loading",
                "load": {"step": st["step"] or "weights", "text": text, "index": idx + 1, "steps": len(STEPS),
                         "elapsed_s": round(elapsed, 1), "last_s": last,
                         "eta_s": round(max(0.0, last - elapsed), 1) if last else None}}
    if st["error"]:
        return {"model_state": "failed", "load": {"error": st["error"]}}
    if getattr(engine, "unloaded", False) or not hasattr(engine, "alive"):
        return {"model_state": "paused", "load": {"last_s": _read_last()}}
    return {"model_state": "paused", "load": {"last_s": _read_last(), "stopped": True}}


class LoadingMock:
    """--engine mock --mock-load-s N: a mock engine that can be paused and resumed, and takes N seconds to "load"
    through the real steps (for the web page's loading screen and Pause button, without a GPU)."""

    def __init__(self, inner, load_s: float, start_loaded: bool = False):
        self.inner, self.load_s = inner, float(load_s)
        self.unloaded = not start_loaded
        self.starting = False
        self.max_context = inner.max_context

    def __getattr__(self, name):            # generate, tok, scripts ...: the wrapped mock's
        return getattr(self.inner, name)

    def alive(self) -> bool:
        return not self.unloaded and not self.starting

    def unload(self):
        self.unloaded = True

    def close(self):
        self.unloaded = True

    def restart(self):
        self.starting = True
        try:
            for key, _ in STEPS:
                step(key, "mock" if key == "loaded" else None)
                time.sleep(self.load_s / len(STEPS))
            self.unloaded = False
        finally:
            self.starting = False
