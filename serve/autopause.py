"""Opt-in auto-pause: give the graphics card back when another program (a game) starts using it.

"auto_pause_mib": N in the run config (the chat page's Setup tab sets it for the launcher; 0 or absent = off).  Every
few seconds, while the model is loaded and has been idle for QUIET_S, the card's readings (NVML) are compared with
what they were once it went quiet:
  * its used memory grew by N MiB or more, or
  * its load stayed at BUSY_UTIL % or more for BUSY_S seconds (this server's engine does nothing while idle);
then the model is unloaded the normal way (Service.unload: never while a request runs or waits) and the page says
why.  The next message or API request loads it again, as after Pause.  A heuristic: on Windows the card's memory
cannot grow past what the engine left free, so a game is mostly seen by its load.
"""
from __future__ import annotations

import threading
import time

from serve import loadstate

QUIET_S = 30.0          # idle this long before the card is watched (a request's own load and memory settle first)
BUSY_UTIL = 40          # % load ...
BUSY_S = 20.0           # ... for this long
INTERVAL_S = 5.0


def nvml_reader(index: int):
    """() -> {"used_mib", "util"} of the engine's card, or None (no NVIDIA card / driver)"""
    from serve.telemetry import _Nvml
    nv = _Nvml(index)
    if not nv.ok():
        return lambda: None

    def read():
        r = nv.read()
        if r.get("mem_used") is None and r.get("util") is None:
            return None
        return {"used_mib": (r["mem_used"] >> 20) if r.get("mem_used") is not None else None, "util": r.get("util")}
    return read


class AutoPause:
    def __init__(self, svc, read=None, clock=time.time, say=print):
        self.svc, self.read, self.clock, self.say = svc, read, clock, say
        self.base = None            # the used memory once quiet
        self.busy_since = None
        self.seen_activity = None   # svc.last_request_at when the base was taken: a newer request takes a new base

    def reset(self):
        self.base, self.busy_since = None, None

    def step(self) -> str | None:
        """one check.  -> the reason, when it paused the model"""
        svc = self.svc
        mib = int(getattr(svc, "auto_pause_mib", 0) or 0)
        engine = getattr(svc, "engine", None)
        if mib <= 0 or not hasattr(engine, "unload") or not svc.loaded() or getattr(engine, "starting", False):
            self.reset()
            return None
        with svc.status_lock:
            busy = svc.status.get("busy") or svc.status.get("queued")
        last = svc.last_request_at or getattr(svc, "started_at", 0) or 0
        now = self.clock()
        if busy or now - last < QUIET_S or last != self.seen_activity and self.base is not None:
            self.reset()
            if not busy and now - last >= QUIET_S:
                self.seen_activity = last
            return None
        if self.read is None:
            self.read = nvml_reader(int(getattr(svc, "gpu_index", 0) or 0))
        r = self.read()
        if not r:
            return None
        if self.base is None:
            self.base, self.seen_activity = r.get("used_mib"), last
            return None
        why = None
        used, util = r.get("used_mib"), r.get("util")
        if used is not None and self.base is not None and used - self.base >= mib:
            why = f"another program took {(used - self.base) / 1024:.1f} GB of the graphics card's memory"
        if util is not None and util >= BUSY_UTIL:
            self.busy_since = self.busy_since or now
            if why is None and now - self.busy_since >= BUSY_S:
                why = f"another program keeps the graphics card busy ({util}% load)"
        else:
            self.busy_since = None
        if why is None:
            return None
        if svc.unload() != "unloaded":      # a request arrived meanwhile: never mid-request
            self.reset()
            return None
        self.reset()
        loadstate.paused(f"Paused by itself: {why}.")
        self.say(f"[strata] auto-pause: {why}; the model is unloaded (the next request loads it again)", flush=True)
        return why


def start(svc, interval: float = INTERVAL_S) -> AutoPause:
    """the watcher thread (it does nothing while auto_pause_mib is 0, so the page can switch it on without a restart)"""
    ap = AutoPause(svc)

    def loop():
        while True:
            time.sleep(interval)
            try:
                ap.step()
            except Exception as e:          # noqa: BLE001  never take the server down over this
                print(f"[strata] auto-pause: {e}", flush=True)
    threading.Thread(target=loop, daemon=True, name="strata-auto-pause").start()
    return ap
