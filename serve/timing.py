"""serve/timing.py - where a request's time goes outside the engine's prompt timer (R-list "9.3 s a session"): opt-in
server-side timestamps, one log line per request with the time between them in ms.

On with STRATA_TIMING=1 in the environment or "timing": true in the run config; off by default.  Off, every call site
is behind `if timing.ON:` - nothing else runs.  On, the HTTP handler starts a request's ladder (start) on its thread,
the server stamps the steps it passes (stamp: the first time each one is reached counts; a step on another thread -
a background response - is not stamped) and the handler logs it when the request ends (finish):

    received     the request's line and headers were read (the handler started on it)
    parsed       its body was read and decoded as JSON
    converted    the client's format converted to the template's messages and the request checked (prepare entered)
    images       pictures a tool returned that cannot be read were noted (prepare)
    rendered     the chat template rendered: the prompt's text (chars=)
    tokenized    the prompt tokenized (tokens=)
    prepared     prepare() returned: context checks, strata_prefix
    turn         the engine's turn taken (the FIFO; a queued request waits here)
    loaded       the engine checked loaded (or started)
    sent         the GEN line written to the engine and flushed
    first line   the engine's first line read (RESUME, PP or T)
    first token  the first generated token read
    first sent   the first piece of the answer written to the client (the whole answer, without streaming)
    done         the request ended

    [strata] timing /v1/messages: parsed +1, converted +4, images +0, rendered +38, tokenized +92, prepared +0,
             turn +0, loaded +0, sent +3, first line +240, first token +812, first sent +1, done +5230 = 6421 ms
             (chars 81234, tokens 20311)

The stamp list follows brainstorm_1107/agent_sessions.md (round 2, S0-S16), server side only.
"""
from __future__ import annotations

import os
import threading
import time

ON = os.environ.get("STRATA_TIMING", "").strip().lower() in ("1", "true", "on", "yes")
STEPS = ("received", "parsed", "converted", "images", "rendered", "tokenized", "prepared", "turn", "loaded", "sent",
         "first line", "first token", "first sent", "done")
_local = threading.local()


def start(what: str) -> None:
    """A new ladder for the request this thread handles (`what`: its path), stamped `received` now."""
    _local.ladder = {"what": what, "marks": {"received": time.perf_counter()}, "info": {}}


def stamp(step: str, **info) -> None:
    """This thread's request reached `step` (the first time counts); `info`: numbers for the line (chars, tokens)."""
    ladder = getattr(_local, "ladder", None)
    if ladder is None:
        return
    ladder["marks"].setdefault(step, time.perf_counter())
    for k, v in info.items():
        ladder["info"].setdefault(k, v)


def line(ladder: dict) -> str:
    marks = sorted(ladder["marks"].items(), key=lambda kv: kv[1])
    parts, prev = [], marks[0][1]
    for step, t in marks[1:]:
        parts.append(f"{step} +{(t - prev) * 1000:.0f}")
        prev = t
    total = (marks[-1][1] - marks[0][1]) * 1000
    info = ", ".join(f"{k} {v}" for k, v in ladder["info"].items())
    return (f"[strata] timing {ladder['what']}: " + ", ".join(parts) + f" = {total:.0f} ms" +
            (f" ({info})" if info else ""))


def finish() -> str | None:
    """Stamp `done`, log this thread's ladder and forget it; -> the line (None when no ladder was started)."""
    ladder = getattr(_local, "ladder", None)
    if ladder is None:
        return None
    _local.ladder = None
    ladder["marks"].setdefault("done", time.perf_counter())
    text = line(ladder)
    print(text, flush=True)
    return text
