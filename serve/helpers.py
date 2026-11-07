"""Claude Code's helper calls (0.94): its small side requests, recognised by narrow signatures, and the main-first
order that lets the main conversation go ahead of a prompt suggestion.

Claude Code sends, next to the main conversation, requests of its own (to the "background" model, which with
ANTHROPIC_BASE_URL set is this server):

- a quota probe at start: one user message "quota", max_tokens 1;
- a session title / new-topic check: a short JSON-answer prompt with no tools;
- a Bash command-prefix check before a Bash call (`<policy_spec>` ... `Command:`);
- a file-path extraction after a Bash call (`Command:` ... `Output:` ... `<filepaths>`);
- a prompt suggestion after a reply: the whole main conversation plus a `[SUGGESTION MODE: ...]` user message.

Each rule below is a narrow match on that request's shape (third-party reverse engineering of Claude Code; see
docs/DETAILS.md "Claude Code's helper calls").  A request that matches none is served exactly as before.

What the server does with each class (Service.helper_calls; "claude_code_helpers": false turns all of it off):

- quota: answered at once, without the model (the engine is not even started);
- title, topic, bash_prefix, file_paths: thinking off (the launcher's config leaves anthropic_thinking at "model",
  so they thought for hundreds of tokens before writing a few words);
- suggestion: thinking unchanged (a fork of the main conversation: changing thinking changes the prompt's first
  tokens and the whole conversation would be read again), a small thinking budget, and pre-emptible: a main request
  that arrives takes the turn first, and a running suggestion is stopped (MainFirst).
"""
from __future__ import annotations

import threading
import time
import uuid

from serve.frontend import _text_of

# the classes, in the order they are checked
QUOTA, TITLE, TOPIC, BASH_PREFIX, FILE_PATHS, SUGGESTION = (
    "quota", "title", "topic", "bash_prefix", "file_paths", "suggestion")
THINKING_OFF = (TITLE, TOPIC, BASH_PREFIX, FILE_PATHS)
SUGGESTION_MARK = "[SUGGESTION MODE:"
SUGGESTION_BUDGET = 256          # a suggestion's thinking budget unless the request gives its own


def _messages(req: dict) -> list:
    m = req.get("messages")
    return m if isinstance(m, list) and all(isinstance(x, dict) for x in m) else []


def _user_texts(msgs: list) -> list[str]:
    return [_text_of(m.get("content")) if not isinstance(m.get("content"), str) else m["content"]
            for m in msgs if m.get("role") == "user"]


def classify(req: dict) -> str | None:
    """The helper class of an Anthropic Messages request, or None (the main conversation, or anything else)."""
    if not isinstance(req, dict):
        return None
    msgs = _messages(req)
    if not msgs:
        return None
    tools = req.get("tools")
    no_tools = not tools
    last = msgs[-1]
    last_text = last.get("content") if isinstance(last.get("content"), str) else _text_of(last.get("content"))
    # a prompt suggestion forks the main conversation (tools and all): only its last user message marks it
    if last.get("role") == "user" and isinstance(last_text, str) and last_text.lstrip().startswith(SUGGESTION_MARK):
        return SUGGESTION
    if not no_tools:
        return None
    system = req.get("system")
    system = system if isinstance(system, str) else _text_of(system) if isinstance(system, list) else ""
    users = _user_texts(msgs)
    if (req.get("max_tokens") == 1 and len(msgs) == 1 and msgs[0].get("role") == "user"
            and len(users) == 1 and users[0].strip().lower() == "quota"):
        return QUOTA
    if len(msgs) > 2:
        return None
    low = system.lower()
    if "isnewtopic" in low and "new conversation topic" in low:
        return TOPIC
    if "title" in low and ("sentence-case title" in low or (
            "return json" in low and "field" in low and ("coding session" in low or "this session" in low))):
        return TITLE
    if len(msgs) == 1 and len(users) == 1:
        text = users[0]
        if "<policy_spec>" in text and "Command:" in text:
            return BASH_PREFIX
        both = text + "\n" + system
        if "Command:" in text and "Output:" in text and "<filepaths>" in both.lower():
            return FILE_PATHS
    return None


def text_events(model: str, text: str, stop: str = "end_turn", input_tokens: int = 1, output_tokens: int = 1):
    """A finished answer as Anthropic stream events (anthropic_collect reads them for a JSON reply)."""
    mid = "msg_" + uuid.uuid4().hex[:24]
    yield "message_start", {"type": "message_start", "message": {
        "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}}
    yield "content_block_start", {"type": "content_block_start", "index": 0,
                                  "content_block": {"type": "text", "text": ""}}
    yield "content_block_delta", {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": text}}
    yield "content_block_stop", {"type": "content_block_stop", "index": 0}
    yield "message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                            "usage": {"input_tokens": input_tokens, "cache_read_input_tokens": 0,
                                      "output_tokens": output_tokens}}
    yield "message_stop", {"type": "message_stop"}


def quota_events(model: str):
    """The quota probe's answer: one word, max_tokens reached - what a model asked for one token gives, without
    the model."""
    return text_events(model, "ok", "max_tokens")


# ------------------------------------------------------------------------------------------------ the helper lane
LANE_CLASSES = (TITLE, TOPIC, FILE_PATHS)       # the Bash prefix check decides permissions: the main model's


class HelperLane:
    """0.94 (opt-in): a small local model (serve/embeddings.Sidecar: an OpenAI-compatible server on a second card or
    the CPU) answers Claude Code's small helper calls, so they neither wait for nor park the main conversation.

        "helper_lane": {"cmd": "llama-server", "args": ["-m", "Qwen3-1.7B-Q8_0.gguf", "--jinja", "-c", "8192",
                        "-ngl", "99", "--port", "{port}"], "model": "qwen3-1.7b",
                        "classes": ["title", "topic", "file_paths"]}

    The Bash command-prefix check is not sent there unless listed: Claude Code uses its answer to decide whether a
    command needs your permission, and a small model gets that wrong more often.  A lane that fails falls back to
    the main model."""

    def __init__(self, sidecar, classes=None, max_tokens: int = 512):
        self.sidecar = sidecar
        classes = list(classes) if classes is not None else list(LANE_CLASSES)
        bad = [c for c in classes if c not in THINKING_OFF]
        if bad:
            raise ValueError(f"helper_lane.classes: {', '.join(map(str, bad))} cannot go to the helper lane "
                             f"(choose from {', '.join(THINKING_OFF)})")
        self.classes = set(classes)
        self.max_tokens = int(max_tokens)

    def takes(self, kind) -> bool:
        return kind in self.classes

    def chat_request(self, req: dict) -> dict:
        """An Anthropic helper request -> the sidecar's OpenAI chat request (text only: helpers carry no tools)."""
        system = req.get("system")
        system = system if isinstance(system, str) else _text_of(system) if isinstance(system, list) else ""
        msgs = [{"role": "system", "content": system}] if system else []
        for m in req.get("messages") or []:
            c = m.get("content")
            msgs.append({"role": m.get("role"), "content": c if isinstance(c, str) else _text_of(c)})
        cap = req.get("max_tokens")
        cap = min(int(cap), self.max_tokens) if isinstance(cap, int) and cap > 0 else self.max_tokens
        out = {"model": self.sidecar.model, "messages": msgs, "max_tokens": cap, "stream": False,
               "chat_template_kwargs": {"enable_thinking": False}}
        for k in ("temperature", "top_p", "stop"):
            if req.get(k) is not None:
                out[k] = req[k]
        if req.get("stop_sequences"):
            out["stop"] = req["stop_sequences"]
        return out

    def answer(self, req: dict) -> tuple[str, str, dict] | None:
        """(text, Anthropic stop reason, usage) from the lane, or None when it failed (then the main model answers)."""
        try:
            status, out = self.sidecar.post("/v1/chat/completions", self.chat_request(req))
        except RuntimeError as e:
            print(f"[strata] helper lane: {e}; the main model answers", flush=True)
            return None
        if status != 200 or not isinstance(out, dict) or not out.get("choices"):
            print(f"[strata] helper lane answered {status}; the main model answers", flush=True)
            return None
        choice = out["choices"][0]
        text = (choice.get("message") or {}).get("content") or ""
        stop = "max_tokens" if choice.get("finish_reason") == "length" else "end_turn"
        return text, stop, out.get("usage") or {}


def apply(kind: str | None, req: dict, kwargs: dict) -> tuple[dict, dict]:
    """(request, template kwargs) for a helper class: thinking off for the small helpers; for a suggestion, the
    pre-emptible mark and a thinking budget.  Anything else is returned as it came."""
    if kind in THINKING_OFF:
        kwargs = {k: v for k, v in kwargs.items() if k != "reasoning_effort"}
        kwargs["enable_thinking"] = False
    elif kind == SUGGESTION:
        req = dict(req, strata_preemptible=True)
        if req.get("reasoning_budget_tokens") is None:
            req["reasoning_budget_tokens"] = SUGGESTION_BUDGET
    return req, kwargs


class MainFirst:
    """The order between main requests and pre-emptible ones (prompt suggestions) on the single engine turn.

    - A pre-emptible request does not take the turn while a main request waits for it.
    - A main request that arrives stops every pre-emptible one: a queued one gives its place up at once, a decoding
      one is cancelled, and one still reading its prompt is cancelled at its first token - the engine drops its
      conversation when a prompt read is cancelled (ds41_generate.cpp, cancel during prefill), and a suggestion reads
      only the few tokens past the main conversation, so it finishes that read first.
    """

    def __init__(self):
        self.lock = threading.Condition()
        self.main_waiting = 0
        self.entries: dict[int, dict] = {}       # id -> {"cancel", "state": queued|reading|decoding, "stop"}
        self.preempted = 0                       # how many were stopped (GET /metrics could show it)

    def enter(self, cancel: threading.Event, preemptible: bool, stoppable: bool = True) -> dict:
        """`preemptible`: the request waits while a main request waits; `stoppable` (a prompt suggestion): a main
        request also stops it.  A background response is preemptible but not stoppable."""
        e = {"cancel": cancel, "state": "queued", "stop": False, "pre": preemptible, "stoppable": stoppable}
        with self.lock:
            if preemptible:
                self.entries[id(e)] = e
            else:
                self.main_waiting += 1
                for other in self.entries.values():
                    self._stop(other)
            self.lock.notify_all()
        return e

    def _stop(self, e: dict) -> None:
        if e["stop"] or not e["stoppable"]:
            return
        e["stop"] = True
        self.preempted += 1
        if e["state"] != "reading":
            e["cancel"].set()
        print(f"[strata] helper call: a main request arrived; the prompt suggestion "
              f"{'stops at its first token' if e['state'] == 'reading' else 'is cancelled'}", flush=True)

    def acquire(self, e: dict, fifo, timeout: float) -> bool:
        """Take the engine's turn (`fifo`) within `timeout` seconds: True when taken.  A main request simply waits
        for it; a pre-emptible one never takes it while a main request waits, nor once it was told to stop."""
        if not e["pre"]:
            return fifo.acquire(timeout=timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if e["cancel"].is_set():
                return False
            with self.lock:
                blocked = self.main_waiting > 0 or e["stop"]
                if blocked:
                    self.lock.wait(min(0.25, max(0.0, deadline - time.monotonic())))
                    continue
            if fifo.acquire(timeout=min(0.25, max(0.01, deadline - time.monotonic()))):
                with self.lock:
                    if not (self.main_waiting > 0 or e["stop"]):
                        return True
                fifo.release()                   # a main request came while this one took the turn: its turn
        return False

    def took(self, e: dict) -> None:
        with self.lock:
            e["state"] = "reading"
            if not e["pre"]:
                self.main_waiting -= 1
                self.lock.notify_all()

    def first_token(self, e: dict) -> None:
        if e["state"] == "reading":
            with self.lock:
                e["state"] = "decoding"
                if e["stop"]:
                    e["cancel"].set()

    def leave(self, e: dict) -> None:
        with self.lock:
            if e["pre"]:
                self.entries.pop(id(e), None)
            elif e["state"] == "queued":         # gave up while queued
                self.main_waiting -= 1
            e["state"] = "done"
            self.lock.notify_all()
