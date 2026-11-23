"""serve/reasoning_memory.py - put back the thinking a client dropped from an earlier assistant turn (after upstream
Xolotl 3f0cb0c2, MIT).

Many agent clients (OpenCode and other OpenAI-chat clients, some Anthropic ones) send their history back without the
model's thinking.  DeepSeek V4.1's encoder renders a turn's thinking whenever the request has tools (and, without
tools, for the turns after the last user message), so the prompt then differs from the KV the engine holds at the
first such turn, and everything after it is read again every turn.

This keeps the thinking of recent replies, each under (the conversation it answered, its answer).  A history turn
without thinking gets it back only when BOTH hold:
- every message before it is the same as the messages the reply was generated after (a hash chain over the request's
  messages, with the thinking already put back into the earlier turns), and
- its answer text (outer whitespace aside) and tool calls (name and arguments) are the reply's.
A turn the client sent with its own thinking, or one no reply here wrote, is left as it came.  The thinking put back
is the reply's own text, as the chat completions response gave it.

Bounded: KEEP replies and KEEP_CHARS characters of thinking, least recently used first out.
"""
from __future__ import annotations

import collections
import hashlib
import json
import threading

KEEP = 4096                 # replies remembered
KEEP_CHARS = 32 << 20       # their thinking, in characters (~32-128 MB at most)
PENDING = 8                 # prompts prepared and not yet run (prepare -> run hand-off)


def _given(text) -> bool:
    return isinstance(text, str) and bool(text.strip())


def _text(content) -> str:
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content.strip() if isinstance(content, str) else ""


def answer_key(content, calls=()) -> bytes:
    """An assistant turn's answer: its text and its tool calls (name, arguments), as the reply wrote them (ToolCall)
    or as a client sends them back (OpenAI / the frontend's {"function": {...}})."""
    norm = []
    for c in calls or ():
        if hasattr(c, "name") and hasattr(c, "arguments"):
            name, args = c.name, c.arguments
        elif isinstance(c, dict):
            f = c.get("function") if isinstance(c.get("function"), dict) else c
            name, args = f.get("name"), f.get("arguments")
        else:
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        norm.append([name, args])
    return hashlib.sha256(json.dumps([_text(content), norm], sort_keys=True, default=repr).encode()).digest()


def _message_bytes(m) -> bytes:
    return json.dumps(m, sort_keys=True, default=repr).encode()


class ReasoningMemory:
    def __init__(self, keep: int = KEEP, keep_chars: int = KEEP_CHARS):
        self.keep, self.keep_chars = keep, keep_chars
        self.on = True                                    # the config's "restore_reasoning": false turns it off
        self.kept: collections.OrderedDict[tuple[bytes, bytes], str] = collections.OrderedDict()
        self.chars = 0
        self.pending: collections.OrderedDict[int, tuple[list, bytes]] = collections.OrderedDict()
        self.mu = threading.Lock()
        self.model = None                                 # R-6: the model key the kept thinking was written under

    def __len__(self):
        return len(self.kept)

    def clear(self):
        with self.mu:
            self.kept.clear()
            self.pending.clear()
            self.chars = 0

    def bind(self, model_key) -> bool:
        """R-6: the kept thinking belongs to the model that wrote it (the run config's "model_key").  Bound to another
        key, the memory starts empty.  -> True when something was dropped.  Today the key is set once per server
        process (a model change is a new start); this keeps it right if a model is ever swapped in place."""
        if model_key == self.model:
            return False
        had = bool(self.kept or self.pending)
        self.clear()
        self.model = model_key
        return had

    def restore(self, messages) -> tuple[list, bytes | None, int, bool]:
        """-> (messages, context, put back, missed).  `messages` with the thinking put back into the turns that lost
        it (changed turns are copies: the caller's dicts stay as they came); `context` identifies the conversation
        so far, for remember(); `missed`: the last assistant turn came without thinking and none was found."""
        if not self.on:
            return messages, None, 0, False
        h = hashlib.sha256()
        out, n, missed = None, 0, False
        with self.mu:
            for i, m in enumerate(messages):
                if (self.kept and isinstance(m, dict) and m.get("role") == "assistant"
                        and not _given(m.get("reasoning_content"))):
                    key = (h.digest(), answer_key(m.get("content"), m.get("tool_calls")))
                    r = self.kept.get(key)
                    if r is not None:
                        self.kept.move_to_end(key)        # a conversation in use keeps its turns' thinking
                        out = list(messages) if out is None else out
                        m = out[i] = {**m, "reasoning_content": r}
                        n += 1
                        missed = False
                    else:
                        missed = True
                elif isinstance(m, dict) and m.get("role") == "assistant":
                    missed = False
                h.update(_message_bytes(m))
                h.update(b"\0")
        return (messages if out is None else out), h.digest(), n, missed

    def prepared(self, ids, context):
        """prepare() -> run(): the prompt `ids` was rendered from the conversation `context`."""
        if context is None:
            return
        with self.mu:
            self.pending[id(ids)] = (ids, context)
            while len(self.pending) > PENDING:
                self.pending.popitem(last=False)

    def context_of(self, ids):
        with self.mu:
            got = self.pending.pop(id(ids), None)
        return got[1] if got is not None and got[0] is ids else None

    def remember(self, context, content, calls, reasoning):
        """A finished reply: its thinking, under the conversation it answered and its answer."""
        if context is None or not self.on or not _given(reasoning) or (not _text(content) and not calls):
            return
        key = (context, answer_key(content, calls))
        with self.mu:
            old = self.kept.pop(key, None)
            if old is not None:
                self.chars -= len(old)
            self.kept[key] = reasoning
            self.chars += len(reasoning)
            while self.kept and (len(self.kept) > self.keep or self.chars > self.keep_chars):
                _, r = self.kept.popitem(last=False)
                self.chars -= len(r)
