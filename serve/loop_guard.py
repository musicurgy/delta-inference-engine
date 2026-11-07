"""serve/loop_guard.py - the tool-call loop guard (BACKLOG #69): an agent that makes the same tool call - the same name
and the same arguments - again and again, turn after turn, gets one short note before its next turn: it is repeating
itself, so it should try something else or stop.

The design follows Project Maya's serve/loop_guard.py (MIT; its `tool_repeat` and `tool_note`): the conversation the
client sends is read for the repeated call, the note is a message of its own after the last tool result (inside the
result a model ran the call again; as a message of its own it moved on), and only this request's prompt changes - the
client's conversation is not touched, so its next request matches the cached prompt up to that result.  Simpler than
Maya's: no blocking of calls, no results compared - a run of the same call N times in a row is enough.

    repeated(messages, n)   -> (tool names, times) when the request answers the results of a call that is the same as
                               the ones before it, N times in a row (else None)
    note(names, times)      the note's text
    setting(value)          the run config's "loop_guard": absent or true -> REPEATS, {"repeats": N} -> N, false -> off

Calls are compared by name and arguments in one form (args_key: a client's JSON string and a parsed dict compare
equal; key order does not matter).  An assistant turn with several calls counts as one step: the same batch again is a
repeat.  An assistant turn without a call ends the run (the model said something in between), and so does a different
call.  DeepSeek writes its calls as DSML (#844 in Maya); the client sends them back as JSON like any other, so nothing
here depends on the template.
"""
from __future__ import annotations

import json

REPEATS = 4     # the same call this many times in a row: a loop (2 at the least; a single call is no repeat)


def args_key(value) -> str:
    """A call's arguments in one form: a client's JSON string and the parser's dict compare equal (Maya's args_key)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            pass
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _calls(message: dict) -> tuple:
    """An assistant message's tool calls as ((name, args_key), ...), in their order."""
    out = []
    for c in message.get("tool_calls") or []:
        fn = c.get("function") if isinstance(c, dict) and isinstance(c.get("function"), dict) else c
        if not isinstance(fn, dict):
            continue
        out.append((str(fn.get("name")), args_key(fn.get("arguments"))))
    return tuple(out)


def repeated(messages: list[dict], repeats: int = REPEATS) -> tuple[str, int] | None:
    """The template messages of a request: does it answer tool results of a call (or a batch of calls) that the model
    made `repeats` times or more in a row?  -> (the calls' names, how many times) or None."""
    if not repeats or not messages:
        return None
    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), None)
    if last is None or not messages[last].get("tool_calls"):
        return None
    if not any(m.get("role") == "tool" for m in messages[last + 1:]):
        return None                                   # the request does not answer that call's results
    key, times = _calls(messages[last]), 0
    if not key:
        return None
    for m in reversed(messages[:last + 1]):
        if m.get("role") != "assistant":
            continue                                  # tool results, and the client's own messages between
        if not m.get("tool_calls") or _calls(m) != key:
            break
        times += 1
    if times < repeats:
        return None
    names = ", ".join(dict.fromkeys(name for name, _ in key))
    return names, times


def note(names: str, times: int) -> str:
    return (f"[Note from the server, not the user: you have made the same tool call ({names}, with the same "
            f"arguments) {times} times in a row. Repeating it will not give a different result. Try something "
            "different, or stop and say what is blocking you.]")


def setting(value) -> int:
    """The run config's "loop_guard" -> the repeats that make a loop (0: off).  ValueError for a bad value."""
    if value is None or value is True:
        return REPEATS
    if value is False:
        return 0
    if isinstance(value, dict) and set(value) <= {"repeats"}:
        n = value.get("repeats", REPEATS)
        if isinstance(n, int) and not isinstance(n, bool) and n >= 2:
            return n
    raise ValueError('"loop_guard" must be false, true or {"repeats": N} with N 2 or more, not ' + repr(value))
