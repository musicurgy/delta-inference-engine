"""Server tools (0.94): tools a request declares by `type` that this server runs itself, inside the answer, through
the same loop that runs MCP tools for the web app (server.run_with_mcp).

A kind is registered with `register(ServerKind(...))`:

- `prefixes`: the Anthropic `type` values it takes (`tool_search_tool_regex_20251119` matches `tool_search_tool_regex_`);
- `available()`: whether it can run here (web search needs a configured backend, for example);
- `definition(entry, ctx)`: the function tool the model is offered in its place ({"name", "description", "parameters"});
- `run(arguments, ctx, cancel)`: -> (text the model reads, the API's result block content);
- `result_type`: the Anthropic block that carries its result (`tool_search_tool_result`, `web_search_tool_result`, ...);
- `replay(block, ctx)`: the text the model read, rebuilt from that block when a client sends the turn back - the same
  text, so the conversation cache keeps everything up to it.

NOTE for the merge with package P2 (which removes the server tools this server cannot run): this is the minimal
registry P6 needed.  A request's server-typed tool whose kind is not registered or not available is left exactly as
it was before 0.94 (P2 decides what happens to those).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

# the request's tools carry this key when they are deferred (Anthropic's defer_loading): left out of the prompt's tool
# list (Service.encode_prompt), kept for the parser and for tool search
DEFERRED = "defer_loading"
TOOL_SEARCH = True                     # the config's "tool_search": "off" renders deferred tools as before 0.94


@dataclass
class ServerKind:
    name: str
    prefixes: tuple
    result_type: str
    definition: Callable
    run: Callable
    replay: Callable
    available: Callable = lambda: True
    usage_key: str | None = None       # usage.server_tool_use.<key> counts its calls


KINDS: list[ServerKind] = []


def register(kind: ServerKind) -> None:
    KINDS[:] = [k for k in KINDS if k.name != kind.name] + [kind]


def kind_of(entry) -> ServerKind | None:
    """The registered, available kind of an Anthropic tool entry ({"type": "web_search_20250305", "name": ...})."""
    t = entry.get("type") if isinstance(entry, dict) else None
    if not isinstance(t, str) or t in ("custom", "function"):
        return None
    for k in KINDS:
        if t.startswith(k.prefixes) and k.available():
            return k
    return None


def kind_by_result(block_type: str) -> ServerKind | None:
    for k in KINDS:
        if k.result_type == block_type:
            return k
    return None


@dataclass
class Context:
    """What a request's server tools need: every tool it declared (flat: name, description, parameters, and the
    DEFERRED key on deferred ones), its server tools by name, and their per-request settings and counters."""
    tools: list = field(default_factory=list)
    server: dict = field(default_factory=dict)          # tool name -> (ServerKind, the request's entry)
    uses: dict = field(default_factory=dict)            # tool name -> calls so far

    def by_name(self) -> dict:
        return {t["name"]: t for t in self.tools if isinstance(t, dict) and t.get("name")}


def anthropic_tools(raw: list[dict]) -> tuple[list[dict], Context]:
    """A messages request's tools -> (the flat tools for the template and the parser, the request's Context).  A tool
    that is not deferred and not a registered server tool comes out exactly as before 0.94."""
    tools, server = [], {}
    deferring = TOOL_SEARCH and any(t.get(DEFERRED) is True for t in raw if kind_of(t) is None)
    for t in raw:
        k = kind_of(t)
        if k is not None:
            server[t["name"]] = (k, t)
            continue
        flat = {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
        if deferring and t.get(DEFERRED) is True:
            flat[DEFERRED] = True
        tools.append(flat)
    # with nothing that could find them (no search tool, no other tool), deferred tools are shown as before
    if deferring and not server and all(t.get(DEFERRED) for t in tools):
        for t in tools:
            t.pop(DEFERRED, None)
    ctx = Context(tools=tools, server=server)
    defs = [k.definition(entry, ctx) for k, entry in server.values()]
    return tools + defs, ctx


def shown(tools):
    """The tools the prompt lists: the deferred ones left out.  The same object when none is deferred."""
    if not tools or not any(isinstance(t, dict) and t.get(DEFERRED) for t in tools):
        return tools
    out = [t for t in tools if not (isinstance(t, dict) and t.get(DEFERRED))]
    return out or None


def tool_definition_text(names: list[str], ctx: Context) -> str:
    """What the model reads when tools are loaded (a tool search result, or a client's tool_reference blocks): each
    definition in the shape the prompt's own tool list has, at this point of the conversation - never at the top, so
    every earlier token (and the conversation cache) stays as it was."""
    tools = ctx.by_name()
    found = [n for n in dict.fromkeys(names) if n in tools]
    missing = [n for n in dict.fromkeys(names) if n not in tools]
    if not found:
        text = "No matching tools were found."
    else:
        lines = [json.dumps({"type": "function", "function": {k: tools[n][k] for k in ("name", "description",
                                                                                      "parameters")}},
                            ensure_ascii=False) for n in found]
        text = ("These tools are now available; call them like the tools listed at the start:\n" + "\n".join(lines))
    if missing:
        text += "\nUnknown tools: " + ", ".join(missing)
    return text


class Runner:
    """The request's server tools in the shape run_with_mcp drives (settings / routes / call), as the MCP hub has."""

    def __init__(self, ctx: Context, max_rounds: int = 8):
        self.ctx = ctx
        self.settings = {"max_rounds": max_rounds}

    def names(self) -> set:
        return set(self.ctx.server)

    def routes(self) -> dict:
        return {name: (None, name) for name in self.ctx.server}

    def call(self, name: str, arguments: dict, cancel: threading.Event | None = None) -> dict:
        t0 = time.perf_counter()
        kind, entry = self.ctx.server[name]
        n = self.ctx.uses.get(name, 0) + 1
        self.ctx.uses[name] = n
        try:
            text, block = kind.run(arguments if isinstance(arguments, dict) else {}, self.ctx, cancel, entry=entry,
                                   use=n)
            ok = not (isinstance(block, dict) and str(block.get("type", "")).endswith("_error"))
        except Exception as e:  # noqa: BLE001 - a tool's failure is the model's to read, not the request's end
            text, block, ok = f"error: {e}", {"type": kind.result_type + "_error", "error_code": "unavailable"}, False
        return {"ok": ok, "text": text, "chars": len(text), "truncated": False,
                "ms": round((time.perf_counter() - t0) * 1000.0, 1), "block": block, "kind": kind.name,
                "result_type": kind.result_type, "usage_key": kind.usage_key}


def server_tool_id() -> str:
    return "srvtoolu_" + uuid.uuid4().hex[:24]


def replay_text(block: dict, ctx: Context) -> str | None:
    """The text the model read for a server tool's result block that a client sent back, or None (a block of a kind
    this server does not run: left out of the prompt, as before)."""
    k = kind_by_result(block.get("type"))
    return k.replay(block, ctx) if k is not None else None


def runs_tool(name: str, ctx: Context) -> bool:
    return name in ctx.server
