"""Tool search (0.94): Anthropic's `tool_search_tool_regex_*` and `tool_search_tool_bm25_*` server tools, run here.

A request marks tools `defer_loading: true`; they stay out of the prompt's tool list (server_tools.shown).  The model
calls the search tool with a query; the server matches the deferred tools' names, descriptions and argument names and
descriptions, answers with `tool_reference` blocks, and the model reads the matching tools' full definitions in the
tool result (server_tools.tool_definition_text) - in the conversation, where they were found, so the prompt's start
and the conversation cache stay as they were.  Claude Code's own client-side ToolSearch returns `tool_reference`
blocks in an ordinary tool_result, which frontend.anthropic_to_messages expands the same way.
"""
from __future__ import annotations

import math
import re

from serve import server_tools as ST

MAX_RESULTS = 5
MAX_PATTERN = 200
TEXT_CAP = 4000                       # the characters of one tool that are searched (bounds a pathological pattern)


def _schema_text(schema, depth=0) -> list[str]:
    out = []
    if not isinstance(schema, dict) or depth > 3:
        return out
    for name, spec in (schema.get("properties") or {}).items() if isinstance(schema.get("properties"), dict) else ():
        out.append(str(name))
        if isinstance(spec, dict):
            if isinstance(spec.get("description"), str):
                out.append(spec["description"])
            out += _schema_text(spec, depth + 1)
            if isinstance(spec.get("items"), dict):
                out += _schema_text(spec["items"], depth + 1)
    return out


def tool_text(t: dict) -> str:
    """The searchable text of a tool: its name, description, and argument names and descriptions."""
    parts = [t.get("name") or "", t.get("description") or ""] + _schema_text(t.get("parameters"))
    return "\n".join(p for p in parts if p)[:TEXT_CAP]


def catalog(ctx: ST.Context) -> list[dict]:
    """The tools a search looks through: the deferred ones (all of the request's tools when none is deferred)."""
    tools = [t for t in ctx.tools if isinstance(t, dict)]
    deferred = [t for t in tools if t.get(ST.DEFERRED)]
    return deferred or tools


def regex_search(pattern: str, tools: list[dict], limit: int = MAX_RESULTS) -> list[str]:
    """Names of the tools whose text matches `pattern` (Python re, case-insensitive); a name match ranks first.
    ValueError("pattern_too_long" / "invalid_pattern") for a pattern this refuses."""
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("invalid_pattern")
    if len(pattern) > MAX_PATTERN:
        raise ValueError("pattern_too_long")
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error:
        raise ValueError("invalid_pattern") from None
    named, other = [], []
    for t in tools:
        if rx.search(t.get("name") or ""):
            named.append(t["name"])
        elif rx.search(tool_text(t)):
            other.append(t["name"])
    return (named + other)[:limit]


_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text or ""):
        out += [p.lower() for p in _CAMEL.split(w) if p]
    return out


def bm25_search(query: str, tools: list[dict], limit: int = MAX_RESULTS, k1: float = 1.2, b: float = 0.75) -> list[str]:
    """Names of the tools that best match a natural-language query (Okapi BM25 over tool_text; the name's words count
    twice).  Only tools that share a word with the query are returned."""
    q = set(words(query))
    if not q or not tools:
        return []
    docs = [words(t.get("name") or "") * 2 + words(tool_text(t)) for t in tools]
    avg = sum(len(d) for d in docs) / len(docs) or 1.0
    df = {w: sum(1 for d in docs if w in d) for w in q}
    n = len(docs)
    scored = []
    for i, d in enumerate(docs):
        s = 0.0
        for w in q:
            f = d.count(w)
            if not f:
                continue
            idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
        if s > 0:
            scored.append((-s, i))
    scored.sort()
    return [tools[i]["name"] for _, i in scored[:limit]]


def _definition(kind: str):
    def definition(entry: dict, ctx: ST.Context) -> dict:
        names = [t["name"] for t in catalog(ctx) if t.get(ST.DEFERRED)]
        listed = ", ".join(names[:200]) + (f" and {len(names) - 200} more" if len(names) > 200 else "")
        how = ("a Python regular expression (case-insensitive) matched against tool names, descriptions and argument "
               "names" if kind == "regex" else "a few words describing what you want to do")
        desc = ("Search for tools that are not listed above and load them. The matching tools' definitions are "
                "returned, and you can call them afterwards like any other tool."
                + (f" Tools you can load: {listed}." if names else ""))
        return {"name": entry["name"], "description": desc, "parameters": {
            "type": "object", "properties": {"query": {"type": "string", "description": how}}, "required": ["query"]}}
    return definition


def _run(kind: str):
    def run(arguments: dict, ctx: ST.Context, cancel=None, entry=None, use=1):
        query = arguments.get("query")
        try:
            names = regex_search(query, catalog(ctx)) if kind == "regex" else \
                bm25_search(query if isinstance(query, str) else "", catalog(ctx))
        except ValueError as e:
            code = str(e)
            return (f"error: {code.replace('_', ' ')}",
                    {"type": "tool_search_tool_result_error", "error_code": code})
        print(f"[strata] tool search ({kind}) {str(query)[:80]!r}: {', '.join(names) or 'nothing found'}", flush=True)
        return (ST.tool_definition_text(names, ctx),
                {"type": "tool_search_tool_search_result",
                 "tool_references": [{"type": "tool_reference", "tool_name": n} for n in names]})
    return run


def replay(block: dict, ctx: ST.Context) -> str:
    content = block.get("content") if isinstance(block.get("content"), dict) else {}
    if str(content.get("type", "")).endswith("_error"):
        return f"error: {str(content.get('error_code') or 'unavailable').replace('_', ' ')}"
    refs = content.get("tool_references") if isinstance(content.get("tool_references"), list) else []
    return ST.tool_definition_text([r.get("tool_name") for r in refs if isinstance(r, dict) and r.get("tool_name")],
                                   ctx)


for _kind in ("regex", "bm25"):
    ST.register(ST.ServerKind(name=f"tool_search_{_kind}", prefixes=(f"tool_search_tool_{_kind}",),
                              result_type="tool_search_tool_result", definition=_definition(_kind), run=_run(_kind),
                              replay=replay, available=lambda: ST.TOOL_SEARCH, usage_key="tool_search_requests"))
