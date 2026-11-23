#!/usr/bin/env python3
"""Tests for serve_ds41.py (CPU, seconds).  Run under Strata's venv:
  /home/mal/AI/Strata/.venv/bin/python tools/v41/serve/test_serve_ds41.py [TOKENIZER_DIR]
TOKENIZER_DIR (export_tokenizer.py's output) enables the tokenizer and the HTTP tests (server --engine mock)."""
import json
import os
import socket
import subprocess
import sys
import time
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import serve_ds41 as D  # noqa: E402  (patches the server's modules)

E, F = D.E, D.F
FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:300]))
    if not ok:
        FAILS.append(name)


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
                         "required": ["path"]}}}]
FLAT = [t["function"] for t in TOOLS]   # as the server hands them to the parser (a wrapped tool has no "name": f096)


def t_template():
    T = D.Ds41Template()
    chats = [
        [{"role": "user", "content": "hi"}],
        [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"},
         {"role": "assistant", "content": "hello", "reasoning_content": "greet"}, {"role": "user", "content": "again"}],
        [{"role": "user", "content": "read a.txt"},
         {"role": "assistant", "content": "", "reasoning_content": "use the tool",
          "tool_calls": [{"id": "c1", "type": "function",
                          "function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt", "lines": 3})}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "AAA"}],
    ]
    for i, msgs in enumerate(chats):
        tools = TOOLS if i == 2 else None
        for mode, kw in (("thinking", {}), ("chat", {"enable_thinking": False})):
            ref_msgs = [dict(m) for m in msgs]
            if tools:
                ref_msgs.insert(0, {"role": "system", "content": "", "tools": tools})
            ref = E.encode_messages(ref_msgs, thinking_mode=mode)
            got = T.render(msgs, tools=tools, **kw)
            if tools:   # the server's flat tool format must render the same
                flat = T.render(msgs, tools=[t["function"] for t in tools], **kw)
                check(f"template chat{i} {mode} flat tools == OpenAI tools", flat == got)
            check(f"template chat{i} {mode} == encode_messages", got == ref, (got[-200:], ref[-200:]))
            hist = T.render(msgs, tools=tools, add_generation_prompt=False, **kw)
            check(f"template chat{i} {mode} no-gen is a prefix", got.startswith(hist) and len(hist) < len(got))
    # content parts, effort
    got = T.render([{"role": "user", "content": [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}]}])
    # 11-15 sweep: text parts are joined with a blank line, as DeepSeek's encoder joins a list's text items
    check("template content parts joined", got == E.encode_messages([{"role": "user", "content": "x\n\ny"}], "thinking"))
    for eff, n in (("max", 100), (30, 30), ("low", 50), ("high", 75), ("xhigh", 75)):
        got = T.render([{"role": "user", "content": "q"}], reasoning_effort=eff)
        check(f"effort {eff!r} -> {n}", f"Reasoning Effort: {n} " in got, got[:120])
    got = T.render([{"role": "user", "content": "q"}])
    check("effort default -> 75", "Reasoning Effort: 75 " in got)


def t_prefill():
    """A conversation ending with an assistant message is continued (Anthropic prefill, DeepSeek's trailing assistant):
    the prompt ends with that text, not with <｜end▁of▁sentence｜>; earlier assistant turns keep their end token."""
    T = D.Ds41Template()
    eos = E.eos_token
    for kw in ({}, {"enable_thinking": False}):
        p = T.render([{"role": "user", "content": "Write a haiku."}, {"role": "assistant", "content": "Autumn wind"}], **kw)
        check(f"prefill continues the last assistant message {kw}", p.endswith("Autumn wind") and not p.endswith(eos),
              repr(p[-40:]))
        p = T.render([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                      {"role": "user", "content": "c"}], **kw)
        check(f"earlier assistant turns still end with the end token {kw}", ("b" + eos) in p, repr(p[-60:]))
    check("effort 'ultra' is DeepSeek's top level", D.effort_kwargs("ultra") == {"reasoning_effort": 100})


def t_effort_kwargs():
    for v, want in (("high", {"reasoning_effort": 75}), ("max", {"reasoning_effort": 100}), ("25", {"reasoning_effort": 25}),
                    ("none", {"enable_thinking": False}), ("0", {"enable_thinking": False}), (False, {"enable_thinking": False}),
                    ("medium", {"reasoning_effort": 62}), (None, {})):
        check(f"effort_kwargs({v!r})", D.effort_kwargs(v) == want, D.effort_kwargs(v))
    try:
        D.effort_kwargs("turbo"); check("effort_kwargs rejects unknown", False)
    except ValueError:
        check("effort_kwargs rejects unknown", True)


def dsml(calls):
    inv = []
    for name, args in calls:
        ps = []
        for k, v in args.items():
            s = isinstance(v, str)
            ps.append(f'<{E.dsml_token}{E.tool_parameter_tag_name} name="{k}" string="{"true" if s else "false"}">'
                      f'{v if s else json.dumps(v)}</{E.dsml_token}{E.tool_parameter_tag_name}>')
        inv.append(f'<{E.dsml_token}{E.tool_call_tag_name} name="{name}">\n' + "\n".join(ps) +
                   f'\n</{E.dsml_token}{E.tool_call_tag_name}>')
    return f"\n\n{D.CALLS_OPEN}\n" + "\n".join(inv) + f"\n{D.CALLS_CLOSE}"


def run_parser(text, thinking, split, **kw):
    p = F.OutputParser(thinking=thinking, tools=FLAT, **kw)
    ev = []
    for i in range(0, len(text), split):
        ev += p.feed(text[i:i + split])
    ev += p.finish()
    content = "".join(e.text for e in ev if e.kind == "content")
    reasoning = "".join(e.text for e in ev if e.kind == "reasoning")
    calls = [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
    return reasoning, content, calls


def t_parser():
    calls = [("read_file", {"path": "a b.txt", "lines": 3}),
             ("read_file", {"path": "x</y>", "lines": [1, 2], "opts": {"k": True}})]
    body = "I'll look." + dsml(calls)
    for thinking in (True, False):
        text = ("checking files" + E.thinking_end_token if thinking else "") + body
        ref = E.parse_message_from_completion_text(text + E.eos_token, "thinking" if thinking else "chat")
        ref_calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]]
        bad = []
        for split in range(1, 40):
            r, c, k = run_parser(text, thinking, split)
            if k != ref_calls or c.strip() != ref["content"].strip() or (thinking and r != ref["reasoning_content"]):
                bad.append((split, r, c, k))
        check(f"DSML parse == DeepSeek's parser at every split 1..39 (thinking={thinking})", not bad, bad[:1] or ref)
    # plain answer, and text that only looks like a wrapper start
    r, c, k = run_parser("plain answer <｜DSML not a tag", False, 3)
    check("non-DSML text passes through", c == "plain answer <｜DSML not a tag" and not k, (c, k))
    # upstream #106: a turn cut (max_tokens, cancel) anywhere inside the DSML sends no DSML as content
    raw = "Checking." + dsml([("read_file", {"path": "a b.txt", "lines": 3})])
    for thinking in (False, True):
        text = ("plan" + E.thinking_end_token if thinking else "") + raw
        bad, done = [], text.index(D.INVOKE_CLOSE) + len(D.INVOKE_CLOSE)   # the call is whole from here
        for cut in range(text.index("<" + E.dsml_token), len(text) + 1):
            for step in (1, 7, 4096):
                for reason in ("length", "cancel"):
                    p = F.OutputParser(thinking=thinking, tools=FLAT)
                    ev = []
                    for i in range(0, cut, step):
                        ev += p.feed(text[i:min(i + step, cut)])
                    ev += p.finish(reason)
                    c = "".join(e.text for e in ev if e.kind == "content")
                    calls = [e.call.name for e in ev if e.kind == "tool_call"]
                    if c.strip() != "Checking." or calls != (["read_file"] if cut >= done else []):
                        bad.append((cut - len(text), step, reason, c[-40:], calls))
        check(f"a cut inside DSML sends no DSML text, every cut / step 1, 7, 4096 (thinking={thinking})", not bad, bad[:2])
    # upstream #110: the ways a low-bit quant spells the tags - whole they are one call, cut anywhere no markup in the content
    def spelled(o, c):
        return (f'{o} calls>\n{o} invoke name="read_file">\n{o} parameter name="path" string="true">a.txt{c} parameter>\n'
                f'{o} parameter name="lines" string="false">3{c} parameter>\n{c} invoke>\n{c} calls>')
    forms = [("the token", "<" + E.dsml_token, "</" + E.dsml_token), ("bare", "<", "</"), ("backslash", "<\\", "</\\"),
             ("U+FF3C", "<＼", "</＼"), ("misspelled", "<＼DSML＼", "</＼DSML＼"),
             ("mixed", "<", "</" + E.dsml_token)]
    want = [("read_file", {"path": "a.txt", "lines": 3})]
    for name, o, cl in forms:
        for thinking in (True, False):
            head = ("hm" + E.thinking_end_token if thinking else "") + "Reading.\n"
            whole, bad = head + spelled(o, cl), []
            for step in (1, 7, 4096):
                r, c, k = run_parser(whole, thinking, step)
                if k != want or c.strip() != "Reading.":
                    bad.append((step, c, k))
            check(f"{name}: one call and the text before it (thinking={thinking})", not bad, bad[:1])
            bad = []
            for cut in range(len(head), len(whole) + 1):
                for step in (1, 7, 4096):
                    p, ev = F.OutputParser(thinking=thinking, tools=FLAT), []
                    for i in range(0, cut, step):
                        ev += p.feed(whole[:cut][i:i + step])
                    ev += p.finish("length")
                    c = "".join(e.text for e in ev if e.kind == "content")
                    if c.strip() != "Reading.":
                        bad.append((cut, step, c))
            check(f"{name}: cut anywhere, no markup in the content (thinking={thinking})", not bad, bad[:2])
    # an agent trace (UD-IQ2_XXS, greedy, bias 0.28): a stray word in the marker's place inside a calls block
    o, cl = "<" + E.dsml_token, "</" + E.dsml_token
    stray = (f'Checking.\n\n{o} calls>\n<name invoke name="read_file">\n{o} parameter name="path" string="true">a.txt{cl} parameter>\n'
             f'{o} parameter name="lines" string="false">3{cl} parameter>\n{cl} invoke>\n{cl} calls>')
    bad = []
    for cut in range(len("Checking."), len(stray) + 1):
        for step in (1, 7, 4096):
            p, ev = F.OutputParser(thinking=False, tools=FLAT), []
            for i in range(0, cut, step):
                ev += p.feed(stray[:cut][i:i + step])
            ev += p.finish("stop" if cut == len(stray) else "length")
            c = "".join(e.text for e in ev if e.kind == "content")
            k = [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
            if c.strip() != "Checking." or (cut == len(stray) and k != want):
                bad.append((cut, step, c[-40:], k))
    check("a stray word for the marker inside a calls block: one call, cut anywhere no markup in the content", not bad, bad[:2])
    r, c, k = run_parser(f'{o} calls>\n{o} invoke name="read_file">\n{o} parameter name="path" string="true">'
                         f'a <name invoke name="x"> b{cl} parameter>\n{cl} invoke>\n{cl} calls>', False, 3)
    check("a stray-word tag inside a value is text", k == [("read_file", {"path": 'a <name invoke name="x"> b'})], (c, k))
    for text in ('See <name invoke name="read_file"> here.', '<name invoke name="read_file">\n', "<b>bold</b> and <i>x</i>",
                 "a <div class=\"x\"> b"):
        r, c, k = run_parser(text, False, 3)
        check(f"prose stays text (stray word): {text[:24]!r}", c == text and not k, (c, k))
    for text in ('Never run < invoke name="rm"> here.', 'Use `< invoke name="read_file">` for that.',
                 '```\n< invoke name="read_file">\n```\n', "if a < b and c > d: pass", "x <invoke> y", "1 < 2 < 3", "a <\\ b"):
        r, c, k = run_parser(text, False, 3)
        check(f"prose stays text: {text[:24]!r}", c == text and not k, (c, k))
    r, c, k = run_parser('Reading:\n< invoke name="bash">\n< parameter name="command" string="true">ls</ parameter>\n</ invoke>', False, 5)
    check("an undeclared tool's call at the start of a line is a call", k == [("bash", {"command": "ls"})], (c, k))
    flat = [t["function"] for t in TOOLS]              # as the server hands them to the parser
    def run_flat(text):
        p, ev = F.OutputParser(thinking=False, tools=flat), []
        for i in range(0, len(text), 5):
            ev += p.feed(text[i:i + 5])
        ev += p.finish()
        return "".join(e.text for e in ev if e.kind == "content"), [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"]
    c, k = run_flat('< invoke name="read_file">\n< parameter name="lines">3</ parameter>\n< parameter name="path">a</ parameter>\n</ invoke>')
    check("no string= attribute: by the declared type", k == [("read_file", {"lines": 3, "path": "a"})], (c, k))
    c, k = run_flat('Now: < invoke name="read_file">\n< parameter name="path">a</ parameter>\n</ invoke>')
    check("a declared tool's bare call in mid-line is a call", k == [("read_file", {"path": "a"})] and c.strip() == "Now:", (c, k))
    r, c, k = run_parser(dsml([("read_file", {"path": "a < b </ c", "lines": 3})]), False, 4)
    check("tags inside a value are not tags", k == [("read_file", {"path": "a < b </ c", "lines": 3})], (c, k))
    r, c, k = run_parser("ok <" + E.dsml_token + " inv", False, 4096)
    check("a natural stop inside a tag keeps it as text", c == "ok <" + E.dsml_token + " inv", c)
    esc = "a&lt;" + D.P_CLOSE[1:] + "b"
    check("DwarfStar escape undone", D.unescape(esc) == "a" + D.P_CLOSE + "b", D.unescape(esc))


def t_param_schema():
    """upstream b48bf015: parameters decode by their schema, unions and nullable strings too; DSML's string= still counts"""
    null_str = {"anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}]}
    spec = {"name": "note", "parameters": {"type": "object", "properties": {
        "path": null_str, "one": {"oneOf": [{"type": "string"}, {"type": "null"}]}, "listed": {"type": ["string", "null"]},
        "plain": {"type": "string"}, "mixed": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        "count": {"anyOf": [{"type": "integer"}, {"type": "null"}]}, "n": {"type": "integer"}, "free": {}}}}
    union = {"name": "auto", "parameters": {"type": "object", "properties": {}, "oneOf": [{"$ref": "#/$defs/v"}, {"$ref": "#/$defs/w"}],
             "$defs": {"id": {"type": "string"}, "v": {"type": "object", "properties": {"id": {"$ref": "#/$defs/id"}}},
                       "w": {"oneOf": [{"type": "object", "properties": {"name": {"type": "string"}, "every": {"type": "integer"}}}]}}}}

    def arg(sp, name, value, s="false"):
        body = f' name="{sp["name"]}">{D.P_OPEN} name="{name}" string="{s}">{value}{D.P_CLOSE}'
        return D.ds41_parse_tool_call(body, sp).arguments[name]
    for n in ("path", "one", "listed", "plain", "mixed"):
        check(f"schema: {n} stays text", arg(spec, n, "123") == "123" and arg(spec, n, "x/y") == "x/y", n)
    for n in ("path", "one", "listed"):
        check(f"schema: {n} null is None", arg(spec, n, "null") is None, n)
    check("schema: null stays text on a plain string", arg(spec, "plain", "null") == "null")
    check("schema: a number where the type says so", arg(spec, "count", "7") == 7 and arg(spec, "n", "5") == 5 and arg(spec, "count", "null") is None)
    check("schema: no type decodes JSON", arg(spec, "free", "[1, 2]") == [1, 2])
    check("schema: string=true wins", arg(spec, "n", "5", "true") == "5")
    # 0.95 (Abyss 66e4aa6): a boolean parameter written in Python's spelling is a bool; only where the schema says boolean
    bools = {"name": "opt", "parameters": {"type": "object", "properties": {
        "flag": {"type": "boolean"}, "maybe": {"anyOf": [{"type": "boolean"}, {"type": "null"}]}, "s": {"type": "string"},
        "either": {"type": ["boolean", "string"]}, "free": {}}}}
    got = [arg(bools, "flag", v) for v in ("True", "False", " TRUE\n", "true", "false", '"True"')]
    check("schema: a boolean written True / False is a bool", got == [True, False, True, True, False, True], got)
    got = (arg(bools, "maybe", "False"), arg(bools, "maybe", "null"), arg(bools, "flag", "yes"), arg(bools, "s", "True"),
           arg(bools, "either", "True"), arg(bools, "free", "True"), arg(spec, "n", "True"),
           D.ds41_parse_tool_call(f' name="x">{D.P_OPEN} name="a" string="false">True{D.P_CLOSE}').arguments["a"])
    check("schema: True stays text where the schema does not say boolean", got == (False, None, "yes", "True", "True", "True", "True", "True"), got)
    check("schema: without a schema, JSON", D.ds41_parse_tool_call(f' name="x">{D.P_OPEN} name="a" string="false">3{D.P_CLOSE}').arguments == {"a": 3})
    # llama.cpp c90aa996b: a raw newline / tab inside a JSON value's string (the model wrote the character) is JSON
    got = arg(spec, "free", '[{"old": "x = 1\n", "new": "x = 2\tdone"}]')
    check("schema: raw control characters in a JSON value's string", got == [{"old": "x = 1\n", "new": "x = 2\tdone"}], got)
    got = D.ds41_parse_tool_call(f' name="x">{D.P_OPEN} name="a" string="false">{{"k": "a\nb"}}{D.P_CLOSE}').arguments
    check("schema: raw control characters, without a schema", got == {"a": {"k": "a\nb"}}, got)
    check("schema: a union's variants type their parameters", arg(union, "id", "42") == "42" and arg(union, "every", "8") == 8 and arg(union, "name", "123") == "123")

    class P:
        schemas = {"note": spec}
    for n, want in (("plain", "true"), ("mixed", "true"), ("free", "true"), ("missing", "true"), ("n", "false"), ("count", "false"), ("path", "false")):
        check(f"schema: a tag without string= on {n}", D._param_is_string(P, "note", n) == want, n)
    # a tag with no string=: the whole way through the parser
    for value, want in (("7", 7), ("null", None)):
        text = f'{D.CALLS_OPEN}\n< invoke name="note">\n< parameter name="count">{value}</ parameter>\n</ invoke>\n{D.CALLS_CLOSE}'
        p = F.OutputParser(thinking=False, tools=[spec])
        ev = p.feed(text) + p.finish()
        got = [e.call.arguments for e in ev if e.kind == "tool_call"]
        check(f"schema: bare tag decodes {value}", got == [{"count": want}], got)


def t_undeclared_call():
    """upstream 31c02061: a call opened with the DSML token comes back as a call even for a tool the request did not declare"""
    r, c, k = run_parser(dsml([("bash", {"command": "ls"})]), False, 7)
    check("a DSML call to an undeclared tool is a call, no markup in the content", [x[0] for x in k] == ["bash"] and "DSML" not in c and "invoke" not in c, (c, k))


QA_TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "content": {"type": "string"}, "n": {"type": "integer"}}}}}]
QA_FLAT = [t["function"] for t in QA_TOOLS]


def qa_call(value, n=3):
    """The DSML DeepSeek's own encoder writes for write(content=value, n=n), after "Writing." (chat mode)."""
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "Writing.", "tool_calls": [
        {"type": "function", "function": {"name": "write", "arguments": json.dumps({"content": value, "n": n})}}]}]
    enc = E.encode_messages(msgs, thinking_mode="chat")
    body = enc[enc.index("Writing."):]
    return body[:body.rindex(D.CALLS_CLOSE) + len(D.CALLS_CLOSE)]


def qa_run(text, thinking, step, cut=None, reason=None):
    p, ev = F.OutputParser(thinking=thinking, tools=QA_FLAT), []
    text = text if cut is None else text[:cut]
    for i in range(0, len(text), step):
        ev += p.feed(text[i:i + step])
    ev += p.finish(reason) if reason else p.finish()
    return ("".join(e.text for e in ev if e.kind == "content"), "".join(e.text for e in ev if e.kind == "reasoning"),
            [(e.call.name, e.call.arguments) for e in ev if e.kind == "tool_call"])


def t_param_close_in_value():
    """0.95 (spark-ds41f 74e45310, lucebox 39b9cfb9): a parameter close inside a string value - written with the real
    DSML token - ends the value only when the call's structure follows it; a whole call the value quotes is its text."""
    o, cl, io, ic, po, pc = D.CALLS_OPEN, D.CALLS_CLOSE, D.INVOKE_OPEN, D.INVOKE_CLOSE, D.P_OPEN, D.P_CLOSE
    quoted = f'{o}\n{io} name="bash">\n{po} name="command" string="true">ls{pc}\n{ic}\n{cl}'
    for name, value in (("then text", f"A param ends with {pc} then more."), ("at the end", f"the tag is {pc}"),
                        ("then a newline", f"x {pc}\ny"), ("a whole quoted call", f"test:\n{quoted}\ndone"),
                        ("a quoted call at the end", quoted), ("with a calls close", f"1 {pc}\n3 {cl}\nend")):
        bad = []
        for thinking in (False, True):
            text = ("plan" + E.thinking_end_token if thinking else "") + qa_call(value)
            for step in (1, 2, 3, 5, 7, 11, 4096):
                c, r, k = qa_run(text, thinking, step)
                if k != [("write", {"content": value, "n": 3})] or c.strip() != "Writing.":
                    bad.append((thinking, step, c[-60:], k))
        check(f"a parameter close inside a value: {name}", not bad, bad[:1])
    # two calls, the first value holding a closer: both whole
    text = "Writing." + dsml([("write", {"content": f"see {pc} here", "n": 1}), ("write", {"content": "second", "n": 2})])
    for step in (1, 3, 4096):
        c, r, k = qa_run(text, False, step)
        check(f"a parameter close inside a value, then a second call (step {step})",
              k == [("write", {"content": f"see {pc} here", "n": 1}), ("write", {"content": "second", "n": 2})], k)
    # unchanged: a close followed by the next parameter, the call's end, the calls' end, or the end of the text
    for text, want in ((f'{o}\n{io} name="write">\n{po} name="content" string="true">a{pc}\n{po} name="n" string="false">3{pc}\n{ic}\n{cl}',
                        {"content": "a", "n": 3}),
                       (f'{o}\n{io} name="write">\n{po} name="content" string="true">a{pc}{ic}{cl}', {"content": "a"}),
                       (f'{o}\n{io} name="write">\n{po} name="content" string="true">a{pc}\n{cl}', None),
                       (f'{o}\n{io} name="write">\n{po} name="content" string="true">a{pc}  \n', None)):
        for step in (1, 4, 4096):     # (None: a call without its invoke close is no call, as before)
            c, r, k = qa_run("W." + text, False, step)
            check(f"a parameter close the structure follows still ends the value ({len(text)}, step {step})",
                  k == ([("write", want)] if want else []) and D.LITERAL_LT not in c, (c, k))


def t_dsml_leak():
    """R-5: DSML left in an answer is counted (once per answer) and logged; the events themselves never change."""
    def leaks(text, thinking=False, step=1, reason=None):
        before = D.DSML_LEAKS["answers"]
        p, ev = F.OutputParser(thinking=thinking, tools=FLAT), []
        for i in range(0, len(text), step):
            ev += p.feed(text[i:i + step])
        ev += p.finish(reason) if reason else p.finish()
        return D.DSML_LEAKS["answers"] - before, "".join(e.text for e in ev if e.kind == "content")

    call = "Reading." + dsml([("read_file", {"path": "a b.txt", "lines": 3})])
    for step in (1, 5, 4096):
        n, c = leaks(call, step=step)
        check(f"a parsed call is no leak (step {step})", n == 0 and "DSML" not in c, (n, c))
        n, c = leaks("plan " + E.dsml_token + " calls" + E.thinking_end_token + call, thinking=True, step=step)
        check(f"DSML in the thinking is no leak (step {step})", n == 0, n)
        n, c = leaks("The " + E.dsml_token + " token, and " + E.dsml_token + " again.", step=step)
        check(f"the DSML token as answer text: counted once (step {step})", n == 1 and E.dsml_token in c, (n, c))
        n, c = leaks('In XML: <invoke name="x"> is a tag.', step=step)
        check(f"a marker-less invoke tag in prose: counted, text unchanged (step {step})",
              n == 1 and c == 'In XML: <invoke name="x"> is a tag.', (n, c))
    n, c = leaks("Done. <" + E.dsml_token + " inv", reason="stop")
    check("a natural stop inside a tag keeps the text and counts it", n == 1 and c.endswith(E.dsml_token + " inv"), (n, c))
    n, c = leaks("plain answer <｜DSML not a tag")
    check("text that only starts like a tag is no leak", n == 0, n)


def t_tool_qa():
    """ds4's tool-call QA list (antirez 46e46e7, aa8fef4, d077fa6, fc6414c; MIT): split delimiters at every byte, literal
    tool / thinking markers inside an argument, escapes, and a truncated call that stays truncated."""
    literals = {"thinking tags": E.thinking_start_token + " x " + E.thinking_end_token, "calls close": D.CALLS_CLOSE,
                "calls open": D.CALLS_OPEN, "invoke close": D.INVOKE_CLOSE, "bare invoke close": "</ invoke>",
                "invoke open": f'{D.INVOKE_OPEN} name="bash">', "parameter open": f'{D.P_OPEN} name="n" string="false">',
                "end of sentence": E.eos_token, "HTML entities": "&lt;div&gt; &amp; &quot;", "lone <": "a < b <"}
    # (not covered: an invoke close followed by a calls close or another call's opening inside one value - that is
    # also what a quant that left out the value's closing tag writes, and upstream #110's recovery wins there)
    for name, lit in literals.items():
        value = f"before {lit} after\n"
        body, want = qa_call(value), [("write", {"content": value, "n": 3})]
        ref = E.parse_message_from_completion_text(body + E.eos_token, "chat")
        ref_ok = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in ref["tool_calls"]] == want
        for thinking in (False, True):
            text, bad = ("plan" + E.thinking_end_token if thinking else "") + body, []
            for step in range(1, 24):
                c, r, k = qa_run(text, thinking, step)
                if k != want or c.strip() != "Writing." or (thinking and r != "plan"):
                    bad.append((step, c[-60:], k))
            check(f"QA: literal {name} inside an argument, every split 1..23 (thinking={thinking})", ref_ok and not bad,
                  bad[:1] or "DeepSeek's parser differs")
    # a cut anywhere inside such a call: no call, no markup, no placeholder in the content; a natural stop keeps the text
    value = f"x {D.INVOKE_CLOSE} {D.CALLS_OPEN} y"
    body, bad = qa_call(value), []
    done = body.rindex(D.INVOKE_CLOSE) + len(D.INVOKE_CLOSE)        # the call is whole from here
    for cut in range(body.index(D.CALLS_OPEN), len(body)):
        for step in (1, 5, 4096):
            c, r, k = qa_run(body, False, step, cut, "length")
            if c.strip() != "Writing." or k != ([("write", {"content": value, "n": 3})] if cut >= done else []):
                bad.append(("length", cut, step, c[-40:], k))
            c, r, k = qa_run(body, False, step, cut, "stop")   # (a natural stop may still recover a call)
            if D.LITERAL_LT in c or any(D.LITERAL_LT in json.dumps(a) for _, a in k):
                bad.append(("stop", cut, step, c[-40:], k))
    check("QA: a cut inside a call with literal tags in its value: no call, no DSML, no placeholder", not bad, bad[:2])
    # upstream #110's recovery stays: a value's closing tag left out, the call ends at its invoke close
    o, cl = D.CALLS_OPEN, D.CALLS_CLOSE
    rf = [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}}}}]
    for text, want in ((f'R.\n\n{o}\n{D.INVOKE_OPEN} name="read_file">\n{D.P_OPEN} name="path" string="true">a.txt{D.INVOKE_CLOSE}\n{cl}',
                        [{"path": "a.txt"}]),
                       (f'R.\n\n{o}\n{D.INVOKE_OPEN} name="read_file">\n{D.P_OPEN} name="path" string="true">a.txt{D.INVOKE_CLOSE}\n'
                        f'{D.INVOKE_OPEN} name="read_file">\n{D.P_OPEN} name="path" string="true">b.txt{D.P_CLOSE}\n{D.INVOKE_CLOSE}\n{cl}',
                        [{"path": "a.txt"}, {"path": "b.txt"}]),
                       (f'R.\n\n{o}\n{D.INVOKE_OPEN} name="read_file">\n{D.P_OPEN} name="path" string="true">a.txt{D.INVOKE_CLOSE}\n'
                        f'<name invoke name="read_file">\n{D.P_OPEN} name="path" string="true">b.txt{D.P_CLOSE}\n{D.INVOKE_CLOSE}\n{cl}',
                        [{"path": "a.txt"}, {"path": "b.txt"}]),         # the next call opened with a stray word
                       (f'R.\n\n{o}\n{D.INVOKE_OPEN} name="read_file">\n{D.P_OPEN} name="path" string="true">a.txt{D.INVOKE_CLOSE}',
                        [{"path": "a.txt"}])):
        bad = []
        for step in (1, 3, 7, 4096):
            p, ev = F.OutputParser(thinking=False, tools=[t["function"] for t in rf]), []
            for i in range(0, len(text), step):
                ev += p.feed(text[i:i + step])
            ev += p.finish()
            k = [e.call.arguments for e in ev if e.kind == "tool_call"]
            if k != want:
                bad.append((step, k))
        check(f"QA: a value's closing tag left out, the call still ends ({len(want)} call(s))", not bad, bad[:1])
    # escapes: DwarfStar's escaped closing spelling is undone one level; everything else stays as written
    for value, want in (("&lt;" + D.P_CLOSE[1:], D.P_CLOSE), ("&amp;lt;" + D.P_CLOSE[1:], "&lt;" + D.P_CLOSE[1:]),
                        ("&lt;" + D.INVOKE_CLOSE[1:], "&lt;" + D.INVOKE_CLOSE[1:]), ("a \\n b \\\" c", "a \\n b \\\" c")):
        c, r, k = qa_run(qa_call(value), False, 3)
        check(f"QA: escape {value[:12]!r}", k == [("write", {"content": want, "n": 3})], k)
    # a call cut by max_tokens through the server: finish_reason stays "length", no tool_calls, whole or streamed
    import serve.server as S
    tok = S.ByteTokenizer()
    tok.SPECIALS = tok.SPECIALS + [E.eos_token]       # V4.1's end of turn: one id, as the server asserts
    body = qa_call("x " + D.INVOKE_CLOSE + " y")
    cut = len(body[:body.index(" y")].encode())
    for stream in (False, True):
        svc = S.Service(S.MockEngine(tok, body, max_context=1 << 20), tok, D.Ds41Template())
        httpd = S.serve(svc, port=0)
        try:
            req = {"model": "x", "max_tokens": cut, "stream": stream, "tools": QA_TOOLS,
                   "messages": [{"role": "user", "content": "write it"}], "reasoning_effort": "none"}
            r = urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions", json.dumps(req).encode(),
                {"Content-Type": "application/json"}), timeout=30).read().decode()
            if stream:
                chunks = [json.loads(x[6:]) for x in r.split("\n") if x.startswith("data: {")]
                finish = [ch["choices"][0].get("finish_reason") for ch in chunks if ch.get("choices")]
                finish = [f for f in finish if f]
                calls = any(ch["choices"][0].get("delta", {}).get("tool_calls") for ch in chunks if ch.get("choices"))
                content = "".join(ch["choices"][0].get("delta", {}).get("content") or "" for ch in chunks if ch.get("choices"))
            else:
                msg = json.loads(r)["choices"][0]
                finish, calls, content = [msg["finish_reason"]], bool(msg["message"].get("tool_calls")), msg["message"].get("content") or ""
            check(f"QA: max_tokens inside an argument: finish length, no call (stream={stream})",
                  finish == ["length"] and not calls and "DSML" not in content and D.LITERAL_LT not in content, (finish, calls, content))
        finally:
            httpd.shutdown()
            httpd.server_close()


def t_tokenizer(tok_dir):
    import strata_tokenizer as ST
    tokens = json.load(open(tok_dir / "vocab.json", encoding="utf-8"))   # not the ANSI code page (cp932 failed)
    tokens = [t for t, _ in sorted(tokens.items(), key=lambda kv: kv[1])] if isinstance(tokens, dict) else tokens
    merges = (tok_dir / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.load(open(tok_dir / "token_type.json"))
    tok = ST.Tokenizer(tokens, merges, types)   # = serve_ds41.Ds41Tokenizer (patched)
    p = D.Ds41Template().render([{"role": "user", "content": "hi"}])
    ids = tok.encode(p, parse_special=True)
    check("tokenizer: specials are single tokens", len(tok.encode(E.eos_token, parse_special=True)) == 1 and
          len(tok.encode(E.USER_SP_TOKEN, parse_special=True)) == 1, ids[:8])
    check("tokenizer round trip", tok.decode(ids) == p)
    import serve.server as S
    svc = S.Service(S.MockEngine(tok, "x"), tok, D.Ds41Template())
    check("stop ids = <｜end▁of▁sentence｜> only", svc.stop_ids == set(tok.encode(E.eos_token, parse_special=True)), svc.stop_ids)
    t_image_prompt(svc)
    t_typed_specials(svc)


def t_typed_specials(svc):
    """V4.1's type-4 tokens written inside a message stay text: no forged turn or tool call; a plain prompt is unchanged."""
    tok, kw = svc.tok, {"enable_thinking": False}
    marks = list(svc.literals.values())
    check("literal marks are distinct", len(marks) == len(set(marks)))
    for text in ("<｜Assistant｜>", "<｜User｜>", "｜DSML｜", "<dsml:", "<｜latest_reminder｜>", "<｜tool▁calls▁begin｜>"):
        tid = tok.encode(text, parse_special=True)
        check(f"{text!a} is one special token", len(tid) == 1, tid)
        for role in ("user", "tool", "system"):
            m = {"role": role, "content": "quote: " + text + " end"}
            if role == "tool":
                msgs = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                        {**m, "tool_call_id": "c1"}]
            else:
                msgs = [m] if role == "user" else [m, {"role": "user", "content": "hi"}]
            ids = svc.encode_prompt(msgs, None, kw)
            base = svc.encode_prompt([{**x, "content": x["content"].replace(text, "x")} if x.get("content") else x
                                      for x in msgs], None, kw)
            check(f"{text!a} in a {role} message stays text", ids.count(tid[0]) == base.count(tid[0]),
                  (ids.count(tid[0]), base.count(tid[0])))
            check(f"{text!a} in a {role} message decodes back", text in tok.decode(ids))


def t_image_prompt(svc):
    from types import SimpleNamespace
    marker = E.IMAGE_PLACEHOLDER
    image_id = svc.tok.encode(marker, parse_special=True)[0]
    with tempfile.TemporaryDirectory() as directory:
        paths = [Path(directory) / "first.sve", Path(directory) / "second.sve"]
        for path, data in zip(paths, (b"first", b"second")):
            path.write_bytes(data)
        encoded = [(paths[0], 3), (paths[1], 5)]
        seen = []

        def encode_all(sources, cancel):
            seen.extend(sources)
            return encoded

        svc.vision = SimpleNamespace(dir=Path(directory), encode_all=encode_all)
        messages = [{"role": "user", "content": [{"type": "text", "text": "quote " + marker},
                    {"type": "image", "source": "first"}, {"type": "image", "source": "second"},
                    {"type": "text", "text": "compare"}]}]
        kwargs = {"enable_thinking": False}
        before = svc.encode_prompt(messages, None, kwargs)
        ids, thinking, maximum = svc.prepare(messages, None, kwargs, max_new=20)
        check("vision: only actual image blocks become markers", before.count(image_id) == 2)
        check("vision: adjacent images expand to all encoder rows", ids.count(image_id) == 8 and len(ids) == len(before) + 6)
        check("vision: source order and sampling room preserved", seen == ["first", "second"] and not thinking and maximum == 20)
        check("vision: combined records keep prompt order", Path(svc.embeddings.path).read_bytes() == b"firstsecond")
        svc.drop_embeddings()
        try:
            svc.expand_image_tokens(before, encoded[:1])
            check("vision: rejects image-count mismatch", False)
        except ValueError:
            check("vision: rejects image-count mismatch", True)
        svc.vision = None


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def t_http(tok_dir):
    port = free_port()
    script = "need the file" + E.thinking_end_token + dsml([("read_file", {"path": "a.txt", "lines": 3})])
    py = sys.executable
    proc = subprocess.Popen([py, str(HERE / "serve_ds41.py"), "--engine", "mock", "--port", str(port),
                             "--tokenizer", str(tok_dir), "--script", script, "--script", script, "--script", "done" + E.thinking_end_token + "The file says AAA."],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1); break
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(0.5)
        msgs = [{"role": "user", "content": "read a.txt"}]
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": msgs, "tools": TOOLS})
        ch = r["choices"][0]
        tc = (ch["message"].get("tool_calls") or [{}])[0].get("function", {})
        check("HTTP OpenAI tool call", ch.get("finish_reason") == "tool_calls" and tc.get("name") == "read_file" and
              json.loads(tc.get("arguments", "{}")) == {"path": "a.txt", "lines": 3}, r)
        r = post(port, "/v1/messages", {"model": "x", "max_tokens": 200, "messages": msgs,
                                         "tools": [{"name": "read_file", "description": "Read a file",
                                                    "input_schema": TOOLS[0]["function"]["parameters"]}]})
        tu = [b for b in r.get("content", []) if b.get("type") == "tool_use"]
        check("HTTP Anthropic tool_use", tu and tu[0]["name"] == "read_file" and tu[0]["input"] == {"path": "a.txt", "lines": 3}, r)
        r = post(port, "/v1/chat/completions", {"model": "x", "messages": msgs})
        check("HTTP plain answer after the scripted call", "AAA" in (r["choices"][0]["message"].get("content") or ""), r)
        try:                      # 0.95: a role V4.1's template does not know is the request's fault - a 400, not a reset
            post(port, "/v1/chat/completions", {"model": "x", "messages": [{"role": "function", "content": "x"}]})
            status = 200
        except urllib.error.HTTPError as e:
            status = e.code
        except Exception as e:    # noqa: BLE001 - a dropped connection
            status = repr(e)
        check("HTTP unknown role -> 400", status == 400, status)
        for fn in ({"arguments": "{}"}, {"name": None, "arguments": "{}"}):   # 0.95 (SGLang #49475): no name, a 400
            try:
                post(port, "/v1/chat/completions", {"model": "x", "messages": [
                    {"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": fn}]}, {"role": "tool", "tool_call_id": "c1", "content": "r"}]})
                status = 200
            except urllib.error.HTTPError as e:
                status = e.code
            except Exception as e:    # noqa: BLE001 - a dropped connection
                status = repr(e)
            check(f"HTTP assistant tool call without a name -> 400 ({fn})", status == 400, status)
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill(); out = proc.communicate()[0]
        if FAILS:
            print("---- server output (tail) ----\n" + out[-3000:])


if __name__ == "__main__":
    # the checks print DeepSeek's markers (U+FF5C ...): a cp1252 console would stop at the first one and skip the rest
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    t_template()
    t_prefill()
    t_effort_kwargs()
    t_parser()
    t_param_schema()
    t_undeclared_call()
    t_tool_qa()
    t_param_close_in_value()
    t_dsml_leak()
    if len(sys.argv) > 1:
        t_tokenizer(Path(sys.argv[1]))
        t_http(Path(sys.argv[1]))
    print(f"{'FAILED ' + str(len(FAILS)) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
