"""serve/test_toolsearch.py - deferred tools and tool search (serve/server_tools.py, serve/toolsearch.py): deferred
tools stay out of the prompt, tool_reference blocks expand in place, the tool_search_tool_regex / _bm25 server tools run
inside the answer, and a replayed turn renders the prompt the server read - against the mock engine (no GPU).

    python -m unittest serve.test_toolsearch -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server_tools as ST  # noqa: E402
from serve import toolsearch as TS  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_request, anthropic_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def tool(name, desc, **props):
    return {"name": name, "description": desc, "input_schema": {
        "type": "object", "properties": {k: {"type": "string", "description": v} for k, v in props.items()}}}


BASH = tool("Bash", "Run a shell command", command="the command")
WEATHER = dict(tool("get_weather", "Current weather for a city", city="the city name"), defer_loading=True)
DB = dict(tool("mcp__db__query", "Run a read-only SQL query on the project database", sql="the SQL text"),
          defer_loading=True)
MAIL = dict(tool("mcp__mail__send", "Send an email message", to="recipient address", body="the text"),
            defer_loading=True)
REGEX = {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
BM25 = {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}


def call_script(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"</think>\n\nLet me look.\n\n<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class ScriptedEngine(MockEngine):
    def __init__(self, tok, scripts, **k):
        super().__init__(tok, list(scripts), **k)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def prompt_text(self, i):
        return bytes(t for t in self.prompts[i] if t < 256).decode("utf-8", "replace")


class Search(unittest.TestCase):
    def setUp(self):
        self.tools = [ST.anthropic_tools([t])[0][0] for t in (BASH, WEATHER, DB, MAIL)]

    def test_regex(self):
        self.assertEqual(TS.regex_search("weather", self.tools), ["get_weather"])
        self.assertEqual(TS.regex_search("mcp__", self.tools), ["mcp__db__query", "mcp__mail__send"])
        self.assertEqual(TS.regex_search("SQL", self.tools), ["mcp__db__query"])          # description, any case
        self.assertEqual(TS.regex_search("recipient", self.tools), ["mcp__mail__send"])   # an argument's description
        for bad, code in (("(", "invalid_pattern"), ("a" * 201, "pattern_too_long"), ("", "invalid_pattern")):
            with self.assertRaises(ValueError) as e:
                TS.regex_search(bad, self.tools)
            self.assertEqual(str(e.exception), code)

    def test_bm25(self):
        self.assertEqual(TS.bm25_search("send an email", self.tools)[0], "mcp__mail__send")
        self.assertEqual(TS.bm25_search("database query", self.tools)[0], "mcp__db__query")
        self.assertEqual(TS.bm25_search("weather in Paris", self.tools)[0], "get_weather")
        self.assertEqual(TS.bm25_search("zzz qqq", self.tools), [])
        self.assertEqual(TS.words("getWeather mcp__db__query"), ["get", "weather", "mcp", "db", "query"])


class Render(unittest.TestCase):
    def setUp(self):
        ST.TOOL_SEARCH = True
        self.template = ChatTemplate(ROOT / "serve/chat_template.jinja")
        self.svc = Service(MockEngine(ByteTokenizer(), "x"), ByteTokenizer(), self.template)

    def tearDown(self):
        ST.TOOL_SEARCH = True

    def prompt(self, req):
        messages, tools, kw = anthropic_to_messages(req)
        return bytes(t for t in self.svc.encode_prompt(messages, tools, kw) if t < 256).decode()

    def test_requests_without_deferred_tools_render_as_before(self):
        req = {"system": "s", "tools": [BASH, tool("Read", "read a file", path="p")],
               "messages": [{"role": "user", "content": "hi"}]}
        messages, tools, kw = anthropic_to_messages(req)
        before = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
                  for t in req["tools"]]
        self.assertEqual(tools, before)
        self.assertEqual(self.svc.encode_prompt(messages, tools, kw),
                         self.svc.tok.encode(self.template.render(messages, tools=before, **kw), parse_special=True))

    def test_deferred_tools_are_left_out_but_parsed(self):
        req = {"tools": [BASH, WEATHER, DB], "messages": [{"role": "user", "content": "hi"}]}
        p = self.prompt(req)
        self.assertIn("Run a shell command", p)
        self.assertNotIn("Current weather", p)
        self.assertNotIn("read-only SQL", p)
        _, tools, _ = anthropic_to_messages(req)
        self.assertEqual([t["name"] for t in tools], ["Bash", "get_weather", "mcp__db__query"])  # the parser's map
        ST.TOOL_SEARCH = False                       # "tool_search": "off": everything listed, as before 0.94
        self.assertIn("Current weather", self.prompt(req))

    def test_only_deferred_tools_and_no_search_are_all_listed(self):
        req = {"tools": [WEATHER, DB], "messages": [{"role": "user", "content": "hi"}]}
        self.assertIn("Current weather", self.prompt(req))

    def test_tool_reference_in_a_tool_result_expands_in_place(self):
        # Claude Code's own ToolSearch: a client tool whose result names tools
        search = tool("ToolSearch", "Find deferred tools", query="q")
        req = {"tools": [BASH, search, WEATHER, DB], "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "ToolSearch",
                                               "input": {"query": "weather"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [
                {"type": "tool_reference", "tool_name": "get_weather"}]}]}]}
        p = self.prompt(req)
        head, _, tail = p.partition("These tools are now available")
        self.assertNotIn("Current weather", head)      # not in the tool list at the top
        self.assertIn('"name": "get_weather", "description": "Current weather for a city"', tail)
        self.assertNotIn("read-only SQL", p)

    def test_search_tool_is_offered_with_the_deferred_names(self):
        req = {"tools": [BASH, WEATHER, DB, REGEX], "messages": [{"role": "user", "content": "hi"}]}
        _, tools, _, ctx = anthropic_request(req)
        self.assertIn("tool_search_tool_regex", ctx.server)
        offered = [t for t in tools if t["name"] == "tool_search_tool_regex"][0]
        self.assertIn("get_weather, mcp__db__query", offered["description"])
        self.assertEqual(offered["parameters"]["required"], ["query"])
        self.assertIn("tool_search_tool_regex", self.prompt(req))


class Http(unittest.TestCase):
    def start(self, scripts):
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, scripts, max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, body, path="/v1/messages"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
                return r.status, json.loads(data) if not body.get("stream") else data.decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_regex_search_runs_in_the_answer_and_replays_to_the_same_prompt(self):
        self.start([call_script("tool_search_tool_regex", query="weather"),
                    call_script("get_weather", city="Paris")])
        req = {"model": "m", "max_tokens": 500, "tools": [BASH, WEATHER, DB, REGEX],
               "messages": [{"role": "user", "content": "weather in Paris?"}]}
        s, b = self.post(req)
        self.assertEqual(s, 200, b)
        kinds = [c["type"] for c in b["content"]]
        self.assertEqual(kinds, ["text", "server_tool_use", "tool_search_tool_result", "text", "tool_use"], b)
        use, result, call = b["content"][1], b["content"][2], b["content"][4]
        self.assertEqual((use["name"], use["input"]), ("tool_search_tool_regex", {"query": "weather"}))
        self.assertTrue(use["id"].startswith("srvtoolu_"))
        self.assertEqual(result["tool_use_id"], use["id"])
        self.assertEqual(result["content"], {"type": "tool_search_tool_search_result",
                                             "tool_references": [{"type": "tool_reference",
                                                                  "tool_name": "get_weather"}]})
        self.assertEqual((call["name"], call["input"]), ("get_weather", {"city": "Paris"}))   # parsed with its schema
        self.assertEqual(b["stop_reason"], "tool_use")
        self.assertEqual(b["usage"]["server_tool_use"], {"tool_search_requests": 1})
        first, second = self.engine.prompt_text(0), self.engine.prompt_text(1)
        self.assertNotIn("Current weather", first)
        self.assertTrue(second.startswith(first.rsplit("<|im_start|>assistant", 1)[0]))
        self.assertIn('"name": "get_weather"', second)
        # the client sends the turn back with the tool's result: its prompt starts with what the server read
        replay = dict(req, messages=req["messages"] + [
            {"role": "assistant", "content": b["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "sunny"}]}])
        messages, tools, kw = anthropic_to_messages(replay)
        replayed = bytes(t for t in self.svc.encode_prompt(messages, tools, kw) if t < 256).decode()
        self.assertTrue(replayed.startswith(second.rsplit("<|im_start|>assistant", 1)[0]),
                        (second[-300:], replayed[len(second) - 300:len(second) + 50]))

    def test_bm25_stream_and_a_bad_pattern(self):
        self.start([call_script("tool_search_tool_regex", query="("), "</think>\n\nNo tool for that."])
        req = {"model": "m", "max_tokens": 500, "stream": True, "tools": [BASH, DB, REGEX, BM25],
               "messages": [{"role": "user", "content": "query the db"}]}
        s, text = self.post(req)
        self.assertEqual(s, 200)
        blocks = [json.loads(line[6:])["content_block"] for line in text.splitlines()
                  if line.startswith("data: ") and '"content_block_start"' in line]
        err = [b for b in blocks if b["type"] == "tool_search_tool_result"][0]
        self.assertEqual(err["content"], {"type": "tool_search_tool_result_error", "error_code": "invalid_pattern"})
        self.assertIn("error: invalid pattern", self.engine.prompt_text(1))
        self.assertIn("event: message_stop", text)

    def test_count_tokens_leaves_deferred_tools_out(self):
        self.start(["</think>\n\nok"])
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        s, small = self.post(dict(base, tools=[BASH, WEATHER, DB, MAIL]), "/v1/messages/count_tokens")
        s2, big = self.post(dict(base, tools=[BASH] + [{k: v for k, v in t.items() if k != "defer_loading"}
                                                       for t in (WEATHER, DB, MAIL)]), "/v1/messages/count_tokens")
        self.assertEqual((s, s2), (200, 200))
        self.assertLess(small["input_tokens"], big["input_tokens"])


if __name__ == "__main__":
    unittest.main()
