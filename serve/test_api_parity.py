"""0.94 P2: API parity and agent correctness, against the mock engine (no GPU).

    python -m unittest serve.test_api_parity -v

C3 server tools left out, C4 structured output and tool control, C8 Anthropic ping events, the quick-win endpoints
and the count_tokens memo.
"""
from __future__ import annotations

import json
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve import frontend, structured
from serve import server as server_mod
from serve.frontend import ChatTemplate, anthropic_to_messages
from serve.server import ByteTokenizer, MockEngine, Service, serve
from serve.test_server import CallingEngine

try:
    import jsonschema  # noqa: F401
    HAVE_JSONSCHEMA = True
except ImportError:
    HAVE_JSONSCHEMA = False

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "serve/chat_template.jinja"
SEARCH = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
CALL = "<tool_call>\n<function=search>\n<parameter=q>\n{}\n</parameter>\n</function>\n</tool_call>"
TWO_CALLS = "</think>\n\n" + CALL.format("one") + "\n" + CALL.format("two")
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


class Http(unittest.TestCase):
    """A server on the mock engine; `script` is what the engine writes (one token per byte)."""
    script: str | list = "</think>\n\nhello"

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = self.make_engine()
        self.svc = Service(self.engine, self.tok, ChatTemplate(TEMPLATE))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def make_engine(self):
        return MockEngine(self.tok, self.script, max_context=16384)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, method=None, headers=None):
        """-> (status, JSON body or the raw text of a stream, headers)."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            r = e
        with r:
            raw = r.read().decode()
            if "event-stream" in (r.headers.get("Content-Type") or "") or not raw:
                return r.status, raw, r.headers
            return r.status, json.loads(raw), r.headers

    def script_is(self, script):
        self.engine = self.svc.engine = MockEngine(self.tok, script, max_context=16384)

    @staticmethod
    def sse(raw):
        """[(event name or None, data)] of an SSE stream; comments as (":", text)."""
        out, name = [], None
        for line in raw.splitlines():
            if line.startswith(":"):
                out.append((":", line))
            elif line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                payload = line[6:]
                out.append((name, json.loads(payload) if payload.startswith("{") else payload))
                name = None
        return out


# ------------------------------------------------------------------------------------------------ C3
class ServerTools(Http):
    TOOLS = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
             {"type": "web_fetch_20250910", "name": "web_fetch"},
             {"type": "code_execution_20250825", "name": "code_execution"},
             {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
             {"name": "read", "description": "read a file", "input_schema": SEARCH},
             {"type": "bash_20250124", "name": "bash"}]

    def test_server_tools_are_left_out_once_logged(self):
        frontend._SERVER_TOOLS_NOTED.clear()
        with mock.patch("builtins.print") as out:
            _, tools, _ = anthropic_to_messages({"messages": [{"role": "user", "content": "hi"}], "tools": self.TOOLS})
            anthropic_to_messages({"messages": [{"role": "user", "content": "hi"}], "tools": self.TOOLS})
        # client tools (a schema of their own, or Anthropic's client tool types such as bash) stay, and so does the
        # tool search this server runs (serve/toolsearch.py; web search and fetch are off until "web_tools")
        self.assertEqual([t["name"] for t in tools], ["read", "bash", "tool_search_tool_regex"])
        lines = [c.args[0] for c in out.call_args_list if "server tools" in c.args[0]]
        self.assertEqual(len(lines), 1)                     # said once per type, not per request
        for kind in ("web_search_20250305", "web_fetch_20250910", "code_execution_20250825"):
            self.assertIn(kind, lines[0])
        self.assertNotIn("tool_search_tool_regex", lines[0])

    def test_only_server_tools_render_as_no_tools(self):
        msgs = [{"role": "user", "content": "hi"}]
        with_server = anthropic_to_messages({"messages": msgs, "tools": self.TOOLS[:1]})
        plain = anthropic_to_messages({"messages": msgs})
        self.assertIsNone(with_server[1])
        self.assertEqual(self.svc.encode_prompt(*with_server), self.svc.encode_prompt(*plain))

    def test_a_registered_server_tool_is_kept(self):
        # one registry (serve/server_tools.py): a configured web search is kept and run, an unconfigured fetch dropped
        from serve import websearch
        websearch.configure({"search": "searxng", "searxng_url": "http://127.0.0.1:9"})
        self.addCleanup(websearch.configure, None)
        _, tools, _ = anthropic_to_messages({"messages": [{"role": "user", "content": "hi"}], "tools": self.TOOLS[:2]})
        self.assertEqual([t["name"] for t in tools], ["web_search"])
        self.assertEqual(tools[0]["parameters"]["required"], ["query"])     # offered with its query schema
        websearch.configure(None)
        _, tools, _ = anthropic_to_messages({"messages": [{"role": "user", "content": "hi"}], "tools": self.TOOLS[:2]})
        self.assertIsNone(tools)

    def test_messages_request_with_server_tools_answers(self):
        code, b, _ = self.request("/v1/messages", {"model": "m", "max_tokens": 50, "tools": self.TOOLS,
                                                    "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200, b)
        self.assertNotIn("web_search", self.tok.decode(self.engine.last_prompt))


# ------------------------------------------------------------------------------------------------ C4
class Formats(Http):
    script = '{"a": 1}'
    SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}

    def messages(self, **extra):
        body = {"model": "m", "max_tokens": 200, "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "give me a"}],
                "output_config": {"format": {"type": "json_schema", "schema": self.SCHEMA}}, **extra}
        return self.request("/v1/messages", body)

    def test_messages_output_config_format(self):
        code, b, _ = self.messages()
        self.assertEqual(code, 200, b)
        self.assertEqual([c["type"] for c in b["content"]], ["text"])
        self.assertEqual(json.loads(b["content"][0]["text"]), {"a": 1})
        self.assertIn("OUTPUT FORMAT REQUIREMENT", self.tok.decode(self.engine.last_prompt))

    def test_messages_legacy_output_format_and_stream(self):
        code, raw, _ = self.messages(output_config=None, output_format={"type": "json_schema", "schema": self.SCHEMA},
                                     stream=True)
        self.assertEqual(code, 200, raw)
        events = [(n, d) for n, d in self.sse(raw) if n not in (":", None)]
        names = [n for n, _ in events]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        text = "".join(d["delta"]["text"] for n, d in events if n == "content_block_delta")
        self.assertEqual(json.loads(text), {"a": 1})

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
    def test_messages_answer_that_fails_is_an_anthropic_error(self):
        self.script_is('{"a": "x"}')
        code, b, _ = self.messages()
        self.assertEqual(code, 502)
        self.assertEqual(b["type"], "error")
        self.assertIn("JSON Schema", b["error"]["message"])
        code, raw, _ = self.messages(stream=True)
        self.assertEqual(code, 200)
        self.assertIn(("error",), [(n,) for n, _ in self.sse(raw)])

    def test_messages_bad_format_is_a_400_naming_output_config(self):
        code, b, _ = self.messages(output_config={"format": {"type": "json_schema", "schema": {"type": "array"}}})
        self.assertEqual(code, 400)
        self.assertEqual(b["type"], "error")
        self.assertEqual(b["error"]["type"], "invalid_request_error")
        self.assertIn("output_config.format", b["error"]["message"])

    def test_directive_goes_after_the_last_user_turn(self):
        system = {"role": "system", "content": "You are terse."}
        history = [system, {"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"},
                   {"role": "user", "content": "second"}]
        fmt = {"type": "json_object"}
        plain = self.tok.decode(self.svc.encode_prompt(history, None, {}))
        late, _ = structured.prepare_format(fmt, history)
        late = self.tok.decode(self.svc.encode_prompt(late, None, {}))
        at = plain.index("second") + len("second")
        self.assertEqual(late[:at], plain[:at])            # the whole prompt up to the user's words is shared
        self.assertGreater(late.index("OUTPUT FORMAT REQUIREMENT"), at)
        early, _ = structured.prepare_format(fmt, history, position="start")
        early = self.tok.decode(self.svc.encode_prompt(early, None, {}))
        self.assertLess(early.index("OUTPUT FORMAT REQUIREMENT"), early.index("first"))
        # after tool results: a user turn of its own; without a user turn at the end (a prefill): the system message
        tool_turn = history + [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search", "arguments": {"q": "x"}}}]}, {"role": "tool", "content": "found"}]
        out, _ = structured.prepare_format(fmt, tool_turn)
        self.assertEqual(out[:-1], tool_turn)
        self.assertEqual(out[-1]["role"], "user")
        prefill = history + [{"role": "assistant", "content": "{"}]
        out, _ = structured.prepare_format(fmt, prefill)
        self.assertIn("OUTPUT FORMAT REQUIREMENT", out[0]["content"])

    def test_a_request_without_a_format_reaches_the_engine_unchanged(self):
        msgs = [{"role": "user", "content": "hi"}]
        self.assertEqual(structured.prepare_format(None, msgs), (msgs, None))
        self.script_is("</think>\n\nhello")
        code, b, _ = self.request("/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": 20})
        self.assertEqual(code, 200, b)
        self.assertEqual(self.engine.last_prompt, self.svc.encode_prompt(msgs, None, {}))


class FormatWithTools(Http):
    script = "</think>\n\n" + CALL.format("weather")
    TOOLS = [{"type": "function", "function": {"name": "search", "parameters": SEARCH}}]
    FORMAT = {"type": "json_schema", "json_schema": {"name": "a", "schema": Formats.SCHEMA}}

    def chat(self, **extra):
        return self.request("/v1/chat/completions", {
            "model": "m", "messages": [{"role": "user", "content": "weather?"}], "max_tokens": 300,
            "tools": self.TOOLS, "response_format": self.FORMAT, **extra})

    def test_a_tool_call_is_returned_as_is(self):
        code, b, _ = self.chat()
        self.assertEqual(code, 200, b)
        choice = b["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"q": "weather"})
        self.assertIn("only to your final answer", self.tok.decode(self.engine.last_prompt))
        code, raw, _ = self.chat(stream=True)
        self.assertEqual(code, 200)
        self.assertIn('"tool_calls"', raw)
        self.assertIn('"finish_reason": "tool_calls"', raw)

    def test_the_final_answer_is_checked(self):
        self.script_is('</think>\n\n{"a": 3}')
        code, b, _ = self.chat()
        self.assertEqual(code, 200, b)
        self.assertEqual(json.loads(b["choices"][0]["message"]["content"]), {"a": 3})
        self.script_is("</think>\n\nnot json")
        self.assertEqual(self.chat()[0], 502)

    def test_mcp_with_a_format_is_still_refused(self):
        self.assertEqual(self.chat(strata_mcp=True)[0], 400)


class ResponsesToolChoice(Http):
    TOOLS = [{"type": "function", "name": "search", "description": "search the web",
              "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
             {"type": "function", "name": "done", "parameters": {"type": "object", "properties": {}}}]

    def make_engine(self):
        return CallingEngine(self.tok)

    def respond(self, **extra):
        return self.request("/v1/responses", {"model": "m", "input": "2+2?", "max_output_tokens": 400,
                                              "tools": self.TOOLS, **extra})

    def calls(self, b):
        return [(o["name"], json.loads(o["arguments"])) for o in b["output"] if o["type"] == "function_call"]

    def test_auto_answers_in_text(self):
        code, b, _ = self.respond()
        self.assertEqual(code, 200, b)
        self.assertEqual(self.calls(b), [])

    def test_required_and_named_force_a_call(self):
        for choice in ("required", {"type": "function", "name": "search"},
                       {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "search"}]}):
            with self.subTest(choice=choice):
                code, b, _ = self.respond(tool_choice=choice)
                self.assertEqual(code, 200, b)
                self.assertEqual(self.calls(b), [("search", {"q": "2+2"})])
                self.assertTrue(self.tok.decode(self.engine.prompts[-1]).endswith("<function=")
                                or self.tok.decode(self.engine.prompts[-1]).endswith("<function=search>\n"))

    def test_effort_uses_the_models_own_table(self):
        # a model adapter (serve_ds41.py) replaces frontend.effort_kwargs: /v1/responses reads the replacement too,
        # so Codex's "max" is the model's max, as on the chat and messages routes
        from serve.responses import template_kwargs
        with mock.patch.object(frontend, "effort_kwargs", lambda v: {"reasoning_effort": {"max": 100}.get(v, 75)}):
            self.assertEqual(template_kwargs({"reasoning": {"effort": "max"}}, {}), {"reasoning_effort": 100})

    def test_namespaced_tool_by_its_own_name(self):
        from serve.responses import request_tools, tool_choice
        req = {"tools": [{"type": "namespace", "name": "mcp__fs", "tools": [{"type": "function", "name": "read"}]}],
               "tool_choice": {"type": "function", "name": "read"}}
        _, names, _ = request_tools(req)
        self.assertEqual(tool_choice(req, names), {"type": "function", "name": "mcp__fs.read"})


class OneCallPerTurn(Http):
    script = TWO_CALLS
    TOOLS = [{"type": "function", "function": {"name": "search", "parameters": SEARCH}}]

    def chat(self, **extra):
        return self.request("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "go"}],
                                                     "max_tokens": 400, "tools": self.TOOLS, **extra})

    def test_chat_parallel_tool_calls_false(self):
        code, b, _ = self.chat()
        self.assertEqual(len(b["choices"][0]["message"]["tool_calls"]), 2)
        code, b, _ = self.chat(parallel_tool_calls=False)
        self.assertEqual(code, 200, b)
        msg = b["choices"][0]["message"]
        self.assertEqual([json.loads(c["function"]["arguments"]) for c in msg["tool_calls"]], [{"q": "one"}])
        self.assertEqual(b["choices"][0]["finish_reason"], "tool_calls")
        self.assertLess(b["usage"]["completion_tokens"], len(TWO_CALLS))      # the engine was stopped there
        code, raw, _ = self.chat(parallel_tool_calls=False, stream=True)
        self.assertEqual(raw.count('"id": "call_'), 1)

    def test_anthropic_disable_parallel_tool_use(self):
        body = {"model": "m", "max_tokens": 400, "messages": [{"role": "user", "content": "go"}],
                "tools": [{"name": "search", "input_schema": SEARCH}]}
        code, b, _ = self.request("/v1/messages", body)
        self.assertEqual(len([c for c in b["content"] if c["type"] == "tool_use"]), 2)
        code, b, _ = self.request("/v1/messages", {**body, "tool_choice": {"type": "auto",
                                                                           "disable_parallel_tool_use": True}})
        self.assertEqual(code, 200, b)
        uses = [c for c in b["content"] if c["type"] == "tool_use"]
        self.assertEqual([u["input"] for u in uses], [{"q": "one"}])
        self.assertEqual(b["stop_reason"], "tool_use")

    def test_responses_parallel_tool_calls_false(self):
        body = {"model": "m", "input": "go", "max_output_tokens": 400,
                "tools": [{"type": "function", "name": "search", "parameters": SEARCH}], "parallel_tool_calls": False}
        code, b, _ = self.request("/v1/responses", body)
        self.assertEqual(code, 200, b)
        self.assertEqual([json.loads(o["arguments"]) for o in b["output"] if o["type"] == "function_call"],
                         [{"q": "one"}])

    def test_the_flag_alone_changes_nothing_for_the_engine(self):
        self.chat()
        before = list(self.engine.last_prompt)
        self.chat(parallel_tool_calls=False)
        self.assertEqual(self.engine.last_prompt, before)


@unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is not installed")
class StrictTools(Http):
    script = "</think>\n\n" + CALL.format("c")
    ENUM = {"type": "object", "properties": {"q": {"type": "string", "enum": ["a", "b"]}}, "required": ["q"]}

    def chat(self, strict, **extra):
        tool = {"type": "function", "function": {"name": "search", "parameters": self.ENUM, "strict": strict}}
        return self.request("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "go"}],
                                                     "max_tokens": 400, "tools": [tool], **extra})

    def test_chat(self):
        self.assertEqual(self.chat(False)[0], 200)
        code, b, _ = self.chat(True)
        self.assertEqual(code, 502)
        self.assertEqual(b["error"]["code"], "structured_output_failed")
        self.assertIn("strict tool search", b["error"]["message"])
        code, raw, _ = self.chat(True, stream=True)
        self.assertIn("structured_output_failed", raw)
        self.script_is("</think>\n\n" + CALL.format("a"))
        self.assertEqual(self.chat(True)[0], 200)

    def test_messages(self):
        body = {"model": "m", "max_tokens": 400, "messages": [{"role": "user", "content": "go"}],
                "tools": [{"name": "search", "input_schema": self.ENUM, "strict": True}]}
        code, b, _ = self.request("/v1/messages", body)
        self.assertEqual(code, 502)
        self.assertEqual(b["type"], "error")
        code, raw, _ = self.request("/v1/messages", {**body, "stream": True})
        self.assertIn("event: error", raw)

    def test_responses(self):
        body = {"model": "m", "input": "go", "max_output_tokens": 400,
                "tools": [{"type": "function", "name": "search", "parameters": self.ENUM, "strict": True}]}
        code, b, _ = self.request("/v1/responses", body)
        self.assertEqual(code, 502, b)
        code, raw, _ = self.request("/v1/responses", {**body, "stream": True})
        self.assertIn("response.failed", raw)


class _Error:
    absolute_path = ("q",)
    message = "'c' is not one of ['a', 'b']"


class _EnumValidator:
    """jsonschema's iter_errors for the ENUM schema (so the pipeline is tested without the optional package)."""

    def iter_errors(self, value):
        if value.get("q") not in ("a", "b"):
            yield _Error()


class StrictPipeline(Http):
    """strict tools through every route with a stand-in validator: a bad call ends the request, a good one passes."""
    script = "</think>\n\n" + CALL.format("c")

    def setUp(self):
        super().setUp()
        fake = lambda tools: {t.get("function", t)["name"]: _EnumValidator() for t in tools or []   # noqa: E731
                              if t.get("function", t).get("strict") is True}
        patcher = mock.patch.object(server_mod, "strict_tool_validators", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_every_route(self):
        chat = {"model": "m", "messages": [{"role": "user", "content": "go"}], "max_tokens": 400,
                "tools": [{"type": "function", "function": {"name": "search", "parameters": SEARCH, "strict": True}}]}
        msgs = {"model": "m", "max_tokens": 400, "messages": [{"role": "user", "content": "go"}],
                "tools": [{"name": "search", "input_schema": SEARCH, "strict": True}]}
        resp = {"model": "m", "input": "go", "max_output_tokens": 400,
                "tools": [{"type": "function", "name": "search", "parameters": SEARCH, "strict": True}]}
        for path, body in (("/v1/chat/completions", chat), ("/v1/messages", msgs), ("/v1/responses", resp)):
            with self.subTest(path=path):
                code, b, _ = self.request(path, body)
                self.assertEqual(code, 502, b)
                self.assertIn("strict tool search", json.dumps(b))
                code, raw, _ = self.request(path, {**body, "stream": True})
                self.assertIn("strict tool search", raw)
        self.script_is("</think>\n\n" + CALL.format("a"))
        for path, body in (("/v1/chat/completions", chat), ("/v1/messages", msgs), ("/v1/responses", resp)):
            with self.subTest(path=path, ok=True):
                self.assertEqual(self.request(path, body)[0], 200)


# ------------------------------------------------------------------------------------------------ C8
class SlowPrefillEngine(MockEngine):
    """A long prompt read: heartbeats (the engine's PP lines) before the first token."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        for _ in range(4):
            time.sleep(0.02)
            yield None
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class AnthropicPings(Http):
    def make_engine(self):
        return SlowPrefillEngine(self.tok, "</think>\n\nhello", max_context=16384)

    def test_a_slow_prefill_sends_ping_events(self):
        with mock.patch.object(server_mod, "ANTHROPIC_PING_S", 0.0):
            code, raw, _ = self.request("/v1/messages", {"model": "m", "max_tokens": 50, "stream": True,
                                                         "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200)
        events = self.sse(raw)
        names = [n for n, _ in events if n != ":"]
        self.assertEqual(names[0], "message_start")
        self.assertGreaterEqual(names.count("ping"), 4)
        self.assertEqual([d for n, d in events if n == "ping"][0], {"type": "ping"})
        self.assertLess(names.index("ping"), names.index("content_block_start"))

    def test_quiet_gaps_shorter_than_the_interval_stay_comments(self):
        code, raw, _ = self.request("/v1/messages", {"model": "m", "max_tokens": 50, "stream": True,
                                                     "messages": [{"role": "user", "content": "hi"}]})
        names = [n for n, _ in self.sse(raw)]
        self.assertNotIn("ping", names)
        self.assertIn(":", names)

    def test_chat_completions_keeps_comments(self):
        with mock.patch.object(server_mod, "ANTHROPIC_PING_S", 0.0):
            code, raw, _ = self.request("/v1/chat/completions", {"model": "m", "max_tokens": 50, "stream": True,
                                                                 "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 200)
        self.assertIn(": keep-alive", raw)
        self.assertNotIn("ping", raw)


# ------------------------------------------------------------------------------------------------ endpoints
class Endpoints(Http):
    def test_responses_input_tokens_is_the_prompt_responses_reads(self):
        body = {"model": "m", "instructions": "Be brief.", "input": "how many tokens?",
                "tools": [{"type": "function", "name": "search", "parameters": SEARCH}]}
        code, b, _ = self.request("/v1/responses/input_tokens", body)
        self.assertEqual(code, 200, b)
        self.assertEqual(b["object"], "response.input_tokens")
        code, r, _ = self.request("/v1/responses", {**body, "max_output_tokens": 20})
        self.assertEqual(code, 200, r)
        self.assertEqual(b["input_tokens"], r["usage"]["input_tokens"])
        code, b, _ = self.request("/v1/responses/input_tokens", {"input": "x", "previous_response_id": "resp_1"})
        self.assertEqual(code, 400)
        self.assertEqual(b["error"]["param"], "previous_response_id")
        # an unknown previous_response_id (0.94: none is stored here) and an unknown response to cancel
        self.assertEqual(self.request("/v1/responses/resp_1/cancel", {})[0], 404)

    def test_tokenize_detokenize(self):
        code, b, _ = self.request("/tokenize", {"content": "hi there"})
        self.assertEqual(code, 200, b)
        self.assertEqual(b["tokens"], self.tok.encode("hi there", parse_special=True))
        code, p, _ = self.request("/tokenize", {"content": "hé", "with_pieces": True})
        self.assertEqual(code, 200, p)
        self.assertEqual([x["id"] for x in p["tokens"]], self.tok.encode("hé", parse_special=True))
        self.assertEqual(p["tokens"][0]["piece"], "h")
        code, d, _ = self.request("/detokenize", {"tokens": b["tokens"]})
        self.assertEqual((code, d), (200, {"content": "hi there"}))
        self.assertEqual(self.request("/detokenize", {"tokens": ["a"]})[0], 400)
        self.assertEqual(self.request("/tokenize", {"content": 5})[0], 400)

    def test_apply_template_is_the_prompt_chat_reads(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        code, b, _ = self.request("/apply-template", {"messages": msgs})
        self.assertEqual(code, 200, b)
        self.request("/v1/chat/completions", {"model": "m", "messages": msgs, "max_tokens": 5})
        self.assertEqual(self.tok.encode(b["prompt"], parse_special=True), self.engine.last_prompt)

    def test_models_metadata(self):
        code, b, _ = self.request("/v1/models")
        self.assertEqual(code, 200, b)
        m = b["data"][0]
        ctx = self.svc.reported_ctx()
        for key in ("context_length", "max_context_length", "max_model_len", "context_window", "max_input_tokens"):
            self.assertEqual(m[key], ctx, key)
        self.assertEqual((m["type"], m["object"], m["id"]), ("model", "model", self.svc.model))
        self.assertTrue(m["display_name"])
        self.assertRegex(m["created_at"], r"^\d{4}-\d\d-\d\dT")
        self.assertTrue(m["capabilities"]["tools"])
        self.assertFalse(m["capabilities"]["vision"])
        self.assertEqual(m["meta"]["n_ctx"], ctx)               # the old fields stay
        self.assertEqual((b["has_more"], b["first_id"], b["last_id"]), (False, m["id"], b["data"][-1]["id"]))

    def test_codex_catalog(self):
        code, b, _ = self.request("/v1/codex/models.json")
        self.assertEqual(code, 200, b)
        m = b["models"][0]
        self.assertEqual(m["slug"], self.svc.model)
        self.assertEqual(m["context_window"], self.svc.reported_ctx())
        # the fields Codex's ModelInfo requires (no serde default)
        for key in ("slug", "display_name", "description", "supported_reasoning_levels", "shell_type", "visibility",
                    "supported_in_api", "priority", "availability_nux", "upgrade", "support_verbosity",
                    "default_verbosity", "apply_patch_tool_type", "truncation_policy", "experimental_supported_tools",
                    "base_instructions"):
            self.assertIn(key, m)
        self.assertEqual([x["effort"] for x in m["supported_reasoning_levels"]], ["low", "medium", "high"])
        self.assertEqual(m["default_reasoning_level"], "high")
        self.assertEqual(m["apply_patch_tool_type"], "freeform")

    def test_codex_catalog_levels_follow_the_models_effort_table(self):
        def ds41(value):                                     # V4.1's table: max is more than high
            return {"reasoning_effort": {"low": 50, "medium": 62, "high": 75, "max": 100}[value]}
        with mock.patch.object(frontend, "effort_kwargs", ds41):
            self.assertEqual(server_mod.effort_levels(), ["low", "medium", "high", "max"])

    def test_head_probe(self):
        for path, want in (("/api/hello", 200), ("/", 200), ("/health", 200), ("/nothing", 404)):
            with self.subTest(path=path):
                code, body, _ = self.request(path, method="HEAD")
                self.assertEqual((code, body), (want, ""))
        self.svc.api_key = "k"
        self.addCleanup(setattr, self.svc, "api_key", None)
        self.assertEqual(self.request("/api/hello", method="HEAD")[0], 401)
        self.assertEqual(self.request("/api/hello", method="HEAD", headers={"x-api-key": "k"})[0], 200)
        self.assertEqual(self.request("/health", method="HEAD")[0], 200)

    def test_anthropic_errors_use_the_envelope(self):
        code, b, _ = self.request("/v1/messages", {"model": "m", "messages": "nope"})
        self.assertEqual(code, 400)
        self.assertEqual(b["type"], "error")
        self.assertEqual(b["error"]["type"], "invalid_request_error")
        self.svc.api_key = "k"
        self.addCleanup(setattr, self.svc, "api_key", None)
        code, b, _ = self.request("/v1/messages?beta=true", {"model": "m", "messages": []})
        self.assertEqual((code, b["type"], b["error"]["type"]), (401, "error", "authentication_error"))
        code, b, _ = self.request("/v1/chat/completions", {"model": "m", "messages": []})
        self.assertNotIn("type", b)                         # OpenAI's routes keep OpenAI's shape
        self.assertEqual(server_mod.anthropic_error(409, {"error": {"type": "model_busy", "message": "busy"}}),
                         {"type": "error", "error": {"type": "api_error", "code": "model_busy", "message": "busy"}})
        self.assertEqual(server_mod.anthropic_error(503, {"error": {"type": "server_error", "message": "x"}})
                         ["error"]["type"], "overloaded_error")


# ------------------------------------------------------------------------------------------------ R8
class CountTokensMemo(Http):
    BODY = {"model": "m", "messages": [{"role": "user", "content": "count me"}]}

    def test_the_same_body_is_counted_once(self):
        with mock.patch.object(self.svc, "encode_prompt", wraps=self.svc.encode_prompt) as enc:
            first = self.request("/v1/messages/count_tokens", self.BODY)
            second = self.request("/v1/messages/count_tokens", self.BODY)
            self.assertEqual(enc.call_count, 1)
            self.assertEqual(first[:2], second[:2])
            other = self.request("/v1/messages/count_tokens", {**self.BODY, "system": "longer"})
            self.assertEqual(enc.call_count, 2)
            self.assertGreater(other[1]["input_tokens"], first[1]["input_tokens"])
            # the shared Chat settings change what is rendered: a new count
            self.svc.shared["reasoning_effort"] = "low"
            self.addCleanup(self.svc.shared.pop, "reasoning_effort", None)
            self.request("/v1/messages/count_tokens", self.BODY)
            self.assertEqual(enc.call_count, 3)
            self.request("/v1/responses/input_tokens", {"input": "count me"})
            self.request("/v1/responses/input_tokens", {"input": "count me"})
            self.assertEqual(enc.call_count, 4)

    def test_the_memo_is_bounded_and_errors_are_not_kept(self):
        with mock.patch.object(server_mod, "COUNT_CACHE_MAX", 3):
            for i in range(6):
                self.request("/v1/messages/count_tokens", {**self.BODY, "system": str(i)})
        self.assertEqual(len(self.svc.count_cache), 3)
        bad = {**self.BODY, "stop_sequences": 5}
        self.assertEqual(self.request("/v1/messages/count_tokens", bad)[0], 400)
        self.assertEqual(self.request("/v1/messages/count_tokens", bad)[0], 400)

    def test_count_includes_the_format_instruction(self):
        fmt = {"output_config": {"format": {"type": "json_schema", "schema": Formats.SCHEMA}}}
        plain = self.request("/v1/messages/count_tokens", self.BODY)[1]["input_tokens"]
        shaped = self.request("/v1/messages/count_tokens", {**self.BODY, **fmt})[1]["input_tokens"]
        self.assertGreater(shaped, plain)


if __name__ == "__main__":
    unittest.main()
