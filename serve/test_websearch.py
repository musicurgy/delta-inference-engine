"""serve/test_websearch.py - web search and web fetch as server tools (serve/websearch.py) against a mock SearXNG and a
mock web site on 127.0.0.1 and the mock engine: no real search is sent, no GPU.

    python -m unittest serve.test_websearch -v
"""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server_tools as ST  # noqa: E402
from serve import websearch as W  # noqa: E402
from serve.frontend import ChatTemplate, anthropic_request, anthropic_to_messages  # noqa: E402
from serve.responses import input_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, public_url, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = [{"url": "https://example.org/a", "title": "Result A", "content": "<b>Alpha</b> text " + "x" * 900,
            "publishedDate": "2026-10-01"},
           {"url": "https://blocked.example.com/b", "title": "Result B", "content": "Beta"},
           {"url": "https://docs.example.net/c", "title": "Result C", "content": "Gamma"}]
PAGE = ("<html><head><title>The page</title><style>.x{}</style><script>var secret=1</script></head>"
        "<body><h1>Hello</h1><p>First   paragraph.</p><div>Second</div></body></html>")


class Site(BaseHTTPRequestHandler):
    queries: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path)
        if u.path == "/search":
            q = urllib.parse.parse_qs(u.query)
            Site.queries.append(q)
            body, ctype = json.dumps({"results": RESULTS}).encode(), "application/json"
        elif u.path == "/page":
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        elif u.path == "/plain":
            body, ctype = ("line\n" * 10000).encode(), "text/plain"
        elif u.path == "/bounce":
            self.send_response(302)
            self.send_header("Location", "/private")
            self.end_headers()
            return
        elif u.path == "/private":
            body, ctype = b"secret", "text/plain"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def no_private(url):
    """A stand-in for public_url that lets the test site through but not its /private page."""
    if "/private" in url:
        raise ValueError("points into this PC")


def call_script(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"</think>\n\nSearching.\n\n<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class ScriptedEngine(MockEngine):
    def __init__(self, tok, scripts, **k):
        super().__init__(tok, list(scripts), **k)
        self.prompts = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.prompts.append(list(ids))
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)

    def prompt_text(self, i):
        return bytes(t for t in self.prompts[i] if t < 256).decode("utf-8", "replace")


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.site = ThreadingHTTPServer(("127.0.0.1", 0), Site)
        threading.Thread(target=cls.site.serve_forever, daemon=True).start()
        cls.root = f"http://127.0.0.1:{cls.site.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.site.shutdown()
        cls.site.server_close()

    def setUp(self):
        W.configure({"search": "searxng", "searxng_url": self.root, "fetch": True, "snippet_chars": 100},
                    public_url=no_private)
        self.addCleanup(W.configure, None)


class Config(unittest.TestCase):
    def test_bad_blocks(self):
        for block, word in (({"search": "google"}, "one of"), ({"search": "brave"}, "brave_api_key"),
                            ({"search": "searxng", "searxng_url": "ftp://x"}, "http"),
                            ({"search": "tavily", "tavily_api_key": "k", "max_uses": 0}, "max_uses"), ([], None)):
            with self.subTest(block=block):
                if word is None:
                    self.assertEqual(W.configure(block), {})
                    continue
                with self.assertRaises(ValueError) as e:
                    W.configure(block)
                self.assertIn(word, str(e.exception))
        W.configure(None)
        self.assertFalse(W.searching() or W.fetching())

    def test_off_leaves_the_tool_as_before(self):
        W.configure(None)
        req = {"tools": [{"type": "web_search_20250305", "name": "web_search"}], "messages": [
            {"role": "user", "content": "hi"}]}
        _, tools, _, ctx = anthropic_request(req)
        self.assertEqual(ctx.server, {})
        self.assertEqual(tools, [{"name": "web_search", "description": "", "parameters": {}}])

    def test_brave_and_tavily_requests(self):
        seen = []

        def fake(url, headers=None, data=None):
            seen.append((url, headers, data))
            return {"web": {"results": [{"url": "https://a.b/", "title": "T", "description": "D", "age": "1d"}]}} \
                if "brave" in url else {"results": [{"url": "https://c.d/", "title": "U", "content": "E"}]}
        with mock.patch.object(W, "_json", fake):
            W.configure({"search": "brave", "brave_api_key": "BK"})
            self.assertEqual(W.search("cats")[0], {"url": "https://a.b/", "title": "T", "snippet": "D",
                                                   "page_age": "1d"})
            self.assertEqual(seen[-1][1]["X-Subscription-Token"], "BK")
            self.assertIn("q=cats", seen[-1][0])
            W.configure({"search": "tavily", "tavily_api_key": "TK"})
            self.assertEqual(W.search("dogs")[0]["title"], "U")
            self.assertEqual(seen[-1][1]["Authorization"], "Bearer TK")
            self.assertEqual(json.loads(seen[-1][2])["query"], "dogs")
        W.configure(None)


class Functions(Base):
    def test_search_trims_filters_and_replays(self):
        r = W.search("alpha", blocked=["example.com"])
        self.assertEqual([x["title"] for x in r], ["Result A", "Result C"])
        self.assertTrue(r[0]["snippet"].startswith("Alpha text"))           # tags gone
        self.assertLessEqual(len(r[0]["snippet"]), 105)
        self.assertEqual([x["title"] for x in W.search("alpha", allowed=["example.net"])], ["Result C"])
        blocks = W.search_blocks(r)
        self.assertTrue(blocks[0]["encrypted_content"].startswith(W.ENC_PREFIX))
        self.assertEqual(W.results_text(W.results_of_blocks(blocks)), W.results_text(r))
        self.assertEqual(Site.queries[-1]["format"], ["json"])
        with self.assertRaises(W.WebError) as e:
            W.search("q" * 401)
        self.assertEqual(e.exception.code, "query_too_long")

    def test_fetch(self):
        url = self.root + "/page"
        page = W.fetch(url, seen=f"read {url} please")
        self.assertEqual(page["title"], "The page")
        self.assertEqual(page["text"], "Hello\n\nFirst paragraph.\n\nSecond")
        self.assertNotIn("secret", page["text"])
        W.CONFIG["fetch_max_chars"] = 50
        plain = W.fetch(self.root + "/plain", seen=self.root + "/plain")
        self.assertTrue(plain["text"].endswith("[cut at 50 characters]"))
        for target, seen, code in ((url, "nothing", "url_not_allowed"),                 # not in the conversation
                                   (self.root + "/bounce", self.root + "/bounce", "url_not_allowed"),   # redirect
                                   ("file:///etc/passwd", "file:///etc/passwd", "invalid_input"),
                                   (self.root + "/missing", self.root + "/missing", "url_not_accessible")):
            with self.subTest(target=target):
                with self.assertRaises(W.WebError) as e:
                    W.fetch(target, seen=seen)
                self.assertEqual(e.exception.code, code)

    def test_fetch_refuses_this_pc_with_the_real_check(self):
        W.configure({"fetch": True}, public_url=public_url)
        url = self.root + "/page"
        with mock.patch.dict("os.environ", {"STRATA_IMAGE_FETCH_PRIVATE": ""}):
            with self.assertRaises(W.WebError) as e:
                W.fetch(url, seen=url)
        self.assertEqual(e.exception.code, "url_not_allowed")


class Http(Base):
    def start(self, scripts):
        tok = ByteTokenizer()
        self.engine = ScriptedEngine(tok, scripts, max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_claude_code_websearch(self):
        # Claude Code's WebSearch: a request with only the server tool, the model searches and answers
        self.start([call_script("web_search", query="alpha news"), "</think>\n\nAlpha happened [1]."])
        req = {"model": "m", "max_tokens": 400, "tools": [{"type": "web_search_20250305", "name": "web_search",
                                                             "max_uses": 3, "blocked_domains": ["example.com"]}],
               "messages": [{"role": "user", "content": "Perform a web search for the query: alpha news"}]}
        s, b = self.post("/v1/messages", req)
        self.assertEqual(s, 200, b)
        self.assertEqual([c["type"] for c in b["content"]],
                         ["text", "server_tool_use", "web_search_tool_result", "text"])
        res = b["content"][2]["content"]
        self.assertEqual([r["url"] for r in res], ["https://example.org/a", "https://docs.example.net/c"])
        self.assertEqual(b["usage"]["server_tool_use"], {"web_search_requests": 1})
        self.assertEqual(b["stop_reason"], "end_turn")
        second = self.engine.prompt_text(1)
        self.assertIn("[1] Result A", second)
        replay = dict(req, messages=req["messages"] + [{"role": "assistant", "content": b["content"]}])
        messages, tools, kw = anthropic_to_messages(replay)
        text = bytes(t for t in self.svc.encode_prompt(messages, tools, kw) if t < 256).decode()
        self.assertTrue(text.startswith(second.rsplit("<|im_start|>assistant", 1)[0]))

    def test_max_uses_and_fetch_in_one_answer(self):
        self.start([call_script("web_search", query="one"), call_script("web_search", query="two"),
                    call_script("web_fetch", url=self.root + "/page"), "</think>\n\nDone."])
        req = {"model": "m", "max_tokens": 400, "tools": [
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 1},
            {"type": "web_fetch_20250910", "name": "web_fetch"}],
            "messages": [{"role": "user", "content": f"look at {self.root}/page and search"}]}
        s, b = self.post("/v1/messages", req)
        self.assertEqual(s, 200, b)
        results = [c for c in b["content"] if c["type"].endswith("_tool_result")]
        self.assertIsInstance(results[0]["content"], list)
        self.assertEqual(results[1]["content"], {"type": "web_search_tool_result_error",
                                                 "error_code": "max_uses_exceeded"})
        self.assertEqual(results[2]["type"], "web_fetch_tool_result")
        self.assertEqual(results[2]["content"]["content"]["source"]["data"], "Hello\n\nFirst paragraph.\n\nSecond")
        self.assertEqual(b["usage"]["server_tool_use"], {"web_search_requests": 1, "web_fetch_requests": 1})
        self.assertIn("Page " + self.root + "/page (The page):", self.engine.prompt_text(3))
        # the replay renders the fetched page and the error exactly as read
        replay = dict(req, messages=req["messages"] + [{"role": "assistant", "content": b["content"]}])
        messages, tools, kw = anthropic_to_messages(replay)
        text = bytes(t for t in self.svc.encode_prompt(messages, tools, kw) if t < 256).decode()
        last = self.engine.prompt_text(3)
        self.assertTrue(text.startswith(last.rsplit("<|im_start|>assistant", 1)[0]))

    def test_codex_web_search(self):
        self.start([call_script("web_search", query="alpha"), "</think>\n\nIt is alpha."])
        req = {"model": "m", "input": "what is alpha?", "tools": [{"type": "web_search"}]}
        s, b = self.post("/v1/responses", req)
        self.assertEqual(s, 200, b)
        kinds = [o["type"] for o in b["output"]]
        self.assertIn("web_search_call", kinds)
        call = next(o for o in b["output"] if o["type"] == "web_search_call")
        self.assertEqual((call["status"], call["action"]), ("completed", {"type": "search", "query": "alpha"}))
        self.assertEqual(b["output"][-1]["content"][0]["text"], "It is alpha.")
        # Codex sends the items back: the search renders as the model read it
        items = [{"type": "message", "role": "user", "content": "what is alpha?"}] + b["output"]
        messages = input_messages({"input": items})
        tool = next(m for m in messages if m["role"] == "tool")
        self.assertIn("[1] Result A", tool["content"])
        self.assertEqual(next(m for m in messages if m.get("tool_calls"))["tool_calls"][0]["function"],
                         {"name": "web_search", "arguments": {"query": "alpha"}})


if __name__ == "__main__":
    unittest.main()
