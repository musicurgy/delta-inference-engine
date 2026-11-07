"""A stand-in for llama-server --embedding / --reranking (and a chat model) in the tests (serve/test_embeddings.py,
serve/test_helpers.py): /health, POST /v1/chat/completions (echoes the last message as a JSON title),
POST /v1/embeddings (a 4-number vector per input, from its length) and POST /v1/rerank (a document's score is the number
of query words in it).  Standard library only.

    python serve/embeddings_fake_server.py --port 5310
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path == "/v1/embeddings":
            inp = req["input"] if isinstance(req["input"], list) else [req["input"]]
            data = [{"object": "embedding", "index": i, "embedding": [float(len(str(x))), 1.0, 2.0, 2.0]}
                    for i, x in enumerate(inp)]
            self.reply(200, {"object": "list", "data": data, "model": req.get("model"),
                             "usage": {"prompt_tokens": 3 * len(inp), "total_tokens": 3 * len(inp)}})
        elif self.path == "/v1/chat/completions":         # the helper lane: echo what it was given
            last = req["messages"][-1]["content"]
            thinking = (req.get("chat_template_kwargs") or {}).get("enable_thinking")
            self.reply(200, {"choices": [{"message": {"role": "assistant", "content": json.dumps(
                {"title": last[:20], "thinking": thinking, "system": req["messages"][0]["role"]})},
                "finish_reason": "stop"}], "usage": {"prompt_tokens": 11, "completion_tokens": 4}})
        elif self.path == "/v1/rerank":
            words = set(req["query"].lower().split())
            res = [{"index": i, "relevance_score": float(sum(w in words for w in d.lower().split()))}
                   for i, d in enumerate(req["documents"])]
            self.reply(200, {"model": req.get("model"), "results": res, "usage": {"prompt_tokens": 7}})
        else:
            self.reply(404, {"error": "not found"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ThreadingHTTPServer(("127.0.0.1", ap.parse_args().port), H).serve_forever()
