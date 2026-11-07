"""serve/test_embeddings.py - /v1/embeddings and /v1/rerank through a lazy sidecar (serve/embeddings.py), with the
stand-in server serve/embeddings_fake_server.py (no model, no GPU).

    python -m unittest serve.test_embeddings -v
"""
from __future__ import annotations

import base64
import json
import socket
import struct
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import embeddings as EM  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "serve" / "embeddings_fake_server.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Device(unittest.TestCase):
    def test_pick(self):
        self.assertEqual(EM.pick_device("auto", [0], gpus=[0, 1]), 1)      # the spare card
        self.assertEqual(EM.pick_device("auto", [0, 2], gpus=[0, 1, 2]), 1)
        self.assertIsNone(EM.pick_device("auto", [0], gpus=[0]))           # one card: the CPU
        self.assertIsNone(EM.pick_device("auto", [0], gpus=[]))
        self.assertIsNone(EM.pick_device("cpu", [0], gpus=[0, 1]))
        self.assertEqual(EM.pick_device(1, [0], gpus=[]), 1)
        s = EM.Sidecar("embeddings", {"cmd": "x", "gpu": "cpu"})
        self.assertEqual(s.env()["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(EM.Sidecar("rerank", {"cmd": "x", "gpu": 1}).env()["CUDA_VISIBLE_DEVICES"], "1")
        with self.assertRaises(ValueError):
            EM.Sidecar("embeddings", {"args": []})

    def test_shapes(self):
        with self.assertRaises(ValueError):
            EM.embeddings_request({"input": [1, "a"]})
        with self.assertRaises(ValueError):
            EM.embeddings_request({"input": "a", "dimensions": 0})
        with self.assertRaises(ValueError):
            EM.rerank_request({"query": "q", "documents": [{"no": 1}]})
        self.assertEqual(EM.rerank_request({"query": "q", "documents": ["a", {"text": "b"}]})["documents"], ["a", "b"])
        self.assertEqual(EM._shorten([3.0, 4.0, 1.0], 2), [0.6, 0.8])


class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "x"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        for kind in ("embeddings", "rerank"):
            cls.svc.sidecars[kind] = EM.Sidecar(kind, {"cmd": sys.executable, "args": [FAKE, "--port", "{port}"],
                                                       "port": free_port(), "gpu": "cpu", "model": f"tiny-{kind}"})
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        for s in cls.svc.sidecars.values():
            s.stop()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_embeddings(self):
        s, b = self.post("/v1/embeddings", {"model": "whatever", "input": ["ab", "abcd"]})
        self.assertEqual(s, 200, b)
        self.assertEqual(b["model"], "tiny-embeddings")
        self.assertEqual([d["embedding"] for d in b["data"]], [[2.0, 1.0, 2.0, 2.0], [4.0, 1.0, 2.0, 2.0]])
        self.assertEqual(b["usage"], {"prompt_tokens": 6, "total_tokens": 6})
        s, b = self.post("/v1/embeddings", {"input": "abc", "dimensions": 2, "encoding_format": "base64"})
        self.assertEqual(s, 200, b)
        vec = struct.unpack("<2f", base64.b64decode(b["data"][0]["embedding"]))
        self.assertAlmostEqual(vec[0], 3 / 10 ** 0.5, places=5)
        s, b = self.post("/v1/embeddings", {"input": {"x": 1}})
        self.assertEqual(s, 400, b)

    def test_rerank(self):
        docs = ["the cat sat", "dogs and cats", "a cat and a dog sat"]
        for path in ("/v1/rerank", "/rerank"):
            s, b = self.post(path, {"query": "cat sat", "documents": docs, "top_n": 2, "return_documents": True})
            self.assertEqual(s, 200, b)
            self.assertEqual([r["index"] for r in b["results"]], [0, 2])
            self.assertEqual(b["results"][0]["document"], {"text": "the cat sat"})
            self.assertEqual(b["model"], "tiny-rerank")

    def test_listed_on_v1_models(self):
        with urllib.request.urlopen(self.base + "/v1/models", timeout=30) as r:
            data = {m["id"]: m for m in json.loads(r.read())["data"]}
        self.assertEqual(data["tiny-embeddings"]["architecture"]["output_modalities"], ["embedding"])
        self.assertEqual(data["tiny-rerank"]["endpoints"], ["/v1/rerank", "/rerank"])

    def test_not_configured_is_a_404(self):
        side = self.svc.sidecars.pop("rerank")
        try:
            s, b = self.post("/v1/rerank", {"query": "q", "documents": ["a"]})
            self.assertEqual(s, 404)
            self.assertIn("rerank", b["error"]["message"])
        finally:
            self.svc.sidecars["rerank"] = side

    def test_lazy_start_and_stop(self):
        side = self.svc.sidecars["embeddings"]
        side.stop()
        self.assertFalse(side.alive())
        s, _ = self.post("/v1/embeddings", {"input": "a"})
        self.assertEqual(s, 200)
        self.assertTrue(side.alive())
        with urllib.request.urlopen(side.base_url + "/health", timeout=5) as r:
            self.assertEqual(json.loads(r.read())["cuda_visible_devices"], "")    # the CPU: no card visible


if __name__ == "__main__":
    unittest.main()
