"""`/v1/embeddings` and `/v1/rerank` (0.94): a small embedding or reranking model in a lazy sidecar.

Coding agents (Roo / Kilo codebase indexing, Continue), and RAG front ends (Open WebUI, LibreChat, AnythingLLM) want an
OpenAI-style `/v1/embeddings` and a Jina / Cohere-style `/v1/rerank` next to the chat model.  The big model is the wrong
tool for that, so Strata runs a small one beside it - any OpenAI-compatible server named by `cmd` / `args` in the
config, for example llama.cpp's llama-server with Qwen3-Embedding-0.6B (`--embedding`) or Qwen3-Reranker-0.6B
(`--reranking`), both Apache-2.0:

    "embeddings": {"cmd": "llama-server", "args": ["-m", "Qwen3-Embedding-0.6B-Q8_0.gguf", "--embedding",
                   "--pooling", "last", "-c", "8192", "-ngl", "99", "--port", "{port}"],
                   "model": "qwen3-embedding-0.6b"},
    "rerank":     {"cmd": "llama-server", "args": ["-m", "Qwen3-Reranker-0.6B-Q8_0.gguf", "--reranking",
                   "-c", "8192", "-ngl", "99", "--port", "{port}"], "model": "qwen3-reranker-0.6b"}

- Off until used: the sidecar starts with the first request and stops after `idle_s` (600) without one.
- Where it runs (`"gpu"`): "auto" (the default) takes a second NVIDIA card when the PC has one that the main model does
  not use, else the CPU (at below-normal priority, so the model's expert threads keep theirs); a number names a card
  (as nvidia-smi numbers them); "cpu" forces the CPU.  It never shares the main model's card unless told to.
- The main model's queue is not involved: an embedding request does not wait for a chat answer, nor delay one.

Self-contained: no serve.server import (server.py imports this).
"""
from __future__ import annotations

import base64
import os
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request

HEALTH_TIMEOUT_S = 2.0
FORWARD_TIMEOUT_S = 300.0
DEFAULT_PORTS = {"embeddings": 5310, "rerank": 5320}
BELOW_NORMAL = 0x00004000          # Windows BELOW_NORMAL_PRIORITY_CLASS


def nvidia_gpus() -> list[int]:
    """The NVIDIA cards as nvidia-smi numbers them ([] when there is none or nvidia-smi is missing)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()] if out.returncode == 0 else []


def pick_device(want, main_gpus: list[int], gpus=None) -> int | None:
    """The card the sidecar runs on, or None for the CPU.  `want`: "auto", "cpu" or a card number."""
    if want in (None, "", "auto"):
        spare = [g for g in (nvidia_gpus() if gpus is None else gpus) if g not in set(main_gpus or [0])]
        return spare[0] if spare else None
    if str(want).lower() == "cpu":
        return None
    return int(want)


class Sidecar:
    """One lazily-started OpenAI-compatible server (embeddings or rerank)."""

    def __init__(self, kind: str, cfg: dict, main_gpus=None, log=None, gpus=None):
        if not isinstance(cfg, dict) or not cfg.get("cmd"):
            raise ValueError(f'"{kind}" needs "cmd" (and "args") naming the server to start')
        self.kind = kind
        self.cmd = str(cfg["cmd"])
        self.args = [str(a) for a in cfg.get("args", [])]
        self.port = int(cfg.get("port", DEFAULT_PORTS.get(kind, 5330)))
        self.health_path = str(cfg.get("health", "/health"))
        self.start_timeout_s = float(cfg.get("start_timeout_s", 180))
        self.idle_s = float(cfg.get("idle_s", 600))
        self.model = str(cfg.get("model") or kind)
        self.device = pick_device(cfg.get("gpu", "auto"), main_gpus or [], gpus)
        self.proc = None
        self.lock = threading.Lock()
        self.last_used = 0.0
        self._log = log

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def where(self) -> str:
        return "the CPU" if self.device is None else f"GPU {self.device}"

    def env(self) -> dict:
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "" if self.device is None else str(self.device)
        return env

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self) -> str:
        with self.lock:
            if self.alive():
                self.last_used = time.time()
                return self.base_url
            args = [self.cmd, *[a.format(port=self.port) for a in self.args]]
            print(f"[strata] starting the {self.kind} server ({self.model}) on {self.where()}, port {self.port}",
                  flush=True)
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | (BELOW_NORMAL if os.name == "nt" and
                                                                   self.device is None else 0)
            try:
                self.proc = subprocess.Popen(  # noqa: S603 - the command comes from the config
                    args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=self._log or subprocess.DEVNULL,
                    env=self.env(), creationflags=flags,
                    preexec_fn=(lambda: os.nice(5)) if os.name != "nt" and self.device is None else None)
            except OSError as e:
                self.proc = None
                raise RuntimeError(f"the {self.kind} server could not start: {e}") from None
            try:
                from serve.winjob import contain       # ends with Strata, as the engine does
                contain(self.proc)
            except Exception:  # noqa: BLE001 - best effort
                pass
            self.last_used = time.time()
            deadline = time.time() + self.start_timeout_s
            while time.time() < deadline:
                if not self.alive():
                    self.proc = None
                    raise RuntimeError(f"the {self.kind} server exited during startup")
                if self._healthy():
                    self.last_used = time.time()
                    print(f"[strata] the {self.kind} server is up", flush=True)
                    return self.base_url
                time.sleep(0.25)
            self._stop_locked()
            raise RuntimeError(f"the {self.kind} server did not answer {self.health_path} in "
                               f"{self.start_timeout_s:.0f} s")

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + self.health_path, timeout=HEALTH_TIMEOUT_S) as r:  # noqa: S310
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def _stop_locked(self):
        proc, self.proc = self.proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def stop(self) -> None:
        with self.lock:
            self._stop_locked()

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        import json
        self.ensure()
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=FORWARD_TIMEOUT_S) as r:  # noqa: S310
                status, data = r.status, r.read()
        except urllib.error.HTTPError as e:
            status, data = e.code, e.read()
        finally:
            self.last_used = time.time()
        try:
            return status, json.loads(data or b"{}")
        except ValueError:
            return 502, {"error": {"type": "server_error", "message": f"the {self.kind} server answered "
                                   f"{data[:200]!r}"}}


def start_idle_stop(sidecars: dict) -> None:
    """Stop each sidecar after its idle_s without a request (its file stays in the page cache)."""
    if not sidecars:
        return

    def loop():
        while True:
            time.sleep(5.0)
            for s in sidecars.values():
                try:
                    if s.alive() and time.time() - s.last_used > s.idle_s:
                        s.stop()
                        print(f"[strata] the {s.kind} server stopped after {s.idle_s:.0f} s idle", flush=True)
                except Exception as e:  # noqa: BLE001 - the thread keeps running
                    print(f"[strata] {s.kind}: {e}", flush=True)
    threading.Thread(target=loop, daemon=True, name="strata-sidecar-idle").start()


# ------------------------------------------------------------------------------------------------ the API shapes
def embeddings_request(req: dict) -> dict:
    """The client's /v1/embeddings body -> the sidecar's: input checked, floats asked for (dimensions and base64 are
    done here, so any OpenAI-compatible sidecar serves them)."""
    inp = req.get("input")
    if isinstance(inp, str):
        ok = True
    elif isinstance(inp, list) and inp:
        ok = all(isinstance(x, str) for x in inp) or all(isinstance(x, int) for x in inp) or \
            all(isinstance(x, list) and all(isinstance(t, int) for t in x) for x in inp)
    else:
        ok = False
    if not ok:
        raise ValueError("input must be a string, a list of strings, or token ids")
    fmt = req.get("encoding_format", "float")
    if fmt not in ("float", "base64", None):
        raise ValueError('encoding_format must be "float" or "base64"')
    dims = req.get("dimensions")
    if dims is not None and (isinstance(dims, bool) or not isinstance(dims, int) or dims < 1):
        raise ValueError("dimensions must be a whole number of at least 1")
    return {"input": inp, "encoding_format": "float"}


def _shorten(vec: list[float], dims: int | None) -> list[float]:
    """Matryoshka (MRL) models such as Qwen3-Embedding: the first `dims` values, normalized again."""
    if not dims or dims >= len(vec):
        return vec
    v = vec[:dims]
    n = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / n for x in v]


def embeddings_response(req: dict, out: dict, model: str) -> dict:
    data = []
    for i, row in enumerate(out.get("data") or []):
        vec = _shorten([float(x) for x in row.get("embedding") or []], req.get("dimensions"))
        if req.get("encoding_format") == "base64":
            vec = base64.b64encode(struct.pack(f"<{len(vec)}f", *vec)).decode("ascii")
        data.append({"object": "embedding", "index": row.get("index", i), "embedding": vec})
    usage = out.get("usage") or {}
    n = int(usage.get("prompt_tokens") or 0)
    return {"object": "list", "data": data, "model": model,
            "usage": {"prompt_tokens": n, "total_tokens": int(usage.get("total_tokens") or n)}}


def rerank_request(req: dict) -> dict:
    query, docs = req.get("query"), req.get("documents")
    if not isinstance(query, str) or not query:
        raise ValueError("query must be a non-empty string")
    if not isinstance(docs, list) or not docs:
        raise ValueError("documents must be a non-empty list")
    texts = []
    for i, d in enumerate(docs):
        t = d if isinstance(d, str) else d.get("text") if isinstance(d, dict) else None
        if not isinstance(t, str):
            raise ValueError(f'documents[{i}] must be a string or {{"text": ...}}')
        texts.append(t)
    top_n = req.get("top_n")
    if top_n is not None and (isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1):
        raise ValueError("top_n must be a whole number of at least 1")
    return {"query": query, "documents": texts}


def rerank_response(req: dict, out: dict, model: str, texts: list[str]) -> dict:
    rows = out.get("results") if isinstance(out.get("results"), list) else out.get("data") or []
    res = []
    for r in rows:
        i = int(r.get("index", 0))
        score = r.get("relevance_score", r.get("score"))
        item = {"index": i, "relevance_score": float(score) if score is not None else 0.0}
        if req.get("return_documents"):
            item["document"] = {"text": texts[i] if 0 <= i < len(texts) else ""}
        res.append(item)
    res.sort(key=lambda r: -r["relevance_score"])
    if req.get("top_n"):
        res = res[:int(req["top_n"])]
    usage = out.get("usage") or {}
    return {"object": "list", "model": model, "results": res,
            "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
                      "total_tokens": int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)}}
