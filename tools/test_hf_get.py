#!/usr/bin/env python3
"""Tests for tools/v41/serve/hf_get.py (the download runner) with a fake "hf" command - no network, no model files.
Run: python -m unittest tools.test_hf_get"""
import io
import json
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "v41" / "serve"))
import hf_get  # noqa: E402

FAKE = textwrap.dedent(r'''
    import json, os, sys, time
    args = sys.argv[2:]                       # after "download"
    d = args[args.index("--local-dir") + 1]
    name = args[1]
    mode = os.environ["FAKE_MODE"]
    size = int(os.environ["FAKE_SIZE"])
    log = os.path.join(d, "calls.jsonl")
    calls = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    with open(log, "a") as f:
        f.write(json.dumps({"xet_off": os.environ.get("HF_HUB_DISABLE_XET") == "1",
                            "bars_off": os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"}) + "\n")
    part = os.path.join(d, ".cache", "huggingface", "download")
    os.makedirs(part, exist_ok=True)
    def finish():
        with open(os.path.join(d, name), "wb") as f:
            f.write(b"x" * size)
        sys.exit(0)
    if mode == "ok":
        finish()
    if mode == "stall" and calls < int(os.environ.get("FAKE_STALLS", "1")):
        with open(os.path.join(part, "abc.incomplete"), "wb") as f:
            f.write(b"x" * 10)
        time.sleep(60)
    if mode == "missing" and calls == 0:
        sys.exit(0)                           # "finished", nothing written
    if mode == "fail" and calls < int(os.environ.get("FAKE_FAILS", "1")):
        sys.exit(3)
    finish()
''')


class Item:
    def __init__(self, path, size):
        self.path, self.size = path, size


class FakeApi:
    def __init__(self, items):
        self.items = items

    def get_paths_info(self, repo, files, **kw):
        return [Item(p, n) for p, n in self.items if p in files]

    def list_repo_tree(self, repo, recursive=True, **kw):
        return [Item(p, n) for p, n in self.items]


class HfGetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        self.fake = os.path.join(self.dir, "fake_hf.py")
        Path(self.fake).write_text(FAKE)
        self.models = os.path.join(self.dir, "models")
        os.makedirs(self.models)
        os.environ["DS41_NO_TOAST"] = "1"
        os.environ["DS41_NO_KEEPAWAKE"] = "1"
        os.environ.pop("HF_HUB_DISABLE_XET", None)
        os.environ["HF_XET_CACHE"] = os.path.join(self.dir, "xet")    # an empty Xet cache: only the folder moves

    def tearDown(self):
        self.tmp.cleanup()

    def run_fake(self, mode, size=1000, **env):
        os.environ.update({"FAKE_MODE": mode, "FAKE_SIZE": str(size), **{k: str(v) for k, v in env.items()}})
        out = io.StringIO()
        rc = hf_get.run(["org/repo", "model.gguf", "--local-dir", self.models], hf_cmd=[sys.executable, self.fake],
                        stall_s=1.5, poll_s=0.2, backoff=(0, 0, 0), api=FakeApi([("model.gguf", size)]), out=out)
        calls = [json.loads(x) for x in Path(self.models, "calls.jsonl").read_text().splitlines()]
        return rc, calls, out.getvalue()

    def test_ok(self):
        rc, calls, text = self.run_fake("ok")
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["bars_off"])                 # our own progress line instead of hf's bars
        self.assertIn("(0.0 MB)", text)

    def test_stall_restarts_then_xet_off(self):
        rc, calls, text = self.run_fake("stall", FAKE_STALLS=2)
        self.assertEqual(rc, 0)
        self.assertEqual([c["xet_off"] for c in calls], [False, False, True])
        self.assertIn("has not moved", text)
        self.assertIn("HF_HUB_DISABLE_XET=1", text)

    def test_exit_0_with_files_missing_is_retried(self):
        rc, calls, text = self.run_fake("missing")
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        self.assertIn("not complete", text)

    def test_failures_retried_then_given_up(self):
        rc, calls, text = self.run_fake("fail", FAKE_FAILS=1)
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 2)
        Path(self.models, "calls.jsonl").unlink()
        Path(self.models, "model.gguf").unlink()
        rc, calls, text = self.run_fake("fail", FAKE_FAILS=10)
        self.assertEqual(rc, 3)
        self.assertEqual(len(calls), 4)                      # the first try + one per back-off step
        self.assertTrue(calls[-1]["xet_off"])
        self.assertIn("did not work after several tries", text)


class PartsTest(unittest.TestCase):
    def test_parse(self):
        p = hf_get.parse(["unsloth/X", "--revision", "60ad", "--include", "UD-IQ1_S/*", "--local-dir", "D:/m"])
        self.assertEqual((p["repo"], p["revision"], p["include"], p["local_dir"], p["files"]),
                         ("unsloth/X", "60ad", ["UD-IQ1_S/*"], "D:/m", []))
        p = hf_get.parse(["a/b", "f1", "f2", "--local-dir=D"])
        self.assertEqual((p["files"], p["local_dir"]), (["f1", "f2"], "D"))

    def test_plan_include(self):
        api = FakeApi([("UD-IQ1_S/a.gguf", 5), ("UD-IQ1_S/b.gguf", 7), ("UD-Q2/c.gguf", 9)])
        p = hf_get.parse(["u/X", "--include", "UD-IQ1_S/*", "--local-dir", "."])
        self.assertEqual(hf_get.plan(p, api), [("UD-IQ1_S/a.gguf", 5), ("UD-IQ1_S/b.gguf", 7)])

    def test_plan_offline(self):
        class Broken:
            def get_paths_info(self, *a, **k):
                raise OSError("offline")
        self.assertIsNone(hf_get.plan(hf_get.parse(["a/b", "f", "--local-dir", "."]), Broken()))

    def test_words(self):
        self.assertEqual(hf_get.duration(30), "30 s")
        self.assertEqual(hf_get.duration(600), "10 min")
        self.assertEqual(hf_get.duration(4800), "1 h 20 min")
        line = hf_get.progress_line(142e9, 365.7e9, 45e6)
        self.assertEqual(line, "142.0 of 365.7 GB (38%) - 45 MB/s - about 1 h 23 min left")
        self.assertEqual(hf_get.progress_line(1e9, None, None), "1.0 GB so far")

    def test_rate(self):
        r = hf_get.Rate(window=10)
        self.assertIsNone(r.add(0, 0))
        self.assertEqual(r.add(4, 400), 100)
        self.assertEqual(r.add(20, 2000), 2000 / 20)        # the window keeps two points at least
        self.assertAlmostEqual(r.add(30, 3000), (3000 - 2000) / 10)


if __name__ == "__main__":
    unittest.main()
