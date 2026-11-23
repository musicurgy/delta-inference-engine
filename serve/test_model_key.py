"""R-6: state that outlives a request and depends on the model's weights is keyed by the model ("model_key" in the run
config, tools/v41/serve/ds41_local.py; else a sampled hash of the model file) and dropped when the model changes:
the last start's duration (serve/loadstate.py), the reasoning memory (serve/reasoning_memory.py) and the learned
expert profile (server.learned_profile_args).  No GPU, no model files.

    python -m unittest serve.test_model_key -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import loadstate  # noqa: E402
from serve.reasoning_memory import ReasoningMemory  # noqa: E402
from serve.server import engine_args, model_key_of  # noqa: E402


class LastStart(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (loadstate.last_start_path, loadstate.model_key)
        loadstate.last_start_path = os.path.join(self.tmp.name, "cfg.last-start.json")

    def tearDown(self):
        loadstate.last_start_path, loadstate.model_key = self.saved
        self.tmp.cleanup()

    def write(self, rec):
        Path(loadstate.last_start_path).write_text(json.dumps(rec), encoding="utf-8")

    def test_another_models_record_is_not_this_models_estimate(self):
        loadstate.model_key = "a" * 16
        self.write({"seconds": 58, "model": "b" * 16})
        self.assertIsNone(loadstate._read_last())
        self.write({"seconds": 58, "model": "a" * 16})
        self.assertEqual(loadstate._read_last(), 58.0)
        self.write({"seconds": 58})                      # a record from before the key: kept, as before
        self.assertEqual(loadstate._read_last(), 58.0)

    def test_unkeyed_config_reads_every_record(self):
        loadstate.model_key = None
        self.write({"seconds": 40, "model": "b" * 16})
        self.assertEqual(loadstate._read_last(), 40.0)

    def test_done_records_the_key(self):
        loadstate.model_key = "c" * 16
        loadstate.begin()
        loadstate.done()
        rec = json.loads(Path(loadstate.last_start_path).read_text(encoding="utf-8"))
        self.assertEqual(rec["model"], "c" * 16)
        loadstate.model_key = None
        loadstate.begin()
        loadstate.done()
        self.assertNotIn("model", json.loads(Path(loadstate.last_start_path).read_text(encoding="utf-8")))


class Memory(unittest.TestCase):
    def test_another_model_starts_empty(self):
        m = ReasoningMemory()
        self.assertFalse(m.bind("k1"))                   # first key: nothing to drop
        m.remember(b"ctx", "answer", (), "thinking")
        self.assertEqual(len(m), 1)
        self.assertFalse(m.bind("k1"))                   # same model: kept
        self.assertEqual(len(m), 1)
        self.assertTrue(m.bind("k2"))
        self.assertEqual(len(m), 0)
        self.assertEqual(m.model, "k2")

    def test_unkeyed_is_unchanged(self):
        m = ReasoningMemory()
        m.remember(b"ctx", "answer", (), "thinking")
        self.assertFalse(m.bind(None))                   # a config without "model_key": as before
        self.assertEqual(len(m), 1)


class LearnedProfile(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        sys.path.insert(0, str(ROOT / "tools"))
        import make_profile
        self.mp = make_profile
        for name in ("base.bin", "learned.bin"):
            make_profile.write_profile(self.d / name, [(i % make_profile.N_LAYER, 0) for i in range(4)], n_expert=512)

    def cfg(self, key=None):
        c = {"args": ["--expert-profile", "base.bin"], "cwd": str(self.d), "expert_profile_save": "learned.bin"}
        if key:
            c["model_key"] = key
        return c

    def run_args(self, cfg):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            a = engine_args(cfg)
        return a, out.getvalue()

    def test_a_profile_without_a_record_keeps_the_header_rule(self):
        self.assertEqual(self.run_args(self.cfg())[0][:2], ["--expert-profile", "learned.bin"])
        self.assertFalse((self.d / "learned.bin.model-key").exists())   # no key: nothing written

    def test_the_same_model_starts_from_it(self):
        a, _ = self.run_args(self.cfg("k1"))
        self.assertEqual(a[:2], ["--expert-profile", "learned.bin"])
        self.assertEqual((self.d / "learned.bin.model-key").read_text(encoding="utf-8").strip(), "k1")
        self.assertEqual(self.run_args(self.cfg("k1"))[0][:2], ["--expert-profile", "learned.bin"])

    def test_another_model_of_the_same_shape_does_not(self):
        self.run_args(self.cfg("k1"))
        a, out = self.run_args(self.cfg("k2"))
        self.assertEqual(a[:2], ["--expert-profile", "base.bin"])
        self.assertIn("saved for another model", out)
        self.assertFalse((self.d / "learned.bin").exists())
        self.assertEqual((self.d / "learned.bin.model-key").read_text(encoding="utf-8").strip(), "k2")
        self.assertIn("--expert-profile-save", a)

    def test_the_load_key_is_not_the_profiles_key(self):
        """the drafter is in "load_key" only: a start with another drafter keeps the learned profile"""
        c = self.cfg("k1")
        c["load_key"] = "with-drafter"
        self.run_args(c)
        c = self.cfg("k1")
        c["load_key"] = "without-drafter"
        a, out = self.run_args(c)
        self.assertEqual(a[:2], ["--expert-profile", "learned.bin"])
        self.assertNotIn("saved for another model", out)
        self.assertTrue((self.d / "learned.bin").exists())

    def test_the_key_of_a_config_without_one(self):
        m = self.d / "model.gguf"
        m.write_bytes(b"x" * (3 << 20))
        k = model_key_of({"cwd": str(self.d)}, ["--native", "model.gguf"])
        self.assertRegex(k, r"^[0-9a-f]{16}$")
        self.assertEqual(model_key_of({"model_key": "given"}, ["--native", "model.gguf"]), "given")
        m.write_bytes(b"x" * (3 << 20) + b"y")
        self.assertNotEqual(model_key_of({"cwd": str(self.d)}, ["--native", "model.gguf"]), k)
        self.assertIsNone(model_key_of({"cwd": str(self.d)}, ["--native", "missing.gguf"]))
        self.assertIsNone(model_key_of({}, ["--ctx", "8"]))


if __name__ == "__main__":
    unittest.main()
