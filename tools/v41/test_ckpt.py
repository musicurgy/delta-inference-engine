"""R-2 / R-6: the checkpoint table (data/ds41/checkpoints.json), tools/v41/ckpt.py, and the launcher's checkpoint check
and model key (ds41_local.checkpoint_report / model_state).  No GPU, no download, no model file needed (the 1017
fingerprint is re-computed from the expected-hash file named by DS41_EXPECTED_1017 when it is set).

    python tools/v41/test_ckpt.py
"""
from __future__ import annotations

import io
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "serve"))
import ckpt  # noqa: E402

EXPECTED_1017 = Path(os.environ.get("DS41_EXPECTED_1017", "") or "/nonexistent/expected.json")


def gguf(path: Path, kv: dict) -> None:
    """A GGUF with string metadata only (no tensors): enough for ckpt.which()'s metadata path."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, v in kv.items():
        out += s(k) + struct.pack("<I", 8) + s(v)
    path.write_bytes(out)


def index(path: Path, layout: dict) -> None:
    """A safetensors index: {tensor name: shard}."""
    path.write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": layout}), encoding="utf-8")


class Table(unittest.TestCase):
    def test_rows_have_what_the_entry_points_read(self):
        t = ckpt.load()
        self.assertEqual(t["default"], "1017")
        self.assertEqual(set(t["order"]), set(t["checkpoints"]))
        for cid, r in t["checkpoints"].items():
            with self.subTest(cid):
                self.assertRegex(r["fingerprint"], r"^[0-9a-f]{16}$")
                ds = r["deepseek"]
                for k in ("repo", "shards", "index", "small_dir", "mtp_prefix", "mtp_shards_hint"):
                    self.assertIn(k, ds)
                for k in ("drafter", "drafter_uniform", "drafter_source", "keep", "vision", "tokenizer_dir"):
                    self.assertIn(k, r["derived"])
                ckpt.bat_vars(cid)   # every value can go into a batch file
        fps = [r["fingerprint"] for r in t["checkpoints"].values()]
        self.assertEqual(len(fps), len(set(fps)))

    def test_1017_is_todays_files(self):
        """the default row names exactly the files 0.95 hard-coded (output unchanged for the 1017 model)"""
        v = ckpt.bat_vars("1017")
        self.assertEqual(v["CK_DS"], "deepseek-ai/DeepSeek-V4.1-Flash-1017")
        self.assertEqual(v["CK_Q2"], "DeepSeek-V4.1-Flash-1017-Q2.gguf")
        self.assertEqual(v["CK_DRAFT_IMX"], "V41-1017-MTP-iq2xxs-imx.gguf")
        self.assertEqual(v["CK_KEEP"], "tools/v41/serve/dspark_expert_counts_1017.txt")
        self.assertEqual(v["CK_MTP_HINT"].split(), [f"model-000{n}-of-00048.safetensors" for n in (44, 45, 46)])
        self.assertEqual(ckpt.derived("rank", "1017"), "data/ds41/rank-1017-share.bin")
        self.assertEqual(v["CK_IMATRIX"], "tools\\v41\\serve\\drafter-imatrix.bin")
        for role in ("rank", "keep", "drafter_imatrix"):
            self.assertTrue((ROOT / ckpt.derived(role, "1017")).exists(), role)

    def test_the_batch_files_defaults_are_the_default_row(self):
        """Start DeepSeek.bat / setup-ds41-models.bat / first-start.bat keep 1017's names for when the table cannot be
        read: they must be the table's default row, value for value"""
        v = ckpt.bat_vars()
        for bat in ("Start DeepSeek.bat", "setup-ds41-models.bat", "tools/v41/serve/first-start.bat"):
            text = (ROOT / bat).read_text(encoding="utf-8")
            sets = re.findall(r'^set "(CK_[A-Z_0-9]+)=([^"]*)"\s*$', text, re.M)
            self.assertTrue(sets, bat)
            self.assertIn("tools\\v41\\ckpt.py bat", text, bat)
            for k, x in sets:
                with self.subTest(bat=bat, var=k):
                    self.assertEqual(x, v[k])

    def test_no_1017_name_is_left_in_the_launchers_commands(self):
        for bat in ("Start DeepSeek.bat", "setup-ds41-models.bat", "tools/v41/serve/first-start.bat"):
            for ln in (ROOT / bat).read_text(encoding="utf-8").splitlines():
                if ln.startswith("rem") or ln.startswith('set "CK_'):
                    continue
                with self.subTest(bat=bat, line=ln):
                    self.assertNotRegex(ln, r"1017|0004[4-6]")

    def test_a_checkpoint_without_a_drafter_imatrix_gets_none(self):
        """R-2: 1017's drafter-imatrix.bin is 1017's; a row without its own (v41, a new checkpoint on day 0) exports
        an empty CK_IMATRIX, so setup requantises its drafter uniformly"""
        self.assertIsNone(ckpt.derived("drafter_imatrix", "v41"))
        self.assertEqual(ckpt.bat_vars("v41")["CK_IMATRIX"], "")
        imx = str(ROOT / "tools/v41/serve/drafter-imatrix.bin")
        self.assertEqual(ckpt.imatrix_ok(imx, "1017"), (True, ""))
        ok, why = ckpt.imatrix_ok(imx, "v41")
        self.assertFalse(ok)
        self.assertIn("checkpoint 1017", why)
        # beside an engine build (no stamp there): recognised by its name, still 1017's
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "drafter-imatrix.bin"
            p.write_bytes(b"x")
            self.assertFalse(ckpt.imatrix_ok(str(p), "v41")[0])
            self.assertTrue(ckpt.imatrix_ok(str(p), "1017")[0])
            mine = Path(d) / "drafter-imatrix-1201.bin"   # a new row's own matrix, no stamp: its row named it
            mine.write_bytes(b"x")
            self.assertTrue(ckpt.imatrix_ok(str(mine), "v41")[0])
        self.assertEqual(ckpt.main(["imatrix-ok", imx, "--id", "v41"]), 1)
        self.assertEqual(ckpt.main(["imatrix-ok", imx, "--id", "1017"]), 0)

    def test_setup_picks_the_matrix_by_the_row(self):
        text = (ROOT / "setup-ds41-models.bat").read_text(encoding="utf-8")
        cmds = [ln for ln in text.splitlines() if not ln.startswith("rem") and "IMX=" in ln and "set" in ln]
        self.assertTrue(cmds)
        for ln in cmds:
            with self.subTest(ln):
                self.assertNotIn("drafter-imatrix.bin", ln)   # only through CK_IMATRIX
        self.assertIn("ckpt.py imatrix-ok", text)

    def test_active_row_and_a_bad_name(self):
        self.assertEqual(ckpt.active_id(env={}), "1017")
        self.assertEqual(ckpt.active_id(env={"DS41_CHECKPOINT": "v41"}), "v41")
        with self.assertRaises(ckpt.TableError):
            ckpt.active_id(env={"DS41_CHECKPOINT": "1125"})

    def test_a_value_cmd_cannot_carry_is_refused(self):
        t = json.loads(json.dumps(ckpt.load()))
        t["checkpoints"]["1017"]["derived"]["drafter"] = "a&b.gguf"
        with self.assertRaises(ckpt.TableError):
            ckpt.bat_vars("1017", t)

    def test_a_broken_table_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            p.write_text('{"default": "x", "checkpoints": {}}', encoding="utf-8")
            with self.assertRaises(ckpt.TableError):
                ckpt.load(str(p))


class Index(unittest.TestCase):
    def test_mtp_shards_come_from_the_index(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.safetensors.index.json"
            index(p, {"embed.weight": "model-00002-of-00048.safetensors",
                      "mtp.0.attn.wq.weight": "model-00044-of-00048.safetensors",
                      "mtp.1.attn.wq.weight": "model-00045-of-00048.safetensors",
                      "mtp.2.ffn.experts.0.w1.weight": "model-00046-of-00048.safetensors",
                      "layers.14.engram.embed.weight": "model-00048-of-00048.safetensors"})
            self.assertEqual(ckpt.mtp_shards(str(p)), [f"model-000{n}-of-00048.safetensors" for n in (44, 45, 46)])
            # a new checkpoint that moves (or adds) MTP tensors: the list follows the index, never 44-46 by name
            index(p, {"mtp.0.x": "model-00050-of-00052.safetensors", "mtp.2.y": "model-00052-of-00052.safetensors",
                      "mtp.1.z": "model-00050-of-00052.safetensors"})
            self.assertEqual(ckpt.mtp_shards(str(p)), ["model-00050-of-00052.safetensors", "model-00052-of-00052.safetensors"])
            index(p, {"layers.0.x": "model-00001-of-00002.safetensors"})
            with self.assertRaises(ValueError):
                ckpt.mtp_shards(str(p))

    def test_the_cli_prints_one_line_for_the_batch_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "i.json"
            index(p, {"mtp.0.a": "model-00045-of-00048.safetensors", "mtp.0.b": "model-00044-of-00048.safetensors"})
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(ckpt.main(["mtp-shards", str(p)]), 0)
            self.assertEqual(out.getvalue(), "model-00044-of-00048.safetensors model-00045-of-00048.safetensors\n")
            self.assertEqual(ckpt.main(["mtp-shards", str(Path(d) / "missing.json")]), 1)   # the hint stays


class Fingerprint(unittest.TestCase):
    def test_order_and_extra_files_do_not_matter(self):
        a = {"model-00002-of-00002.safetensors": "BB", "model-00001-of-00002.safetensors": "aa", "README.md": "x"}
        b = {"model-00001-of-00002.safetensors": "aa", "model-00002-of-00002.safetensors": "bb"}
        self.assertEqual(ckpt.fingerprint_of_hashes(a), ckpt.fingerprint_of_hashes(b))
        b["model-00002-of-00002.safetensors"] = "cc"
        self.assertNotEqual(ckpt.fingerprint_of_hashes(a), ckpt.fingerprint_of_hashes(b))

    @unittest.skipUnless(EXPECTED_1017.exists(), "no local 1017 hash list")
    def test_1017_matches_its_published_shard_hashes(self):
        self.assertEqual(ckpt.fingerprint_of_expected(str(EXPECTED_1017)), ckpt.row("1017")["fingerprint"])


class Which(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, name: str) -> str:
        p = self.d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        return str(p)

    def test_by_name(self):
        for name, cid in (("DeepSeek-V4.1-Flash-1017-Q2.gguf", "1017"), ("ds4-q2-slim.gguf", "1017"),
                          ("DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00003-of-00006.gguf", "1017"),
                          ("V41-1017-MTP-iq2xxs-imx.gguf", "1017"), ("engram-hot-2g.bin", "1017"),
                          ("V41-GSQ.gguf", "v41"), ("V41-MTP-Vision.Q2_K.gguf", "v41"),
                          ("dspark_expert_counts.txt", "v41"), ("dspark_expert_counts_1017.txt", "1017"),
                          ("my-model.gguf", None)):
            with self.subTest(name):
                self.assertEqual(ckpt.which(self.touch(name))[0], cid)

    def test_a_new_dated_checkpoint_is_not_taken_for_an_existing_one(self):
        """day 0: a row appended after the others (here '1201') - its files, stamped or not, are its own or unknown,
        never v41's or 1017's; and each row's own file names are claimed by no other row"""
        t = json.loads(json.dumps(ckpt.load()))
        t["checkpoints"]["1201"] = {**t["checkpoints"]["v41"], "fingerprint": "fedcba9876543210",
                                    "match": ["(?<![0-9])1201(?![0-9])"], "meta_match": ["DeepSeek-V4\\.1-Flash-1201"]}
        t["order"] = t["order"] + ["1201"]
        for name in ("DeepSeek-V4.1-Flash-1201-Q2.gguf", "DeepSeek-V4.1-Flash-1201-UD-IQ1_S-00001-of-00004.gguf",
                     "DeepSeek-V4.1-Flash-1201-Vision.gguf", "V41-1201-MTP-iq2xxs.gguf", "rank-1201-share.bin"):
            with self.subTest(name):
                self.assertEqual(ckpt.which(self.touch(name), t)[0], "1201")
                del t["checkpoints"]["1201"]["match"]
                self.assertIsNone(ckpt.which(self.touch(name), t, meta=False)[0])   # unknown, not v41 / 1017
                t["checkpoints"]["1201"]["match"] = ["(?<![0-9])1201(?![0-9])"]
        # "1017" is a word of its own, not any run of digits
        self.assertIsNone(ckpt.which(self.touch("model-210170.gguf"), t)[0])
        # the base's names (a letter after "DeepSeek-V4.1-Flash-") are still v41's
        for name in ("DeepSeek-V4.1-Flash-UD-IQ1_S-00001-of-00004.gguf", "DeepSeek-V4.1-Flash-Vision.gguf"):
            self.assertEqual(ckpt.which(self.touch(name), t)[0], "v41")
        # every row's own file names: its own row or unknown, never another's
        for cid, r in ckpt.load()["checkpoints"].items():
            names = [v for k, v in (r.get("derived") or {}).items() if isinstance(v, str) and not k.endswith("repo")]
            q2 = (r.get("models") or {}).get("ds4_q2") or {}
            names += [q2[k] for k in ("file", "slim") if q2.get(k)]
            for n in names:
                with self.subTest(row=cid, name=n):
                    self.assertIn(ckpt.which(self.touch(os.path.basename(n)), meta=False)[0], (cid, None))

    def test_a_stamp_wins_over_the_name(self):
        p = self.touch("V41-1017-MTP-iq2xxs-imx.gguf")
        ckpt.write_stamp(p, "drafter", "v41")
        self.assertEqual(ckpt.which(p)[:3], ("v41", ckpt.row("v41")["fingerprint"], "stamp"))
        # a split model: the first part's stamp stands for every part
        first = self.touch("x-00001-of-00003.gguf")
        ckpt.write_stamp(first, "model", "1017")
        self.assertEqual(ckpt.which(str(self.d / "x-00002-of-00003.gguf"))[0], "1017")

    def test_a_stamp_of_an_unknown_checkpoint_by_its_fingerprint(self):
        p = self.touch("drafter.gguf")
        Path(p + ckpt.STAMP_SUFFIX).write_text(json.dumps({"checkpoint": "renamed", "fingerprint":
                                                           ckpt.row("1017")["fingerprint"]}), encoding="utf-8")
        self.assertEqual(ckpt.which(p)[0], "1017")

    def test_by_gguf_metadata(self):
        p = self.d / "custom.gguf"
        gguf(p, {"general.name": "DeepSeek-V4.1-Flash-1017", "general.architecture": "deepseek41"})
        self.assertEqual(ckpt.which(str(p))[::2], ("1017", "metadata"))
        gguf(p, {"general.base_model.0.repo_url": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"})
        self.assertEqual(ckpt.which(str(p))[0], "v41")
        gguf(p, {"general.name": "Qwen"})
        self.assertEqual(ckpt.which(str(p))[0], None)

    def test_a_folder_stamp(self):
        d = self.d / "engram-new"
        d.mkdir()
        ckpt.write_stamp(str(d), "engram", "v41")
        self.assertTrue((d / ckpt.DIR_STAMP).exists())
        self.assertEqual(ckpt.which(str(d))[0], "v41")


class Check(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def f(self, name: str) -> str:
        p = self.d / name
        p.write_bytes(b"x")
        return str(p)

    def test_matching_files_are_quiet(self):
        mid, w = ckpt.check(self.f("DeepSeek-V4.1-Flash-1017-Q2.gguf"),
                            {"drafter": self.f("V41-1017-MTP-iq2xxs-imx.gguf"),
                             "keep": str(ROOT / "tools/v41/serve/dspark_expert_counts_1017.txt"),
                             "rank": str(ROOT / "data/ds41/rank-1017-share.bin"), "unknown": self.f("other.bin"),
                             "missing": str(self.d / "nothing.gguf")})
        self.assertEqual((mid, w), ("1017", []))

    def test_another_checkpoints_file_warns(self):
        mid, w = ckpt.check(self.f("DeepSeek-V4.1-Flash-1017-Q2.gguf"),
                            {"keep": str(ROOT / "tools/v41/serve/dspark_expert_counts.txt"),
                             "drafter": self.f("V41-MTP-iq2xxs-imx.gguf")})
        self.assertEqual(mid, "1017")
        self.assertEqual(len(w), 2)
        self.assertIn("checkpoint v41", w[0])

    def test_a_new_checkpoints_model_with_1017_derived_files(self):
        """the day-0 mistake R-2 is about: the new model, 1017's drafter / rank / Engram hot set"""
        t = json.loads(json.dumps(ckpt.load()))
        t["checkpoints"]["1125"] = {**t["checkpoints"]["v41"], "fingerprint": "0123456789abcdef",
                                    "match": ["1125"], "meta_match": []}
        t["order"] = ["1125"] + t["order"]
        mid, w = ckpt.check(self.f("DeepSeek-V4.1-Flash-1125-UD-IQ2_XXS-00001-of-00006.gguf"),
                            {"rank": str(ROOT / "data/ds41/rank-1017-share.bin"), "engram_hot": self.f("engram-hot-2g.bin"),
                             "drafter": self.f("V41-1017-MTP-iq2xxs-imx.gguf")}, t)
        self.assertEqual(mid, "1125")
        self.assertEqual(sorted(x.split()[0] for x in w), ["drafter", "engram_hot", "rank"])

    def test_an_unknown_model_is_not_checked(self):
        self.assertEqual(ckpt.check(self.f("mine.gguf"), {"drafter": self.f("V41-MTP.gguf")}), (None, []))

    def test_the_tokenizer_is_compared_by_its_own_hash(self):
        tok = self.d / "tokenizer"
        tok.mkdir()
        ckpt.write_stamp(str(tok), "tokenizer", "v41")   # V4.1 and 1017 share tokenizer.json byte for byte
        self.assertEqual(ckpt.check(self.f("DeepSeek-V4.1-Flash-1017-Q2.gguf"), {"tokenizer": str(tok)})[1], [])
        st = json.loads((tok / ckpt.DIR_STAMP).read_text(encoding="utf-8"))
        st["tokenizer_sha256"] = "0" * 64
        (tok / ckpt.DIR_STAMP).write_text(json.dumps(st), encoding="utf-8")
        self.assertEqual(len(ckpt.check(self.f("DeepSeek-V4.1-Flash-1017-Q2.gguf"), {"tokenizer": str(tok)})[1]), 1)


class Converters(unittest.TestCase):
    def test_the_stamp_goes_beside_the_final_name(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "V41-NEW-MTP.gguf.part")
            Path(out).write_bytes(b"x")
            with mock.patch.dict(os.environ, {"V41_SRC_URL": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017"}):
                os.environ.pop("DS41_CHECKPOINT", None)
                sp = ckpt.stamp_output(out, "drafter_source")
            self.assertEqual(sp, os.path.join(d, "V41-NEW-MTP.gguf") + ckpt.STAMP_SUFFIX)
            self.assertEqual(json.loads(Path(sp).read_text(encoding="utf-8"))["checkpoint"], "1017")

    def test_the_base_repo_is_not_taken_for_1017(self):
        env = {"V41_SRC_URL": "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash"}
        self.assertEqual(ckpt.source_checkpoint(env=env), "v41")
        self.assertEqual(ckpt.source_checkpoint(env={"DS41_CHECKPOINT": "1017", **env}), "1017")
        self.assertIsNone(ckpt.source_checkpoint(env={"V41_SRC_URL": "https://example.invalid/x"}))

    def test_unknown_source_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"V41_SRC_URL": "x"}):
            os.environ.pop("DS41_CHECKPOINT", None)
            self.assertIsNone(ckpt.stamp_output(os.path.join(d, "a.gguf"), "vision"))
            self.assertEqual(os.listdir(d), [])


class ModelKey(unittest.TestCase):
    def test_changes_with_the_weights_quant_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "m-00001-of-00002.gguf"
            b = Path(d) / "m-00002-of-00002.gguf"
            a.write_bytes(b"a" * 4096)
            b.write_bytes(b"b" * (3 << 20))
            k = ckpt.model_key([str(a)], "fp")
            self.assertEqual(k, ckpt.model_key([str(a)], "fp"))
            self.assertNotEqual(k, ckpt.model_key([str(a)], "other"))
            b.write_bytes(b"b" * ((3 << 20) - 1) + b"c")   # the second part's last MiB: every part counts
            self.assertNotEqual(k, ckpt.model_key([str(a)], "fp"))
            self.assertEqual(ckpt.model_key([str(a), None, ""], "fp"), ckpt.model_key([str(a)], "fp"))


class Launcher(unittest.TestCase):
    """ds41_local: the checkpoint line, the warnings, the model key and the state cleared when it changes"""

    def setUp(self):
        import ds41_local
        self.L = ds41_local
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def cfg(self, model: str, env: dict | None = None) -> dict:
        m = self.d / model
        m.write_bytes(b"m" * 1024)
        return {"args": ["-m", str(m).replace("\\", "/"), "--rank", str(ROOT / "data/ds41/rank-1017-share.bin")],
                "cwd": str(ROOT), "env": env or {}, "tokenizer": str(self.d / "tokenizer")}

    def run_report(self, cfg, env):
        out = io.StringIO()
        with mock.patch.dict(os.environ, env, clear=False), redirect_stdout(out):
            for k in ("STRATA_SERVE_DSPARK", "STRATA_DSPARK_KEEP", "STRATA_ENGRAM_HOT"):
                if k not in env:
                    os.environ.pop(k, None)
            self.L.checkpoint_report(cfg)
        return out.getvalue()

    def test_1017_with_its_own_files(self):
        drafter = self.d / "V41-1017-MTP-iq2xxs-imx.gguf"
        drafter.write_bytes(b"d")
        cfg = self.cfg("DeepSeek-V4.1-Flash-1017-Q2.gguf")
        text = self.run_report(cfg, {"STRATA_SERVE_DSPARK": str(drafter),
                                     "STRATA_DSPARK_KEEP": "tools/v41/serve/dspark_expert_counts_1017.txt:112,80,48"})
        self.assertIn("checkpoint 1017 (", text)
        self.assertNotIn("WARNING", text)
        self.assertEqual(cfg["checkpoint"]["id"], "1017")
        self.assertRegex(cfg["model_key"], r"^[0-9a-f]{16}$")
        self.assertRegex(cfg["load_key"], r"^[0-9a-f]{16}$")

    def test_the_drafter_changes_the_load_key_only(self):
        """the learned expert profile and the reasoning memory are keyed by the main model alone: a drafter made
        again (uniform -> imatrix), or a start without one, must not change "model_key" (the server would delete the
        profile)"""
        cfg = self.cfg("DeepSeek-V4.1-Flash-1017-Q2.gguf")
        self.run_report(cfg, {})
        without = dict(cfg)
        uni = self.d / "V41-1017-MTP-iq2xxs.gguf"
        uni.write_bytes(b"u" * 2048)
        imx = self.d / "V41-1017-MTP-iq2xxs-imx.gguf"
        imx.write_bytes(b"i" * 2048)
        keys = []
        for d in (uni, imx):
            c = self.cfg("DeepSeek-V4.1-Flash-1017-Q2.gguf")
            self.run_report(c, {"STRATA_SERVE_DSPARK": str(d)})
            keys.append(c)
        self.assertEqual({without["model_key"], keys[0]["model_key"], keys[1]["model_key"]}, {without["model_key"]})
        self.assertEqual(len({without["load_key"], keys[0]["load_key"], keys[1]["load_key"]}), 3)
        # another main model: another model key
        other = self.cfg("DeepSeek-V4.1-Flash-1017-UD-IQ1_S-00001-of-00004.gguf")
        self.run_report(other, {})
        self.assertNotEqual(other["model_key"], without["model_key"])

    def test_a_windows_path_in_the_keep_setting(self):
        files = self.L.derived_in_use({"args": [], "env": {"STRATA_SERVE_DSPARK": "D:/m/d.gguf",
                                                           "STRATA_DSPARK_KEEP": "C:/x/counts.txt:96,64,32"}})
        self.assertEqual(files["keep"], "C:/x/counts.txt")
        files = self.L.derived_in_use({"args": [], "env": {"STRATA_SERVE_DSPARK": "D:/m/d.gguf",
                                                           "STRATA_DSPARK_KEEP": "C:/x/counts.txt"}})
        self.assertEqual(files["keep"], "C:/x/counts.txt")

    def test_a_mismatch_is_a_warning_not_a_refusal(self):
        cfg = self.cfg("DeepSeek-V4.1-Flash-1017-Q2.gguf")
        text = self.run_report(cfg, {"STRATA_SERVE_DSPARK": str(ROOT / "data/ds41/rank-1017-share.bin"),
                                     "STRATA_DSPARK_KEEP": "tools/v41/serve/dspark_expert_counts.txt:64"})
        self.assertIn("WARNING - keep dspark_expert_counts.txt was built from checkpoint v41", text)

    def test_the_model_key_clears_the_last_start_on_a_change(self):
        out = self.d / "local" / "ds41-windows.json"
        out.parent.mkdir()
        last = out.parent / "ds41-windows.last-start.json"
        last.write_text('{"seconds": 58}', encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.L.model_state({"model_key": "a" * 16, "args": ["-m", "x.gguf"]}, str(out))   # first: recorded only
            self.assertTrue(last.exists())
            self.L.model_state({"model_key": "a" * 16, "args": ["-m", "x.gguf"]}, str(out))
            self.assertTrue(last.exists())
            self.L.model_state({"model_key": "b" * 16, "args": ["-m", "y.gguf"]}, str(out))
        self.assertFalse(last.exists())
        self.assertIn("the model changed", buf.getvalue())
        self.assertEqual(json.loads((out.parent / "model-key.json").read_text(encoding="utf-8"))["model_key"], "b" * 16)
        # same model, another drafter: the load-time estimate goes, the model key stays
        last.write_text('{"seconds": 58}', encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.L.model_state({"model_key": "b" * 16, "load_key": "c" * 16, "args": ["-m", "y.gguf"]}, str(out))
        self.assertFalse(last.exists())
        self.assertIn("the drafter / EXL3 files changed", buf.getvalue())
        rec = json.loads((out.parent / "model-key.json").read_text(encoding="utf-8"))
        self.assertEqual((rec["model_key"], rec["load_key"]), ("b" * 16, "c" * 16))


if __name__ == "__main__":
    unittest.main()
