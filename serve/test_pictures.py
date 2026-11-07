"""0.94 pictures: on by default, where the encoder runs (CPU / the multi-GPU card / a chosen card), OpenAI's
`detail: low|high`, and the encoder's per-picture token cap.

    python -m unittest serve.test_pictures
"""
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from serve.frontend import ChatTemplate, image_details_of, images_of, openai_to_messages
from serve.responses import input_messages
import serve.server as server
from serve.server import ByteTokenizer, MockEngine, Service

ROOT = Path(__file__).resolve().parents[1]
V41_SERVE = ROOT / "tools" / "v41" / "serve"
V41_VISION = ROOT / "tools" / "v41" / "vision"
sys.path.insert(0, str(V41_SERVE))
import ds41_local as L  # noqa: E402  (no side effects at import)

PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


class Detail(unittest.TestCase):
    """OpenAI's image_url detail reaches the encoder as "low" / "high"; "auto" and anything else are the default."""

    def chat(self, detail):
        url = {"url": PNG} if detail is None else {"url": PNG, "detail": detail}
        msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": url}, {"type": "text", "text": "what is it"},
            {"type": "image_url", "image_url": PNG}]}]})
        return msgs

    def test_chat_completions(self):
        for detail, want in (("low", "low"), ("HIGH", "high"), ("auto", None), (None, None), ("tiny", None)):
            with self.subTest(detail=detail):
                msgs = self.chat(detail)
                self.assertEqual(images_of(msgs), [PNG, PNG])
                self.assertEqual(image_details_of(msgs), [want, None])

    def test_responses_input_image(self):
        msgs = input_messages({"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": PNG, "detail": "low"}, {"type": "input_text", "text": "and"},
            {"type": "input_image", "image_url": PNG}]}]})
        self.assertEqual(image_details_of(msgs), ["low", None])

    def test_text_only_messages_have_none(self):
        self.assertEqual(image_details_of([{"role": "user", "content": "hi"}]), [])


class FakeEncoder:
    """The encoder's stdin / stdout: records each line and answers OK with 7 rows."""

    def __init__(self):
        self.lines = []
        self.out = []
        outer = self

        class In(io.StringIO):
            def write(self, s):
                outer.lines.append(s)
                outer.out.append("OK 7 7 1 1\n")
                return len(s)

        self.stdin = In()
        self.stdout = SimpleNamespace(readline=lambda: outer.out.pop(0) if outer.out else "")

    def poll(self):
        return None


class EncoderCap(unittest.TestCase):
    """Vision.encode: "low" asks the encoder for a capped picture (ENCMAX), cached apart from the full one, with a key
    that differs from the first character (the engine gets key[:16] for its picture-aware cache, #2228)."""

    def vision(self, d, detail_tokens):
        v = server.Vision.__new__(server.Vision)
        v.dir, v.cache, v.lock = Path(d), {}, threading.Lock()
        v.proc, v.stopped, v.lazy_vision, v.on_start, v.last_used = FakeEncoder(), False, False, None, 0.0
        v.DETAIL_TOKENS = detail_tokens
        return v

    def test_low_is_capped_and_cached_apart(self):
        with tempfile.TemporaryDirectory() as d:
            v = self.vision(d, {"low": 256})
            full, _ = v.encode(PNG)
            low, n = v.encode(PNG, detail="low")
            again, _ = v.encode(PNG, detail="low")
            high, _ = v.encode(PNG, detail="high")
            self.assertEqual(n, 7)
            self.assertTrue(v.proc.lines[0].startswith("ENC ") and v.proc.lines[1].startswith("ENCMAX 256 "))
            self.assertEqual(len(v.proc.lines), 2)               # the second low and the high one come from the cache
            self.assertEqual((again, high), (low, full))
            self.assertNotEqual(Path(low).stem[:16], Path(full).stem[:16])

    def test_an_encoder_without_a_cap_ignores_detail(self):
        with tempfile.TemporaryDirectory() as d:
            v = self.vision(d, {})                                # strata-vision: no ENCMAX
            a, _ = v.encode(PNG, detail="low")
            b, _ = v.encode(PNG)
            self.assertEqual(a, b)
            self.assertEqual([x.split()[0] for x in v.proc.lines], ["ENC"])

    def test_encode_all_passes_each_detail(self):
        with tempfile.TemporaryDirectory() as d:
            v = self.vision(d, {"low": 256})
            v.encode_all([PNG, PNG], details=["low", None])
            self.assertEqual([x.split()[0] for x in v.proc.lines], ["ENCMAX", "ENC"])


class PrepareDetails(unittest.TestCase):
    """Service.prepare hands the details to encode_all only when a picture has one (older encoders keep their call)."""

    def run_prepare(self, detail):
        tok = ByteTokenizer()
        seen = []
        with tempfile.TemporaryDirectory() as d:
            rows = Path(d) / "0123456789abcdef0123456789abcdef.sve"
            rows.write_bytes(b"rows")

            def encode_all(sources, cancel=None, **kw):
                seen.append(kw)
                return [(rows, 3)] * len(sources)

            svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"),
                          vision=SimpleNamespace(dir=Path(d), encode_all=encode_all))
            url = {"url": PNG, **({"detail": detail} if detail else {})}
            msgs, _, _ = openai_to_messages({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": url}, {"type": "text", "text": "what"}]}]})
            svc.prepare(msgs, None, {})
            svc.embeddings.path.unlink(missing_ok=True)
        return seen

    def test_low(self):
        self.assertEqual(self.run_prepare("low"), [{"details": ["low"]}])

    def test_none(self):
        self.assertEqual(self.run_prepare(None), [{}])


GPUS = [{"index": 0, "name": "RTX 3090", "mib": 24576, "display": True},
        {"index": 1, "name": "RTX 3060", "mib": 12288, "display": False}]


class Placement(unittest.TestCase):
    """ds41_local.vision_place: never the engine's card by default; the multi-GPU card when the switch is on."""

    def place(self, mode, multi=False, gpus=GPUS, engine=0):
        with mock.patch.object(L, "page_settings", lambda: {"multi_gpu": multi}):
            return L.vision_place(mode, engine, gpus)[:2]

    def test_default_is_on(self):
        with mock.patch.dict(os.environ, {"DS41_VISION": ""}), mock.patch("builtins.open", side_effect=OSError):
            self.assertEqual(L.vision_choice(), "on")

    def test_on(self):
        self.assertEqual(self.place("on"), ("cpu", None))                       # one card, or multi-GPU off
        self.assertEqual(self.place("on", multi=True), ("gpu", 1))              # the helper's card
        self.assertEqual(self.place("auto", multi=True, gpus=GPUS[:1]), ("cpu", None))   # switch on, no second card
        self.assertEqual(self.place("on", multi=True, engine=1), ("gpu", 0))    # whichever card is not the engine's

    def test_overrides(self):
        self.assertEqual(self.place("cpu", multi=True), ("cpu", None))
        self.assertEqual(self.place("gpu:1"), ("gpu", 1))
        self.assertEqual(self.place("gpu1"), ("gpu", 1))
        self.assertEqual(self.place("gpu:0"), ("gpu", None))                    # the engine's own card
        self.assertEqual(self.place("main"), ("gpu", None))
        self.assertEqual(self.place("gpu"), ("gpu", None))                      # 0.92's spelling of main
        self.assertEqual(self.place("gpu:5"), ("cpu", None))                    # no such card
        self.assertEqual(self.place("off", multi=True), ("off", None))
        self.assertEqual(self.place("sideways"), (None, None))

    def test_plan_for_the_launchers(self):
        with mock.patch.object(L, "vision_choice", lambda: "gpu:1"), mock.patch.object(L, "torch_build", lambda: "cpu"), \
                mock.patch.object(L, "page_settings", dict), \
                mock.patch.dict(sys.modules, {"setup_panel": SimpleNamespace(list_gpus=lambda: GPUS)}):
            self.assertEqual(L.vision_plan(), "cuda cpu")
        with mock.patch.object(L, "vision_choice", lambda: "on"), mock.patch.object(L, "torch_build", lambda: None), \
                mock.patch.object(L, "page_settings", dict):
            self.assertEqual(L.vision_plan(), "cpu none")
        with mock.patch.object(L, "vision_choice", lambda: "off"), mock.patch.object(L, "torch_build", lambda: "cuda"):
            self.assertEqual(L.vision_plan(), "off cuda")

    def section(self, mode, build, multi=True):
        with mock.patch.object(L, "vision_choice", lambda: mode), mock.patch.object(L, "torch_build", lambda: build), \
                mock.patch.object(L, "page_settings", lambda: {"multi_gpu": multi}), \
                mock.patch.dict(sys.modules, {"setup_panel": SimpleNamespace(list_gpus=lambda: GPUS,
                                                                             helper_gpu=lambda g, e: GPUS[1])}), \
                mock.patch.object(L.os.path, "exists", lambda p: True), \
                mock.patch.dict(os.environ, {"DS41_GGUF": "model.gguf"}), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as said:
            return L.vision_section("models", 0), said.getvalue()

    def test_section_on_the_second_card(self):
        v, said = self.section("on", "cuda")
        self.assertEqual((v["gpu"], v["cuda_device"], v["dtype"]), (True, 1, "bf16"))
        self.assertNotIn("threads", v)

    def test_section_cpu_build_falls_back_to_the_cpu(self):
        v, said = self.section("on", "cpu")
        self.assertFalse(v["gpu"])
        self.assertNotIn("cuda_device", v)
        self.assertIn("2.0 GB", said)

    def test_cpu_encoder_lazy_under_110_gib(self):
        # 1107 review: the CPU encoder's 3.9 GB stays out of RAM on a 64 / 96 GB PC until the first picture
        for gib, want in ((63.8, True), (95.7, True), (127.7, False), (191.8, False)):
            with mock.patch.object(L, "total_ram_gib", lambda g=gib: g), mock.patch.dict(os.environ, {"DS41_VISION_LAZY": ""}):
                v, _ = self.section("cpu", "cpu")
            self.assertEqual(bool(v.get("lazy")), want, gib)
            if want:
                self.assertEqual((v["lazy_idle_s"], v["lazy_reserve_mib"]), (L.VISION_LAZY_IDLE_S, 0))
        with mock.patch.object(L, "total_ram_gib", lambda: 63.8), mock.patch.dict(os.environ, {"DS41_VISION_LAZY": "0"}):
            self.assertNotIn("lazy", self.section("cpu", "cpu")[0])
        with mock.patch.object(L, "total_ram_gib", lambda: 63.8):
            self.assertNotIn("lazy", self.section("on", "cuda")[0])          # a card encoder: never lazy here

    def test_section_main_card_has_no_device(self):
        v, _ = self.section("main", "cuda", multi=False)
        self.assertTrue(v["gpu"])
        self.assertNotIn("cuda_device", v)                  # the engine's environment: its own card

    def test_section_off_and_unknown(self):
        self.assertIsNone(self.section("off", "cuda")[0])
        v, said = self.section("sideways", "cuda")
        self.assertIsNone(v)
        self.assertIn("text only", said)

    def test_helper_gives_room_on_a_shared_card(self):
        plan = {"gpu": 1, "gib": 9.5, "args": ["--gib", "9.5", "--name", "ds41_remote"]}
        shared = L.share_card(plan, {"gpu": True, "cuda_device": 1})
        self.assertEqual((shared["gib"], shared["args"][1]), (7.5, "7.5"))
        self.assertEqual(plan["args"][1], "9.5")                                  # the original is not changed
        self.assertIs(L.share_card(plan, {"gpu": False}), plan)
        self.assertIs(L.share_card(plan, None), plan)
        self.assertIsNone(L.share_card(None, {"cuda_device": 1}))

    def test_encoder_env_names_the_card(self):
        env = server.vision_env({"vision": {"cuda_device": 1}}, {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertEqual((env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"]), ("1", "PCI_BUS_ID"))


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy"), "needs PyTorch")
class EncoderProtocol(unittest.TestCase):
    """ds41_vision.py's resident loop: ENCMAX <n> passes the cap to the encoder; a bad line is an ERR line."""

    def test_encmax(self):
        sys.path.insert(0, str(V41_VISION))
        import ds41_vision as DV
        import torch
        caps = []

        class Enc:
            n_embd = 4

            def encode(self, data, max_tokens=None):
                caps.append(max_tokens)
                return torch.zeros(6, 4), 2, 2, 4, 4

        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "a b.img"                        # a space in the path: split at the last space
            img.write_bytes(b"x")
            lines = f"ENC {img} {d}/a.sve\nENCMAX 256 {img} {d}/b.sve\nENCMAX x {img}\nQUIT\n"
            with mock.patch("sys.stdin", io.StringIO(lines)), mock.patch("sys.stdout", new_callable=io.StringIO) as out:
                DV.serve(Enc(), None)
            said = out.getvalue().splitlines()
            self.assertEqual(said[0], "READY 4")
            self.assertTrue(said[1].startswith("OK 6 ") and said[2].startswith("OK 6 "))
            self.assertTrue(said[3].startswith("ERR expected: ENCMAX"))
            self.assertEqual(caps, [None, 256])
            self.assertEqual(DV.read_sve(f"{d}/b.sve")[0].shape, (6, 4))


class Ds41Detail(unittest.TestCase):
    """serve_ds41's encoder offers "low" (256 tokens) under its own cap.  In a child process: importing serve_ds41
    patches the server module for the DS41 model."""

    def test_detail_tokens(self):
        code = ("import sys; sys.path.insert(0, sys.argv[1]); import serve_ds41 as D, json; "
                "v = D.Ds41Vision({'exe': 'x', 'mmproj': 'm', 'model': 'g'}, lazy=True); "
                "w = D.Ds41Vision({'exe': 'x', 'mmproj': 'm', 'model': 'g', 'max_tokens': 200}, lazy=True); "
                "print(json.dumps([v.DETAIL_TOKENS, w.DETAIL_TOKENS])); v.shutdown(); w.shutdown()")
        r = subprocess.run([sys.executable, "-c", code, str(V41_SERVE)], capture_output=True, text=True, cwd=ROOT,
                           timeout=120)
        if r.returncode and "No module named" in r.stderr:
            self.skipTest(r.stderr.strip().splitlines()[-1])
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(json.loads(r.stdout.strip().splitlines()[-1]), [{"low": 256}, {}])


if __name__ == "__main__":
    unittest.main()
