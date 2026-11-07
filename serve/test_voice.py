"""local (Indra's Mirror): the voice sidecar - config defaults, health, and the passthrough forward.

The start/stop idle loop is the same shape as the tested lazy vision; the spawn path needs a real
audio server, so it is exercised here only for the clean-failure case.
"""
import http.server
import json
import threading
import unittest

from serve.voice import VoiceServer


class _StubHandler(http.server.BaseHTTPRequestHandler):
    seen = None

    def do_GET(self):
        self._send(200, b"ok", "text/plain") if self.path == "/health" else self._send(404, b"{}", "application/json")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        _StubHandler.seen = (self.path, self.headers.get("Content-Type"), body)
        self._send(200, json.dumps({"text": "hello world", "bytes": len(body)}).encode(), "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class VoiceServerTest(unittest.TestCase):
    def setUp(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.shutdown)

    def test_config_defaults(self):
        v = VoiceServer("asr", {"cmd": "/bin/true"})
        self.assertEqual(v.port, 5110)
        self.assertEqual(v.reserve_mib, 1800)
        self.assertEqual(v.base_url, "http://127.0.0.1:5110")
        self.assertFalse(v.alive())          # not started
        v.stop()                             # safe when it was never started

    def test_tts_default_port(self):
        self.assertEqual(VoiceServer("tts", {"cmd": "/bin/true"}).port, 5210)

    def test_forward_passes_the_body_through_and_parses_the_reply(self):
        v = VoiceServer("asr", {"cmd": "/bin/true", "port": self.port})
        self.assertTrue(v._healthy())
        status, out, _ctype = v.forward("/v1/audio/transcriptions", b"RIFFxxxx", "audio/wav")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(out)["text"], "hello world")
        self.assertEqual(_StubHandler.seen, ("/v1/audio/transcriptions", "audio/wav", b"RIFFxxxx"))

    def test_ensure_fails_cleanly_when_the_command_is_missing(self):
        v = VoiceServer("asr", {"cmd": "/nonexistent/definitely-not-here", "port": self.port,
                                "start_timeout_s": 2})
        with self.assertRaises(RuntimeError):
            v.ensure()
        self.assertFalse(v.alive())

    def test_ensure_counts_the_start_as_use(self):
        # the idle loop stops a child once time.time() - last_used >= idle_s; before this, last_used stayed 0
        # until the caller touched it AFTER ensure(), so a tick during a slow start killed the first request
        import time
        v = VoiceServer("asr", {"cmd": "/bin/sleep", "args": ["30"], "port": self.port, "start_timeout_s": 5})
        self.addCleanup(v.stop)
        v.ensure()                               # the stub answers /health, so this returns at once
        self.assertTrue(v.alive())
        self.assertLess(time.time() - v.last_used, 5.0)

class VoiceServerOptions(unittest.TestCase):
    def test_env_and_log_reach_the_child(self):
        from unittest import mock
        v = VoiceServer("tts", {"cmd": "py", "args": ["speech.py", "--port", "{port}"], "port": 1,
                                "env": {"CUDA_VISIBLE_DEVICES": 1}, "show_log": True, "start_timeout_s": 0.01})
        with mock.patch("subprocess.Popen") as popen:
            popen.return_value.poll.return_value = None
            with self.assertRaises(RuntimeError):            # nothing answers /health on port 1
                v.ensure()
        args, kw = popen.call_args
        self.assertEqual(args[0], ["py", "speech.py", "--port", "1"])
        self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertIn("PATH", {k.upper() for k in kw["env"]})      # the rest of the environment is kept
        self.assertIsNone(kw["stderr"])                                # the child's lines in the server window
        quiet = VoiceServer("asr", {"cmd": "x"})
        self.assertEqual((quiet.env, quiet.show_log), ({}, False))


class StrataProxy(unittest.TestCase):
    """The chat page's microphone: JSON to /v1/audio/transcriptions through Strata to the speech child; /health says
    which voice servers there are."""

    def setUp(self):
        import sys
        from pathlib import Path
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        root = Path(__file__).resolve().parents[1]
        self.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        threading.Thread(target=self.stub.serve_forever, daemon=True).start()
        self.addCleanup(self.stub.server_close)
        self.addCleanup(self.stub.shutdown)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(root / "serve/chat_template.jinja"))
        asr = VoiceServer("asr", {"cmd": sys.executable, "port": self.stub.server_address[1]})
        asr.ensure = lambda: asr.base_url                             # the stub is already up
        self.svc.voice_servers = {"asr": asr}
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def call(self, method, path, body=None, headers=None):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=30)
        try:
            c.request(method, path, body=body, headers=headers or {})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        finally:
            c.close()

    def test_health_names_the_voice_servers(self):
        self.assertEqual(self.call("GET", "/health")[1]["voice"], {"asr": True, "tts": False})

    def test_json_from_the_own_page_is_forwarded(self):
        port = self.httpd.server_address[1]
        body = json.dumps({"file": "data:audio/wav;base64,UklGRg=="}).encode()
        status, out = self.call("POST", "/v1/audio/transcriptions", body,
                                {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"})
        self.assertEqual((status, out["text"]), (200, "hello world"))
        self.assertEqual(_StubHandler.seen[:2], ("/v1/audio/transcriptions", "application/json"))

    def test_no_tts_configured(self):
        status, out = self.call("POST", "/v1/audio/speech", b"{}", {"Content-Type": "application/json"})
        self.assertEqual(status, 404)
        self.assertIn("no tts server", out["error"]["message"])


class LauncherVoiceSection(unittest.TestCase):
    """tools/v41/serve/ds41_local.py: voice-choice.txt / DS41_VOICE -> the server's "voice" section."""

    def setUp(self):
        import importlib
        import os
        import sys
        import tempfile
        from pathlib import Path
        here = str(Path(__file__).resolve().parents[1] / "tools" / "v41" / "serve")
        if here not in sys.path:
            sys.path.insert(0, here)
        self.L = importlib.import_module("ds41_local")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.m = self.tmp.name
        self.os = os

    def put(self, *parts):
        p = self.os.path.join(self.m, "speech", *parts)
        self.os.makedirs(self.os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()

    def section(self, mode, sherpa=True, gpus=(), cuda=False, engine_gpu=0):
        from unittest import mock
        import io
        import contextlib
        out = io.StringIO()
        real = self.L.importlib.util.find_spec
        def find_spec(name):
            return (object() if sherpa else None) if name == "sherpa_onnx" else real(name)
        with mock.patch.dict(self.os.environ, {"DS41_VOICE": mode}), \
                mock.patch.object(self.L.importlib.util, "find_spec", side_effect=find_spec), \
                mock.patch.object(self.L, "sherpa_cuda", return_value=cuda), \
                mock.patch("setup_panel.list_gpus", return_value=list(gpus)), contextlib.redirect_stdout(out):
            return self.L.voice_section(self.m, engine_gpu), out.getvalue()

    def test_off_by_default_and_not_set_up(self):
        self.assertEqual(self.section("off")[0], None)
        cfg, said = self.section("cpu")
        self.assertIsNone(cfg)
        self.assertIn("Enable Voice.bat", said)

    def test_cpu(self):
        self.put(self.L.VOICE_ASR, "encoder.int8.onnx")
        self.put(self.L.VOICE_TTS, "model.onnx")
        self.put(self.L.VOICE_VAD)
        cfg, _ = self.section("cpu")
        self.assertEqual(cfg["lazy_idle_s"], self.L.VOICE_IDLE_S)
        asr, tts = cfg["asr"], cfg["tts"]
        self.assertEqual((asr["port"], tts["port"]), (5110, 5210))
        self.assertTrue(asr["args"][1].endswith("serve/speech.py"))
        for c in (asr, tts):
            a = c["args"]
            self.assertEqual((a[a.index("--device") + 1], a[a.index("--cpus") + 1]), ("cpu", "e"))
            self.assertNotIn("env", c)
        self.assertIn("--vad", asr["args"])
        self.assertNotIn("--vad", tts["args"])
        self.assertIsNone(self.section("cpu", sherpa=False)[0])        # the runtime is missing

    def test_gpu(self):
        self.put(self.L.VOICE_ASR, "encoder.int8.onnx")
        self.put(self.L.VOICE_TTS, "model.onnx")
        two = [{"index": 0, "name": "RTX 3090", "mib": 24576}, {"index": 1, "name": "RTX 3060", "mib": 12288}]
        cfg, _ = self.section("gpu", gpus=two, cuda=True)
        self.assertEqual(cfg["tts"]["env"], {"CUDA_VISIBLE_DEVICES": "1"})      # read-aloud on the second card
        self.assertIn("cuda", cfg["tts"]["args"])
        self.assertIn("cpu", cfg["asr"]["args"])                               # dictation stays on the CPU
        self.assertNotIn("env", cfg["asr"])
        cfg, said = self.section("gpu", gpus=two[:1], cuda=True)      # one card: the CPU
        self.assertNotIn("env", cfg["tts"])
        self.assertIn("there is none", said)
        cfg, said = self.section("gpu:1", gpus=two, cuda=False)       # the CPU build of sherpa-onnx
        self.assertIn("CUDA build", said)
        self.assertIn("cpu", cfg["tts"]["args"])


if __name__ == "__main__":
    unittest.main()
