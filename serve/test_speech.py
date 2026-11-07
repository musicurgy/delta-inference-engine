"""serve/test_speech.py - the speech child (serve/speech.py): audio in and out, the OpenAI request and answer shapes, the
CPU placement, and its HTTP server with stand-in models (no sherpa-onnx needed).  With sherpa-onnx installed and
STRATA_SPEECH_MODELS naming a folder that holds the two models, one real round trip runs too (speak, then transcribe).

    python -m unittest serve.test_speech -v
"""
from __future__ import annotations

import base64
import http.client
import importlib.util
import io
import json
import math
import os
import struct
import sys
import threading
import unittest
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import speech as S  # noqa: E402


def sine_wav(seconds=0.5, rate=16000, freq=440.0, bits=16, channels=1) -> bytes:
    n = int(seconds * rate)
    x = [math.sin(2 * math.pi * freq * i / rate) * 0.5 for i in range(n)]
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(rate)
        if bits == 16:
            frames = b"".join(struct.pack("<h", int(v * 32767)) * channels for v in x)
        else:
            frames = b"".join(struct.pack("<B", int(v * 127 + 128)) * channels for v in x)
        w.writeframes(frames)
    return out.getvalue()


def float_wav(samples, rate=16000) -> bytes:
    data = np.asarray(samples, dtype="<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE" + b"fmt " +
            struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data)


def multipart(fields: dict, boundary="XyZ") -> tuple[bytes, str]:
    out = b""
    for k, v in fields.items():
        if isinstance(v, tuple):
            name, data = v
            out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\n"
                    f"Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
        else:
            out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    return out + f"--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


class Audio(unittest.TestCase):
    def test_read_wav_16_bit(self):
        x, rate = S.read_wav(sine_wav())
        self.assertEqual((rate, len(x), x.dtype), (16000, 8000, np.float32))
        self.assertAlmostEqual(float(np.max(np.abs(x))), 0.5, places=2)

    def test_read_wav_8_bit_stereo_and_float(self):
        x, rate = S.read_wav(sine_wav(rate=8000, bits=8, channels=2))
        self.assertEqual((rate, len(x)), (8000, 4000))
        self.assertAlmostEqual(float(np.max(np.abs(x))), 0.5, places=1)
        y, _ = S.read_wav(float_wav([0.0, 0.25, -0.75]))
        np.testing.assert_allclose(y, [0.0, 0.25, -0.75])

    def test_not_wav(self):
        with self.assertRaises(S.AudioError):
            S.read_wav(b"OggS....")
        with mock.patch.object(S.shutil, "which", return_value=None):
            with self.assertRaisesRegex(S.AudioError, "ffmpeg"):
                S.decode_audio(b"ID3\x03mp3 bytes")

    def test_wav_out_and_formats(self):
        x = np.zeros(2400, dtype=np.float32)
        wav = S.wav_bytes(x, 24000)
        self.assertEqual(S.read_wav(wav)[1], 24000)
        pcm, ctype = S.encode_audio(x, 24000, "pcm")
        self.assertEqual((len(pcm), ctype), (4800, "audio/pcm"))
        with mock.patch.object(S.shutil, "which", return_value=None):    # no ffmpeg: WAV instead of mp3
            data, ctype = S.encode_audio(x, 24000, "mp3")
        self.assertEqual((data[:4], ctype), (b"RIFF", "audio/wav"))


class Requests(unittest.TestCase):
    def test_openai_multipart(self):
        body, ctype = multipart({"file": ("a.wav", b"RIFFdata"), "model": "whisper-1", "response_format": "text"})
        audio, opts = S.transcription_request(body, ctype)
        self.assertEqual(audio, b"RIFFdata")
        self.assertEqual((opts["model"], opts["response_format"]), ("whisper-1", "text"))

    def test_web_chat_json(self):
        for f in ("data:audio/wav;base64," + base64.b64encode(b"RIFF1").decode(), base64.b64encode(b"RIFF1").decode()):
            audio, opts = S.transcription_request(json.dumps({"file": f, "language": "de"}).encode(),
                                                  "application/json")
            self.assertEqual((audio, opts), (b"RIFF1", {"language": "de"}))
        with self.assertRaises(ValueError):
            S.transcription_request(b"{}", "application/json")
        with self.assertRaises(ValueError):
            S.transcription_request(b"x", "text/plain")

    def test_formats(self):
        segs = [(0.0, 1.5, "Hello there."), (2.0, 3.25, "Second part.")]
        self.assertEqual(json.loads(S.format_transcript(segs, "json", 3.3, "")[0]),
                         {"text": "Hello there. Second part."})
        self.assertEqual(S.format_transcript(segs, "text", 3.3, "")[0], b"Hello there. Second part.")
        srt = S.format_transcript(segs, "srt", 3.3, "")[0].decode()
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHello there.\n", srt)
        vtt = S.format_transcript(segs, "vtt", 3.3, "")[0].decode()
        self.assertTrue(vtt.startswith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n"))
        v = json.loads(S.format_transcript(segs, "verbose_json", 3.3, "en")[0])
        self.assertEqual((v["language"], v["duration"], v["segments"][1]["start"]), ("en", 3.3, 2.0))


class CpuPlacement(unittest.TestCase):
    HYBRID = [(0, i, 1) for i in range(12)] + [(0, i, 0) for i in range(12, 20)]   # i5-13600K: 6 P (HT) + 8 E

    def test_efficiency_cores(self):
        self.assertEqual(S.efficiency_cores(self.HYBRID), list(range(12, 20)))
        self.assertEqual(S.efficiency_cores([(0, i, 0) for i in range(16)]), [])     # one kind of core: none

    def test_cpu_lists(self):
        self.assertEqual(S.parse_cpus("12-15, 18,3"), [3, 12, 13, 14, 15, 18])
        self.assertEqual(S.parse_cpus(""), [])
        self.assertEqual(S.compact([3, 12, 13, 14, 15, 18]), "3,12-15,18")

    @unittest.skipUnless(os.name == "nt", "Windows' CPU sets")
    def test_this_pc(self):
        sets = S.cpu_sets()
        self.assertTrue(sets)
        self.assertEqual(len({lp for _, lp, _ in sets}), os.cpu_count())


class FakeAsr:
    def __init__(self):
        self.seen = None

    def transcribe(self, samples, rate):
        self.seen = (len(samples), rate)
        return [(0.0, len(samples) / rate, "hello world")]


class FakeTts:
    def speak(self, text, voice=None, speed=1.0):
        self.seen = (text, voice, speed)
        return np.zeros(2400, dtype=np.float32), 24000


class OverHttp(unittest.TestCase):
    def serve(self, kind, model):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(kind, model, lambda m: None))
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_address[1]

    def call(self, port, method, path, body=None, ctype="application/json"):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            c.request(method, path, body=body, headers={"Content-Type": ctype} if body is not None else {})
            r = c.getresponse()
            return r.status, r.read(), r.getheader("Content-Type")
        finally:
            c.close()

    def test_transcriptions(self):
        asr = FakeAsr()
        port = self.serve("asr", asr)
        self.assertEqual(self.call(port, "GET", "/health")[0], 200)
        body, ctype = multipart({"file": ("a.wav", sine_wav(1.0)), "model": "whisper-1"})
        status, out, _ = self.call(port, "POST", "/v1/audio/transcriptions", body, ctype)
        self.assertEqual((status, json.loads(out)), (200, {"text": "hello world"}))
        self.assertEqual(asr.seen, (16000, 16000))
        status, out, ctype = self.call(port, "POST", "/v1/audio/transcriptions", json.dumps(
            {"file": base64.b64encode(sine_wav()).decode(), "response_format": "text"}).encode())
        self.assertEqual((status, out, ctype), (200, b"hello world", "text/plain; charset=utf-8"))
        status, out, _ = self.call(port, "POST", "/v1/audio/transcriptions", json.dumps(
            {"file": base64.b64encode(b"not audio").decode()}).encode())
        self.assertEqual(status, 400)
        self.assertEqual(self.call(port, "POST", "/v1/audio/speech", b"{}")[0], 404)   # the other kind's path

    def test_speech(self):
        tts = FakeTts()
        port = self.serve("tts", tts)
        status, out, ctype = self.call(port, "POST", "/v1/audio/speech", json.dumps(
            {"input": "Hi there.", "voice": "nova", "response_format": "wav", "speed": 1.25}).encode())
        self.assertEqual((status, out[:4], ctype), (200, b"RIFF", "audio/wav"))
        self.assertEqual(tts.seen, ("Hi there.", "nova", 1.25))
        self.assertEqual(self.call(port, "POST", "/v1/audio/speech", json.dumps({"input": " "}).encode())[0], 400)
        voices = json.loads(self.call(port, "GET", "/v1/audio/voices")[1])["voices"]
        self.assertEqual((len(voices), voices[3]), (54, "af_heart"))


class Voices(unittest.TestCase):
    def test_voice_names(self):
        t = S.Tts.__new__(S.Tts)          # the name mapping only: no model loaded
        t.default = 3
        self.assertEqual(t.voice_id("af_heart"), 3)
        self.assertEqual(t.voice_id("nova"), S.KOKORO_VOICES.index("af_nova"))      # OpenAI's names
        self.assertEqual(t.voice_id("BM_GEORGE"), S.KOKORO_VOICES.index("bm_george"))
        self.assertEqual(t.voice_id("17"), 17)
        self.assertEqual(t.voice_id("no-such-voice"), 3)


MODELS = os.environ.get("STRATA_SPEECH_MODELS", "")


@unittest.skipUnless(importlib.util.find_spec("sherpa_onnx") and MODELS and
                     os.path.isdir(os.path.join(MODELS, "kokoro-multi-lang-v1_0")),
                     "set STRATA_SPEECH_MODELS to a folder with the speech models (and install sherpa-onnx)")
class RealModels(unittest.TestCase):
    def test_round_trip(self):
        tts = S.Tts(os.path.join(MODELS, "kokoro-multi-lang-v1_0"), 4, "cpu")
        samples, rate = tts.speak("The quick brown fox jumps over the lazy dog.")
        asr = S.Asr(os.path.join(MODELS, "parakeet-tdt-0.6b-v3-int8"), 4, "cpu",
                    os.path.join(MODELS, "silero_vad.onnx"))
        text = " ".join(t for _, _, t in asr.transcribe(np.asarray(samples, np.float32), rate)).lower()
        self.assertIn("quick brown fox", text)


if __name__ == "__main__":
    unittest.main()
