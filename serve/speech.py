"""The speech server: dictation (speech -> text) and read-aloud (text -> speech), OpenAI-compatible, run by sherpa-onnx.

serve/voice.py starts this as a child process the first time the chat's microphone or speaker button (or an API client)
needs it, and stops it again after a few idle minutes, so it costs nothing until it is used.  One process per job:

  python serve/speech.py --kind asr --port 5110 --model <folder> [--threads 4] [--cpus e] [--device cpu|cuda]
  python serve/speech.py --kind tts --port 5210 --model <folder> [--voice af_heart] ...

  asr  POST /v1/audio/transcriptions  multipart/form-data (file, language, response_format: json | text |
       verbose_json | srt | vtt), as OpenAI's; or JSON {"file": "<data: URL or base64>", ...} (the web chat).
       NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0; 25 European languages, punctuation and capitals), int8, ~0.7 GB RAM.
       WAV is read here; other formats (mp3, m4a, webm, ...) need ffmpeg on the PATH.
  tts  POST /v1/audio/speech  {"input": "...", "voice": "af_heart", "speed": 1.0, "response_format": "wav" | "pcm"}.
       Kokoro-82M v1.0 (Apache-2.0; 54 voices: English, Spanish, French, Italian, Portuguese, Hindi, Japanese, Chinese).
       mp3 / opus / aac / flac need ffmpeg; without it the answer is WAV (most players read it whatever the name).
  both GET /health (200 once the model is loaded).

--cpus e pins the process to the CPU's efficiency cores (Intel 12th gen and later), so it does not take the
performance cores the model's expert threads run on; --device cuda runs it on a graphics card (the launcher picks a
second card with CUDA_VISIBLE_DEVICES; it needs sherpa-onnx's CUDA build, else it falls back to the CPU).

Standard library + numpy + sherpa_onnx (Apache-2.0).  It imports nothing from the server.
"""
from __future__ import annotations

import argparse
import base64
import hmac
import io
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAMPLE_RATE = 16000
MAX_AUDIO_BYTES = 200 << 20        # an upload larger than this is refused (about 100 minutes of 16 kHz WAV)
CHUNK_S = 25.0                     # long audio is cut into pieces of at most this length (VAD where it has one)

# Kokoro v1.0's speaker ids (sherpa-onnx's voices.bin order) and OpenAI's voice names mapped onto them
KOKORO_VOICES = (
    "af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky am_adam am_echo "
    "am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa bf_alice bf_emma bf_isabella bf_lily bm_daniel "
    "bm_fable bm_george bm_lewis ef_dora em_alex ff_siwis hf_alpha hf_beta hm_omega hm_psi if_sara im_nicola jf_alpha "
    "jf_gongitsune jf_nezumi jf_tebukuro jm_kumo pf_dora pm_alex pm_santa zf_xiaobei zf_xiaoni zf_xiaoxiao zf_xiaoyi "
    "zm_yunjian zm_yunxi zm_yunxia zm_yunyang em_santa").split()
OPENAI_VOICES = {"alloy": "af_alloy", "ash": "am_adam", "ballad": "bm_george", "coral": "af_bella", "echo": "am_echo",
                 "fable": "bm_fable", "nova": "af_nova", "onyx": "am_onyx", "sage": "af_sarah", "shimmer": "af_sky",
                 "verse": "am_michael", "marin": "af_heart", "cedar": "am_liam"}
# espeak-ng's language for each Kokoro voice prefix (the first letter); English and Chinese use the model's lexicons
VOICE_LANG = {"a": "en-us", "b": "en-gb", "e": "es", "f": "fr", "h": "hi", "i": "it", "j": "ja", "p": "pt-br", "z": "zh"}


# ------------------------------------------------------------------------------------------------ CPU placement
def cpu_sets() -> list[tuple[int, int, int]]:
    """Windows: [(group, logical processor, efficiency class)] for every logical processor; [] elsewhere or on error.
    A higher efficiency class is a faster core; on a CPU with one kind of core they are all the same."""
    if os.name != "nt":
        return []
    import ctypes
    import ctypes.wintypes as wt
    try:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        n = wt.ULONG(0)
        k.GetSystemCpuSetInformation(None, 0, ctypes.byref(n), None, 0)
        buf = (ctypes.c_ubyte * n.value)()
        if not n.value or not k.GetSystemCpuSetInformation(buf, n, ctypes.byref(n), None, 0):
            return []
    except (OSError, AttributeError):
        return []
    raw, out, off = bytes(buf), [], 0
    while off + 8 <= len(raw):
        size, kind = struct.unpack_from("<II", raw, off)
        if size <= 0:
            break
        if kind == 0 and off + 19 <= len(raw):           # CpuSetInformation: Id, Group, LogicalProcessorIndex, ...
            group, = struct.unpack_from("<H", raw, off + 12)
            out.append((group, raw[off + 14], raw[off + 18]))
        off += size
    return out


def efficiency_cores(sets=None) -> list[int]:
    """The logical processors (group 0) of the slowest core class, when the CPU has more than one class."""
    sets = cpu_sets() if sets is None else sets
    classes = {e for _, _, e in sets}
    if len(classes) < 2:
        return []
    low = min(classes)
    return sorted(lp for g, lp, e in sets if g == 0 and e == low)


def pin(cpus: str, threads: int, priority: str = "normal") -> str:
    """Apply --cpus (e = the efficiency cores; eN = the last N of them; a list like 12-15,18; empty = leave it) and
    --priority (below: the model's own threads win when both want a core, but then speech is starved while the model
    writes).  The engine's expert pool takes the performance cores first and then the efficiency cores in order, so
    with a pool smaller than the core count the LAST efficiency cores are the free ones.  Returns what was done."""
    done = []
    if os.name == "nt":
        import ctypes
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if priority == "below" and k.SetPriorityClass(k.GetCurrentProcess(), 0x4000):   # BELOW_NORMAL_PRIORITY_CLASS
            done.append("below-normal priority")
        cores = e_cores(cpus) if cpus.startswith("e") else parse_cpus(cpus)
        if cores:
            mask = sum(1 << c for c in cores if c < 64)
            if mask and k.SetProcessAffinityMask(k.GetCurrentProcess(), mask):
                done.append(f"CPUs {compact(cores)}")
        elif cpus.startswith("e"):
            done.append("no efficiency cores on this CPU (any core)")
    elif cpus and not cpus.startswith("e") and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, parse_cpus(cpus))
        done.append(f"CPUs {cpus}")
    done.append(f"{threads} threads")
    return ", ".join(done)


def e_cores(spec: str, sets=None) -> list[int]:
    """'e' -> every efficiency core, 'e4' -> the last four of them ([] without efficiency cores)."""
    cores = efficiency_cores(sets)
    n = spec[1:]
    return cores[-int(n):] if n.isdigit() and int(n) > 0 else cores


def parse_cpus(spec: str) -> list[int]:
    out = []
    for part in (spec or "").replace(" ", "").split(","):
        if re.fullmatch(r"\d+-\d+", part):
            a, b = map(int, part.split("-"))
            out += range(a, b + 1)
        elif part.isdigit():
            out.append(int(part))
    return sorted(set(out))


def compact(cpus: list[int]) -> str:
    runs, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            runs.append(str(start) if start == c else f"{start}-{c}")
            start = None
    return ",".join(runs)


# ------------------------------------------------------------------------------------------------ audio in and out
class AudioError(ValueError):
    pass


def read_wav(data: bytes):
    """A WAV file -> (float32 mono samples in [-1, 1], sample rate).  PCM 8/16/24/32-bit and 32/64-bit float."""
    import numpy as np
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise AudioError("not a WAV file")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos:pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", body, 0)
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise AudioError("a WAV file without its fmt or data chunk")
    tag, channels, rate, _, _, bits = fmt
    if tag == 0xFFFE and len(data) > 0:                     # WAVE_FORMAT_EXTENSIBLE: the real tag is in the GUID
        tag = 3 if bits in (32, 64) and b"\x03\x00\x00\x00\x00\x00\x10\x00" in data[12:80] else 1
    if channels < 1 or rate < 1000:
        raise AudioError("a WAV file with no channels or an impossible sample rate")
    if tag == 3:
        x = np.frombuffer(raw[:len(raw) // (bits // 8) * (bits // 8)], dtype="<f4" if bits == 32 else "<f8")
    elif tag == 1 and bits == 16:
        x = np.frombuffer(raw[:len(raw) // 2 * 2], dtype="<i2") / 32768.0
    elif tag == 1 and bits == 8:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif tag == 1 and bits == 24:
        b = np.frombuffer(raw[:len(raw) // 3 * 3], dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(v >= 1 << 23, v - (1 << 24), v) / float(1 << 23)
    elif tag == 1 and bits == 32:
        x = np.frombuffer(raw[:len(raw) // 4 * 4], dtype="<i4") / float(1 << 31)
    else:
        raise AudioError(f"WAV format {tag} with {bits} bits is not supported")
    x = x.astype(np.float32)
    if channels > 1:
        x = x[:len(x) // channels * channels].reshape(-1, channels).mean(axis=1)
    return x, rate


def decode_audio(data: bytes):
    """Any audio file -> (float32 mono, sample rate): WAV here, everything else through ffmpeg when it is installed."""
    import numpy as np
    if data[:4] == b"RIFF":
        return read_wav(data)
    ff = shutil.which("ffmpeg")
    if not ff:
        raise AudioError("only WAV can be read without ffmpeg; send WAV (16 kHz mono is best) or install ffmpeg "
                         "(winget install Gyan.FFmpeg)")
    try:
        r = subprocess.run([ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar",
                            str(SAMPLE_RATE), "-f", "s16le", "pipe:1"], input=data, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise AudioError(f"ffmpeg could not read it ({e})") from None
    if r.returncode != 0 or not r.stdout:
        raise AudioError("ffmpeg could not read it: " + r.stderr.decode("utf-8", "replace").strip()[-300:])
    return (np.frombuffer(r.stdout[:len(r.stdout) // 2 * 2], dtype="<i2") / 32768.0).astype(np.float32), SAMPLE_RATE


def wav_bytes(samples, rate: int) -> bytes:
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return out.getvalue()


ENCODED = {"mp3": ("mp3", "audio/mpeg"), "opus": ("ogg", "audio/ogg"), "aac": ("adts", "audio/aac"),
           "flac": ("flac", "audio/flac")}


def encode_audio(samples, rate: int, fmt: str) -> tuple[bytes, str]:
    """(bytes, content type) for an OpenAI response_format; WAV when the format needs ffmpeg and it is not there."""
    import numpy as np
    fmt = (fmt or "mp3").lower()
    if fmt == "pcm":                                    # OpenAI: raw 24 kHz 16-bit little-endian mono
        return (np.clip(np.asarray(samples, np.float32), -1, 1) * 32767).astype("<i2").tobytes(), "audio/pcm"
    wav = wav_bytes(samples, rate)
    ff = shutil.which("ffmpeg") if fmt in ENCODED else None
    if ff:
        container, ctype = ENCODED[fmt]
        args = [ff, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]
        args += ["-c:a", "libopus"] if fmt == "opus" else []
        try:
            r = subprocess.run(args + ["-f", container, "pipe:1"], input=wav, capture_output=True, timeout=120)
            if r.returncode == 0 and r.stdout:
                return r.stdout, ctype
        except (OSError, subprocess.TimeoutExpired):
            pass
    return wav, "audio/wav"


# ------------------------------------------------------------------------------------------------ requests
def parse_multipart(body: bytes, content_type: str) -> dict:
    """multipart/form-data -> {name: bytes} (file fields too)."""
    msg = BytesParser(policy=HTTP).parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body)
    if not msg.is_multipart():
        raise ValueError("expected multipart/form-data")
    out = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            out[name] = part.get_payload(decode=True) or b""
            if part.get_filename():
                out[name + ".filename"] = part.get_filename().encode()
    return out


def transcription_request(body: bytes, content_type: str) -> tuple[bytes, dict]:
    """(audio bytes, options) from an OpenAI multipart upload or the web chat's JSON."""
    ct = (content_type or "").lower()
    if ct.startswith("application/json"):
        req = json.loads(body or b"{}")
        f = req.get("file") or req.get("audio") or ""
        if not isinstance(f, str) or not f:
            raise ValueError("send {\"file\": \"<a data: URL or base64 of the audio>\"}")
        if f.startswith("data:"):
            f = f.partition(",")[2]
        return base64.b64decode(f), {k: str(v) for k, v in req.items() if k not in ("file", "audio") and v is not None}
    if ct.startswith("multipart/form-data"):
        fields = parse_multipart(body, content_type)
        if "file" not in fields:
            raise ValueError("the form has no \"file\" field")
        return fields.pop("file"), {k: v.decode("utf-8", "replace") for k, v in fields.items()}
    raise ValueError("send multipart/form-data (OpenAI's form) or application/json")


def timestamp(t: float, sep: str) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d}{sep}{ms % 1000:03d}"


def format_transcript(segments: list[tuple[float, float, str]], fmt: str, duration: float, language: str):
    """(body, content type) in an OpenAI response_format."""
    text = " ".join(s for _, _, s in segments if s).strip()
    fmt = (fmt or "json").lower()
    if fmt == "text":
        return text.encode(), "text/plain; charset=utf-8"
    if fmt in ("srt", "vtt"):
        lines = ["WEBVTT", ""] if fmt == "vtt" else []
        for i, (a, b, s) in enumerate(segments, 1):
            if fmt == "srt":
                lines.append(str(i))
            sep = "," if fmt == "srt" else "."
            lines += [f"{timestamp(a, sep)} --> {timestamp(b, sep)}", s, ""]
        return "\n".join(lines).encode(), "text/plain; charset=utf-8"
    if fmt == "verbose_json":
        body = {"task": "transcribe", "language": language or "auto", "duration": round(duration, 3), "text": text,
                "segments": [{"id": i, "start": round(a, 3), "end": round(b, 3), "text": s}
                             for i, (a, b, s) in enumerate(segments)]}
    else:
        body = {"text": text}
    return json.dumps(body, ensure_ascii=False).encode(), "application/json"


# ------------------------------------------------------------------------------------------------ the models
class Asr:
    def __init__(self, folder: str, threads: int, device: str, vad: str = ""):
        import sherpa_onnx as so

        def f(*names):
            for n in names:
                p = os.path.join(folder, n)
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"{names[0]} is missing in {folder}")
        self.rec = so.OfflineRecognizer.from_transducer(
            encoder=f("encoder.int8.onnx", "encoder.onnx"), decoder=f("decoder.int8.onnx", "decoder.onnx"),
            joiner=f("joiner.int8.onnx", "joiner.onnx"), tokens=f("tokens.txt"), num_threads=threads,
            model_type="nemo_transducer", provider=device)
        self.vad_path = vad if vad and os.path.exists(vad) else ""
        self.lock = threading.Lock()

    def segments(self, samples, rate: int):
        """[(start s, end s, samples, rate)] pieces of at most CHUNK_S: speech found by Silero VAD where it is set up, else
        fixed cuts (a short clip is one piece)."""
        import numpy as np
        dur = len(samples) / rate
        if dur <= CHUNK_S:
            return [(0.0, dur, samples, rate)]
        if rate != SAMPLE_RATE:                                 # VAD and the cuts work at 16 kHz
            n = int(round(len(samples) * SAMPLE_RATE / rate))
            samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples).astype(np.float32)
            rate = SAMPLE_RATE
        if self.vad_path:
            import sherpa_onnx as so
            cfg = so.VadModelConfig(silero_vad=so.SileroVadModelConfig(model=self.vad_path, min_silence_duration=0.4,
                                                                       max_speech_duration=CHUNK_S), sample_rate=rate)
            vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=int(dur) + 60)
            out = []
            for i in range(0, len(samples), 512):
                vad.accept_waveform(samples[i:i + 512])
                while not vad.empty():
                    seg = vad.front
                    out.append((seg.start / rate, (seg.start + len(seg.samples)) / rate,
                                np.asarray(seg.samples, dtype=np.float32), rate))
                    vad.pop()
            vad.flush()
            while not vad.empty():
                seg = vad.front
                out.append((seg.start / rate, (seg.start + len(seg.samples)) / rate,
                            np.asarray(seg.samples, dtype=np.float32), rate))
                vad.pop()
            return out
        step = int(CHUNK_S * rate)
        return [(i / rate, min(len(samples), i + step) / rate, samples[i:i + step], rate)
                for i in range(0, len(samples), step)]

    def transcribe(self, samples, rate: int) -> list[tuple[float, float, str]]:
        out = []
        with self.lock:
            for a, b, piece, piece_rate in self.segments(samples, rate):
                s = self.rec.create_stream()
                s.accept_waveform(piece_rate, piece)
                self.rec.decode_stream(s)
                text = s.result.text.strip()
                if text:
                    out.append((a, b, text))
        return out


class Tts:
    def __init__(self, folder: str, threads: int, device: str, voice: str = "af_heart"):
        import sherpa_onnx as so
        model = next((os.path.join(folder, n) for n in ("model.onnx", "model.int8.onnx")
                      if os.path.exists(os.path.join(folder, n))), None)
        if model is None:
            raise FileNotFoundError(f"model.onnx is missing in {folder}")
        lex = ",".join(os.path.join(folder, n) for n in ("lexicon-us-en.txt", "lexicon-zh.txt")
                       if os.path.exists(os.path.join(folder, n)))
        self.so, self.folder, self.threads, self.device, self.lex = so, folder, threads, device, lex
        self.model = model
        self.default = self.voice_id(voice)
        self.engines: dict = {}
        self.lock = threading.Lock()
        self.engine("en-us")                                    # load now: /health means ready

    def engine(self, lang: str):
        """One OfflineTts per espeak language (Kokoro's non-English voices need their own phonemizer setting)."""
        if lang not in self.engines:
            so, d = self.so, self.folder
            kok = so.OfflineTtsKokoroModelConfig(
                model=self.model, voices=os.path.join(d, "voices.bin"), tokens=os.path.join(d, "tokens.txt"),
                lexicon=self.lex, data_dir=os.path.join(d, "espeak-ng-data"),
                dict_dir=os.path.join(d, "dict") if os.path.isdir(os.path.join(d, "dict")) else "",
                lang="" if lang in ("en-us", "zh") else lang)
            self.engines[lang] = so.OfflineTts(so.OfflineTtsConfig(
                model=so.OfflineTtsModelConfig(kokoro=kok, num_threads=self.threads, provider=self.device),
                max_num_sentences=1))
        return self.engines[lang]

    def voice_id(self, voice) -> int:
        v = str(voice or "").strip().lower()
        v = OPENAI_VOICES.get(v, v)
        if v.isdigit() and int(v) < len(KOKORO_VOICES):
            return int(v)
        return KOKORO_VOICES.index(v) if v in KOKORO_VOICES else getattr(self, "default", 3)

    def speak(self, text: str, voice=None, speed: float = 1.0):
        sid = self.voice_id(voice) if voice else self.default
        lang = VOICE_LANG.get(KOKORO_VOICES[sid][0], "en-us")
        with self.lock:
            audio = self.engine(lang).generate(text, sid=sid, speed=max(0.25, min(4.0, speed)))
        return audio.samples, audio.sample_rate


# ------------------------------------------------------------------------------------------------ the HTTP server
TOKEN_ENV = "STRATA_VOICE_TOKEN"          # serve/voice.py starts this server with a random token in this variable
TOKEN_HEADER = "X-Strata-Voice-Token"     # and sends it with every request it forwards


def make_handler(kind: str, model, log, token: str = ""):
    """0.94 (cross-site requests): any web page could POST to 127.0.0.1:5110 / 5210 - a text/plain or multipart body
    needs no CORS preflight - and keep the efficiency cores busy.  So a request that carries an Origin (browsers send
    one with every POST; Strata's proxy and curl do not) or names another Host (DNS rebinding) is refused, read-aloud
    takes application/json only, and with `token` (Strata's own start) every POST must carry it in TOKEN_HEADER."""
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def refused(self, post: bool) -> bool:
            """Answer 403 and True when the request is not from this PC's own programs (see make_handler)."""
            port = self.server.server_address[1]
            host = (self.headers.get("Host") or "").strip().lower()
            if self.headers.get("Origin") is not None:
                why = "requests from web pages are not accepted (send them through Strata's /v1/audio)"
            elif host and host not in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"):
                why = f"Host {host[:80]!r} is not this PC's address"
            elif post and token and not hmac.compare_digest(self.headers.get(TOKEN_HEADER, "").encode(),
                                                           token.encode()):
                why = f"this server takes requests from Strata only (the {TOKEN_HEADER} header)"
            else:
                return False
            if post:
                self.close_connection = True
                # read a small body before answering: closing a Windows socket with unread data sends a reset, and the
                # caller then sees "connection aborted" instead of this 403.  A large one is left unread (no work for it)
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = 0
                if 0 < n <= 1 << 20:
                    try:
                        self.rfile.read(n)
                    except OSError:
                        pass
            self.error(403, why, "permission_error")
            return True

        def send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def error(self, code: int, message: str, kind_: str = "invalid_request_error"):
            self.send(code, json.dumps({"error": {"type": kind_, "message": message}}).encode(), "application/json")

        def do_GET(self):
            if self.refused(post=False):
                return
            if self.path.split("?")[0] == "/health":
                self.send(200, json.dumps({"status": "ok", "kind": kind}).encode(), "application/json")
            elif self.path.split("?")[0] == "/v1/audio/voices" and kind == "tts":
                self.send(200, json.dumps({"voices": KOKORO_VOICES}).encode(), "application/json")
            else:
                self.error(404, "not found")

        def do_POST(self):
            if self.refused(post=True):
                return
            path = self.path.split("?")[0].rstrip("/")
            if kind == "tts" and not self.headers.get("Content-Type", "").lower().startswith("application/json"):
                self.close_connection = True
                self.error(415, "send the request as application/json")
                return
            try:
                n = int(self.headers.get("Content-Length", 0))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_AUDIO_BYTES:
                self.error(413, f"the body must be at most {MAX_AUDIO_BYTES >> 20} MB")
                return
            body = self.rfile.read(n)
            t0 = time.monotonic()
            try:
                if kind == "asr" and path == "/v1/audio/transcriptions":
                    audio, opts = transcription_request(body, self.headers.get("Content-Type", ""))
                    samples, rate = decode_audio(audio)
                    dur = len(samples) / rate
                    segs = model.transcribe(samples, rate)
                    out, ctype = format_transcript(segs, opts.get("response_format", "json"), dur,
                                                   opts.get("language", ""))
                    log(f"transcribed {dur:.1f} s of audio in {time.monotonic() - t0:.2f} s")
                    self.send(200, out, ctype)
                elif kind == "tts" and path == "/v1/audio/speech":
                    req = json.loads(body or b"{}")
                    text = str(req.get("input") or "").strip()
                    if not text:
                        raise ValueError("\"input\" (the text to speak) is empty")
                    if len(text) > 8192:
                        raise ValueError("\"input\" is longer than 8192 characters; send it in parts")
                    samples, rate = model.speak(text, req.get("voice"), float(req.get("speed") or 1.0))
                    out, ctype = encode_audio(samples, rate, req.get("response_format") or "mp3")
                    log(f"spoke {len(samples) / rate:.1f} s of audio in {time.monotonic() - t0:.2f} s")
                    self.send(200, out, ctype)
                else:
                    self.error(404, f"this {kind} server has no POST {path}")
            except (ValueError, AudioError) as e:
                self.error(400, str(e))
            except Exception as e:   # noqa: BLE001 - the model failed: the client gets the reason
                self.error(500, f"{type(e).__name__}: {e}", "server_error")
    return Handler


def cuda_dlls() -> list[str]:
    """sherpa-onnx's CUDA build loads CUDA 12 and cuDNN 9 from the PATH: NVIDIA's pip packages (nvidia-cublas-cu12, ...)
    put them in site-packages/nvidia/*/bin, which nothing adds to the PATH.  Returns the folders added."""
    import site
    added = []
    for base in [*site.getsitepackages(), site.getusersitepackages()]:
        root = os.path.join(base, "nvidia")
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            d = os.path.join(root, name, "bin")
            if os.path.isdir(d) and d not in added:
                added.append(d)
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
        for d in added:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(d)
    return added


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=("asr", "tts"), required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True, help="the model's folder")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--cpus", default="e", help="e (the efficiency cores), eN (the last N of them), a list like 12-19, or '' (any)")
    ap.add_argument("--priority", default="normal", choices=("normal", "below"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--vad", default="", help="asr: silero_vad.onnx, to cut long recordings at pauses")
    ap.add_argument("--voice", default="af_heart", help="tts: the default voice")
    a = ap.parse_args(argv)

    def log(msg):
        print(f"[speech {a.kind}] {msg}", file=sys.stderr, flush=True)
    where = pin(a.cpus, a.threads, a.priority)
    if a.device == "cuda":
        cuda_dlls()
        where = f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '0')} (as CUDA_VISIBLE_DEVICES names it), " + where
    t0 = time.monotonic()
    try:
        model = Asr(a.model, a.threads, a.device, a.vad) if a.kind == "asr" else Tts(a.model, a.threads, a.device, a.voice)
    except Exception as e:   # noqa: BLE001
        log(f"could not load the model from {a.model}: {e}")
        return 1
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(a.kind, model, log, os.environ.get(TOKEN_ENV, "")))
    log(f"ready on port {a.port} ({os.path.basename(os.path.normpath(a.model))}, {where}; "
        f"loaded in {time.monotonic() - t0:.1f} s)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
