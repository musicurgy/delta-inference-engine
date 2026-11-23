"""serve/test_stream_stalled_client.py - a streaming client that stops reading gives the engine's turn back.

The thread that writes the SSE stream also holds the engine's turn.  A client that stops reading without closing (a
suspended or hung process, a dropped network) fills the socket buffers, and the write blocks until the OS gives the
connection up: the engine idle, every other request waiting.  A write that makes no progress for STREAM_WRITE_TIMEOUT_S
now ends the request as a disconnect.  After upstream Xolotl acab3782 / Niko1221/Strata c08fd2ec.

No GPU: the engine is a MockEngine that streams many tokens.

    python -m unittest serve.test_stream_stalled_client -v
"""
import json
import socket
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class StalledClient(unittest.TestCase):
    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        tok = ByteTokenizer()
        # many different tokens: one token repeated would trip the repeat guard and end the reply early
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", 0.5, create=True):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                   "max_tokens": 400000, "reasoning_effort": "none"}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                              b"Content-Type: application/json\r\n"
                              b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                # ... and never reads.  The request takes the turn, the buffers fill, and the write must let go.
                end = time.monotonic() + 5
                while time.monotonic() < end and not svc.fifo.locked():
                    time.sleep(0.01)
                self.assertTrue(svc.fifo.locked(), "the stuck client's request holds the engine's turn")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.fifo.locked():
                    time.sleep(0.05)
                self.assertFalse(svc.fifo.locked(), "the turn is still held by a client that does not read")
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()

    def test_a_client_that_reads_gets_the_whole_stream(self):
        """The timeout changes nothing for a client that reads: the same text with and without it."""
        tok = ByteTokenizer()
        script = "The quick brown fox, 0123456789, jumps over the lazy dog."
        got = {}
        for limit in (0, 0.5):
            engine = MockEngine(tok, script)
            svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
            httpd = serve(svc, port=0)
            try:
                with mock.patch.object(server, "STREAM_WRITE_TIMEOUT_S", limit, create=True):
                    req = urllib.request.Request(
                        f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                        data=json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                         "reasoning_effort": "none"}).encode(),
                        headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=30) as r:
                        raw = r.read().decode("utf-8")
            finally:
                httpd.shutdown()
                httpd.server_close()
            text = ""
            for line in raw.splitlines():
                if line.startswith("data: {"):
                    for ch in json.loads(line[6:]).get("choices", []):
                        text += (ch.get("delta") or {}).get("content") or ""
            self.assertTrue(raw.rstrip().endswith("data: [DONE]"))
            got[limit] = text
        self.assertEqual(got[0], script)
        self.assertEqual(got[0.5], script)



class SlowSocket:
    """A socket that takes a few bytes per send() (a slow client that keeps reading), or none at all (one that
    stopped): send() then raises TimeoutError, as a socket with a timeout does after waiting that long."""

    def __init__(self, per_send=1000, stalled=False):
        self.per_send, self.stalled, self.got, self.sends = per_send, stalled, bytearray(), 0

    def send(self, data):
        if self.stalled:
            raise TimeoutError("timed out")
        self.sends += 1
        n = min(len(data), self.per_send)
        self.got.extend(data[:n])
        return n

    def sendall(self, data):        # what the stock wfile calls: the timeout would be for the whole call
        raise AssertionError("sendall's timeout covers the whole write")

    def fileno(self):
        return -1


class ProgressWrites(unittest.TestCase):
    """The timeout is for a write that makes no progress, not for the whole of a large one (socket.sendall's timeout
    covers the whole call since Python 3.5): a slow client that keeps reading gets all of a large event."""

    def test_a_large_write_goes_out_in_as_many_sends_as_it_takes(self):
        sock = SlowSocket(per_send=1000)
        payload = bytes(range(256)) * 400
        self.assertEqual(server.ProgressWriter(sock).write(payload), len(payload))
        self.assertEqual(bytes(sock.got), payload)
        self.assertEqual(sock.sends, -(-len(payload) // 1000))

    def test_a_reader_that_stops_ends_the_write(self):
        with self.assertRaises(TimeoutError):
            server.ProgressWriter(SlowSocket(stalled=True)).write(b"data: x\n\n")

    def test_a_stream_writes_through_it(self):
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "hello there"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        writes = []
        real = server.ProgressWriter.write

        def counted(self, b):
            writes.append(len(b))
            return real(self, b)

        try:
            with mock.patch.object(server.ProgressWriter, "write", counted):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                    data=json.dumps({"messages": [{"role": "user", "content": "go"}], "stream": True,
                                     "reasoning_effort": "none"}).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    body = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertIn("[DONE]", body)
        self.assertTrue(writes)



class TimeoutSetting(unittest.TestCase):
    def test_a_bad_value_keeps_the_default(self):
        from serve import server
        f = server.stream_write_timeout
        self.assertEqual(f(None), 60.0)
        self.assertEqual(f("30"), 30.0)
        self.assertEqual(f("0"), 0.0)
        self.assertEqual(f(""), 0.0)            # empty: no limit, as before
        for bad in ("abc", "-5", "nan", "inf", "1e999"):
            self.assertEqual(f(bad), 60.0, bad)

if __name__ == "__main__":
    unittest.main()
