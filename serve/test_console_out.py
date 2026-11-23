"""serve/console_out.py: a console that stops taking output (a Windows console in QuickEdit selection, an unread pipe)
must not block the server's print() calls - and nothing written may be lost or reordered once it takes output again.

    python -m unittest serve.test_console_out
"""
import io
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serve import console_out  # noqa: E402
from serve.console_out import QueuedStream, _Queue  # noqa: E402


class HeldConsole(io.StringIO):
    """A console in selection mode: write() waits until released."""

    def __init__(self, log=None):
        super().__init__()
        self.released = threading.Event()
        self.log = log

    def write(self, text):
        self.released.wait()
        if self.log is not None:
            self.log.append((self, text))
        return super().write(text)


def write_in_thread(stream, text, timeout=2.0) -> bool:
    """True if stream.write(text) returned within timeout."""
    done = threading.Event()

    def go():
        stream.write(text)
        done.set()
    threading.Thread(target=go, daemon=True).start()
    return done.wait(timeout)


class TestQueuedStream(unittest.TestCase):
    def test_an_unread_pipe_blocks_a_plain_write_not_a_queued_one(self):
        text = "".join(f"line {i:06d} " + "x" * 100 + "\n" for i in range(10000))     # ~1.1 MB: far over a pipe buffer
        for queued in (False, True):
            r, w = os.pipe()
            reader = os.fdopen(r, "r", encoding="utf-8", newline="")
            real = os.fdopen(w, "w", encoding="utf-8", newline="")
            queue = _Queue(1 << 30) if queued else None
            stream = QueuedStream(real, queue) if queued else real
            got = []
            try:
                returned = write_in_thread(stream, text, timeout=1.0)
                self.assertEqual(returned, queued, "plain write to an unread pipe should block (control)" if not queued
                                 else "a queued write waited for the console")
                t = threading.Thread(target=lambda: got.append(reader.read()), daemon=True)
                t.start()                                         # the console takes output again
                if queued:
                    self.assertTrue(queue.drain(10))
                    queue.close()
                    real.close()
                    t.join(10)
                    self.assertEqual(got, [text], "lost or reordered output")
            finally:
                if not queued:
                    t.join(10)                                    # (the blocked write finishes once read)
                for f in (real, reader):
                    try:
                        f.close()
                    except OSError:
                        pass

    def test_stdout_and_stderr_keep_their_order(self):
        log = []
        out, err = HeldConsole(log), HeldConsole(log)
        queue = _Queue(1 << 20)
        o, e = QueuedStream(out, queue), QueuedStream(err, queue)
        for stream, text in ((o, "a"), (e, "b"), (o, "c"), (e, "d")):
            self.assertTrue(write_in_thread(stream, text))
        out.released.set()
        err.released.set()
        self.assertTrue(queue.drain(5))
        self.assertEqual([(s is out, t) for s, t in log], [(True, "a"), (False, "b"), (True, "c"), (False, "d")])

    def test_drain_is_bounded_while_the_console_is_held(self):
        held = HeldConsole()
        queue = _Queue(1 << 20)
        QueuedStream(held, queue).write("waiting\n")
        t0 = time.monotonic()
        self.assertFalse(queue.drain(0.3))
        self.assertLess(time.monotonic() - t0, 2)
        held.released.set()
        self.assertTrue(queue.drain(5))
        self.assertEqual(held.getvalue(), "waiting\n")

    def test_past_the_limit_the_oldest_goes_and_the_drop_is_said(self):
        held = HeldConsole()
        queue = _Queue(10)
        stream = QueuedStream(held, queue)
        stream.write("1111")                        # taken by the writer at once, which then waits on the console
        time.sleep(0.2)
        for piece in ("2222", "3333", "4444", "5555"):
            stream.write(piece)
        held.released.set()
        self.assertTrue(queue.drain(5))
        text = held.getvalue()
        self.assertTrue(text.startswith("1111"), text)
        self.assertIn("dropped while the console was blocked", text)
        self.assertTrue(text.endswith("44445555"), text)
        self.assertEqual(queue.dropped_chunks, 2)

    def test_the_real_streams_attributes_show_through(self):
        real = io.StringIO()
        stream = QueuedStream(real, _Queue(1 << 20))
        self.assertIs(stream.getvalue.__self__, real)
        self.assertFalse(stream.isatty())
        print("x", file=stream, flush=True)          # print's flush=True must not wait either
        self.assertTrue(console_out.drain() and stream._queue.drain(5))
        self.assertEqual(real.getvalue(), "x\n")


class Own(io.StringIO):
    pass


class TestDecoupled(unittest.TestCase):
    def own_streams(self):
        out, err = Own(), Own()
        return mock.patch.multiple(sys, stdout=out, __stdout__=out, stderr=err, __stderr__=err), out, err

    def test_the_processs_own_streams_are_queued_then_restored(self):
        patch, out, err = self.own_streams()
        with patch, mock.patch.dict(os.environ, {"STRATA_BLOCKING_CONSOLE": ""}):
            with console_out.decoupled():
                self.assertIsInstance(sys.stdout, QueuedStream)
                self.assertIsInstance(sys.stderr, QueuedStream)
                print("to out")
                print("to err", file=sys.stderr)
            self.assertIs(sys.stdout, out)
            self.assertIs(sys.stderr, err)
        self.assertEqual((out.getvalue(), err.getvalue()), ("to out\n", "to err\n"))

    def test_a_replaced_stream_is_left_alone(self):
        patch, out, err = self.own_streams()
        capture = io.StringIO()
        with patch, mock.patch.object(sys, "stderr", capture), \
                mock.patch.dict(os.environ, {"STRATA_BLOCKING_CONSOLE": ""}):
            with console_out.decoupled():
                self.assertIsInstance(sys.stdout, QueuedStream)
                self.assertIs(sys.stderr, capture)

    def test_strata_blocking_console_keeps_the_plain_streams(self):
        patch, out, err = self.own_streams()
        with patch, mock.patch.dict(os.environ, {"STRATA_BLOCKING_CONSOLE": "1"}):
            with console_out.decoupled():
                self.assertIs(sys.stdout, out)

    def test_the_server_runs_with_a_queued_console(self):
        import serve.server as S
        seen = []
        patch, out, err = self.own_streams()
        with patch, mock.patch.dict(os.environ, {"STRATA_BLOCKING_CONSOLE": ""}), \
                mock.patch.object(S, "_main", lambda: seen.append(type(sys.stdout)) or 7):
            self.assertEqual(S.main(), 7)
            self.assertIs(sys.stdout, out)
        self.assertEqual(seen, [QueuedStream])


if __name__ == "__main__":
    unittest.main()
