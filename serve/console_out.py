"""serve/console_out.py - the server's console output must never hold a request.

On Windows a click into the server's console window starts a QuickEdit selection, and until it ends (Esc, or a right
click) every write to that console blocks - `print()` included, in whatever thread called it, and some of those threads
own the engine's turn, a cancel or a queue admission: requests froze until somebody pressed Esc.  A paused terminal, or
a launcher that stopped reading the server's pipe, does the same.  So while the server runs, sys.stdout and sys.stderr
(when they are the process's own) are replaced by stand-ins whose write() puts the text on one queue and returns; one
daemon thread writes it to the real streams in the order it was written (stdout and stderr together, so a `2>&1` log
keeps its order) and is the only thread that ever waits for the console.

Nothing is lost in normal use: only past STRATA_CONSOLE_QUEUE_MB (64 MiB) of text waiting for a console that stays
blocked is the oldest dropped, and one line says so when the console takes output again.  drain() hands what is queued
to the console before an exit that skips Python's cleanup (os._exit).  STRATA_BLOCKING_CONSOLE=1 keeps the plain
streams.  Only diagnostics go through here: the engine's pipe and the HTTP replies do not.

After Niko1221/Strata b776e1cd (MIT): one shared writer for both streams, and no dropping short of a very long block.
"""
from __future__ import annotations

import collections
import contextlib
import os
import sys
import threading
import time


class _Queue:
    """Text waiting for the console, and the one thread that writes it."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._q: collections.deque = collections.deque()
        self._bytes = 0
        self._cv = threading.Condition()
        self._busy = False                         # the writer holds a piece it has not finished writing
        self._closed = False
        self.dropped_chunks = self.dropped_bytes = 0
        self._dropped_said = 0
        self._thread = threading.Thread(target=self._run, name="strata-console", daemon=True)
        self._thread.start()

    def put(self, stream, text: str) -> None:
        with self._cv:
            closed = self._closed
            if not closed:
                self._q.append((stream, text))
                self._bytes += len(text)
                while self._bytes > self.max_bytes and len(self._q) > 1:
                    _, old = self._q.popleft()
                    self._bytes -= len(old)
                    self.dropped_chunks += 1
                    self.dropped_bytes += len(old)
                self._cv.notify()
        if closed:                                 # a late write through a kept stand-in: straight through, as before
            stream.write(text)

    def _run(self) -> None:
        while True:
            with self._cv:
                self._busy = False
                self._cv.notify_all()
                while not self._q and not self._closed:
                    self._cv.wait()
                if not self._q:
                    return
                stream, text = self._q.popleft()
                self._bytes -= len(text)
                self._busy = True
                lost, self._dropped_said = self.dropped_chunks - self._dropped_said, self.dropped_chunks
            try:
                if lost:
                    stream.write(f"[strata] {lost} pieces of console output were dropped while the console was "
                                 "blocked\n")
                stream.write(text)
                stream.flush()
            except (OSError, ValueError, UnicodeError):
                pass                               # a closed or broken console: the text is lost, nothing waits

    def drain(self, timeout: float) -> bool:
        """Wait (at most timeout s) until everything queued has been written.  True if it has."""
        end = time.monotonic() + timeout
        with self._cv:
            while self._q or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class QueuedStream:
    """Stands in for sys.stdout / sys.stderr: write() queues and returns; everything else is the real stream's."""

    def __init__(self, real, queue: _Queue):
        self._real, self._queue = real, queue

    def write(self, text) -> int:
        if not isinstance(text, str):
            text = str(text)
        if text:
            self._queue.put(self._real, text)
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        return None                                # the writer flushes after every piece

    def writable(self) -> bool:
        return True

    def __getattr__(self, name):                   # encoding, errors, fileno, isatty, buffer, reconfigure ...
        return getattr(self._real, name)


_active: _Queue | None = None


def drain(timeout: float = 5.0) -> bool:
    """Before os._exit: hand what is queued to the console (bounded: a console that stays blocked does not hold the
    exit).  True when nothing is left."""
    q = _active
    return q.drain(timeout) if q is not None else True


@contextlib.contextmanager
def decoupled():
    """While the server runs, the process's own stdout / stderr go through one queue (a stream somebody else already
    replaced, such as a test's capture, is left alone).  At the end the queue is written out (for up to 10 s) and the
    streams are restored."""
    global _active
    if os.environ.get("STRATA_BLOCKING_CONSOLE") == "1" or _active is not None:
        yield
        return
    try:
        mb = float(os.environ.get("STRATA_CONSOLE_QUEUE_MB") or 64)
    except ValueError:
        mb = 64.0
    own = [(name, getattr(sys, name, None)) for name in ("stdout", "stderr")]
    own = [(name, cur) for name, cur in own if cur is not None and cur is getattr(sys, "__" + name + "__", None)]
    if not own:
        yield
        return
    queue = _active = _Queue(max(1, int(mb * (1 << 20))))
    for name, cur in own:
        setattr(sys, name, QueuedStream(cur, queue))
    try:
        yield
    finally:
        queue.drain(10.0)
        for name, cur in own:
            if isinstance(getattr(sys, name, None), QueuedStream):
                setattr(sys, name, cur)
        queue.close()
        _active = None
