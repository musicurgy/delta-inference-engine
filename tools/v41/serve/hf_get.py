#!/usr/bin/env python3
"""hf_get.py - "hf download" that you can leave alone (setup-ds41-models.bat and Choose Model.bat run it).

  hf_get.py REPO [FILES...] [--include PATTERN] [--revision REV] --local-dir DIR [other hf download options]
      The same arguments as "hf download".  On top of it:
        * the PC stays awake while it runs (SetThreadExecutionState, as the server's prevent_sleep; the display may
          still turn off; closing a laptop lid or pressing the power button still sleeps it);
        * one progress line for the whole download: "142.0 of 365.7 GB (39%) - 45 MB/s - about 1 h 20 min left";
        * a download that stops moving (no new bytes on disk for DS41_STALL_MIN minutes, default 10) is ended and
          started again - it resumes where it was; from the second problem on it runs without Hugging Face's Xet
          transfer (HF_HUB_DISABLE_XET=1), the usual cure for Xet stalls;
        * a failed run is tried again after a pause (15 s, 30 s, 1, 2, 5 min) instead of stopping the setup;
        * "finished" with files still missing (it happens with Xet) counts as a failure and is tried again;
        * a Windows notification when a long download stops for good (DS41_NO_TOAST=1: none).
  hf_get.py --notify TITLE TEXT      a Windows notification (setup-ds41-models.bat: "set up, starting now")
  hf_get.py --awake CMD [ARGS...]    run a command with the PC kept awake (the long SHA-256 check)
Exit code: 0 done, else the last run's (or 1).
"""
from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")   # no usage reports to Hugging Face (downloads only)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
GB = 1e9
BACKOFF_S = (15, 30, 60, 120, 300)          # between failed runs; after the last one the download gives up
XET_OFF_AFTER = 2                           # problems (stalls or failures) before Xet is switched off


# ------------------------------------------------------------------------------------------------ keep awake, notify
def keep_awake():
    """the PC stays awake until close() (Windows; elsewhere nothing).  -> an object with close()"""
    class Nothing:
        def close(self):
            pass
    if os.name != "nt" or os.environ.get("DS41_NO_KEEPAWAKE") == "1":
        return Nothing()
    try:
        sys.path.insert(0, ROOT)
        from serve.keepawake import KeepAwake, _windows_call
        ka = KeepAwake(_windows_call())
        ka.acquire()
        return ka
    except Exception:                       # noqa: BLE001  never stop a download over this
        return Nothing()


_TOAST_PS = r"""
$ErrorActionPreference = 'Stop'
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] > $null
$t = [Security.SecurityElement]::Escape($env:DS41_TOAST_TITLE)
$b = [Security.SecurityElement]::Escape($env:DS41_TOAST_TEXT)
$x = New-Object Windows.Data.Xml.Dom.XmlDocument
$x.LoadXml("<toast><visual><binding template='ToastGeneric'><text>$t</text><text>$b</text></binding></visual></toast>")
$n = [Windows.UI.Notifications.ToastNotification]::new($x)
$id = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($id).Show($n)
"""


def notify(title: str, text: str) -> bool:
    """a Windows notification (PowerShell's own app id: no install, no admin) and a beep in the console.
    DS41_NO_TOAST=1: the beep only.  -> True when the notification was shown"""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    if os.name != "nt" or os.environ.get("DS41_NO_TOAST") == "1":
        return False
    env = {**os.environ, "DS41_TOAST_TITLE": title, "DS41_TOAST_TEXT": text}
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command",
                            _TOAST_PS], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# ------------------------------------------------------------------------------------------------ arguments
VALUE_OPTS = {"--local-dir", "--revision", "--repo-type", "--cache-dir", "--token", "--max-workers"}
LIST_OPTS = {"--include", "--exclude"}


def parse(args: list[str]) -> dict:
    """what hf_get needs from "hf download"'s arguments: repo, files, include/exclude, revision, local dir"""
    out = {"repo": None, "files": [], "include": [], "exclude": [], "revision": None, "local_dir": None,
           "repo_type": None}
    i, cur = 0, None
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            name, eq, val = a.partition("=")
            cur = None
            if name in VALUE_OPTS:
                if not eq:
                    i += 1
                    val = args[i] if i < len(args) else ""
                out[name[2:].replace("-", "_")] = val
            elif name in LIST_OPTS:
                cur = name[2:]
                if eq:
                    out[cur].append(val)
        elif cur:
            out[cur].append(a)
        elif out["repo"] is None:
            out["repo"] = a
        else:
            out["files"].append(a)
        i += 1
    return out


def plan(p: dict, api=None):
    """[(path in the repo, bytes)] this download fetches, from Hugging Face's file list; None when it cannot be read
    (then the progress line shows the bytes and the speed only)"""
    try:
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi()
        kw = {"revision": p["revision"]} if p["revision"] else {}
        if p["repo_type"]:
            kw["repo_type"] = p["repo_type"]
        if p["files"]:
            items = api.get_paths_info(p["repo"], p["files"], **kw)
        else:
            items = [x for x in api.list_repo_tree(p["repo"], recursive=True, **kw) if hasattr(x, "size")]
        out = []
        for x in items:
            path, size = getattr(x, "path", None), getattr(x, "size", None)
            if path is None or size is None:
                continue
            if p["include"] and not any(fnmatch.fnmatch(path, pat) for pat in p["include"]):
                continue
            if p["exclude"] and any(fnmatch.fnmatch(path, pat) for pat in p["exclude"]):
                continue
            out.append((path, int(size)))
        return out or None
    except Exception:                       # noqa: BLE001  offline, an old huggingface_hub: no total
        return None


# ------------------------------------------------------------------------------------------------ bytes on disk
def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _tree_bytes(folder: str, suffix: str = "", limit: int = 200000) -> int:
    """bytes of the files under folder (ending in suffix), at most `limit` files looked at"""
    total, seen, stack = 0, 0, [folder]
    while stack and seen < limit:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    seen += 1
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.name.endswith(suffix):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


def xet_cache_dir() -> str:
    if os.environ.get("HF_XET_CACHE"):
        return os.environ["HF_XET_CACHE"]
    home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "xet")


def done_bytes(local_dir: str, files) -> int:
    """the planned files' bytes already in place, plus the unfinished downloads (.incomplete) in the folder"""
    part = _tree_bytes(os.path.join(local_dir, ".cache", "huggingface", "download"), ".incomplete")
    if not files:
        return part
    return part + sum(min(_size(os.path.join(local_dir, f)), n) for f, n in files)


def missing(local_dir: str, files) -> list[str]:
    """planned files that are not there in full"""
    return [f for f, n in (files or []) if _size(os.path.join(local_dir, f)) < n]


# ------------------------------------------------------------------------------------------------ words
def duration(s: float) -> str:
    s = max(0, int(s))
    if s < 90:
        return f"{max(1, s)} s"
    if s < 3600:
        return f"{round(s / 60)} min"
    h, m = divmod(round(s / 60), 60)
    return f"{h} h {m} min" if m else f"{h} h"


def amount(n: float) -> str:
    return f"{n / GB:.1f} GB" if n >= 0.95 * GB else f"{n / 1e6:.1f} MB"


def progress_line(done: int, total: int | None, rate: float | None) -> str:
    """'142.0 of 365.7 GB (39%) - 45 MB/s - about 1 h 20 min left'"""
    if total:
        unit, div = ("GB", GB) if total >= 0.95 * GB else ("MB", 1e6)
        parts = [f"{done / div:.1f} of {total / div:.1f} {unit} ({min(100, int(100 * done / total))}%)"]
    else:
        parts = [f"{amount(done)} so far"]
    if rate is not None:
        parts.append(f"{rate / 1e6:.0f} MB/s" if rate >= 1e6 else f"{rate / 1e3:.0f} kB/s")
        if total and rate > 0 and total > done:
            parts.append(f"about {duration((total - done) / rate)} left")
    return " - ".join(parts)


class Rate:
    """bytes per second over the last `window` seconds"""

    def __init__(self, window: float = 60.0):
        self.window, self.points = window, []

    def add(self, t: float, n: int) -> float | None:
        self.points.append((t, n))
        while len(self.points) > 2 and t - self.points[1][0] >= self.window:
            self.points.pop(0)
        t0, n0 = self.points[0]
        return (n - n0) / (t - t0) if t - t0 >= 2 else None


# ------------------------------------------------------------------------------------------------ the runner
def hf_command() -> list[str]:
    exe = os.path.join(os.path.dirname(sys.executable), "hf.exe" if os.name == "nt" else "hf")
    return [exe] if os.path.exists(exe) else ["hf"]


def _kill(proc: subprocess.Popen):
    """end the downloader and its children (hf.exe starts python.exe under it)"""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def run(args: list[str], hf_cmd: list[str] | None = None, stall_s: float | None = None, poll_s: float = 2.0,
        backoff=BACKOFF_S, api=None, out=None, toast_after_s: float = 120.0, label: str | None = None) -> int:
    """hf download ARGS, watched (see the module's docstring).  -> 0, or the last exit code"""
    out = out or sys.stdout
    p = parse(args)
    local_dir = p["local_dir"] or "."
    if stall_s is None:
        try:
            stall_s = float(os.environ.get("DS41_STALL_MIN", "10")) * 60
        except ValueError:
            stall_s = 600.0
    files = plan(p, api)
    total = sum(n for _, n in files) if files else None
    what = label or (p["files"][0] if len(p["files"]) == 1 else p["repo"] or "the files")
    tty = hasattr(out, "isatty") and out.isatty()
    xet_off = os.environ.get("HF_HUB_DISABLE_XET") == "1"
    problems, failures, rc = 0, 0, 1
    t_start = time.time()
    awake = keep_awake()
    try:
        while True:
            env = {**os.environ, "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
            if xet_off:
                env["HF_HUB_DISABLE_XET"] = "1"
            if total:
                print(f"Downloading {what} ({amount(total)}) - the PC stays awake meanwhile; if it stops, start "
                      "again and it goes on where it was.", file=out, flush=True)
            else:
                print(f"Downloading {what} - the PC stays awake meanwhile.", file=out, flush=True)
            proc = subprocess.Popen((hf_cmd or hf_command()) + ["download", *args], env=env, stdout=subprocess.DEVNULL)
            rate = Rate()
            last_n, last_change, last_line, shown = None, time.time(), 0.0, ""
            xet_n, xet_at = 0, 0.0
            stalled = False
            while proc.poll() is None:
                time.sleep(poll_s)
                now = time.time()
                n = done_bytes(local_dir, files)
                if not xet_off and now - xet_at >= 30:   # Xet's chunk cache moves too (looked at less often)
                    xet_n, xet_at = _tree_bytes(xet_cache_dir(), limit=50000), now
                n_act = n + (0 if xet_off else xet_n)
                if n_act != last_n:
                    last_n, last_change = n_act, now
                r = rate.add(now, n)
                if tty:
                    shown = progress_line(n, total, r)
                    out.write("\r  " + shown.ljust(78)[:96])
                    out.flush()
                elif now - last_line >= 60:
                    last_line = now
                    print("  " + progress_line(n, total, r), file=out, flush=True)
                if now - last_change >= stall_s:
                    stalled = True
                    break
            if tty and shown:
                out.write("\n")
                out.flush()
            if stalled:
                _kill(proc)
                problems += 1
                print(f"The download has not moved for {duration(stall_s)}: starting it again (it resumes).",
                      file=out, flush=True)
                if problems >= XET_OFF_AFTER and not xet_off:
                    xet_off = True
                    print("  From now on without Hugging Face's Xet transfer (HF_HUB_DISABLE_XET=1).", file=out,
                          flush=True)
                continue
            rc = proc.returncode
            gone = missing(local_dir, files)
            if rc == 0 and not gone:
                return 0
            problems += 1
            if rc == 0:
                print(f"The downloader stopped with {len(gone)} file(s) not complete ({', '.join(gone[:3])}): "
                      "trying again.", file=out, flush=True)
                rc = 1
            if failures >= len(backoff):
                print("The download did not work after several tries (the reason is above).  Check the internet "
                      "connection, then start again: it goes on where it was.", file=out, flush=True)
                if time.time() - t_start >= toast_after_s:
                    notify("DeepSeek: the download stopped", "Start it again: it goes on where it was.")
                return rc
            wait = backoff[failures]
            failures += 1
            if problems >= XET_OFF_AFTER and not xet_off:
                xet_off = True
                print("  Trying without Hugging Face's Xet transfer (HF_HUB_DISABLE_XET=1).", file=out, flush=True)
            print(f"The download stopped (exit code {rc}); trying again in {duration(wait)} "
                  f"(try {failures + 1} of {len(backoff) + 1}) ...", file=out, flush=True)
            time.sleep(wait)
    finally:
        awake.close()


def main(argv: list[str]) -> int:
    if argv[:1] == ["--notify"] and len(argv) >= 3:
        notify(argv[1], argv[2])
        return 0
    if argv[:1] == ["--awake"] and len(argv) >= 2:
        awake = keep_awake()
        try:
            return subprocess.call(argv[1:])
        finally:
            awake.close()
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    return run(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
