"""serve/winjob.py - on Windows, tie the processes Strata starts to the server's own lifetime.

Closing the console window, Task Manager or a crash end the server without running its cleanup, and the engine,
the vision encoder and the MCP servers kept running on their own. `contain(proc)` puts a child in a job object that
is set to kill everything in it when its last handle closes - and the only handle is this process's, which the OS
closes however the server ends. Processes the child starts later join the same job. The server itself stays out of
the job, so a browser that `--open` starts is not tied to it. `start_contained(args, ...)` starts a child already in
the job, so what it starts in its first moments is in it too. Each child (and the server) is also opted out of Windows
power throttling, see `no_throttle` (#691). Elsewhere, and if the job cannot be made, a no-op.
"""
from __future__ import annotations

import os
import subprocess
import threading

_job = None
_lock = threading.Lock()

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _IoCounters(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                       "OtherOperationCount", "ReadTransferCount",
                                                       "WriteTransferCount", "OtherTransferCount")]

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _JobObjectExtendedLimitInformation = 9

    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)

    def _make_job():
        job = _k32.CreateJobObjectW(None, None)     # no security attributes: the handle is not inherited
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(job, _JobObjectExtendedLimitInformation, ctypes.byref(info),
                                            ctypes.sizeof(info)):
            _k32.CloseHandle(job)
            return None
        return job

    class _PowerThrottling(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    _k32.SetProcessInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    _k32.GetCurrentProcess.restype = wintypes.HANDLE

    def no_throttle(handle) -> bool:
        """Opt a process out of Windows power throttling (EcoQoS).  Without it Windows 11 treats a process whose console
        window is minimized as background work and, on a hybrid CPU, moves its threads to the E-cores: generation
        slowed by about 20% while the server window was minimized (#691, i9-13980HX: 36.6 -> 43.2 tok/s)."""
        for mask in (0x1 | 0x4, 0x1):               # EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, both off (Windows 11);
            info = _PowerThrottling(1, mask, 0)     # Windows 10 knows the first only: ERROR_INVALID_PARAMETER for both
            if _k32.SetProcessInformation(handle, 4, ctypes.byref(info), ctypes.sizeof(info)):   # ProcessPowerThrottling
                return True
        return False

    no_throttle(_k32.GetCurrentProcess())           # the server itself (tokenizing, streaming) too
else:
    def no_throttle(handle) -> bool:
        return False


def contain(proc) -> bool:
    """Put a started subprocess.Popen in the kill-on-close job. True if it is in; False (harmlessly) otherwise."""
    if proc is None or not _job_ready():
        return False
    try:
        no_throttle(int(proc._handle))
        return bool(_k32.AssignProcessToJobObject(_job, int(proc._handle)))
    except (AttributeError, OSError, TypeError, ValueError):   # (TypeError: a stand-in, no handle)
        return False


# ---- starting a child inside the job (after Niko1221/Strata 5d87c4d8, MIT)
_REAL_POPEN = subprocess.Popen    # the class itself, before a test can replace the name with a stand-in
CREATE_SUSPENDED = 0x00000004


def _job_ready() -> bool:
    global _job
    if os.name != "nt":
        return False
    with _lock:
        if _job is None:
            _job = _make_job() or 0
        return bool(_job)


def start_contained(args, **kw):
    """`subprocess.Popen(args, **kw)` followed by `contain()`, without the gap between the two.  A child put in the job
    after Popen returns has already run: an MCP server started through `npx` / `uvx` / a `.cmd` shim starts its real
    process in its first milliseconds, and that process stayed outside the job and survived a hard kill of the server.
    So the child is created suspended (the caller's own creationflags kept), put in the job, then resumed.  A refused
    assignment still resumes it (it runs outside the job, as contain() leaves it).  If it cannot be resumed it never
    ran an instruction: it is ended and started again the old way.  No job (not Windows, or the job could not be made):
    plain Popen."""
    if not _job_ready():
        proc = subprocess.Popen(args, **kw)
        contain(proc)
        return proc
    proc = subprocess.Popen(args, **{**kw, "creationflags": (kw.get("creationflags") or 0) | CREATE_SUSPENDED})
    if not isinstance(proc, _REAL_POPEN):          # a test's stand-in: no suspended process behind it
        contain(proc)
        return proc
    try:
        contain(proc)                              # False when refused (a job of the server's own forbids it): runs anyway
        resumed = _resume(proc.pid)
    except BaseException:
        _end(proc)                                 # the caller never gets it: not left suspended
        raise
    if resumed:
        return proc
    _end(proc)
    proc = subprocess.Popen(args, **kw)            # as before this change: started, then contained
    contain(proc)
    return proc


def _end(proc) -> None:
    """End a child the caller never got, and close its pipes."""
    try:
        proc.kill()
        proc.wait(10)
    except (OSError, subprocess.TimeoutExpired):
        pass
    for f in (proc.stdin, proc.stdout, proc.stderr):
        try:
            if f:
                f.close()
        except OSError:
            pass


if os.name == "nt":
    class _ThreadEntry(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG),
                    ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]

    _TH32CS_SNAPTHREAD, _THREAD_SUSPEND_RESUME = 0x4, 0x2
    _k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    _k32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    _k32.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))
    _k32.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))
    _k32.OpenThread.restype = wintypes.HANDLE
    _k32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.ResumeThread.restype = wintypes.DWORD
    _k32.ResumeThread.argtypes = (wintypes.HANDLE,)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)


def _resume(pid) -> bool:
    """Resume the threads of a process created suspended (Popen closes the main thread's handle, so they are found by
    a thread snapshot).  True if at least one was resumed."""
    if os.name != "nt" or not isinstance(pid, int) or pid <= 0:
        return False
    snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPTHREAD, 0)
    if not snap or snap == wintypes.HANDLE(-1).value:
        return False
    resumed = 0
    try:
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(_ThreadEntry)
        ok = _k32.Thread32First(snap, ctypes.byref(entry))
        while ok:
            if entry.th32OwnerProcessID == pid:
                th = _k32.OpenThread(_THREAD_SUSPEND_RESUME, False, entry.th32ThreadID)
                if th:
                    if _k32.ResumeThread(th) != 0xFFFFFFFF:
                        resumed += 1
                    _k32.CloseHandle(th)
            ok = _k32.Thread32Next(snap, ctypes.byref(entry))
    finally:
        _k32.CloseHandle(snap)
    return resumed > 0
