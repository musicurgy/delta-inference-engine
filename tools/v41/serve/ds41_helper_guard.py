"""Multi-GPU: a second card's helper (ds41_helper) that stops must never stop the model.

serve_ds41.py starts the helper before each engine start; the engine, given STRATA_DS41_REMOTE, refuses to start when
the helper is not serving (ds41_moe.cpp: "the helper is not serving").  This guard notices a helper that stopped - at
once (a missing file, no such card: it exits within EARLY_S), or while the engine was starting (no VRAM for its experts:
it sets ready = -1 and exits with 1, and the engine's start fails) - says so in one line, and runs the engine on one
card: STRATA_DS41_REMOTE taken out of its environment, for this start and every later one of this server."""
import os
import subprocess

REMOTE = "STRATA_DS41_REMOTE"
EARLY_S = 2.0       # an argument, file or device error ends the helper before this
EXIT_WAIT_S = 10.0  # after a failed engine start: how long a failed helper may take to exit (CUDA teardown)
ENV_AT = 4          # StrataEngine.__init__(exe, args, cwd, log, env, ...): env's place among the positional arguments


def reason(log_path, start, code):
    """Why the helper stopped: its last "ds41_helper: ..." line in this start's part of its log, else its exit code."""
    why = None
    try:
        with open(log_path, "rb") as f:
            f.seek(start)
            for line in f.read().decode("utf-8", "replace").splitlines():
                if line.startswith("ds41_helper: "):
                    why = line[len("ds41_helper: "):].strip()
    except (OSError, TypeError):
        pass
    if why:
        return why if len(why) <= 200 else why[:197] + "..."
    return f"exit code {code}" if code is not None else "it stopped"


def env_of(a, k):
    """The environment the engine is started with (None: this process's)."""
    return k["env"] if "env" in k else (a[ENV_AT] if len(a) > ENV_AT else None)


def uses_remote(a, k):
    env = env_of(a, k)
    return REMOTE in (os.environ if env is None else env)


def without_remote(a, k):
    """The engine's arguments with STRATA_DS41_REMOTE out of its environment (the caller's dict is not changed)."""
    if not uses_remote(a, k):
        return a, k
    env = {x: v for x, v in (env_of(a, k) or os.environ).items() if x != REMOTE}
    if "env" in k or len(a) <= ENV_AT:
        return a, {**k, "env": env}
    a = list(a)
    a[ENV_AT] = env
    return tuple(a), k


class HelperGuard:
    def __init__(self, say=None):
        self.off = None     # why the helper stopped: from then on every engine start is a one-card start
        self.proc = None
        self.log = None
        self.log_at = 0
        self.say = say or (lambda s: print(s, flush=True))

    def started(self, proc, log, log_at):
        """A helper was just started (its output appended to `log` from byte `log_at`): give it EARLY_S to fail."""
        self.proc, self.log, self.log_at = proc, log, log_at
        try:
            proc.wait(timeout=EARLY_S)
        except subprocess.TimeoutExpired:
            return
        self.stopped()

    def stopped(self):
        self.off = reason(self.log, self.log_at, self.proc.returncode)
        self.say(f"[strata] the second card's helper stopped ({self.off}): running on one card")

    def init(self, init, obj, a, k):
        """Start the engine (init(obj, *a, **k)); after a failed start whose helper had stopped with an error, once more
        on one card.  Any other failure is the engine's own and goes up unchanged."""
        if self.off:
            a, k = without_remote(a, k)
        try:
            return init(obj, *a, **k)
        except Exception as e:
            if self.off or self.proc is None or not uses_remote(a, k):
                raise
            try:
                code = self.proc.wait(timeout=EXIT_WAIT_S)
            except subprocess.TimeoutExpired:
                raise e from None       # the helper is still running: not why the engine failed
            if code == 0:               # it stopped because it was told to (the engine stopping), not on an error
                raise
            self.stopped()
        a, k = without_remote(a, k)
        return init(obj, *a, **k)
