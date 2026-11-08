#!/usr/bin/env python3
"""setup_space.py MODEL_FOLDER - will what setup-ds41-models.bat still has to do fit on that drive?

The steps that are not done yet run one after another; each leaves some files and needs more while it runs (the drafter:
its three source shards and the repacked copy, then both are removed).  The peak over that sequence (a download in
progress counts as far as it got) is compared with the drive's free space.  Exit 0 if it fits, 1 with an explanation if
not - so the setup stops before a download fills the drive (the Hugging Face downloader only warns).
DS41_SKIP_SPACE_CHECK=1 skips the check.

    setup_space.py --drafter-plan MODEL_FOLDER   prints what the drafter step will do: none / make / upgrade / opt-in
                                                 (first-start.bat asks; setup-ds41-models.bat follows the same rules)"""
import glob
import os
import shutil
import subprocess
import sys

GB = 1e9
MARGIN = 5 * GB
Q2_NAME = "DeepSeek-V4.1-Flash-1017-Q2.gguf"
Q2 = 365.7 * GB                                 # the ds4 Q2 GGUF: the model and its Engram tables
MTP_SHARDS, SIDECAR, DRAFTER = 7.9 * GB, 7.9 * GB, 4.2 * GB
VISION = 0.97 * GB                             # pictures: DeepSeek-V4.1-Flash-1017-Vision.gguf
# the drafter: 0.94's (experts shrunk with drafter-imatrix.bin, +0.6% drafts accepted), 0.93's (uniform), their source
IMX_NAME, UNIFORM_NAME, SOURCE_NAME = "V41-1017-MTP-iq2xxs-imx.gguf", "V41-1017-MTP-iq2xxs.gguf", "V41-1017-MTP.gguf"
VOICE = 1.08 * GB                              # voice: speech\ (Parakeet TDT 0.6B v3 int8, Kokoro-82M v1.0, Silero VAD)


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def partial(folder):
    """bytes of unfinished Hugging Face downloads under folder (they are resumed, not started again)"""
    return sum(size(p) for p in glob.glob(os.path.join(folder, ".cache", "huggingface", "download", "*.incomplete")))


def imatrix_usable():
    """setup-ds41-models.bat's rule: drafter-imatrix.bin beside the engine build's v41_sidecar_requant.exe (or in
    tools/v41/serve of a source checkout), and an exe that knows --imatrix (one from before 0.94 answers "unknown arg")"""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(root, "tools", "v41", "serve"))
    try:
        import cpu_level
        folder = cpu_level.pick_build()[0] or "build-ds41"
    except Exception:  # noqa: BLE001
        folder = "build-ds41"
    req = os.path.join(root, folder, "v41_sidecar_requant.exe")
    imx = os.path.join(root, folder, "drafter-imatrix.bin")
    if not os.path.exists(imx):
        imx = os.path.join(root, "tools", "v41", "serve", "drafter-imatrix.bin")
    if not os.path.exists(imx):
        return False
    if not os.path.exists(req):
        return True   # no engine yet: first-start fetches the current one, which knows it
    try:
        r = subprocess.run([req, "--imatrix", "probe"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return "unknown arg" not in (r.stdout + r.stderr)


def drafter_plan(m):
    """none (not used here, 0.94's is there, or 0.93's is there and this engine cannot make the new one) / make (no
    drafter yet) / upgrade (0.93's is there and 0.94's is made now: from the full-size drafter still on disk, or - asked
    for with setup-ds41-models.bat drafter-imx / DS41_DRAFTER_IMX=1 - after downloading its 7.9 GB of shards again) /
    opt-in (0.93's is there and kept: the 7.9 GB download for ~0.3% speed is the user's call)"""
    if os.environ.get("DS41_NO_DRAFTER") == "1" or os.path.exists(os.path.join(m, IMX_NAME)):
        return "none"
    has_uniform = os.path.exists(os.path.join(m, UNIFORM_NAME))
    if not has_uniform:
        return "make"
    if not imatrix_usable():
        return "none"
    if os.path.exists(os.path.join(m, SOURCE_NAME)) or os.environ.get("DS41_DRAFTER_IMX") == "1":
        return "upgrade"
    return "opt-in"


def main():
    if sys.argv[1:2] == ["--drafter-plan"]:
        print(drafter_plan(sys.argv[2]))
        return 0
    m = sys.argv[1]
    os.makedirs(m, exist_ok=True)
    steps = []   # (what, GB more while it runs, GB more once done), in the setup's order
    # an install from before 2026-10-27 (the slim copy + DeepSeek's Engram shards) needs no main model
    old = os.path.exists(os.path.join(m, "ds4-q2-slim.gguf")) and \
        os.path.exists(os.path.join(m, "engram", "model-00048-of-00048.safetensors"))
    # nor does a model Choose Model.bat picked that is there (model-choice.txt: an Unsloth GGUF carries its own Engram)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(root, "model-choice.txt"), encoding="utf-8-sig") as f:
            pick = f.readline().strip().strip('"')
    except OSError:
        pick = ""
    if pick and os.path.exists(pick if os.path.isabs(pick) else os.path.join(m, pick)):
        old = True
    # the drafter first (setup-ds41-models.bat builds it before the main model's download); none under 14 GB of VRAM
    if drafter_plan(m) in ("make", "upgrade"):   # "opt-in": nothing is done unless asked
        if os.path.exists(os.path.join(m, SOURCE_NAME)):
            steps.append(("drafter", DRAFTER, DRAFTER - SIDECAR))
        else:
            steps.append(("drafter (while it is built)", MTP_SHARDS + SIDECAR, DRAFTER))
    # pictures (on unless vision-choice.txt / DS41_VISION says off), before the main model as well
    try:
        with open(os.path.join(root, "vision-choice.txt"), encoding="utf-8-sig") as f:
            vis = f.readline().strip().lower()
    except OSError:
        vis = ""
    vis = os.environ.get("DS41_VISION", "").strip().lower() or vis or "on"     # on by default since 0.94
    if vis not in ("off", "0", "no", "none", "false", "disable", "disabled") and not os.path.exists(os.path.join(m, "DeepSeek-V4.1-Flash-1017-Vision.gguf.ok")):
        steps.append(("pictures (0.97 GB)", VISION, VISION))
    # voice (off unless voice-choice.txt / DS41_VOICE says cpu or gpu: Enable Voice.bat)
    try:
        with open(os.path.join(root, "voice-choice.txt"), encoding="utf-8-sig") as f:
            voc = f.readline().strip().lower()
    except OSError:
        voc = ""
    voc = os.environ.get("DS41_VOICE", "").strip().lower() or voc or "off"
    if (voc == "cpu" or voc.startswith("gpu")) and not os.path.exists(
            os.path.join(m, "speech", "kokoro-multi-lang-v1_0", "model.onnx.ok")):
        steps.append(("voice (1.1 GB)", VOICE, VOICE))
    if not old and not os.path.exists(os.path.join(m, Q2_NAME + ".ok")):
        n = max(0.0, Q2 - size(os.path.join(m, Q2_NAME)) - partial(m))
        steps.append(("main model (366 GB download)", n, n))
    if not steps:
        return 0
    need, kept = 0.0, 0.0
    for _, peak, after in steps:
        need = max(need, kept + peak)
        kept += after
    free = shutil.disk_usage(m).free
    parts = [f"{w} {p / GB:.0f} GB" for w, p, _ in steps]
    print(f"Space: {free / GB:.0f} GB free; still to do: {', '.join(parts)} - at most about {(need + MARGIN) / GB:.0f} GB "
          f"needed at once")
    if free >= need + MARGIN or os.environ.get("DS41_SKIP_SPACE_CHECK") == "1":
        return 0
    print(f"Not enough free space on that drive: about {(need + MARGIN - free) / GB:.0f} GB more is needed.  Free some up,\n"
          f"or set DS41_MODELS in ds41-paths.bat to a folder on a bigger NVMe drive, then start again.\n"
          f"(To try anyway: set DS41_SKIP_SPACE_CHECK=1.)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
