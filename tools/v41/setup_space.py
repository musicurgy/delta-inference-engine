#!/usr/bin/env python3
"""setup_space.py MODEL_FOLDER - will what setup-ds41-models.bat still has to do fit on that drive?

The steps that are not done yet run one after another; each leaves some files and needs more while it runs (the drafter:
its three source shards and the repacked copy, then both are removed).  The peak over that sequence (a download in
progress counts as far as it got) is compared with the drive's free space.  Exit 0 if it fits, 1 with an explanation if
not - so the setup stops before a download fills the drive (the Hugging Face downloader only warns).
DS41_SKIP_SPACE_CHECK=1 skips the check."""
import glob
import os
import shutil
import sys

GB = 1e9
MARGIN = 5 * GB
Q2_NAME = "DeepSeek-V4.1-Flash-1017-Q2.gguf"
Q2 = 365.7 * GB                                 # the ds4 Q2 GGUF: the model and its Engram tables
MTP_SHARDS, SIDECAR, DRAFTER = 7.9 * GB, 7.9 * GB, 4.2 * GB
VISION = 0.97 * GB                             # pictures: DeepSeek-V4.1-Flash-1017-Vision.gguf


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return 0


def partial(folder):
    """bytes of unfinished Hugging Face downloads under folder (they are resumed, not started again)"""
    return sum(size(p) for p in glob.glob(os.path.join(folder, ".cache", "huggingface", "download", "*.incomplete")))


def main():
    m = sys.argv[1]
    os.makedirs(m, exist_ok=True)
    steps = []   # (what, GB more while it runs, GB more once done), in the setup's order
    # an install from before 2026-10-27 (the slim copy + DeepSeek's Engram shards) needs no main model
    old = os.path.exists(os.path.join(m, "ds4-q2-slim.gguf")) and \
        os.path.exists(os.path.join(m, "engram", "model-00048-of-00048.safetensors"))
    # nor does a model Choose Model.bat picked that is there (model-choice.txt: an Unsloth GGUF carries its own Engram)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        with open(os.path.join(root, "model-choice.txt"), encoding="utf-8") as f:
            pick = f.readline().strip().strip('"')
    except OSError:
        pick = ""
    if pick and os.path.exists(pick if os.path.isabs(pick) else os.path.join(m, pick)):
        old = True
    # the drafter first (setup-ds41-models.bat builds it before the main model's download); none under 14 GB of VRAM
    if os.environ.get("DS41_NO_DRAFTER") != "1" and not os.path.exists(os.path.join(m, "V41-1017-MTP-iq2xxs.gguf")):
        if os.path.exists(os.path.join(m, "V41-1017-MTP.gguf")):
            steps.append(("drafter", DRAFTER, DRAFTER - SIDECAR))
        else:
            steps.append(("drafter (while it is built)", MTP_SHARDS + SIDECAR, DRAFTER))
    # pictures (on unless vision-choice.txt / DS41_VISION says off), before the main model as well
    try:
        with open(os.path.join(root, "vision-choice.txt"), encoding="utf-8") as f:
            vis = f.readline().strip().lower()
    except OSError:
        vis = ""
    vis = os.environ.get("DS41_VISION", "").strip().lower() or vis or "cpu"
    if vis in ("cpu", "gpu") and not os.path.exists(os.path.join(m, "DeepSeek-V4.1-Flash-1017-Vision.gguf.ok")):
        steps.append(("pictures (0.97 GB)", VISION, VISION))
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
