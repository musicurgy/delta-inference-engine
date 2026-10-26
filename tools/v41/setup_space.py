#!/usr/bin/env python3
"""setup_space.py MODEL_FOLDER - will what setup-ds41-models.bat still has to do fit on that drive?

The steps that are not done yet run one after another; each leaves some files and needs more while it runs (the main
model: the 366 GB download AND its 163 GB copy, then the download is removed).  The peak over that sequence (a download
in progress counts as far as it got) is compared with the drive's free space.  Exit 0 if it fits, 1 with an explanation
if not - so the setup stops before a download fills the drive (the Hugging Face downloader only warns).
DS41_SKIP_SPACE_CHECK=1 skips the check."""
import glob
import os
import shutil
import sys

GB = 1e9
MARGIN = 5 * GB
Q2, SLIM = 366.0 * GB, 163.0 * GB              # the ds4 Q2 download, then the copy without its Engram tables
SHARD = {"model-00047-of-00048.safetensors": 101.6 * GB, "model-00048-of-00048.safetensors": 101.6 * GB}
MTP_SHARDS, SIDECAR, DRAFTER = 7.9 * GB, 7.9 * GB, 4.2 * GB


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
    if not os.path.exists(os.path.join(m, "ds4-q2-slim.gguf")):
        dl = os.path.join(m, "download")
        got = size(os.path.join(dl, "DeepSeek-V4.1-Flash-1017-Q2.gguf")) + partial(dl)
        part = size(os.path.join(m, "ds4-q2-slim.gguf.part"))
        steps.append(("main model (366 GB download + its 163 GB copy; the download is then removed)",
                      max(0.0, Q2 - got) + max(0.0, SLIM - part), SLIM - part - got))
    eng = os.path.join(m, "engram")
    miss = [s for s in SHARD if not os.path.exists(os.path.join(eng, s))]
    if miss:
        n = max(0.0, sum(SHARD[s] for s in miss) - partial(eng))
        steps.append(("Engram", n, n))
    if not os.path.exists(os.path.join(m, "V41-1017-MTP-iq2xxs.gguf")):
        if os.path.exists(os.path.join(m, "V41-1017-MTP.gguf")):
            steps.append(("drafter", DRAFTER, DRAFTER - SIDECAR))
        else:
            steps.append(("drafter (while it is built)", MTP_SHARDS + SIDECAR, DRAFTER))
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
