#!/usr/bin/env python3
"""ckpt.py - the checkpoint table (data/ds41/checkpoints.json), checkpoint fingerprints and the stamps on derived files.

Every entry point reads its checkpoint's file names from here: the launcher (Start DeepSeek.bat, ds41_local.py), the
model menu (pick_model.py), setup (setup-ds41-models.bat, first-start.bat, setup_space.py) and the converters
(v41_sidecar.py, extract_engram.py).  DS41_CHECKPOINT=<id> selects a row; otherwise the table's "default".

Derived files (drafter, rank, KEEP counts, vision, Engram tables / hot set, tokenizer export) carry a stamp beside them:
<file>.ckpt.json (for a folder: <folder>/ckpt.json) = {"checkpoint", "fingerprint", "role", ...}.  The bytes of the file
itself are not changed, so every reader of those formats (the engine included) is unaffected.  A file without a stamp
is recognised by its name ("match") or, for a GGUF, by its general.* metadata ("meta_match").

    ckpt.py bat [--id ID]                      NAME=VALUE lines for the batch files (CK_*; an empty value = none)
    ckpt.py mtp-shards INDEX.json [--id ID]    the shards holding the MTP (drafter) tensors, from the safetensors index
    ckpt.py shards INDEX.json PREFIX|/REGEX/   the shards holding the tensors whose names match
    ckpt.py stamp PATH --role ROLE [--id ID] [--from SRC]   write PATH's stamp (exit 0 even if it cannot: a warning)
    ckpt.py which PATH                         the checkpoint a file was built from: "<id> <fingerprint> <how>" or "unknown"
    ckpt.py check MODEL [ROLE=PATH ...]        warnings for derived files built from another checkpoint than MODEL
    ckpt.py imatrix-ok PATH [--id ID]          exit 1 when the importance matrix PATH is another checkpoint's (setup)
    ckpt.py fingerprint --repo R [--revision REV] | --expected FILE.json | --hashes FILE (name sha256 per line)
    ckpt.py key PATH [PATH ...]                the model key (R-6) of the files the engine loads
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = os.path.join(ROOT, "data", "ds41", "checkpoints.json")
STAMP_SUFFIX = ".ckpt.json"
DIR_STAMP = "ckpt.json"
SHARD_RE = re.compile(r"model-\d{5}-of-\d{5}\.safetensors")
SPLIT_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)
# roles whose files are compared by the tokenizer they were made from, not by the weights (the tokenizer is the same
# file across checkpoints so far: V4.1 -> 1017 kept it byte for byte)
TOKENIZER_ROLES = ("tokenizer",)

_cache: dict = {}


class TableError(ValueError):
    pass


def load(path: str | None = None) -> dict:
    """The table (cached per path).  TableError when it is missing or broken."""
    path = path or TABLE
    if path in _cache:
        return _cache[path]
    try:
        with open(path, encoding="utf-8-sig") as f:
            t = json.load(f)
    except (OSError, ValueError) as e:
        raise TableError(f"cannot read the checkpoint table {path}: {e}") from None
    rows = t.get("checkpoints") if isinstance(t, dict) else None
    if not isinstance(rows, dict) or not rows or t.get("default") not in rows:
        raise TableError(f"{path}: no 'checkpoints' rows, or 'default' is not one of them")
    t.setdefault("order", list(rows))
    _cache[path] = t
    return t


def active_id(table: dict | None = None, env=None) -> str:
    t = table or load()
    env = os.environ if env is None else env
    cid = (env.get("DS41_CHECKPOINT") or "").strip() or t["default"]
    if cid not in t["checkpoints"]:
        raise TableError(f"DS41_CHECKPOINT={cid!r} is not in the checkpoint table ({', '.join(t['checkpoints'])})")
    return cid


def row(cid: str | None = None, table: dict | None = None) -> dict:
    """The table row of checkpoint `cid` (the active one when None), with "id" added."""
    t = table or load()
    cid = cid or active_id(t)
    if cid not in t["checkpoints"]:
        raise TableError(f"no checkpoint {cid!r} in the table")
    return {**t["checkpoints"][cid], "id": cid}


def derived(role: str, cid: str | None = None, table: dict | None = None):
    """A derived file's name (or path relative to the repository) for checkpoint `cid`, or None."""
    return (row(cid, table).get("derived") or {}).get(role)


# ---- the safetensors index: which shards hold which tensors -------------------------------------------------------

def shards_with(index_path: str, test) -> list[str]:
    """The shard files (sorted, unique) whose tensors' names pass `test` (a prefix string or a callable)."""
    with open(index_path, encoding="utf-8-sig") as f:
        wm = json.load(f).get("weight_map")
    if not isinstance(wm, dict):
        raise ValueError(f"{index_path}: no weight_map")
    ok = test if callable(test) else (lambda k: k.startswith(test))
    return sorted({v for k, v in wm.items() if ok(k)})


def mtp_shards(index_path: str, cid: str | None = None, table: dict | None = None) -> list[str]:
    """The shards with the MTP blocks (the drafter's source), read from the index - not a fixed list."""
    prefix = (row(cid, table).get("deepseek") or {}).get("mtp_prefix") or "mtp."
    got = shards_with(index_path, prefix)
    if not got:
        raise ValueError(f"{index_path}: no tensor starts with {prefix!r}")
    return got


# ---- fingerprints -----------------------------------------------------------------------------------------------------

def fingerprint_of_hashes(hashes: dict) -> str:
    """The checkpoint fingerprint from {shard file name: sha256} (other names are ignored)."""
    sh = {os.path.basename(k): str(v).lower() for k, v in hashes.items() if SHARD_RE.fullmatch(os.path.basename(k))}
    if not sh:
        raise ValueError("no model-NNNNN-of-NNNNN.safetensors hashes")
    return hashlib.sha256("".join(f"{k} {sh[k]}\n" for k in sorted(sh)).encode()).hexdigest()[:16]


def fingerprint_of_expected(path: str) -> str:
    """From an expected-hash file: {"files": {name: {"sha256": ...}}} (e.g. a checkpoint's published shard hashes)."""
    with open(path, encoding="utf-8-sig") as f:
        files = json.load(f)["files"]
    return fingerprint_of_hashes({k: (v.get("sha256") if isinstance(v, dict) else v) for k, v in files.items()})


def fingerprint_of_hf(repo: str, revision: str | None = None) -> str:
    """From Hugging Face's published LFS hashes (one API call, nothing downloaded)."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    return fingerprint_of_hashes({s.rfilename: s.lfs.sha256 for s in info.siblings if getattr(s, "lfs", None)})


def by_fingerprint(fp: str, table: dict | None = None):
    t = table or load()
    return next((cid for cid, r in t["checkpoints"].items() if r.get("fingerprint") and r["fingerprint"] == fp), None)


# ---- stamps on derived files ------------------------------------------------------------------------------------------

def first_shard(path: str) -> str:
    """A split GGUF's first file (its stamp stands for the whole model)."""
    m = SPLIT_RE.search(path)
    return path[:m.start()] + f"-00001-of-{m.group(2)}.gguf" if m else path


def stamp_path(path: str) -> str:
    path = path.rstrip("/\\")
    return os.path.join(path, DIR_STAMP) if os.path.isdir(path) else first_shard(path) + STAMP_SUFFIX


def write_stamp(path: str, role: str, cid: str | None = None, source: str | None = None,
                table: dict | None = None) -> str:
    """Stamp `path` (a file or folder) as built from checkpoint `cid`.  -> the stamp's path."""
    r = row(cid, table)
    d = {"checkpoint": r["id"], "fingerprint": r.get("fingerprint"), "role": role,
         "file": os.path.basename(path.rstrip("/\\")),
         "stamped": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if role in TOKENIZER_ROLES:
        d["tokenizer_sha256"] = (r.get("deepseek") or {}).get("tokenizer_sha256")
    if source:
        d["from"] = source
    sp = stamp_path(path)
    tmp = sp + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
        f.write("\n")
    os.replace(tmp, sp)
    return sp


def source_checkpoint(source: str | None = None, table: dict | None = None, env=None):
    """The checkpoint a converter reads: DS41_CHECKPOINT, else the row whose DeepSeek repository `source` (or the
    V41_SRC_URL setup sets) names, else None."""
    t = table or load()
    env = os.environ if env is None else env
    cid = (env.get("DS41_CHECKPOINT") or "").strip()
    if cid:
        return cid if cid in t["checkpoints"] else None
    src = (source or env.get("V41_SRC_URL") or "").rstrip("/")
    for c in t["order"]:
        repo = ((t["checkpoints"][c].get("deepseek") or {}).get("repo") or "")
        if repo and (src == repo or src.endswith("/" + repo)):
            return c
    return None


def stamp_output(path: str, role: str, source: str | None = None) -> str | None:
    """For the converters: stamp what they wrote, under its final name (a trailing .part / .partial / .tmp, which
    setup renames away, is dropped).  Nothing when the checkpoint is not known (a note on stderr)."""
    final = re.sub(r"\.(part|partial|tmp)$", "", path)
    try:
        cid = source_checkpoint(source)
        if cid is None:
            print(f"ckpt: {os.path.basename(final)} not stamped: its checkpoint is unknown (set DS41_CHECKPOINT)",
                  file=sys.stderr)
            return None
        return write_stamp(final, role, cid, source or os.environ.get("V41_SRC_URL"))
    except (OSError, TableError) as e:
        print(f"ckpt: could not stamp {final}: {e}", file=sys.stderr)
        return None


def read_stamp(path: str):
    try:
        with open(stamp_path(path), encoding="utf-8-sig") as f:
            d = json.load(f)
        return d if isinstance(d, dict) and d.get("checkpoint") else None
    except (OSError, ValueError):
        return None


def _gguf_strings(path: str) -> list[str]:
    """The general.* string values of a GGUF's header ([] when it cannot be read)."""
    try:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serve")
        if here not in sys.path:
            sys.path.insert(0, here)
        import gguf_head
        kv = gguf_head.read_header(first_shard(path))[0]
    except Exception:  # noqa: BLE001  (unreadable: unknown, not an error)
        return []
    out = []
    for k, v in kv.items():
        if k.startswith("general."):
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, list):
                out += [x for x in v if isinstance(x, str)]
    return out


def which(path: str, table: dict | None = None, meta: bool = True):
    """(checkpoint id or None, fingerprint or None, how): the stamp, else the file name, else the GGUF metadata."""
    t = table or load()
    st = read_stamp(path)
    if st:
        cid = st["checkpoint"] if st["checkpoint"] in t["checkpoints"] else by_fingerprint(st.get("fingerprint") or "", t)
        return cid or st["checkpoint"], st.get("fingerprint"), "stamp"
    name = os.path.basename(path.rstrip("/\\"))
    for cid in t["order"]:
        r = t["checkpoints"][cid]
        if any(re.search(p, name, re.I) for p in r.get("match") or ()):
            return cid, r.get("fingerprint"), "name"
    if meta and name.lower().endswith(".gguf") and os.path.isfile(path):
        strings = _gguf_strings(path)
        for cid in t["order"]:
            r = t["checkpoints"][cid]
            if any(re.search(p, s, re.I) for p in r.get("meta_match") or () for s in strings):
                return cid, r.get("fingerprint"), "metadata"
    return None, None, "unknown"


def check(model: str, files: dict, table: dict | None = None) -> tuple[str | None, list[str]]:
    """(the model's checkpoint id or None, warnings): one warning per derived file in `files` ({role: path}) that was
    built from another checkpoint than `model`.  Unknown on either side: no warning (nothing to compare)."""
    t = table or load()
    mid, mfp, mhow = which(model, t)
    warns = []
    if mid is None:
        return None, warns
    mrow = t["checkpoints"].get(mid) or {}
    for role, p in files.items():
        if not p or not os.path.exists(p):
            continue
        did, dfp, dhow = which(p, t, meta=False)
        if did is None:
            continue
        if role in TOKENIZER_ROLES:
            st = read_stamp(p) or {}
            want = (mrow.get("deepseek") or {}).get("tokenizer_sha256")
            have = st.get("tokenizer_sha256") or ((t["checkpoints"].get(did) or {}).get("deepseek") or {}).get("tokenizer_sha256")
            if want and have and want != have:
                warns.append(f"{role} {p} was exported from another tokenizer ({did}) than the model's ({mid})")
            continue
        if did != mid or (dfp and mfp and dfp != mfp):
            warns.append(f"{role} {os.path.basename(p.rstrip('/'))} was built from checkpoint {did} "
                         f"({dfp or '?'}, by its {dhow}), the model {os.path.basename(model)} is {mid} "
                         f"({mfp or '?'}, by its {mhow})")
    return mid, warns


def imatrix_ok(path: str, cid: str | None = None, table: dict | None = None) -> tuple[bool, str]:
    """Setup's last check before it requantises a drafter with the importance matrix `path`: (False, why) when the
    file's stamp or name says it was made for another checkpoint than `cid` (the active one) - a drafter requantised
    with it would be stamped as `cid`'s and the mix would go unseen.  Unknown (no stamp, a name no row claims): its
    row named it, so it is taken."""
    t = table or load()
    cid = row(cid, t)["id"]
    did, _, how = which(path, t, meta=False)
    if did is not None and did != cid:
        return False, f"{os.path.basename(path)} is checkpoint {did}'s (by its {how}), not {cid}'s"
    return True, ""


# ---- the model key (R-6): what any state that depends on the loaded weights is keyed by --------------------------------

def _sample(path: str, h) -> None:
    size = os.path.getsize(path)
    h.update(f"{os.path.basename(path)}\0{size}\0".encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(size - (1 << 20))
            h.update(f.read(1 << 20))


def model_key(paths, fingerprint: str | None = None) -> str:
    """16 hex digits from each file's name, size and first and last MiB (a split GGUF: every part; a folder: its
    .safetensors / .gguf files), plus the checkpoint fingerprint.  Another quant, another checkpoint, a re-made file of
    the same name: another key.  Missing paths are skipped."""
    h = hashlib.sha256((fingerprint or "").encode() + b"\0")
    for p in paths:
        if not p:
            continue
        p = p.rstrip("/\\")
        if os.path.isdir(p):
            parts = sorted(os.path.join(p, f) for f in os.listdir(p) if f.endswith((".safetensors", ".gguf")))
        else:
            m = SPLIT_RE.search(p)
            parts = [p[:m.start()] + f"-{i:05d}-of-{m.group(2)}.gguf" for i in range(1, int(m.group(2)) + 1)] if m else [p]
        for q in parts:
            if os.path.isfile(q):
                _sample(q, h)
    return h.hexdigest()[:16]


# ---- the batch files' variables ---------------------------------------------------------------------------------------

def bat_vars(cid: str | None = None, table: dict | None = None) -> dict:
    """CK_* for the batch files: names of this checkpoint's files ("" = this checkpoint has none)."""
    r = row(cid, table)
    ds, mo, de = r.get("deepseek") or {}, r.get("models") or {}, r.get("derived") or {}
    q2 = mo.get("ds4_q2") or {}
    v = {
        "CK_ID": r["id"], "CK_FP": r.get("fingerprint") or "", "CK_LABEL": r.get("label") or r["id"],
        "CK_DS": ds.get("repo") or "", "CK_SMALL_DIR": ds.get("small_dir") or "deepseek",
        "CK_MTP_HINT": " ".join(ds.get("mtp_shards_hint") or ()),
        "CK_Q2": q2.get("file") or "", "CK_Q2_REPO": q2.get("repo") or "",
        "CK_DRAFT_SRC": de.get("drafter_source") or "", "CK_DRAFT_UNI": de.get("drafter_uniform") or "",
        "CK_DRAFT_IMX": de.get("drafter") or "", "CK_KEEP": de.get("keep") or "",
        "CK_VISION": de.get("vision") or "", "CK_VISION_REPO": de.get("vision_repo") or "",
        "CK_TOKENIZER": de.get("tokenizer_dir") or "tokenizer",
        # the drafter's importance matrix (a path in the repository; setup also looks for its file name beside the
        # engine build): "" = none for this checkpoint, the drafter's experts are requantised uniformly
        "CK_IMATRIX": (de.get("drafter_imatrix") or "").replace("/", "\\"),
    }
    for k, x in v.items():   # a value cmd would read as syntax: refuse it here, not in the middle of a batch file
        if any(c in x for c in '"%^&|<>()!\r\n'):
            raise TableError(f"{k}={x!r}: a character the batch files cannot carry")
    return v


def _arg(argv, name, default=None):
    if name in argv[:-1]:
        return argv[argv.index(name) + 1]
    return default


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    cmd = argv[0]
    try:
        if cmd == "bat":
            for k, x in bat_vars(_arg(argv, "--id")).items():
                print(f"{k}={x}")
            return 0
        if cmd == "mtp-shards":
            print(" ".join(mtp_shards(argv[1], _arg(argv, "--id"))))
            return 0
        if cmd == "shards":
            pat = argv[2]
            test = (lambda k, rx=re.compile(pat[1:-1]): bool(rx.search(k))) if len(pat) > 1 and pat[0] == pat[-1] == "/" else pat
            print(" ".join(shards_with(argv[1], test)))
            return 0
        if cmd == "stamp":
            try:
                print(f"stamped {write_stamp(argv[1], _arg(argv, '--role', 'derived'), _arg(argv, '--id'), _arg(argv, '--from'))}")
            except (OSError, TableError) as e:   # a stamp is a check, not a step setup can fail on
                print(f"ckpt.py: could not stamp {argv[1]}: {e}", file=sys.stderr)
            return 0
        if cmd == "which":
            cid, fp, how = which(argv[1])
            print(f"{cid} {fp or '-'} {how}" if cid else "unknown")
            return 0
        if cmd == "check":
            files = dict(a.split("=", 1) for a in argv[2:] if "=" in a)
            mid, warns = check(argv[1], files)
            for w in warns:
                print(f"WARNING: {w}")
            return 1 if warns else 0
        if cmd == "imatrix-ok":
            try:
                ok, why = imatrix_ok(argv[1], _arg(argv, "--id"))
            except (TableError, OSError) as e:   # the table cannot be read: the row's own name stands (as before)
                print(f"ckpt.py imatrix-ok: {e}", file=sys.stderr)
                return 0
            if not ok:
                print(f"{why} - the drafter is requantised without it", file=sys.stderr)
            return 0 if ok else 1
        if cmd == "fingerprint":
            if _arg(argv, "--expected"):
                print(fingerprint_of_expected(_arg(argv, "--expected")))
            elif _arg(argv, "--hashes"):
                with open(_arg(argv, "--hashes"), encoding="utf-8-sig") as f:
                    print(fingerprint_of_hashes(dict(ln.split()[:2] for ln in f if len(ln.split()) >= 2)))
            elif _arg(argv, "--repo"):
                print(fingerprint_of_hf(_arg(argv, "--repo"), _arg(argv, "--revision")))
            else:
                print("ckpt.py fingerprint: --repo R [--revision REV] | --expected FILE | --hashes FILE", file=sys.stderr)
                return 2
            return 0
        if cmd == "key":
            print(model_key(argv[1:]))
            return 0
    except (TableError, OSError, ValueError, KeyError, IndexError) as e:
        print(f"ckpt.py {cmd}: {e}", file=sys.stderr)
        return 1
    print(f"ckpt.py: unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
