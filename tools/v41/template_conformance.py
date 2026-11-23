#!/usr/bin/env python3
"""R-5: DeepSeek-V4.1 prompt conformance - DeepSeek's official chat_template.jinja against our encoder and our server.

Three renderings of one conversation are compared character for character and, when a tokenizer.json is found, token
for token:

  template  DeepSeek's chat_template.jinja (the HF repo's, commit 08c8657 for 1017; vendored in reference/), rendered
            the way transformers' apply_chat_template renders it (sandboxed Jinja, trim_blocks + lstrip_blocks, its
            tojson and raise_exception)
  encoder   tools/v41/dsv41_encoding.py (a copy of the repo's encoding/encoding.py) called as serve_ds41.py calls it
  serve     the server's own path: an OpenAI (or Anthropic) request -> serve/frontend.py -> serve_ds41.Ds41Template

The template and the encoder must agree on every case.  The server may differ only where a case names the reason
("known" in reference/conformance_cases.json, each one a deliberate choice of ours); a listed difference that has
gone away is also a failure, so the list stays true.  DeepSeek's own golden encodings (encoding/tests) are checked too.

  python tools/v41/template_conformance.py                    # the 1017 reference: every case, PASS / DIFF
  python tools/v41/template_conformance.py --template X.jinja # another template (a new checkpoint's)
  python tools/v41/template_conformance.py --day0 deepseek-ai/DeepSeek-V4.1-Flash-MMDD [--revision main]
      day 0: fetch the new repo's small files (no weights), diff them against 1017's (template, encoding.py, config,
      tokenizer, the DSML markers), then run every case with the NEW template against our encoder and server.
      Exit 0 = prompt format unchanged for us; 1 = a difference to port before serving (tools/v41/DAY0_TEMPLATE.md).

The new repo's encoding.py is only diffed as text, never imported or run; its template runs in Jinja's sandbox.
"""
import argparse
import copy
import difflib
import hashlib
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent                     # tools/v41
REFERENCE = HERE / "reference" / "dsv41-flash-1017"
CASES_FILE = HERE / "reference" / "conformance_cases.json"
OUR_ENCODING = HERE / "dsv41_encoding.py"
# 1017's tokenizer.json (sha256 c90dfa01...) for the token comparison: --tokenizer, else DS41_TOKENIZER_JSON, else the
# first one found of the copies setup-ds41-models.bat downloads into DS41_MODELS (ds41-paths.bat; default ./models)
TOKENIZER_SHA_1017 = "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b"


def tokenizer_candidates():
    models = Path(os.environ.get("DS41_MODELS") or HERE.parent.parent / "models")
    return (str(models / "deepseek" / "tokenizer.json"), str(models / "engram" / "tokenizer.json"))


BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"


def lf(text: str) -> str:
    return text.replace("\r\n", "\n")


def sha256(data) -> str:
    return hashlib.sha256(data if isinstance(data, bytes) else data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------- the renderers
def template_env():
    """Jinja as transformers' apply_chat_template sets it up (chat_template_utils._compile_jinja_template)."""
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    return env


def dict_arguments(messages):
    """Tool call arguments as dicts (the official template refuses JSON strings, as transformers expects)."""
    out = copy.deepcopy(messages)
    for m in out:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                except ValueError:
                    pass
    return out


def string_arguments(messages):
    """Tool call arguments as JSON text, as OpenAI clients send them."""
    out = copy.deepcopy(messages)
    for m in out:
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            if not isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.dumps(fn.get("arguments", {}), ensure_ascii=False)
    return out


class Renderers:
    def __init__(self, template_text: str):
        self.template = template_env().from_string(lf(template_text))
        self._E = self._D = None

    def official(self, case):
        kw = dict(case.get("kwargs") or {})
        return self.template.render(messages=dict_arguments(case["messages"]), tools=case.get("tools"),
                                    add_generation_prompt=True, bos_token=BOS, eos_token=EOS, **kw)

    @property
    def E(self):
        if self._E is None:
            sys.path.insert(0, str(HERE))
            import dsv41_encoding
            self._E = dsv41_encoding
        return self._E

    @property
    def D(self):
        if self._D is None:
            sys.path.insert(0, str(HERE / "serve"))
            import serve_ds41   # noqa: F401  (patches the server's modules: frontend + server, as the server runs)
            self._D = serve_ds41
        return self._D

    def encoder(self, case):
        """dsv41_encoding.encode_messages, given what the template's kwargs say (thinking, effort, drop_thinking)."""
        kw = case.get("kwargs") or {}
        mode = kw.get("thinking_mode")
        if mode is None:
            flag = kw.get("thinking", kw.get("enable_thinking"))
            mode = ("thinking" if flag else "chat") if flag is not None else \
                "chat" if kw.get("reasoning_effort") == "none" else "thinking"
        msgs = copy.deepcopy(case["messages"])
        tools = case.get("tools")
        if tools:   # as the template does: onto a leading system message, else a new empty one
            if msgs and msgs[0].get("role") == "system":
                if not msgs[0].get("tools"):
                    msgs[0] = dict(msgs[0], tools=tools)
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": tools})
        return self.E.encode_messages(msgs, thinking_mode=mode, drop_thinking=kw.get("drop_thinking", True),
                                      reasoning_effort=kw.get("reasoning_effort"))

    def _fresh_tool_order(self):
        F = self.D.F
        with F._TOOL_ORDERS_LOCK:    # first sight of every tool set: the request's own order (frontend._stable_tool_order)
            F._TOOL_ORDERS.clear()

    def serve(self, case):
        """An OpenAI Chat Completions request through the server's frontend and serve_ds41's template."""
        req = {"model": "x", "messages": string_arguments(case["messages"]), **(case.get("request") or {})}
        if case.get("tools"):
            req["tools"] = case["tools"]
        self._fresh_tool_order()
        messages, tools, kwargs = self.D.F.openai_to_messages(req)
        return self.D.Ds41Template().render(messages, tools=tools, add_generation_prompt=True, **kwargs)

    def anthropic(self, case):
        """The case's Anthropic Messages request through the server's frontend and serve_ds41's template."""
        req = {"model": "x", "max_tokens": 1000, **case["anthropic"]}
        self._fresh_tool_order()
        messages, tools, kwargs = self.D.F.anthropic_to_messages(req)
        return self.D.Ds41Template().render(messages, tools=tools, add_generation_prompt=True, **kwargs)


# ---------------------------------------------------------------------------------------------------- comparison
class Tokens:
    def __init__(self, path=None):
        self.tok, self.path = None, None
        paths = [path] if path else [os.environ.get("DS41_TOKENIZER_JSON")] + list(tokenizer_candidates())
        for p in paths:
            if p and Path(p).is_file():
                try:
                    from tokenizers import Tokenizer
                except ImportError:
                    return
                self.tok, self.path = Tokenizer.from_file(str(p)), str(p)
                return

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids if self.tok else None

    def piece(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False) if self.tok else ""


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def describe(a: str, b: str, tokens: Tokens, la="template", lb="other") -> str:
    """Where two renderings part: the character offset with context, and the first differing token."""
    i = first_diff(a, b)
    lo = max(0, i - 60)
    lines = [f"    first difference at char {i} (lengths {len(a)} / {len(b)})",
             f"    {la:>8}: {a[lo:i + 80]!r}", f"    {lb:>8}: {b[lo:i + 80]!r}"]
    ta, tb = tokens.ids(a), tokens.ids(b)
    if ta is not None:
        k = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
        lines.append(f"    tokens {len(ta)} / {len(tb)}; first differing token #{k}: "
                     f"{ta[k:k + 4]} {tokens.piece(ta[k:k + 4])!r} vs {tb[k:k + 4]} {tokens.piece(tb[k:k + 4])!r}")
    return "\n".join(lines)


def same(a: str, b: str, tokens: Tokens) -> bool:
    if a != b:
        return False
    return tokens.ids(a) == tokens.ids(b) if tokens.tok else True


def load_cases(path=CASES_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["cases"]


def golden_cases(folder: Path):
    """DeepSeek's own encoding/tests (test_input_N.json -> test_output_N.txt), read as encoding.load_cases reads them
    and encoded as test_encoding.py does (thinking_mode "chat" unless the case names one)."""
    out = []
    for inp in sorted(folder.glob("test_input_*.json"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        want = folder / inp.name.replace("test_input_", "test_output_").replace(".json", ".txt")
        if not want.is_file():
            continue
        data = json.loads(inp.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = [data]
        elif data and isinstance(data[0], dict) and "role" in data[0]:
            data = [{"messages": data}]
        case = data[0]
        messages = copy.deepcopy(case["messages"])
        if "tools" in case:
            messages[0]["tools"] = case["tools"]
        kw = {"thinking_mode": case.get("thinking_mode") or "chat"}
        if case.get("reasoning_effort") is not None:
            kw["reasoning_effort"] = case["reasoning_effort"]
        out.append({"name": f"golden {inp.stem}", "messages": messages, "kwargs": kw,
                    "want": lf(want.read_bytes().decode("utf-8"))})
    return out


def run(template_text: str, tokens: Tokens, cases=None, golden_dir=None, serve=True, verbose=False, log=print):
    """Every case through every rendering it names.  Returns (failures, lines checked)."""
    R = Renderers(template_text)
    cases = load_cases() if cases is None else cases
    fails, checked = [], 0

    def check(name, ok, detail=""):
        nonlocal checked
        checked += 1
        if not ok:
            fails.append(name)
        if not ok or verbose:
            log(("PASS " if ok else "DIFF ") + name + (("\n" + detail) if detail and not ok else ""))

    for case in cases:
        name, known = case["name"], case.get("known") or {}
        try:
            ref = R.official(case)
        except Exception as e:   # noqa: BLE001 - a template refusal is a result to report, not a crash
            check(f"{name}: template renders", False, f"    {type(e).__name__}: {e}")
            continue
        paths = [("encoder", R.encoder)] if case.get("encoder", True) else []
        if serve and case.get("serve", True):
            paths.append(("serve", R.serve))
        if serve and case.get("anthropic"):
            paths.append(("anthropic", R.anthropic))
        for label, fn in paths:
            try:
                got = fn(case)
            except Exception as e:   # noqa: BLE001
                check(f"{name}: {label}", False, f"    {type(e).__name__}: {e}")
                continue
            ok = same(ref, got, tokens)
            if label in known:
                check(f"{name}: {label} differs as noted ({known[label]})", not ok,
                      "    the noted difference is gone: remove it from the case's \"known\"")
            else:
                check(f"{name}: {label} == template", ok, "" if ok else describe(ref, got, tokens, "template", label))
    for g in golden_cases(golden_dir) if golden_dir else []:
        for label, fn in (("template", R.official), ("encoder", R.encoder)):
            try:
                got = fn(g)
            except Exception as e:   # noqa: BLE001
                check(f"{g['name']}: {label}", False, f"    {type(e).__name__}: {e}")
                continue
            check(f"{g['name']}: {label} == DeepSeek's golden output", same(g["want"], got, tokens),
                  describe(g["want"], got, tokens, "golden", label))
    return fails, checked


# ---------------------------------------------------------------------------------------------------- day 0
SMALL_FILES = ("chat_template.jinja", "config.json", "tokenizer_config.json", "tokenizer.json", "generation_config.json",
               "model.safetensors.index.json", "README.md")
SMALL_DIRS = ("encoding/", "inference/")
SMALL_MAX = 64 << 20


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "dsv41-template-conformance"})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_small(repo: str, revision: str, out: Path, log=print) -> Path:
    """The repo's small files (no weights) into `out`: the template, config, tokenizer, index, encoding/, inference/."""
    api = f"https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision, safe='')}"
    info = json.loads(_get(api))
    sha = info.get("sha", revision)
    sizes = {}
    for s in info.get("siblings", []):
        sizes[s["rfilename"]] = s.get("size")
    names = [n for n in sizes if (n in SMALL_FILES or n.startswith(SMALL_DIRS)) and not n.endswith(
        (".safetensors", ".jpeg", ".jpg", ".png"))]
    out.mkdir(parents=True, exist_ok=True)
    (out / "SOURCE.json").write_text(json.dumps({"repo": repo, "revision": revision, "sha": sha,
                                                 "lastModified": info.get("lastModified"), "fetched": datetime.now().isoformat(timespec="seconds"),
                                                 "files": sorted(names)}, indent=1), encoding="utf-8")
    for n in sorted(names):
        dst = out / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = _get(f"https://huggingface.co/{repo}/resolve/{sha}/{urllib.parse.quote(n)}")
        if len(data) > SMALL_MAX:
            raise RuntimeError(f"{n}: {len(data)} bytes, not a small file")
        dst.write_bytes(data)
        log(f"  fetched {n} ({len(data):,} B)")
    log(f"  {repo} @ {sha[:12]} -> {out}")
    return out


SPECIAL_RE = re.compile(r"<｜[^｜<>\n]{1,40}｜>")
DSML_RE = re.compile(r"｜DSML｜ ?[a-z_]+")


def markers(template_text: str) -> dict:
    """What the prompt format is made of: special tokens and DSML tag names written in the template."""
    t = lf(template_text)
    return {"special": sorted(set(SPECIAL_RE.findall(t))),
            "dsml": sorted(set(m for m in DSML_RE.findall(t))),
            "effort_line": sorted(set(re.findall(r"Reasoning Effort: [^\"]*", t))),
            "think": sorted(set(re.findall(r"</?think>", t)))}


def our_markers(E) -> dict:
    """The same, as our encoder writes them."""
    specials = {E.bos_token, E.eos_token, E.USER_SP_TOKEN, E.ASSISTANT_SP_TOKEN, E.LATEST_REMINDER_SP_TOKEN,
                E.SYSTEM_SP_TOKEN, E.IMAGE_PLACEHOLDER, *E.DS_TASK_SP_TOKENS.values()}
    dsml = {f"{E.dsml_token}{n}" for n in (E.tool_calls_block_name, E.tool_call_tag_name, E.tool_parameter_tag_name)}
    return {"special": sorted(specials), "dsml": sorted(dsml)}


def text_diff(a: str, b: str, la: str, lb: str) -> str:
    return "".join(difflib.unified_diff(lf(a).splitlines(True), lf(b).splitlines(True), la, lb))


def day0(folder: Path, tokens_path=None, log=print) -> int:
    """Compare a checkpoint's small files (already in `folder`) with 1017's and run every case with its template."""
    report = folder / "day0_report"
    report.mkdir(exist_ok=True)
    verdict = []

    def read(p: Path):
        return p.read_bytes().decode("utf-8") if p.is_file() else None

    # 1. hashes: what changed at all
    log("== files against 1017 ==")
    pairs = [("chat_template.jinja", REFERENCE / "chat_template.jinja"), ("config.json", REFERENCE / "config.json"),
             ("tokenizer_config.json", REFERENCE / "tokenizer_config.json"), ("encoding/encoding.py", OUR_ENCODING)]
    for name, ref in pairs:
        new = read(folder / name)
        if new is None:
            log(f"  {name:<24} MISSING in the new repo")
            verdict.append(f"{name} missing")
            continue
        old = read(ref)
        if lf(new) == lf(old):
            log(f"  {name:<24} same as 1017")
            continue
        d = text_diff(old, new, f"1017/{name}", f"new/{name}")
        (report / (name.replace("/", "_") + ".diff")).write_text(d, encoding="utf-8")
        log(f"  {name:<24} CHANGED ({d.count(chr(10))} diff lines -> day0_report/{name.replace('/', '_')}.diff)")
        verdict.append(f"{name} changed")
    tj = folder / "tokenizer.json"
    if tj.is_file():
        h = sha256(tj.read_bytes())
        log(f"  {'tokenizer.json':<24} {'same as 1017' if h == TOKENIZER_SHA_1017 else 'CHANGED (sha256 ' + h[:16] + ')'}")
        if h != TOKENIZER_SHA_1017:
            verdict.append("tokenizer.json changed")
            try:
                new_added = {t["content"]: t["id"] for t in json.loads(tj.read_text(encoding="utf-8"))["added_tokens"]}
                old_path = Tokens().path
                if old_path:
                    old_added = {t["content"]: t["id"] for t in json.loads(Path(old_path).read_text(encoding="utf-8"))["added_tokens"]}
                    moved = sorted(k for k in set(new_added) | set(old_added) if new_added.get(k) != old_added.get(k))
                    log(f"    added/special tokens that differ (content: 1017 id -> new id): "
                        + (", ".join(f"{k!r}: {old_added.get(k)} -> {new_added.get(k)}" for k in moved[:20]) or "none"))
            except Exception as e:   # noqa: BLE001
                log(f"    could not compare added tokens: {e}")
    # 2. the format's markers (the V4 -> V4.1 DSML rename is what vLLM missed for ~12 days)
    tmpl = read(folder / "chat_template.jinja")
    if tmpl is not None:
        sys.path.insert(0, str(HERE))
        import dsv41_encoding as E
        new_m, old_m, ours = markers(tmpl), markers(read(REFERENCE / "chat_template.jinja")), our_markers(E)
        log("== format markers ==")
        for k in new_m:
            if new_m[k] != old_m[k]:
                log(f"  {k}: CHANGED\n    1017: {old_m[k]}\n    new:  {new_m[k]}")
                verdict.append(f"markers {k} changed")
            else:
                log(f"  {k}: same")
        missing = sorted(set(new_m["dsml"]) - set(ours["dsml"])) + sorted(set(new_m["special"]) - set(ours["special"]))
        if missing:
            log(f"  written by the new template, unknown to dsv41_encoding.py / serve_ds41.py's parser: {missing}")
            verdict.append("unknown markers")
    # 3. every case with the new template against our encoder and server; DeepSeek's new goldens through both
    fails = []
    if tmpl is not None:
        log("== conformance: the new template against our encoder and server ==")
        tokens = Tokens(str(tj) if tj.is_file() else tokens_path)
        log(f"  tokenizer: {tokens.path or 'none found (characters only)'}")
        gold = folder / "encoding" / "tests"
        fails, n = run(tmpl, tokens, golden_dir=gold if gold.is_dir() else None, log=lambda s: log("  " + s))
        log(f"  {n - len(fails)}/{n} checks pass")
        if fails:
            verdict.append(f"{len(fails)} conformance differences")
    log("== verdict ==")
    if not verdict:
        log("  PASS: the prompt format is unchanged for us (scenario A for the template); serve as 1017")
        return 0
    log("  ACTION: " + "; ".join(verdict))
    log("  next: tools/v41/DAY0_TEMPLATE.md (port encoding.py into dsv41_encoding.py, re-run this, then the serve tests)")
    return 1


# ---------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=REFERENCE / "chat_template.jinja")
    ap.add_argument("--tokenizer", help="tokenizer.json for the token comparison (default: DS41_TOKENIZER_JSON, else "
                    "$DS41_MODELS/deepseek/tokenizer.json, if found)")
    ap.add_argument("--no-serve", action="store_true", help="template vs encoder only (no server import)")
    ap.add_argument("--day0", metavar="REPO", help="fetch REPO's small files from Hugging Face and diff (see above)")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--local", type=Path, metavar="DIR", help="day 0 on files already in DIR (no download)")
    ap.add_argument("--out", type=Path, help="where --day0 puts the files (default ./day0-<repo name>)")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if a.day0 or a.local:
        folder = a.local
        if a.day0:
            folder = a.out or Path.cwd() / ("day0-" + a.day0.split("/")[-1])
            fetch_small(a.day0, a.revision, folder)
        return day0(folder, a.tokenizer)
    tokens = Tokens(a.tokenizer)
    print(f"template {a.template} (sha256 {sha256(a.template.read_bytes())[:16]}); "
          f"tokenizer {tokens.path or 'none found: characters only'}")
    fails, n = run(a.template.read_bytes().decode("utf-8"), tokens, golden_dir=REFERENCE / "encoding_tests",
                   serve=not a.no_serve, verbose=a.verbose)
    print(f"{n - len(fails)}/{n} checks pass" + (f"; {len(fails)} DIFF" if fails else ""))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
