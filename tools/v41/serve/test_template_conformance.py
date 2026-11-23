#!/usr/bin/env python3
"""R-5 tests (CPU, seconds): DeepSeek's official chat_template.jinja (1017, vendored in tools/v41/reference/) against
dsv41_encoding.py and the server's request path, on every case in reference/conformance_cases.json and on DeepSeek's
own golden encodings.  Fails on any difference that a case does not name as deliberate, and on a named one that is
gone.  Also checks that the harness itself sees a changed template, and the day-0 diff on a local copy of 1017's files.

  python tools/v41/serve/test_template_conformance.py      (Xolotl's .venv: jinja2; tokenizers for token ids)
"""
import io
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import template_conformance as T  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + name + ("" if ok else "  " + str(detail)[:2000]))
    if not ok:
        FAILS.append(name)


REF_TEMPLATE = (T.REFERENCE / "chat_template.jinja").read_bytes().decode("utf-8")


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


def t_reference():
    tokens = T.Tokens()
    print(f"  tokenizer: {tokens.path or 'none found (characters only)'}")
    log = []
    fails, n = quiet(T.run, REF_TEMPLATE, tokens, golden_dir=T.REFERENCE / "encoding_tests", log=log.append)[0]
    check(f"1017 template == encoder == server on every case ({n} checks)", not fails, "\n".join(log))
    check("the cases cover every rendering", n >= 90, n)


def t_template_sha():
    check("vendored template is DeepSeek's 1017 commit 08c8657 (sha256 of its LF text)",
          T.sha256(T.lf(REF_TEMPLATE)) == "95b7033a693b321ceaefb6f89ad07d1e0cf2b095305711f934a4ac649456739d")


def t_sees_changes():
    """The harness must fail on the kinds of change a new checkpoint can bring."""
    mutations = {
        "V4-style DSML tag names (what vLLM missed for ~12 days)": ('{%- set DSML = "｜DSML｜" -%}', '{%- set DSML = "｜DSML｜tool_" -%}'),
        "a different effort sentence": ('" (range 1-100, the higher', '" (range 1-100; the higher'),
        "tool results joined differently": ('"<tool_result>" ~ tc ~ "</tool_result>"', '"<tool_result>\\n" ~ tc ~ "\\n</tool_result>"'),
        "default effort high -> 80": ('"high": 75, "xhigh"', '"high": 80, "xhigh"'),
    }
    for what, (old, new) in mutations.items():
        assert old in REF_TEMPLATE, what
        fails, n = quiet(T.run, REF_TEMPLATE.replace(old, new), T.Tokens(), golden_dir=T.REFERENCE / "encoding_tests",
                         log=lambda s: None)[0]
        check(f"a changed template is caught: {what} ({len(fails)} of {n} checks differ)", len(fails) > 0)


def t_markers():
    sys.path.insert(0, str(HERE.parent))
    import dsv41_encoding as E
    m, ours = T.markers(REF_TEMPLATE), T.our_markers(E)
    check("template's DSML tags are the encoder's", set(m["dsml"]) == set(ours["dsml"]), (m["dsml"], ours["dsml"]))
    check("template's special tokens are known to the encoder", set(m["special"]) <= set(ours["special"]),
          sorted(set(m["special"]) - set(ours["special"])))


def _day0_folder(root: Path, template: str) -> Path:
    d = root / "repo"
    (d / "encoding" / "tests").mkdir(parents=True)
    (d / "chat_template.jinja").write_text(template, encoding="utf-8", newline="")
    for n in ("config.json", "tokenizer_config.json"):
        shutil.copy(T.REFERENCE / n, d / n)
    shutil.copy(T.OUR_ENCODING, d / "encoding" / "encoding.py")
    for f in (T.REFERENCE / "encoding_tests").iterdir():
        shutil.copy(f, d / "encoding" / "tests" / f.name)
    return d


def t_day0_local():
    with tempfile.TemporaryDirectory() as tmp:
        rc, out = quiet(T.day0, _day0_folder(Path(tmp) / "same", REF_TEMPLATE), None, print)
        check("day 0 on 1017's own files: PASS (exit 0)", rc == 0 and "PASS" in out, out[-1500:])
        changed = REF_TEMPLATE.replace('{%- set DSML = "｜DSML｜" -%}', '{%- set DSML = "｜DSML｜tool_" -%}')
        changed = changed.replace("<｜DSML｜ calls>", "<｜DSML｜tool_calls>")
        rc, out = quiet(T.day0, _day0_folder(Path(tmp) / "new", changed), None, print)
        check("day 0 on a DSML rename: exit 1, template CHANGED, markers flagged",
              rc == 1 and re.search(r"chat_template\.jinja +CHANGED", out) and "dsml: CHANGED" in out
              and "unknown markers" in out, out[-2500:])


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    t_template_sha()
    t_markers()
    t_reference()
    t_sees_changes()
    t_day0_local()
    print(f"{'FAILED ' + str(len(FAILS)) if FAILS else 'all passed'}")
    sys.exit(1 if FAILS else 0)
