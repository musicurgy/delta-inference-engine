#!/usr/bin/env python3
"""Strata's API server (~/AI/Strata/serve/server.py: OpenAI + Anthropic endpoints, streaming, sessions, llama.cpp-style
metrics) in front of the DeepSeek-V4.1-Flash engine (`ds41_generate --serve`).  The server is Qwen3.8's; this shim
swaps in V4.1's pieces (same shape as ~/AI/Strata-GLM/tools/glm/serve/serve_glm.py):

- tokenizer: the GGUF's BPE (tokenizer dir from export_tokenizer.py) with llama.cpp's `joyai-llm` pre-split
  (Strata-DS4's strata_tokenizer.py; == HF tokenizers on 465,645 tokens)
- chat template: DeepSeek's own encoder (dsv41_encoding.encode_messages), not Jinja
- reasoning effort: V4.1's numeric 1-100 ("Reasoning Effort: N" system prefix).  Names as DwarfStar's server
  (antirez/ds4 ds4_server.c parse_reasoning_effort_name + ds4_deepseek41_reasoning_effort_text): high / xhigh = 75,
  max = 100, a number 1-100 as given, none / 0 = thinking off; low = 50 (DeepSeek's encoder), medium = 62
- end of turn: <｜end▁of▁sentence｜>
- tool calls: DSML (`<｜DSML｜ invoke name="X">` + `<｜DSML｜ parameter name="K" string="true|false">V</｜DSML｜ parameter>`,
  note the LEADING SPACE in the tag names) parsed as DeepSeek's parse_tool_calls does; the `calls` wrapper is
  stripped from the stream.  A value may carry DwarfStar's escape of its own closing tag (ds4_tool_text.h): undone.

  serve_ds41.py --engine strata --config strata-ds41.json --port 8150
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V41 = HERE.parent
# xolotl: the server is THIS tree's serve/ (it used to import ~/AI/Strata's, a separate upstream checkout at 0.1.39)
STRATA = V41.parent.parent
sys.path.insert(0, str(STRATA))
sys.path.insert(0, str(STRATA / "tools"))
sys.path.insert(0, str(V41.parent))          # tools/: strata_tokenizer.py with the joyai-llm pre-split
sys.path.insert(0, str(V41))

import strata_tokenizer as ST  # noqa: E402
assert "joyai-llm" in getattr(ST, "PRE_PATTERNS", {}), ST.__file__
import dsv41_encoding as E  # noqa: E402
from serve import frontend as F  # noqa: E402
from serve import server as S  # noqa: E402

# ---- tokenizer: the server builds ST.Tokenizer(tokens, merges, types); give it V4.1's pre-split
_Tok = ST.Tokenizer


class Ds41Tokenizer(_Tok):
    def __init__(self, tokens, merges, token_types=None, pre="joyai-llm", **k):
        super().__init__(tokens, merges, token_types, pre="joyai-llm", **k)
        self._literal = {i: t.encode("utf-8") for t, i in self.special_tokens.items()}

    def token_bytes(self, i: int) -> bytes:
        # V4.1's special tokens (<｜User｜>, <｜end▁of▁sentence｜>, ...) are not in the byte alphabet: they decode to
        # their own text, as llama.cpp detokenizes control / user-defined tokens (Qwen's are ASCII, so never hit this)
        lit = self._literal.get(i)
        return lit if lit is not None else super().token_bytes(i)


ST.Tokenizer = Ds41Tokenizer

# ---- reasoning effort: client spellings -> template kwargs (reasoning_effort = an int 1-100, or thinking off)
EFFORT = {"none": 0, "off": 0, "disabled": 0, "false": 0, "minimal": 50, "low": 50, "medium": 62, "high": 75,
          "xhigh": 75, "max": 100, "maximum": 100}


def effort_kwargs(value) -> dict:
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key.isdigit() and 0 <= int(key) <= 100:
        level = int(key)
    elif key in EFFORT:
        level = EFFORT[key]
    else:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium, high, max or 1-100")
    return {"enable_thinking": False} if level == 0 else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget: under 2K low, under 8K medium, else high (as the server's own)."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": 50 if n < 2048 else 62 if n < 8192 else 75}


for mod in (F, S):
    if hasattr(mod, "effort_kwargs"):
        mod.effort_kwargs = effort_kwargs
    if hasattr(mod, "budget_effort"):
        mod.budget_effort = budget_effort


def _level(effort):
    """Whatever reaches the template (ours, or the server's own names) -> 1..100, or None for the default (75)."""
    if effort is None or effort == "":
        return None
    if isinstance(effort, int):
        return max(1, min(100, effort))
    key = str(effort).strip().lower()
    if key.isdigit():
        return max(1, min(100, int(key)))
    return EFFORT.get(key) or None


# ---- chat template: DeepSeek's encoder
GEN_SUFFIXES = (E.ASSISTANT_SP_TOKEN + E.thinking_start_token, E.ASSISTANT_SP_TOKEN + E.thinking_end_token)


def _text(content):
    if content is None or isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "input_text", None))


def to_encoder_messages(messages, tools):
    out = []
    for m in messages:
        m = dict(m)
        content = m.get("content")
        if isinstance(content, list) and any(p.get("type") in ("image", "image_url", "input_image")
                                              for p in content if isinstance(p, dict)):
            m["content"] = [{**p, "type": "image_url" if p.get("type") == "input_image" else
                            "text" if p.get("type") == "input_text" else p.get("type")}
                            for p in content if isinstance(p, dict)]
        else:
            m["content"] = _text(content)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = dict(c.get("function") or {})
                fn.setdefault("arguments", "{}")
                calls.append({"type": "function", "id": c.get("id"), "function": fn})
            m["tool_calls"] = calls
        out.append(m)
    if tools:   # the server passes flat {"name", "description", "parameters"}; the encoder wants OpenAI's wrapper
        tools = [t if "function" in t else {"type": "function", "function": t} for t in tools]
        if out and out[0].get("role") == "system":
            out[0] = dict(out[0], tools=tools)
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools})
    return out


_ChatTemplate = F.ChatTemplate   # the server's own, kept for its capability probe


class Ds41Template:
    """Stands in for F.ChatTemplate: render(messages, tools, add_generation_prompt, **kwargs) -> prompt text."""
    source = "{# DeepSeek-V4.1: rendered by serve_ds41.py with DeepSeek's encoder (dsv41_encoding.py) #}"

    def __init__(self, path=None):
        self._caps = None

    @property
    def caps(self):
        """xolotl: the 0.1.44 server reports the template's capabilities (props: chat_template_caps), detected by
        F.ChatTemplate._detect_caps through render() and the tool-call parser - both DeepSeek's here."""
        if self._caps is None:
            try:
                self._caps = _ChatTemplate._detect_caps(self)
            except Exception:   # noqa: BLE001 - a hint only (as in the server): never fail a request over it
                self._caps = {"supports_tools": True, "supports_tool_calls": True, "supports_system_role": True,
                              "supports_parallel_tool_calls": True, "supports_preserve_reasoning": False}
        return self._caps

    def render(self, messages, tools=None, add_generation_prompt=True, **kw):
        mode = "chat" if kw.get("enable_thinking") is False else "thinking"
        prompt = E.encode_messages(to_encoder_messages(messages, tools), thinking_mode=mode,
                                   reasoning_effort=_level(kw.get("reasoning_effort")))
        if not add_generation_prompt:
            for s in GEN_SUFFIXES:
                if prompt.endswith(s):
                    return prompt[:-len(s)]
        return prompt


F.ChatTemplate = Ds41Template
S.ChatTemplate = Ds41Template

# ---- end of turn
S.IM_END = E.eos_token
_svc_init = S.Service.__init__


def _svc_init_ds41(self, engine, tokenizer, template, *a, **k):
    _svc_init(self, engine, tokenizer, template, *a, **k)
    # REPLACE, not add to: the server also encodes Qwen's "<|endoftext|>", which V4.1's vocabulary splits into plain
    # pieces ("<", "|", ...) - every DSML tag would then end the reply
    eos = tokenizer.encode(E.eos_token, parse_special=True)
    assert len(eos) == 1, eos
    self.stop_ids = set(eos)
    self.effort_end = False      # #458's tail-turn trick is Qwen text; V4.1's effort lives in the first system turn
    self.literals = {**self.literals, E.IMAGE_PLACEHOLDER: "\U000f0e20"}


S.Service.__init__ = _svc_init_ds41


# ---- the dashboard's routing note (Service.mode_note): the config's decode biases until a request names its own
def _config_routing():
    """(route_bias, route_bias_file, protect, pool) from the --config the server was started with; the engine's
    defaults (1.0 / 0.6, no smart router) for what it doesn't set."""
    try:
        cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    except (ValueError, IndexError, OSError):
        return 1.0, 0.6, 0, 0
    args, env = [str(x) for x in cfg.get("args", [])], cfg.get("env") or {}
    def flag(name, dflt):
        try:
            return float(args[args.index(name) + 1])
        except (ValueError, IndexError):
            return dflt
    def rank(name):
        try:
            return max(0, int(env.get(name, 0)))
        except (TypeError, ValueError):
            return 0
    return flag("--route-bias", 1.0), flag("--route-bias-file", 0.6), rank("STRATA_RB_PROTECT"), rank("STRATA_RB_POOL")


_ROUTING = _config_routing()


def _routing_note(rb, rbf, protect, pool):
    if protect or pool:
        return f"smart routing (top {protect}, pool {pool}, {rb:g} / {rbf:g})"
    if rb == 0 and rbf == 0:
        return "exact routing"
    for name, (a, b) in (("light", (0.25, 0.15)), ("fast", (0.5, 0.3))):
        if abs(rb - a) < 1e-9 and abs(rbf - b) < 1e-9:
            return f"{name} routing ({rb:g} / {rbf:g})"
    return f"route bias {rb:g} / {rbf:g}"


_svc_init_routing = S.Service.__init__


def _svc_init_note(self, *a, **k):
    _svc_init_routing(self, *a, **k)
    self.mode_note = _routing_note(*_ROUTING)


S.Service.__init__ = _svc_init_note


def _expand_images_ds41(self, ids, encoded):
    marker = self.tok.encode(E.IMAGE_PLACEHOLDER, parse_special=True)
    if len(marker) != 1:
        raise ValueError("the tokenizer must encode the image marker as one token")
    out, k = [], 0
    for token in ids:
        if token == marker[0]:
            if k == len(encoded):
                raise ValueError("the prompt and its images do not match")
            out.extend([token] * encoded[k][1])
            k += 1
        else:
            out.append(token)
    if k != len(encoded):
        raise ValueError("the prompt and its images do not match")
    return out


S.Service.expand_image_tokens = _expand_images_ds41
_Vision = S.Vision
DETAIL_LOW_TOKENS = 256


class Ds41Vision(_Vision):
    """Run the reference-based encoder with the server's existing cache and ENC protocol."""

    def __init__(self, cfg, log=None, env=None, lazy=False):
        super().__init__(cfg, log=log, env=env, lazy=True)
        args, log, env = self.spawn
        args = [cfg.get("python") or sys.executable, str(V41 / "vision" / "ds41_vision.py"), *args[1:]]
        if cfg.get("reference_dir"):
            args += ["--reference-dir", os.path.abspath(cfg["reference_dir"])]
        if cfg.get("dtype"):     # "bf16" on a GPU: half the weights' VRAM (DeepSeek's reference runs the ViT in bf16)
            args += ["--dtype", str(cfg["dtype"])]
        self.spawn = (args, log, env)
        # OpenAI's detail "low": at most DETAIL_LOW_TOKENS per picture instead of up to 1,024 (a 1024x701 picture: 230
        # tokens instead of 444, read in 5.0 instead of 11.5 s on 6 CPU threads; 11-07).  Still above the model's
        # minimum picture area (295,936 px = 184 tokens for a square)
        top = int(cfg.get("max_tokens") or 1024)
        self.DETAIL_TOKENS = {"low": DETAIL_LOW_TOKENS} if DETAIL_LOW_TOKENS < top else {}
        if not lazy and not self.lazy_vision and not self.elastic:
            try:
                self._start()
            except RuntimeError as e:   # a broken PyTorch (or file) must not stop the model: text works, pictures say why
                print(f"[strata] pictures: the image encoder did not start ({e}); the model starts without pictures - "
                      f"each picture tries it again", flush=True)

    def encode(self, source, keep=(), detail=None):
        # the resident encoder died, timed out (STRATA_VISION_ENCODE_S) or never started: the base class writes to the
        # dead process and pictures stayed broken until a restart of the server - start it again for this picture
        if not self.alive() and not self.lazy_vision and not self.elastic:
            print("[strata] pictures: starting the image encoder again ...", flush=True)
            try:
                self.restart()
            except RuntimeError as e:
                raise ValueError(f"pictures are not working: the image encoder did not start ({e})") from None
        try:
            return super().encode(source, keep, detail)
        except RuntimeError as e:   # a lazy encoder starts inside the base encode(): its failure is the picture's 400
            raise ValueError(f"pictures are not working: the image encoder did not start ({e})") from None


S.Vision = Ds41Vision
_start_lazy_vision = S.Service.start_lazy_vision


def _start_lazy_vision_ds41(self, reserve_mib, idle_s):
    # ds41_generate has no VRAM command: lazy mode's make-room step (an encoder on the engine's own card) would wait
    # 120 s for an answer and then take the engine for dead.  The engine says it is not elastic; the encoder only
    # starts and stops (on the engine's card it then needs the room --vram-margin leaves).
    self.engine.info["vram_elastic"] = False
    _start_lazy_vision(self, reserve_mib, idle_s)


S.Service.start_lazy_vision = _start_lazy_vision_ds41

# ---- tool calls: DSML
CALLS_OPEN = f"<{E.dsml_token}{E.tool_calls_block_name}>"           # <｜DSML｜ calls>
CALLS_CLOSE = f"</{E.dsml_token}{E.tool_calls_block_name}>"
INVOKE_OPEN = f"<{E.dsml_token}{E.tool_call_tag_name}"               # <｜DSML｜ invoke   (name="X">)
INVOKE_CLOSE = f"</{E.dsml_token}{E.tool_call_tag_name}>"
P_OPEN = f"<{E.dsml_token}{E.tool_parameter_tag_name}"
P_CLOSE = f"</{E.dsml_token}{E.tool_parameter_tag_name}>"
_P_RE = re.compile(r' name="(.*?)" string="(true|false)">(.*?)' + re.escape(P_CLOSE), re.DOTALL)


def unescape(value: str) -> str:
    """DwarfStar escapes a value's own closing tag as &lt;... (and &amp; chains before it); DeepSeek's encoder never
    escapes, so only that exact spelling is undone."""
    tail = P_CLOSE[1:]
    return re.sub(r"&((?:amp;)*)lt;" + re.escape(tail),
                  lambda m: ("&" + m.group(1)[:-4] + "lt;" if m.group(1) else "<") + tail, value)


def ds41_call_end(text: str) -> int:
    return text.find(INVOKE_CLOSE)


# ---- a parameter's schema (upstream b48bf015, Basalt #208): `anyOf` / `oneOf` / type lists count, and so does a tool
# whose parameters are themselves a union.  Without this `anyOf: [integer, null]` read as "no type", so a tag with no
# string=... came back as the text "7", and string="false" on a text parameter turned "123" into a number.
def _deref(root, s, depth=0):
    while isinstance(s, dict) and isinstance(s.get("$ref"), str) and s["$ref"].startswith("#/") and depth < 16:
        at = root
        for key in s["$ref"][2:].split("/"):
            at = at.get(key) if isinstance(at, dict) else None
        s, depth = at, depth + 1
    return s


def _alternatives(s):
    return [alt for k in ("anyOf", "oneOf") for alt in (s.get(k) if isinstance(s.get(k), list) else [])]


def _param_types(root, s, depth=0) -> set:
    """The types a parameter may have; empty when nothing says."""
    s = _deref(root, s)
    if not isinstance(s, dict) or depth > 16:
        return set()
    t = s.get("type")
    out = set(t) if isinstance(t, list) else {t} if isinstance(t, str) else set()
    for alt in _alternatives(s):
        out |= _param_types(root, alt, depth + 1)
    return out


def _tool_props(spec) -> tuple:
    """(the tool's parameters, its properties by name): a union's variants take part, the first to name one wins."""
    root = (spec or {}).get("parameters") or {}
    if not isinstance(root, dict):
        return {}, {}
    props, todo, seen = dict(root.get("properties") or {}), _alternatives(root), 0
    while todo and seen < 256:
        seen += 1
        v = _deref(root, todo.pop(0))
        if isinstance(v, dict):
            for k, val in (v.get("properties") or {}).items():
                props.setdefault(k, val)
            todo += _alternatives(v)
    return root, props


def _param_kind(spec, name) -> str:
    """"text" when the value stays text, "text?" when the text null is None too, "json" when it is decoded."""
    root, props = _tool_props(spec)
    if name not in props:
        return "json"
    types = _param_types(root, props[name])
    if "string" not in types:
        return "json"
    return "text?" if "null" in types else "text"


def ds41_parse_tool_call(body: str, schema=None):
    m = re.match(r'\s*name="(.*?)">', body, re.DOTALL)
    if not m:
        raise ValueError("malformed tool call: " + body[:80])
    args = {}
    for key, is_str, value in _P_RE.findall(body[m.end():]):
        value = unescape(value)
        kind = _param_kind(schema, key)
        if is_str == "true" or kind == "text":
            args[key] = value
        elif kind == "text?":
            args[key] = None if value.strip() == "null" else value
        else:
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value
    return F.ToolCall(name=m.group(1), arguments=args)


F.CALL_START, F.CALL_END = INVOKE_OPEN, INVOKE_CLOSE
# xolotl: the 0.1.44 parser treats a call opener as a call only when Qwen's `<function=` follows it (else it is
# prose naming the format); this model's follower is DSML (` name="...">`), so with FUNC_START = "" that guard always passes -
# the pre-0.1.44 behaviour.  Only the opt-in tool_call_recovery reads FUNC_START otherwise (off here).
F.FUNC_START = ""
F.call_end = ds41_call_end
F.parse_tool_call = ds41_parse_tool_call
_op_init, _op_feed, _op_finish = F.OutputParser.__init__, F.OutputParser.feed, F.OutputParser.finish
WRAPPERS = (CALLS_OPEN, CALLS_CLOSE)


# ---- low-bit quants' spellings of the tags (upstream #110): a quant sometimes writes a call's tags without the ｜DSML｜
# token - `< invoke name="write">` - or with a backslash (ASCII or U+FF3C) or a misspelled token (＼DSML＼) in its place,
# and markers can differ within one call.  As text those calls reached clients as the answer, which ends an agent's run.
# Every tag goes through here first and comes out in the one spelling the rest of this file reads.  Without the real
# token a tag opens a call only inside a calls block, for a declared tool, or at the start of a line outside code, so
# `< invoke name="rm">` in prose stays text; `<invoke>` (no marker) is never a tag.  A stray short word in the marker's
# place - `<name invoke name="read_file">`, the UD-IQ2_XXS default's first tag in round 359's agent trace (greedy, bias
# 0.28) - is a tag only between a calls block's other tags (the calls / invoke states), never in prose or in a value.
_M = r"(?P<tok>｜DSML｜\s*)?(?(tok)|(?:[\\＼｜]DSML[\\＼｜]\s*|[\\＼]\s*|(?P<stray>[A-Za-z_]{1,12})?\s+))"
_ANY_TAG = re.compile(rf'<(?P<end>/?){_M}(?:(?:tool_|function_)?(?P<calls>calls)|(?P<word>invoke|parameter)'
                      r'(?:\s+name\s*=\s*"(?P<name>[^"]*)")?(?:\s+string\s*=\s*"(?P<s>true|false)")?)\s*>')
_TAG_HEAD = re.compile(r"^</?(?:｜(?:D(?:S(?:M(?:L(?:｜)?)?)?)?)?|[\\＼](?:D(?:S(?:M(?:L(?:[\\＼｜])?)?)?)?)?)?$")   # "<", "<｜DS", "</＼DSM"
_TAG_REST = re.compile(rf"^</?{_M}(?P<rest>[^<>]*)$")
_TAG_WORDS = ("calls", "tool_calls", "function_calls", "invoke", "parameter")
_TAG_MAX = 600                  # a longer `<...` is text, not held any further


def _maybe_tag(s: str, inside: bool = False) -> bool:
    """`s` starts with '<' and has no '>' yet: it may still become a tag (`inside`: between a calls block's tags)."""
    if len(s) > _TAG_MAX:
        return False
    if _TAG_HEAD.match(s) or (inside and re.fullmatch(r"</?[A-Za-z_]{1,12}", s)):
        return True
    m = _TAG_REST.match(s)
    if m is None or (m["stray"] and not inside):
        return False
    rest = m.group("rest")
    if any(w.startswith(rest) for w in _TAG_WORDS):
        return True
    return any(rest.startswith(w) and rest[len(w)].isspace() and ("\n" not in rest if w[0] in "ip" else not rest[len(w):].strip())
               for w in _TAG_WORDS)                    # the word is whole: invoke's and parameter's attributes are arriving


def _tag_text(self, text: str, out: list):
    out.append(text)
    *done, self.tag_line = (self.tag_line + text).split("\n")
    for line in done:
        if line.lstrip().startswith("```"):
            self.tag_fence = not self.tag_fence


def _param_is_string(self, tool: str, name: str) -> str:
    """A parameter tag written without string=...: as the declared type says, a string if nothing does."""
    spec = self.schemas.get(tool)
    root, props = _tool_props(spec)
    types = _param_types(root, props.get(name))
    return "true" if not types or _param_kind(spec, name) == "text" else "false"   # "text?": the parser reads the null


def _tag(self, m, out: list) -> bool:
    """The tag `m` at this state -> its one spelling in `out`, True; False when here it is not a tag."""
    word, end, st = "calls" if m["calls"] else m["word"], bool(m["end"]), self.tag_state
    if m["stray"] and st not in ("calls", "invoke"):
        return False
    if st == "param":
        if word == "parameter" and end:
            out.append(P_CLOSE); self.tag_state = "invoke"
            return True
        if word == "invoke" and end:                # the value's closing tag was left out
            out.append(P_CLOSE + INVOKE_CLOSE); self.tag_state = "calls"
            return True
        return False
    if word == "parameter":
        if st != "invoke" or end or m["name"] is None:
            return False
        out.append(f'{P_OPEN} name="{m["name"]}" string="{m["s"] or _param_is_string(self, self.tag_tool, m["name"])}">')
        self.tag_state = "param"
        return True
    if word == "invoke":
        if end:
            if st != "invoke":
                return False
            out.append(INVOKE_CLOSE); self.tag_state = "calls"
            return True
        if st == "invoke" or m["name"] is None:
            return False
        if st == "text" and not m["tok"] and (self.tag_fence or self.tag_line.count("`") % 2 or
                                               (m["name"] not in self.schemas and self.tag_line.strip())):
            return False
        out.append(f'{INVOKE_OPEN} name="{m["name"]}">')
        self.tag_state, self.tag_tool = "invoke", m["name"]
        return True
    if end:
        if st == "text":
            return False
        out.append(CALLS_CLOSE); self.tag_state = "text"
        return True
    if st != "text" or (not m["tok"] and (self.tag_fence or self.tag_line.strip())):
        return False
    out.append(CALLS_OPEN); self.tag_state = "calls"
    return True


def _spell_tags(self, delta: str, final: bool) -> str:
    """One delta of the model's text with its tags in the one spelling, holding a `<...` until it is decided."""
    buf, out = self.tag_hold + delta, []
    while buf:
        lt = buf.find("<")
        if lt < 0:
            _tag_text(self, buf, out); buf = ""
            break
        if lt:
            _tag_text(self, buf[:lt], out); buf = buf[lt:]
        gt = buf.find(">")
        if gt < 0 and _maybe_tag(buf, self.tag_state in ("calls", "invoke")):
            if final:
                _tag_text(self, buf, out); buf = ""
            break
        m = _ANY_TAG.fullmatch(buf[:gt + 1]) if gt >= 0 else None
        if m is not None and _tag(self, m, out):
            buf = buf[gt + 1:]
        else:
            _tag_text(self, "<", out); buf = buf[1:]
    self.tag_hold = buf
    return "".join(out)


def _op_init_ds41(self, *a, **k):
    _op_init(self, *a, **k)
    self.stream_tools = False          # the streaming scanner reads Qwen's <function=...> body
    self.dsml_hold = ""
    self.echo_check, self.echo_hold = True, ""
    self.tag_hold, self.tag_state, self.tag_tool, self.tag_line, self.tag_fence = "", "text", None, "", False


# 2026-10-19: after the server closes a thinking budget with REASONING_WRAP_UP, the 1017 sometimes repeats that sentence
# as the first line of its answer (seen in the probe runs: the sentence correctly in the thinking AND again in the
# answer).  Drop an exact repeat at the very start of the answer; anything else passes through unchanged.
WRAP_SENTENCE = S.REASONING_WRAP_UP.split("</think>")[0].strip()


def _drop_wrap_echo(self, evs):
    out = []
    for ev in evs:
        if not self.echo_check or ev.kind != "content":
            if self.echo_check and self.echo_hold and ev.kind != "reasoning":   # flush a held prefix before anything else
                out.append(F.Event("content", self.echo_hold)); self.echo_hold = ""; self.echo_check = False
            out.append(ev)
            continue
        self.echo_hold += ev.text
        head = self.echo_hold.lstrip()
        if not head or WRAP_SENTENCE.startswith(head):
            continue                                   # still could be the echo: hold it
        self.echo_check = False
        if head.startswith(WRAP_SENTENCE):
            rest = head[len(WRAP_SENTENCE):].lstrip("\n")
            if rest:
                out.append(F.Event("content", rest))
        else:
            out.append(F.Event("content", self.echo_hold))
        self.echo_hold = ""
    return out


def _strip_wrappers(self, delta: str, final: bool) -> str:
    """Drop <｜DSML｜ calls> / </｜DSML｜ calls> from the stream, holding a partial one until it is decided."""
    buf = self.dsml_hold + delta
    for w in WRAPPERS:
        buf = buf.replace(w, "")
    keep = 0 if final else self._hold(buf, WRAPPERS)
    self.dsml_hold = buf[len(buf) - keep:] if keep else ""
    return buf[:len(buf) - keep] if keep else buf


def _op_feed_ds41(self, delta: str):
    return _drop_wrap_echo(self, _op_feed(self, _strip_wrappers(self, _spell_tags(self, delta, False), False)))


def _op_finish_ds41(self, *a, **k):   # *a/**k: the 0.1.44 server's finish(reason=...)
    if (a[0] if a else k.get("reason")) not in (None, "stop") and self.state in ("content", "call", "json"):
        # upstream #106: the turn was cut (max_tokens, a cancel, an error) inside DSML - `<｜DSML｜ inv` before a call,
        # a call without its end, or `</｜DSML｜ calls` after a whole one.  None of it is text the model meant, and as
        # content it reached clients as DSML in the answer (an agent read it as the answer).  What is held here is only
        # ever a tag that may still be arriving or an unfinished call (never streamed: stream_tools is off), so a cut
        # drops nothing else; a natural stop keeps it as text, as before.
        self.dsml_hold, self.buf, self.tag_hold = "", "", ""
    tail = _spell_tags(self, "", True)
    out = _op_feed(self, _strip_wrappers(self, tail, True)) if tail or self.dsml_hold else []
    out = _drop_wrap_echo(self, out + _op_finish(self, *a, **k))
    if self.echo_hold:                 # ended while matching: the whole sentence = the echo, a part of it = real text
        if self.echo_hold.strip() != WRAP_SENTENCE:
            out.append(F.Event("content", self.echo_hold))
        self.echo_hold = ""
    return out


F.OutputParser.__init__ = _op_init_ds41
F.OutputParser.feed = _op_feed_ds41
F.OutputParser.finish = _op_finish_ds41
S.OutputParser = F.OutputParser

# ---- tool_choice (forced calls, as upstream 34ac0f1): the server writes the call's opening itself and the model goes
# on from it, so the opening must be DSML the way DeepSeek's encoder writes an assistant call - the frontend's is Qwen's
# `<function=X>`, which this model would continue as plain text.  "required" stops after ` name="` (the model picks
# the tool); a named tool is opened whole.  With thinking the server puts the blank line before it (after </think>);
# without, the prompt ends with </think>, so prepare() adds it there.
_forced_call = F.forced_call


def ds41_forced_call(tool_choice, tools):
    qwen = _forced_call(tool_choice, tools)   # the same decisions, and the same log line for a choice it cannot honour
    if qwen is None:
        return None
    m = re.search(r"<function=([^>\n]+)>", qwen)
    head = CALLS_OPEN + "\n" + INVOKE_OPEN + ' name="'
    return head + m.group(1) + '">\n' if m else head


S.forced_call = ds41_forced_call
_svc_prepare = S.Service.prepare


def _prepare_ds41(self, messages, tools, kwargs, max_new=None, force=None, **kw):
    if force and kwargs.get("enable_thinking", True) is False:
        force = "\n\n" + force
    return _svc_prepare(self, messages, tools, kwargs, max_new, force=force, **kw)


S.Service.prepare = _prepare_ds41

# ---- the chat page's Setup card (setup_panel.py): model, speed or quality, multi-GPU.  Apply needs a new start, because the
# launcher sizes the memory for each model: the server exits with 75 and Start DeepSeek.bat runs it again; the page waits.
import subprocess  # noqa: E402
import setup_panel as _setup  # noqa: E402


class _Setup:
    @staticmethod
    def get():
        return _setup.catalog()

    @staticmethod
    def post(svc, req):
        if req.get("firewall") in ("add", "remove"):    # the user's click; Windows asks (UAC) before anything changes
            on = _setup.change_firewall(req["firewall"] == "add")
            return {"firewall": on}
        if not _setup.needs_restart(req):               # auto-pause: saved, and in effect now
            s = _setup.apply(req)
            svc.auto_pause_mib = int(s.get("auto_pause_mib") or 0)
            return {"restarting": False, "auto_pause_mib": svc.auto_pause_mib}
        with svc.status_lock:
            if svc.status.get("busy") or svc.status.get("queued"):
                raise ValueError("a request is running: try again when it has finished")
        s = _setup.apply(req)

        def restart():
            _time.sleep(0.5)                     # the page gets its answer first
            for close in (getattr(svc.engine, "close", None), svc.vision.shutdown if svc.vision else None):
                try:
                    if close:
                        close()
                except Exception as e:
                    print(f"[strata] while stopping for the restart: {e}", flush=True)
            os._exit(_setup.RESTART_EXIT)
        _th.Thread(target=restart, daemon=True).start()
        out = {"restarting": True}
        if req.get("phone"):                            # the page keeps the key: from now on this PC's page needs it too
            key = svc.api_key[0] if isinstance(svc.api_key, list) and svc.api_key else svc.api_key
            out["api_key"] = (key or "").split(",")[0].strip() or s.get("phone_key") or ""
        return out


S.SETUP = _Setup


def _watch_started(svc):
    """The model is up: the page's choice is kept (a failed start goes back to the one before it)."""
    while not svc.loaded():
        _time.sleep(2)
    _setup.started()


# ---- multi-GPU: ds41_helper beside the engine.  In a Windows job that ends with this process, so closing the window
# frees the second card; started before each engine start (an engine that stops tells the helper to stop too).
_HELPER = None
_helper_job = None


def _start_helper():
    global _HELPER, _helper_job
    plan_file = HERE / "local" / "helper.json"
    if not plan_file.exists() or (_HELPER is not None and _HELPER.poll() is None):
        return
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(plan["gpu"])}
    log = open(HERE / "local" / "helper.log", "ab")
    _HELPER = subprocess.Popen([plan["exe"]] + plan["args"], env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    print(f"[strata] multi-GPU: helper started on {plan['gpu_name']}", flush=True)
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.CreateJobObjectW.restype = ctypes.c_void_p
        k.OpenProcess.restype = ctypes.c_void_p

        class Info(ctypes.Structure):      # JOBOBJECT_EXTENDED_LIMIT_INFORMATION
            _fields_ = [("pad0", ctypes.c_byte * 16), ("flags", ctypes.c_uint32), ("rest", ctypes.c_byte * 124)]
        if _helper_job is None:
            _helper_job = k.CreateJobObjectW(None, None)
            info = Info()
            info.flags = 0x2000            # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            k.SetInformationJobObject(ctypes.c_void_p(_helper_job), 9, ctypes.byref(info), ctypes.sizeof(info))
        h = k.OpenProcess(0x0101, False, _HELPER.pid)      # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        k.AssignProcessToJobObject(ctypes.c_void_p(_helper_job), ctypes.c_void_p(h))


_engine_init = S.StrataEngine.__init__


def _engine_init_helper(self, *a, **k):
    _start_helper()
    _engine_init(self, *a, **k)


S.StrataEngine.__init__ = _engine_init_helper
_svc_init_started = S.Service.__init__


def _svc_init_watch(self, *a, **k):
    _svc_init_started(self, *a, **k)
    _th.Thread(target=_watch_started, args=(self,), daemon=True).start()


S.Service.__init__ = _svc_init_watch

# ---- full text log (llama.cpp-style): every request's thinking / answer / tool calls as they are generated, so a run
# can be watched (`tail -f`) and checked afterwards (loops, what it was thinking).  OFF by default (2026-10-25, the
# user: nothing from a prompt is written to the machine unless a flag asks for it - prompts and answers live in memory
# only and are gone when the server stops).  STRATA_TEXT_LOG=<path> turns it on; "0" / unset = off.
import threading as _th
import time as _time

_TEXT_LOG = os.environ.get("STRATA_TEXT_LOG", "")
if _TEXT_LOG in ("0", "1"):   # "1" is not a path: on needs a file name
    _TEXT_LOG = ""
_text_lock = _th.Lock()
_text_state = {"kind": None}
_KIND_LABEL = {"reasoning": "thinking", "content": "answer", "tool_start": "tool call", "tool_call": "tool call"}


def _text_write(s: str):
    if not _TEXT_LOG or _TEXT_LOG == "0":
        return
    try:
        with _text_lock, open(_TEXT_LOG, "a", encoding="utf-8") as f:
            f.write(s)
    except OSError:
        pass


_svc_note, _svc_run = S.Service._note, S.Service.run


def _note_ds41(self, n, evs, st=None, rate=None):
    out = []
    for ev in evs:
        if not ev.text:
            continue
        label = _KIND_LABEL.get(ev.kind, ev.kind)
        if label != _text_state["kind"]:
            _text_state["kind"] = label
            out.append(f"\n--- {label} ---\n")
        out.append(ev.text)
    if out:
        _text_write("".join(out))
    return _svc_note(self, n, evs, st, rate)


# a model name ending in one of these (a config alias) = that routing for the request, for the strata_tune keys the
# request doesn't name: -exact = no decode biases (quality mode), -fast = the stronger 0.5 / 0.3 nudge (speed mode); the
# plain name gets the config's --route-bias / --route-bias-file
ROUTE_SUFFIXES = {"-exact": {"route_bias": 0.0, "route_bias_file": 0.0},
                  "-fast": {"route_bias": 0.5, "route_bias_file": 0.3}}


def _run_ds41(self, ids, thinking, tools, max_new, sampling, cancel, **kw):   # **kw: the server's newer keywords (0.1.44: force=)
    model = sampling.get("model") if isinstance(sampling, dict) else None
    suffix = next((s for s in ROUTE_SUFFIXES if isinstance(model, str) and model.endswith(s)), None)
    if suffix:
        tune = dict(sampling.get("strata_tune") or {}) if isinstance(sampling.get("strata_tune"), dict) else {}
        for k, v in ROUTE_SUFFIXES[suffix].items():
            tune.setdefault(k, v)
        sampling = {**sampling, "strata_tune": tune}
    tune = sampling.get("strata_tune") if isinstance(sampling, dict) and isinstance(sampling.get("strata_tune"), dict) else {}
    def num(k, dflt):
        v = tune.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else dflt
    self.mode_note = _routing_note(num("route_bias", _ROUTING[0]), num("route_bias_file", _ROUTING[1]),
                                   int(num("route_protect", _ROUTING[2])), int(num("route_pool", _ROUTING[3])))
    t0 = _time.time()
    _text_state["kind"] = None
    _text_write(f"\n\n===== {_time.strftime('%Y-%m-%d %H:%M:%S')}  request: prompt {len(ids):,} tokens, max {max_new}, "
                f"thinking {'on' if thinking else 'off'}, tools {len(tools or [])} =====\n")
    end = "(stream closed)"
    try:
        for item in _svc_run(self, ids, thinking, tools, max_new, sampling, cancel, **kw):
            if item[0] == "done" and isinstance(item[1], dict):
                d = item[1]
                end = f"finish {d.get('finish')}, {d.get('completion_tokens')} tokens"
            yield item
    finally:
        _text_write(f"\n===== end: {end}, {_time.time() - t0:.1f} s =====\n")


S.Service._note = _note_ds41
S.Service.run = _run_ds41

def _port_taken(argv):
    """the port this start asks for, if another program listens on it already - said before the engine loads, in the
    launcher's words (the server's own check comes after a minute of loading and names --port, which a
    Start DeepSeek.bat user does not pass)"""
    import socket
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv[:-1] else 8095
    host = argv[argv.index("--host") + 1] if "--host" in argv[:-1] else "127.0.0.1"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host if host not in ("", "0.0.0.0") else "127.0.0.1", port))
        return None
    except OSError as e:
        return port if (getattr(e, "winerror", None) or e.errno) in (10048, 98) else None
    finally:
        s.close()


if __name__ == "__main__":
    for i, v in enumerate(sys.argv[:-1]):   # paths are the caller's; the server runs from Strata's directory
        if v in ("--config", "--tokenizer", "--mcp-config"):
            sys.argv[i + 1] = os.path.abspath(sys.argv[i + 1])
    taken = _port_taken(sys.argv)
    if taken:
        print(f"[strata] Port {taken} is already in use: DeepSeek is probably running already (another window), or "
              f"another program uses it.\n[strata] Close that, or set DS41_PORT in ds41-paths.bat to another port, then "
              f"start again.", file=sys.stderr, flush=True)
        sys.exit(1)
    os.chdir(str(STRATA))
    try:
        sys.exit(S.main())
    except S.EngineDied as e:   # the engine stopped while starting: its reason and log lines, without Python's traceback
        msg = str(e)
        print(f"\n[strata] {msg}", file=sys.stderr, flush=True)
        if "failed to initialize CUDA" in msg or "no cuda backend" in msg:
            print("[strata] CUDA could not start on this PC: it needs NVIDIA driver 580 or newer and an RTX 20 / GTX 16 "
                  "series card or newer (README.md, \"What you need\").", file=sys.stderr, flush=True)
        print("[strata] For a bug report: \"Start DeepSeek (debug).bat\" packs this and everything else needed into one "
              "zip.", file=sys.stderr, flush=True)
        sys.exit(1)
