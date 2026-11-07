"""JSON response formats at the HTTP boundary: prompt once, validate before delivery.

The native engine has no grammar decoder. Failed generations are errors, never
silently retried or returned as successful structured output.

`jsonschema` is optional (no new hard dependency of the server): json_object needs only the standard library, and
json_schema is checked against its schema when the package is installed - without it, only that the answer is one
JSON object, and the server says so once.
"""
import json
import re
import threading

try:
    import regex as _regex             # requirements.txt: matches with a time limit
except ImportError:                    # pragma: no cover
    _regex = None

PATTERN_S = 0.25                       # 0.94 (ReDoS): one schema `pattern` match; a slower one counts as no match


class StructuredOutputError(RuntimeError):
    pass


class _ObjectOnly:
    """The validator for json_object, and for json_schema without jsonschema: one JSON object."""

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


_jsonschema = None                     # (validators, SchemaError, Registry, NoSuchResource), False when not installed
_jsonschema_lock = threading.Lock()


class _TimedRe:
    """`re` as jsonschema's keyword modules see it (0.94, ReDoS): `pattern`, `patternProperties` and
    `additionalProperties` call re.search(pattern, text), with a client's pattern on the model's answer.  Python's re
    holds the GIL for the whole match, so a pattern like ^(a|a)+$ on a long answer froze every request; this searches
    with the regex module and a time limit, and a match that runs out of time is no match (the answer fails the
    schema, as it would if it did not match).  Everything else is the real re."""

    def __getattr__(self, name):
        return getattr(re, name)

    @staticmethod
    def search(pattern, string, flags=0):
        if _regex is None or not isinstance(string, str):
            return re.search(pattern, string, flags)
        try:
            return _regex.search(pattern, string, flags | _regex.V0, timeout=PATTERN_S, concurrent=True)
        except TimeoutError:
            print(f"[strata] JSON schema: the pattern {str(pattern)[:80]!r} took over {PATTERN_S} s on "
                  f"{len(string):,} characters; counted as no match", flush=True)
            return None


def _time_limited_patterns() -> bool:
    """Point jsonschema's keyword modules at _TimedRe (they import re as a module).  False when one is not there."""
    import importlib
    done = True
    for name in ("jsonschema._keywords", "jsonschema._legacy_keywords", "jsonschema._utils"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            done = False
            continue
        if getattr(module, "re", None) is re:
            module.re = _TimedRe()
    return done


def jsonschema_modules():
    """jsonschema, imported on first use; None when it is not installed (said once, in the server window)."""
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            try:
                from jsonschema import validators
                from jsonschema.exceptions import SchemaError
                from referencing import Registry
                from referencing.exceptions import NoSuchResource
                _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
                _time_limited_patterns()
            except ImportError:
                _jsonschema = False
                print('[strata] response_format json_schema: the Python package jsonschema is not installed, so '
                      'answers are only checked to be one JSON object (python -m pip install "jsonschema>=4.23,<5")',
                      flush=True)
        return _jsonschema or None


def _only_objects(node, root, refs=()):
    """True when every value `node` accepts is a JSON object, so "return one JSON object" stays true.

    `type: object`, an anyOf/oneOf whose branches all qualify (e.g. a root union of object shapes, which llama.cpp's
    grammar path accepts and apps send), an allOf with a qualifying member, or a local `$ref` to one of these.
    Anything that can also be an array, string, number, boolean or null (including `type: ["object", "null"]`) is
    not, and a `$ref` cycle never qualifies.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return _only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches and all(_only_objects(b, root, refs) for b in branches):
            return True
    branches = node.get("allOf")
    return isinstance(branches, list) and any(_only_objects(b, root, refs) for b in branches)


def _check_refs(node, what):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                raise ValueError(f"{what} supports only local schema references (#...)")
            _check_refs(value, what)
    elif isinstance(node, list):
        for value in node:
            _check_refs(value, what)


def schema_validator(schema, what="response_format"):
    """A validator for `schema` (local references only, never fetched); without jsonschema: one JSON object."""
    _check_refs(schema, what)
    modules = jsonschema_modules()
    if modules is None:
        return _ObjectOnly()
    validators, SchemaError, Registry, NoSuchResource = modules

    def no_remote(uri):
        raise NoSuchResource(ref=uri)

    try:
        cls = validators.validator_for(schema)
        cls.check_schema(schema)
        return cls(schema, registry=Registry(retrieve=no_remote))
    except SchemaError as exc:
        raise ValueError(f"invalid {what} schema: {exc.message}") from exc


# C4: where the format instruction goes.  "end" (the default): at the end of the last user turn, so a conversation's
# turns with and without a format render the same prompt up to there and the conversation cache carries across the
# switch.  "start": appended to the first system message, as through 0.93 (the config's "format_instruction").
DIRECTIVE_POSITIONS = ("end", "start")


def prepare_format(response_format, messages, with_tools=False, position="end"):
    if response_format is None:
        return messages, None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return messages, None
    if kind == "json_object":
        schema = {"type": "object"}
        validator = _ObjectOnly()      # the standard library is enough for "one JSON object"
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise ValueError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise ValueError("response_format.json_schema needs a name")
        if "strict" in spec and not isinstance(spec["strict"], bool):
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        if not _only_objects(schema, schema):
            raise ValueError("response_format schema must accept only JSON objects at its root "
                             "(type object, or anyOf/oneOf of object schemas)")
        validator = schema_validator(schema)
    else:
        raise ValueError("response_format.type must be text, json_object or json_schema")
    directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    if with_tools:
        # /v1/responses (#782): the schema is for the final answer; a turn that calls a tool is not an answer
        directive += ("\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is "
                      "what you write once you are done with the tools.")
    messages = [dict(message) for message in messages]
    last = messages[-1].get("role") if messages else None
    if position == "end" and last == "user":
        # after the user's own words, in the same turn: everything before them renders as without a format
        content = messages[-1].get("content")
        if isinstance(content, list):
            messages[-1]["content"] = content + [{"type": "text", "text": "\n\n" + directive}]
        else:
            messages[-1]["content"] = (content or "") + "\n\n" + directive
    elif position == "end" and last == "tool":
        # an agent's turn after tool results: a short user turn of its own after them
        messages.append({"role": "user", "content": directive})
    elif messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def anthropic_format(req: dict):
    """C4: Anthropic's structured output - `output_config.format` (and the older beta's `output_format`) as the chat
    path's response_format, or None.  {"type": "json_schema", "schema": {...}}."""
    cfg = req.get("output_config")
    fmt = cfg.get("format") if isinstance(cfg, dict) else None
    if fmt is None:
        fmt = req.get("output_format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise ValueError("output_config.format must be an object")
    kind = fmt.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": fmt.get("name") or "output", "schema": fmt.get("schema")}}
    if kind == "json_object":
        return {"type": "json_object"}
    raise ValueError("output_config.format.type must be json_schema")


def strict_tool_validators(tools) -> dict:
    """C4: {tool name: validator} for the tools a request marks `"strict": true` (chat {"type": "function",
    "function": {..., "strict": true}}, Responses' flat function tools, Anthropic's tools).  Only an explicit true
    counts.  Without jsonschema installed nothing is checked (said once in the server window)."""
    out = {}
    for t in tools or []:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        if fn.get("strict") is not True or not isinstance(fn.get("name"), str):
            continue
        schema = fn.get("parameters", fn.get("input_schema"))
        if not isinstance(schema, dict) or jsonschema_modules() is None:
            continue
        out[fn["name"]] = schema_validator(schema, f"tool {fn['name']}")
    return out


def check_tool_call(name, arguments, validators) -> None:
    """Raises StructuredOutputError when a strict tool's call does not match its schema."""
    validator = validators.get(name)
    if validator is None:
        return
    try:
        error = next(validator.iter_errors(arguments), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate the call of strict tool {name}: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"the model's call of strict tool {name} failed its schema at {path}: "
                                    f"{error.message}")


def strict_checked(run, validators):
    """Service.run's items, each complete call of a strict tool checked against its schema before it goes on.  A call
    that fails ends the request (StructuredOutputError: a 502, or the stream's error event)."""
    if not validators:
        yield from run
        return
    try:
        for item in run:
            if item[0] == "event" and item[1].kind == "tool_call" and item[1].call is not None:
                check_tool_call(item[1].call.name, item[1].call.arguments, validators)
            yield item
    finally:
        run.close()


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(_extract_json(text), object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    try:
        error = next(validator.iter_errors(value), None)
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical
