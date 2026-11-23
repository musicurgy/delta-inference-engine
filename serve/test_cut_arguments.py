"""serve/test_cut_arguments.py - a tool call max_tokens cut short is replayed with the parameters it holds
(Niko1221/Strata 0c99fcf85).

A client replays what it was sent.  The Qwen path streams a call's arguments as they are written, so a call cut by
max_tokens reaches the client as text that starts like a JSON object and is not valid JSON; tool_arguments() kept it
under a parameter called "arguments", and the model, shown `<parameter=arguments>` + the cut text a dozen times, began
writing {"arguments": "<json>"} calls itself.  Now the complete members and a string value the text ends inside come
back; a tool that really declares "arguments", and text that is not a cut object, keep the old form."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from serve.frontend import anthropic_to_messages, openai_to_messages, tool_arguments  # noqa: E402
from serve.responses import input_messages  # noqa: E402
from serve.server import ChatTemplate  # noqa: E402

WRITE = {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "n": {"type": "integer"}}}}}
RAW = {"type": "function", "function": {"name": "raw", "parameters": {"type": "object", "properties": {
    "arguments": {"type": "string"}}}}}


def chat(args, name="write", tools=(WRITE, RAW)):
    req = {"tools": list(tools), "messages": [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                              "function": {"name": name, "arguments": args}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r"}]}
    return openai_to_messages(req)


class CutArguments(unittest.TestCase):
    def test_cut_calls_give_their_members(self):
        for raw, want in (('{"path": "a.txt", "content": "hello wor', {"path": "a.txt", "content": "hello wor"}),
                          ('{"path": "a.txt", "content": "line\\nnext \\', {"path": "a.txt", "content": "line\nnext "}),
                          ('{"path": "a.txt", "content": "x \\u00', {"path": "a.txt", "content": "x "}),
                          ('{"path": "a.txt", "n": 12', {"path": "a.txt"}),           # a number may be cut
                          ('{"path": "a.txt", "n": 12,', {"path": "a.txt", "n": 12}),
                          ('{"path": "a.txt", "opts": {"k": [1, 2', {"path": "a.txt"}),
                          ('{"path": "a.txt", "con', {"path": "a.txt"}),
                          ('{"path": "a.txt", ', {"path": "a.txt"}),
                          ('{"path"', {}), ('{"pa', {}), ("{", {}), ('  {"path": "a"', {"path": "a"})):
            with self.subTest(raw=raw):
                self.assertEqual(chat(raw)[0][1]["tool_calls"][0]["function"]["arguments"], want)

    def test_old_form_kept(self):
        for raw in ("{cut", "{malformed_json", "prose", "[1, 2", "12", '{"a": 1'):
            with self.subTest(raw=raw):
                name = "raw" if raw == '{"a": 1' else "write"         # a tool that declares "arguments" itself
                self.assertEqual(chat(raw, name)[0][1]["tool_calls"][0]["function"]["arguments"], {"arguments": raw})

    def test_valid_histories_unchanged(self):
        for raw in ('{"path": "a.txt", "content": "x"}', {"path": "a"}, "", None, "[1]", "3", '"s"'):
            with self.subTest(raw=raw):
                self.assertEqual(tool_arguments(raw), chat(raw)[0][1]["tool_calls"][0]["function"]["arguments"])
        tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        msgs, tools, kw = chat('{"path": "a.txt", "content": "x"}')
        self.assertIn("<parameter=content>", tpl.render(msgs, tools=tools, **kw))

    def test_rendered_cut_call_has_no_arguments_parameter(self):
        tpl = ChatTemplate(ROOT / "serve/chat_template.jinja")
        msgs, tools, kw = chat('{"path": "a.txt", "content": "hello wor')
        out = tpl.render(msgs, tools=tools, **kw)
        self.assertNotIn("<parameter=arguments>", out)
        self.assertIn("<parameter=path>", out)
        self.assertIn("hello wor", out)

    def test_messages_and_responses_histories(self):
        req = {"tools": [{"name": "write", "input_schema": WRITE["function"]["parameters"]}], "messages": [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "write",
                                               "input": '{"path": "a.txt", "content": "hel'}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "r"}]}]}
        calls = [m for m in anthropic_to_messages(req)[0] if m.get("tool_calls")][0]["tool_calls"]
        self.assertEqual(calls[0]["function"]["arguments"], {"path": "a.txt", "content": "hel"})
        req = {"tools": [{"type": "function", "name": "write", "parameters": WRITE["function"]["parameters"]}],
               "input": [{"type": "message", "role": "user", "content": "u"},
                         {"type": "function_call", "call_id": "c1", "name": "write", "arguments": '{"path": "a.txt", "n'},
                         {"type": "function_call_output", "call_id": "c1", "output": "r"}]}
        calls = [m for m in input_messages(req) if m.get("tool_calls")][0]["tool_calls"]
        self.assertEqual(calls[0]["function"]["arguments"], {"path": "a.txt"})
        req["input"][1]["arguments"] = "{cut"
        calls = [m for m in input_messages(req) if m.get("tool_calls")][0]["tool_calls"]
        self.assertEqual(calls[0]["function"]["arguments"], {"arguments": "{cut"})


if __name__ == "__main__":
    unittest.main()
