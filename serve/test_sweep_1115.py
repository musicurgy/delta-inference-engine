"""11-15 correctness sweep: DeepSeek-V4.1 request-conversion bugs that other engines have fixed and this server still has.

Each test is @unittest.expectedFailure (the suite stays green) and names the source fix it follows.  Remove the decorator
when the bug is fixed (the test then passes).

The DeepSeek layer (tools/v41/serve/serve_ds41.py) monkeypatches serve.frontend and serve.server on import, so it is
driven in a child process: one run computes every probe, through the real path (frontend.openai_to_messages /
anthropic_to_messages -> serve_ds41.Ds41Template().render, and serve_ds41's patched OutputParser), and the tests read its
JSON.  Nothing here starts a server or an engine.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CHILD = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import serve_ds41 as D
F, E = D.F, D.E
T = D.Ds41Template()
out = {}

def chat(req, **kw):
    m, t, k = F.openai_to_messages(req); k.update(kw)
    return T.render(m, tools=t, **k)

def anth(req, **kw):
    m, t, k = F.anthropic_to_messages(req); k.update(kw)
    return T.render(m, tools=t, **k)

TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object",
          "properties": {"a": {"type": "string"}}}}}]

# 1. parallel tool results sent in finish order (c2 before c1)
out["chat_order"] = chat({"tools": TOOLS, "messages": [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": \"1\"}"}},
        {"id": "c2", "type": "function", "function": {"name": "f", "arguments": "{\"a\": \"2\"}"}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "RESULT-TWO"},
    {"role": "tool", "tool_call_id": "c1", "content": "RESULT-ONE"}]})
out["anth_order"] = anth({"tools": [{"name": "f", "input_schema": {"type": "object", "properties": {}}}], "messages": [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {"a": "1"}},
                                      {"type": "tool_use", "id": "t2", "name": "f", "input": {"a": "2"}}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "RESULT-TWO"},
                                 {"type": "tool_result", "tool_use_id": "t1", "content": "RESULT-ONE"}]}]})

# 2. several text parts in one message
parts = [{"type": "text", "text": "PART-A"}, {"type": "text", "text": "PART-B"}]
out["chat_parts"] = chat({"messages": [{"role": "user", "content": parts}]})
out["chat_parts_ref"] = E.encode_messages([{"role": "user", "content": parts}], thinking_mode="thinking")
tool_hist = [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": parts}]
out["tool_parts"] = chat({"tools": TOOLS, "messages": tool_hist})
out["tool_parts_ref"] = E.encode_messages([{"role": "system", "content": "", "tools": TOOLS}] + tool_hist,
                                          thinking_mode="thinking")
out["anth_parts"] = anth({"messages": [{"role": "user", "content": parts}]})

# 3. two system messages at the start
out["two_systems"] = chat({"messages": [{"role": "system", "content": "SYS-ONE"},
                                        {"role": "system", "content": "SYS-TWO"},
                                        {"role": "user", "content": "hi"}]})

# 4. prefill of reasoning only (a trailing assistant message with reasoning and no answer yet)
out["prefill_reasoning"] = chat({"messages": [{"role": "user", "content": "q"},
                                              {"role": "assistant", "content": "", "reasoning_content": "Let me"}]})

# 5. top-level enable_thinking: false
out["top_enable_thinking"] = F.openai_to_messages({"messages": [{"role": "user", "content": "hi"}],
                                                   "enable_thinking": False})[2]

# 6. DSML written inside the reasoning
P = '<｜DSML｜ parameter name="a" string="true">x</｜DSML｜ parameter>'
block = (D.CALLS_OPEN + '\n<｜DSML｜ invoke name="f">\n' + P + '\n</｜DSML｜ invoke>\n' + D.CALLS_CLOSE)
text = "maybe " + block + " later" + E.thinking_end_token + "answer"
p = F.OutputParser(thinking=True, tools=[t["function"] for t in TOOLS])
ev = []
for i in range(0, len(text), 5):
    ev += p.feed(text[i:i + 5])
ev += p.finish()
out["dsml_in_reasoning"] = {"reasoning": "".join(e.text for e in ev if e.kind == "reasoning"),
                            "block": block, "calls": sum(e.kind == "tool_call" for e in ev)}

print("\n@@JSON@@" + json.dumps(out, ensure_ascii=False))
'''

_RESULTS = None


def results() -> dict:
    global _RESULTS
    if _RESULTS is None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        run = subprocess.run([sys.executable, "-c", CHILD, str(ROOT / "tools" / "v41" / "serve")], cwd=str(ROOT),
                             capture_output=True, text=True, encoding="utf-8", env=env, timeout=300)
        if "@@JSON@@" not in run.stdout:
            raise RuntimeError("probe child failed:\n" + run.stdout[-2000:] + run.stderr[-4000:])
        _RESULTS = json.loads(run.stdout.split("@@JSON@@", 1)[1])
    return _RESULTS


class ToolResultOrder(unittest.TestCase):
    # DeepSeek's encoder (dsv41_encoding.sort_tool_results_by_call_order, the 1017 sync) puts tool results in the
    # order of the calls, by tool_call_id; DSML carries no ids, so the model pairs results with calls BY POSITION.
    # Fixed in spark-ds41f 0178 (order_tool_results), colibri 08420625, ik_llama 6b55d2c7, llama.cpp f534da26e,
    # project-maya 22db5cc; this server's own /v1/responses does it (responses._order_tool_results).
    # openai_to_messages / anthropic_request drop tool_call_id / tool_use_id and the calls' ids, so nothing sorts.
    # was @unittest.expectedFailure (fixed 11-15)
    def test_chat_tool_results_follow_call_order(self):
        p = results()["chat_order"]
        self.assertLess(p.index("RESULT-ONE"), p.index("RESULT-TWO"))

    # was @unittest.expectedFailure (fixed 11-15)
    def test_anthropic_tool_results_follow_call_order(self):
        p = results()["anth_order"]
        self.assertLess(p.index("RESULT-ONE"), p.index("RESULT-TWO"))


class TextParts(unittest.TestCase):
    # The 1017 encoder takes list content as content_blocks and joins parts with "\n\n" (user text blocks and the text
    # of a tool_result).  ik_llama 39ccfcb8 (render_content joins with "\n\n"), llama.cpp 74bca3950 ("content parts
    # stay separate").  serve_ds41._text / frontend._text_of join with "" - "PART-A" "PART-B" reach the model glued.
    # was @unittest.expectedFailure (fixed 11-15)
    def test_chat_user_text_parts_match_encoder(self):
        r = results()
        self.assertEqual(r["chat_parts"], r["chat_parts_ref"])

    # was @unittest.expectedFailure (fixed 11-15)
    def test_chat_tool_result_text_parts_match_encoder(self):
        r = results()
        self.assertEqual(r["tool_parts"], r["tool_parts_ref"])

    # was @unittest.expectedFailure (fixed 11-15)
    def test_anthropic_text_blocks_not_glued(self):
        self.assertNotIn("PART-APART-B", results()["anth_parts"])


class LeadingSystems(unittest.TestCase):
    # A second system message at the very start (index 1, before any user turn) becomes USER text ("SYS-TWO\n\nhi"
    # in the user turn): _late_system_to_user treats every system message past index 0 as mid-conversation.
    # /v1/responses merges leading system/developer messages into one system message (responses.input_messages);
    # DeepSeek's encoder renders it as <｜System｜> (knapcio sglang-43549, spark-ds41f 0645 KEEP).
    # was @unittest.expectedFailure (fixed 11-15)
    def test_second_leading_system_stays_system(self):
        p = results()["two_systems"]
        self.assertLess(p.index("SYS-TWO"), p.index("<｜User｜>"))


class ReasoningPrefill(unittest.TestCase):
    # vLLM #46256 (spark-ds41f 0328, continue_final_message): a final assistant message with no content and no tool
    # calls yet continues its REASONING - the prompt ends inside <think>.  Ours renders <think>Let me</think>: the
    # reasoning is closed for the client and the model goes straight to the answer.
    # was @unittest.expectedFailure (fixed 11-15)
    def test_reasoning_only_prefill_stays_open(self):
        p = results()["prefill_reasoning"]
        self.assertTrue(p.endswith("<think>Let me"), p[-60:])


class ThinkingSwitch(unittest.TestCase):
    # sparkflash D46 (top-level enable_thinking honoured after chat_template_kwargs), MiaAI 93d9e6b / knapcio
    # encoding_compat (enable_thinking <-> thinking aliases).  A top-level "enable_thinking": false is ignored here.
    # was @unittest.expectedFailure (fixed 11-15)
    def test_top_level_enable_thinking_false(self):
        self.assertEqual(results()["top_enable_thinking"].get("enable_thinking"), False)


class DsmlInReasoning(unittest.TestCase):
    # argodrive 037ee39 / antirez ds4 0ead8a8: DSML inside thinking is reasoning text, kept as written.  Ours keeps it
    # in the reasoning (no call - right) but _strip_wrappers removes "<｜DSML｜ calls>" / "</｜DSML｜ calls>" from it,
    # so reasoning_content differs from what the model wrote (and a client's replay of it does too).  Cosmetic.
    @unittest.expectedFailure   # cosmetic, not fixed yet
    def test_dsml_wrapper_kept_in_reasoning(self):
        r = results()["dsml_in_reasoning"]
        self.assertEqual(r["calls"], 0)
        self.assertIn(r["block"], r["reasoning"])


if __name__ == "__main__":
    unittest.main()
