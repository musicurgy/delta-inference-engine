# Day 0: a new checkpoint's prompt format

One command, about a minute (14 MB of small files, no weights):

    tools/v41/day0_template_diff.sh deepseek-ai/DeepSeek-V4.1-Flash-MMDD          # [revision] [out_dir]
    # or: python tools/v41/template_conformance.py --day0 REPO [--revision R] [--out DIR]
    #     python tools/v41/template_conformance.py --local DIR   (files already downloaded)

It prints, and writes diffs to `<out>/day0_report/`:

1. **Files against 1017**: chat_template.jinja, config.json, tokenizer_config.json, encoding/encoding.py (against our
   tools/v41/dsv41_encoding.py), tokenizer.json (sha256, and any added/special token whose id moved).
2. **Format markers** read from the new template: special tokens (`<｜User｜>`, `<｜System｜>`, ...), the DSML tag names
   (`｜DSML｜ calls / invoke / parameter`), the effort sentence, `<think>`. Any marker our encoder and DSML parser do
   not know is flagged: this is the V4 -> V4.1 rename that vLLM missed for ~12 days.
3. **Conformance**: every case in reference/conformance_cases.json rendered with the NEW template against our encoder
   and our server (OpenAI and Anthropic requests), token for token with the new tokenizer.json, plus the new repo's
   own golden encodings (encoding/tests) through both.

The new repo's encoding.py is only diffed as text, never run; the template runs in Jinja's sandbox.

## Exit 0 (PASS)

The prompt format is unchanged for us: serve the new weights with the 1017 encoder (scenario A for the template).
Copy the new files into `reference/dsv41-flash-<MMDD>/` only if they differ in bytes.

## Exit 1: what to do

| Report line | Action |
|---|---|
| encoding/encoding.py CHANGED | Port it into tools/v41/dsv41_encoding.py (as 403e47ec did for 1017: copy, keep CRLF). Re-run with `--template <new>/chat_template.jinja`; then `test_serve_ds41.py` and `test_template_conformance.py` |
| markers dsml CHANGED / unknown markers | The DSML parser in tools/v41/serve/serve_ds41.py reads the names from dsv41_encoding (CALLS_OPEN etc.) and spells low-bit variants in `_ANY_TAG`/`_M`: update both, then run real outputs through test_serve_ds41.py. The answer log's `DSML-looking text left in an answer` count (serve_ds41.DSML_LEAKS) must stay 0 on the agent fixture |
| tokenizer.json CHANGED | New special token ids: export the tokenizer again (tools/v41/serve/export_tokenizer.py) and check `<｜end▁of▁sentence｜>` is still one id (Service init asserts it) |
| conformance differences only | Read the first differing character / token printed per case; usually a new default (effort, drop_thinking) or a new block type |
| config.json CHANGED | Not a template matter: the architecture changed. Check that the engine still loads the new GGUF (tools/v41/engine/test_ds41_geometry `--report` lists every tensor it would refuse) before serving it |

After porting, rerun until exit 0 against the new template, then vendor it: `reference/dsv41-flash-<MMDD>/` + point
`template_conformance.REFERENCE` at it, keeping 1017's folder so the change can be rolled back.

## Known deliberate differences of our server (1017)

Listed per case in reference/conformance_cases.json (`known`); the test fails if one disappears.

- Late system messages become user turns (frontend._late_system_to_user); DeepSeek renders `<｜System｜>`.
- Text parts / blocks of one message are joined with no separator (frontend._text_of, anthropic_request); DeepSeek
  joins with a blank line. Claude Code sends a reminder block and the prompt block in one user turn, so this touches
  most agent turns.
- Tool results stay in the order the client sent them (tool_call ids are not kept); DeepSeek sorts them into call order.
- Effort `medium` = 62 (serve_ds41.EFFORT; Anthropic budgets 2K-8K too); DeepSeek's template maps medium to 75.

Changing any of these changes 1017's prompts, so each needs a GPU A/B (agentic fixture + bake-off) before it ships.
