# DeepSeek reference files (R-5 prompt conformance)

`dsv41-flash-1017/` holds byte-exact copies from https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 at
commit 08c8657f02291e11b6f6b63e0e16014f8e748518 ("Add chat template (#11)", 2026-11-10). MIT License, Copyright (c)
2023 DeepSeek (the repo's LICENSE).

| File | sha256 |
|---|---|
| chat_template.jinja (275 lines) | 95b7033a693b321ceaefb6f89ad07d1e0cf2b095305711f934a4ac649456739d |
| config.json | 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879 |
| tokenizer_config.json | 6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547 |
| encoding_tests/ (= the repo's encoding/tests: DeepSeek's golden encodings) | see the files |
| tokenizer.json (not copied, 6.4 MB) | c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b |
| encoding/encoding.py (not copied: it is ../dsv41_encoding.py, the same text with CRLF) | 13231c592e958fb39e2af6001dae69d90e7a36b265702b9b390236fb387599e2 |

`conformance_cases.json`: the conversations that `../template_conformance.py` renders through the template, the
encoder and the server (system, tools, multi-turn, thinking on / off, tool results, parallel calls, images, mid
system messages, every effort spelling, response_format, namespaces, latest_reminder, task tokens). A case's
`known` names a difference of the server from DeepSeek's template that is deliberate, with the reason.

`.gitattributes` keeps this folder `-text`: no line-ending conversion, so the hashes hold on every checkout.
