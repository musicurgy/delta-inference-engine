#!/usr/bin/env bash
# R-5 day 0, one command: a new DeepSeek checkpoint's prompt format against ours.
#   tools/v41/day0_template_diff.sh deepseek-ai/DeepSeek-V4.1-Flash-MMDD [revision] [out_dir]
# Fetches only the small files (template, config, tokenizer, index, encoding/, inference/; no weights), diffs them
# against 1017's, checks the DSML markers, and renders every conformance case with the NEW template against
# dsv41_encoding.py and the server.  Exit 0 = nothing to port; 1 = see tools/v41/DAY0_TEMPLATE.md.
# PY=... picks the Python (needs jinja2; tokenizers for the token comparison); HF_TOKEN is sent if set.
set -euo pipefail
[ $# -ge 1 ] || { sed -n 2,8p "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
if [ -z "${PY:-}" ]; then
  for c in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python" python3 python; do
    if command -v "$c" >/dev/null 2>&1 || [ -x "$c" ]; then PY=$c; break; fi
  done
fi
REPO=$1; REV=${2:-main}; OUT=${3:-day0-${REPO##*/}}
exec "$PY" "$HERE/template_conformance.py" --day0 "$REPO" --revision "$REV" --out "$OUT"
