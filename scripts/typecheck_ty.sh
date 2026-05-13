#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${TY_BIN:-}" ]]; then
  if command -v ty >/dev/null 2>&1; then
    TY_BIN=ty
  else
    echo "Set TY_BIN to a ty executable, for example a build of dsfaccini/ruff@codex/ty-pydantic-compat." >&2
    exit 127
  fi
fi

TY_OUTPUT_FORMAT="${TY_OUTPUT_FORMAT:-concise}"
TY_PATHS="${TY_PATHS:-pydantic_ai_slim}"
read -r -a ty_paths <<< "$TY_PATHS"

"$TY_BIN" check \
  --python-version 3.13 \
  --output-format "$TY_OUTPUT_FORMAT" \
  "${ty_paths[@]}"
