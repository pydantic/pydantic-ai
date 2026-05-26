#!/usr/bin/env bash
# Pre-warm the harness's uv script environment on the OPEN network.
#
# Runs in each workflow's `pre-agent-steps` — after checkout + Setup uv but
# before the firewalled agent step — into the same uv dirs the in-sandbox
# launcher uses, so the agent run reuses the warm cache instead of depending
# on PyPI access through the AWF firewall.
#
# Strictly non-fatal: on any failure the sandboxed `uv run --script` (with
# the `python` allowlist in each workflow) still installs from scratch.
set -uo pipefail
export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
export XDG_DATA_HOME=/tmp/gh-aw/uv/data
export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
runner="${GITHUB_WORKSPACE}/.github/scripts/pydantic-ai-runner"
uv_bin="$(command -v uv 2>/dev/null || true)"
if [ -z "${uv_bin}" ]; then
  echo "::warning::uv not found for pre-warm; agent will install under the firewall"
  exit 0
fi
echo "[harness-prewarm] using uv=${uv_bin} cache=${UV_CACHE_DIR}"
"${uv_bin}" sync --script "${runner}" \
  || echo "::warning::harness uv pre-warm failed; agent will install under the firewall"
