#!/usr/bin/env bash
# In-sandbox launcher for the Pydantic AI gh-aw shim.
#
# The checked-out workspace is mounted no-exec in the AWF sandbox, so this
# script is installed into gh-aw's exec-able /tmp/gh-aw/bin/ by the workflow's
# pre-step (`install -m 755 ... /tmp/gh-aw/bin/pydantic-ai-runner-launch`).
# It is gh-aw's `engine.command` for every Pydantic AI agentic workflow and
# `uv run --script`s the in-tree runner stub with the agent's argv. The stub
# (`pydantic-ai-runner`) is a tiny `runpy` shim that hands off to the
# `pydantic_ai_gh_aw_shim` package in the same directory.
#
# AWF propagates setup-* tool paths into the container via $GITHUB_PATH and
# /opt/hostedtoolcache — so `uv` and `rg` should be on PATH already. The
# launcher just sets up the uv cache dirs (the default cache dir from
# setup-uv isn't writable by the sandbox user UID 1001).
set -euo pipefail
export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
export XDG_DATA_HOME=/tmp/gh-aw/uv/data
export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
runner="${GITHUB_WORKSPACE}/.github/scripts/pydantic-ai-runner"
echo "[harness-launch] cwd=$(pwd) GITHUB_WORKSPACE=${GITHUB_WORKSPACE:-unset} UV_CACHE_DIR=${UV_CACHE_DIR}" >&2
echo "[harness-launch] runner=${runner} exists=$([ -f "${runner}" ] && echo yes || echo no)" >&2

# Find uv — should be on PATH via AWF's hostedtoolcache propagation.
# Fall back to known paths if not (older AWF versions or non-GHA runners).
uv_bin=""
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
else
  for c in /opt/hostedtoolcache/gh-aw-tools/current/x64/bin/uv /opt/hostedtoolcache/uv/*/*/uv /tmp/gh-aw/bin/uv /usr/local/bin/uv; do
    [ -x "$c" ] && uv_bin="$c" && break
  done
fi
if [ -z "${uv_bin}" ]; then
  echo "[harness-launch] FATAL: uv not found; PATH=${PATH}" >&2
  exit 127
fi
echo "[harness-launch] using uv=${uv_bin}" >&2
exec "${uv_bin}" run --script "${runner}" "$@"
