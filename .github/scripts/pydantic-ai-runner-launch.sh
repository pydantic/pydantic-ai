#!/usr/bin/env bash
# In-sandbox launcher for the Pydantic AI harness.
#
# The checked-out workspace is mounted no-exec in the AWF sandbox, so this
# script is installed into gh-aw's exec-able /tmp/gh-aw/bin/ by the workflow's
# pre-step (`install -m 755 ... /tmp/gh-aw/bin/pydantic-ai-runner-launch`).
# It is gh-aw's `engine.command` for every Pydantic AI agentic workflow and
# `uv run --script`s the in-tree harness with the agent's argv.
#
# setup-uv points UV_CACHE_DIR at ${RUNNER_TEMP}/setup-uv-cache, which is not
# writable by the chrooted sandbox user (UID 1001). Only /tmp/gh-aw is owned
# by that user, so redirect every uv-writable dir there.
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
uv_bin=""
if command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
else
  for c in "${HOME}/.local/bin/uv" "${RUNNER_TOOL_CACHE:-/opt/hostedtoolcache}"/uv/*/*/uv /opt/hostedtoolcache/uv/*/*/uv /home/runner/work/_tool/uv/*/*/uv /usr/local/bin/uv; do
    [ -x "$c" ] && uv_bin="$c" && break
  done
fi
if [ -z "${uv_bin}" ]; then
  echo "[harness-launch] FATAL: uv not found; PATH=${PATH}" >&2
  exit 127
fi
echo "[harness-launch] using uv=${uv_bin}" >&2
exec "${uv_bin}" run --script "${runner}" "$@"
