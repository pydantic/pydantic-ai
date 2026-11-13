#!/usr/bin/env bash
# Stage the harness's source, then pre-warm its uv script environment on the OPEN network.
#
# No workflow runs it: the workflows use the public `pydantic-ai` engine. A workflow
# using `pydantic-ai-runner-launch.sh` runs it in `pre-agent-steps`, after checkout and
# Setup uv but before the firewalled agent step, into the same uv dirs the launcher
# uses, so the agent run reuses the warm cache instead of PyPI through the AWF firewall.
#
# The launcher runs the shim from a checkout of the workflow's own commit,
# never from the workspace. A run that checks out a PR head gets `.github`
# restored from the base branch, so the workspace pairs the base's shim with
# the head's `pydantic_ai_slim`, and the shim dies at import on a head that
# predates a symbol it uses. On `pull_request` and `pull_request_review` runs
# the workflow's commit is the PR's merge commit, so those runs use the PR's
# own shim, not the base's. Staging that checkout is fatal: without it the
# launcher has nothing to run.
#
# The pre-warm itself is non-fatal: on failure the sandboxed `uv run --script`
# (with the `python` allowlist in each workflow) still installs from scratch.
set -uo pipefail
export UV_CACHE_DIR=/tmp/gh-aw/uv/cache
export UV_PYTHON_INSTALL_DIR=/tmp/gh-aw/uv/python
export UV_TOOL_DIR=/tmp/gh-aw/uv/tool
export XDG_DATA_HOME=/tmp/gh-aw/uv/data
export XDG_CACHE_HOME=/tmp/gh-aw/uv/xdg-cache
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_TOOL_DIR" "$XDG_DATA_HOME" "$XDG_CACHE_HOME"
# Must match `pydantic-ai-runner-launch.sh`.
runner_src="${PYDANTIC_AI_RUNNER_SRC:-/tmp/gh-aw/runner-src}"
if ! git -C "${GITHUB_WORKSPACE}" worktree add --detach "${runner_src}" "${GITHUB_SHA}"; then
  echo "::error::could not check out ${GITHUB_SHA} at ${runner_src} for the Pydantic AI gh-aw shim"
  exit 1
fi
runner="${runner_src}/.github/scripts/pydantic-ai-runner"
uv_bin="$(command -v uv 2>/dev/null || true)"
if [ -z "${uv_bin}" ]; then
  echo "::warning::uv not found for pre-warm; agent will install under the firewall"
  exit 0
fi
echo "[harness-prewarm] using uv=${uv_bin} cache=${UV_CACHE_DIR} runner=${runner}"
"${uv_bin}" sync --script "${runner}" \
  || echo "::warning::harness uv pre-warm failed; agent will install under the firewall"
