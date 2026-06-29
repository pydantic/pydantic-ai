#!/usr/bin/env bash
# Install tools that the agent needs inside the AWF sandbox.
#
# Runs in `pre-agent-steps` (on the runner, open network, after checkout)
# before AWF starts the firewalled container. Tools are exposed via:
#
#   1. /opt/hostedtoolcache/gh-aw-tools/current/x64/bin — AWF auto-scans
#      hostedtoolcache bin dirs and merges them into the container PATH.
#   2. $GITHUB_PATH — AWF reads this file at container startup and merges
#      entries into AWF_HOST_PATH (the container's PATH).
#
# This follows the pattern used by elastic/ai-github-actions.
set -euo pipefail

toolcache_bin="/opt/hostedtoolcache/gh-aw-tools/current/x64/bin"
sudo mkdir -p "$toolcache_bin"

# --- ripgrep ---
# The agent's native Grep tool wraps `rg` for fast code search.
echo "[install-sandbox-tools] Installing ripgrep..."
uv tool install ripgrep --force --quiet
rg_path="$(uv tool dir --bin)/rg"
if [ -x "$rg_path" ]; then
  sudo ln -sf "$rg_path" "$toolcache_bin/rg"
  echo "[install-sandbox-tools] rg -> $toolcache_bin/rg"
else
  echo "::warning::ripgrep install succeeded but rg binary not found at $rg_path"
fi

# --- uv ---
# Symlink uv into the toolcache so the launcher and Bash tool can find it.
uv_path="$(command -v uv)"
if [ -n "$uv_path" ]; then
  sudo ln -sf "$uv_path" "$toolcache_bin/uv"
  echo "[install-sandbox-tools] uv -> $toolcache_bin/uv"
fi

# Belt-and-suspenders: also write to $GITHUB_PATH so AWF's GITHUB_PATH
# merge picks it up (works on AWF versions that support this).
echo "$toolcache_bin" >> "${GITHUB_PATH:-/dev/null}"
echo "[install-sandbox-tools] Added $toolcache_bin to GITHUB_PATH"
