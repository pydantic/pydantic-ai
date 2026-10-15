#!/usr/bin/env bash
# Install policy dependencies with the same interpreter as the public engine.
set -euo pipefail
python3 -P -m pip install --quiet --user --disable-pip-version-check \
  'pydantic-ai-slim[retries]==2.56.0' 'fastmcp-slim[client]==3.4.7' 'mcp==1.28.1' 'logfire[httpx]>=4.39.0'
mkdir -p /tmp/gh-aw/workflow-python
cp -R .github/scripts/pydantic_ai_gh_aw_shim /tmp/gh-aw/workflow-python/
# Keep workflow preferences separate from runner and repository preferences.
export XDG_CONFIG_HOME=/tmp/gh-aw/clai2-config
request_limit=400
if [ "${GITHUB_WORKFLOW:-}" = 'Pydantic AI Attention Triage' ]; then
  request_limit=50
fi
"$HOME/.local/bin/clai2" config set run.request_limit "$request_limit"
"$HOME/.local/bin/clai2" config set run.code_mode false
