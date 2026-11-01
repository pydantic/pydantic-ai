#!/usr/bin/env bash
# Install policy dependencies with the same interpreter as the public engine.
set -euo pipefail
# The repository policy uses MCP 1's `McpError`; fastmcp 4 targets the MCP 2 API.
python3 -P -m pip install --quiet --user --disable-pip-version-check \
  'pydantic-ai-slim[logfire,retries]>=2.64.0' 'fastmcp-slim[client]>=3.4.7,<4' 'mcp>=1.28.1,<2'
mkdir -p /tmp/gh-aw/workflow-python
cp -R .github/scripts/pydantic_ai_gh_aw_shim /tmp/gh-aw/workflow-python/
# Keep workflow preferences separate from runner and repository preferences.
export XDG_CONFIG_HOME=/tmp/gh-aw/clai2-config
request_limit=$(PYTHONPATH=/tmp/gh-aw/workflow-python python3 -P -c \
  'from pydantic_ai_gh_aw_shim.cli import run_request_limit; print(run_request_limit())')
"$HOME/.local/bin/clai2" config set run.request_limit "$request_limit"
"$HOME/.local/bin/clai2" config set run.code_mode false
