---
# Shared pre-agent-steps for the Pydantic AI gh-aw shim.
#
# These steps run after checkout but before the agent container starts, so
# they still have open-network access.
#
# Stage launcher: gh-aw's repository checkout happens between pre-steps and
# pre-agent-steps, so this step reads from .github/scripts/ in the workspace.
# The launcher is staged into gh-aw's exec-able /tmp/gh-aw/bin/ path.
#
# Pre-warm: warms the harness's uv script environment on the open network so
# the firewalled agent reuses a warm cache. Non-fatal on failure.
#
# Usage:
#   imports:
#     - shared/pre-agent-steps.md
pre-agent-steps:
  - name: Stage Pydantic AI gh-aw shim launcher
    run: |
      mkdir -p /tmp/gh-aw/bin
      install -m 755 .github/scripts/pydantic-ai-runner-launch.sh /tmp/gh-aw/bin/pydantic-ai-runner-launch
  # Install ripgrep and expose uv+rg inside the AWF chroot.
  # AWF auto-merges /opt/hostedtoolcache/**/bin into the container PATH
  # and also reads $GITHUB_PATH entries added before the engine step.
  - name: Install tools for AWF sandbox (ripgrep)
    run: bash .github/scripts/install-sandbox-tools.sh
  - name: Pre-warm Pydantic AI gh-aw shim uv environment
    run: bash .github/scripts/prewarm-pydantic-ai-runner.sh
---
