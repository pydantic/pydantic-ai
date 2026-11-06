---
# Shared runtime + engine config for the Pydantic AI agentic workflows.
#
# The engine itself is the published definition at
# `src/pydantic_ai_harness/gh-aw/pydantic.md`; each workflow imports it. This
# file adds only what is specific to this repository: the workflow agent
# composition, the Z.AI endpoint, and threat detection on the same engine.
#
# `PAI_AGENT` loads `.github/scripts/gh_aw_agent`, staged in a pre-agent step.
# The engine owns the model, the AWF sandbox, gateway MCP tools and session
# recording; the composition owns the workflow policies.
#
# Threat detection runs on the published engine too: gh-aw v1.0.1 resolves a
# behavior-defined engine for the detection job. The detection module prints the
# model's verdict to stderr so gh-aw's result parser sees the bare
# `THREAT_DETECTION_RESULT:` line.
#
# Usage:
#   imports:
#     - shared/engine-pydantic-ai.md
runtimes:
  uv: {}
max-ai-credits: -1
safe-outputs:
  threat-detection:
    # Detection has an independent budget and the same unknown-model constraint.
    max-ai-credits: -1
    # gh-aw v1.0.1 does not accept an imported engine for the detection job:
    # `getThreatDetectionEngineID` normalizes any non-built-in engine to one of
    # copilot/claude/codex (`declarableDetectionEngineIDs`). Keep the existing
    # Claude CLI detector on the Z.AI endpoint until upstream supports the
    # published engine. Keep main's 20-minute detector; gh-aw 1.0.1 applies the
    # detection job's timeout to its step too, so 22 is the floor plus measured
    # setup headroom.
    engine:
      id: claude
      model: ${{ vars.GH_AW_MODEL }}
      env:
        ANTHROPIC_BASE_URL: https://api.z.ai/api/anthropic
        ANTHROPIC_API_KEY: ${{ secrets.ZAI_API_KEY }}
        GH_AW_HARNESS_MAX_RETRIES: '3'
pre-agent-steps:
  - name: Stage the workflow agent composition
    run: |
      set -euo pipefail
      install -d /tmp/gh-aw/agent-module
      cp -R .github/scripts/gh_aw_agent /tmp/gh-aw/agent-module/
  # gh-aw's repository checkout happens between pre-steps and pre-agent-steps,
  # so this reads the workspace copy of the scripts.
  - name: Install tools for the AWF sandbox (ripgrep, uv)
    run: bash .github/scripts/install-sandbox-tools.sh
post-steps:
  - name: Forward agent spans to the OTLP endpoint
    if: always()
    run: python3 .github/scripts/forward_agent_traces.py
---
