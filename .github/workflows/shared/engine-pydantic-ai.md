---
# Shared runtime + engine config for the Pydantic AI agentic workflows.
#
# The engine itself is the published definition at
# `src/pydantic_ai_harness/gh-aw/pydantic.md`; each workflow imports it. This
# file adds only what is specific to this repository: the workflow agent
# composition, the Z.AI endpoint, and codex-based threat detection.
#
# `PAI_AGENT` loads `.github/scripts/gh_aw_agent`, staged in a pre-agent step.
# The engine owns the model, the AWF sandbox, gateway MCP tools and session
# recording; the composition owns the workflow policies.
#
# Threat detection runs on gh-aw's built-in codex engine, because v1.0.1
# normalizes a custom engine to a built-in one. Its Responses endpoint is the
# Coding Plan's `/api/v1`, reached through the AWF openai proxy so the key stays
# on the host. It emits no OTLP spans.
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
    # gh-aw v1.0.1 cannot run an imported engine in the detection job; it normalizes
    # custom engines to a built-in one. Use the built-in codex engine, which speaks the
    # Responses API, and point it at the Coding Plan's Responses endpoint through the
    # AWF api-proxy. `OPENAI_API_KEY` is a key codex declares, so the Z.AI credential
    # stays on the host and strict mode needs no new repository secret.
    # Keep main's 20-minute detector. gh-aw 1.0.1 applies the detection job's
    # timeout to its step too, so 22 is the floor plus measured setup headroom.
    engine:
      id: codex
      model: ${{ vars.GH_AW_MODEL }}
      env:
        OPENAI_BASE_URL: https://api.z.ai/api/v1
        OPENAI_API_KEY: ${{ secrets.ZAI_API_KEY }}
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
jobs:
  # A separate job keeps the OTLP credential away from the agent job's PATH, which
  # checkout-controlled steps can extend through GITHUB_PATH. The forwarder itself is
  # fetched from the default branch, so neither the checkout nor a PR head supplies it.
  forward_agent_spans:
    needs: [agent]
    if: always()
    runs-on: ubuntu-latest
    timeout-minutes: 5
    permissions:
      actions: read
      contents: read
    steps:
      - name: Download the agent artifact
        continue-on-error: true
        uses: actions/download-artifact@9000827ccba6bdab643e8b6fd33ac0654aef8333 # v8.0.2
        with:
          name: agent
          path: agent
      - name: Forward spans from trusted default-branch code
        env:
          DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
        run: |
          set -euo pipefail
          curl -fsSL "https://raw.githubusercontent.com/${GITHUB_REPOSITORY}/${DEFAULT_BRANCH}/.github/scripts/forward_agent_traces.py" \
            -o "$RUNNER_TEMP/forward_agent_traces.py"
          python3 "$RUNNER_TEMP/forward_agent_traces.py" --path agent/otel-spans.jsonl
---
