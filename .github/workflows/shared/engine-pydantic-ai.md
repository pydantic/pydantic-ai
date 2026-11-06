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
  # gh-aw restores `.github` from the base branch on a PR run, so the workspace copy
  # can pair the lock's engine version with an older composition. The workflow's own
  # commit is the one its lock was compiled from: check that out and stage it, so the
  # composition and the library the engine installs come from one commit.
  - name: Stage the workflow agent composition
    run: |
      set -euo pipefail
      composition_src=/tmp/gh-aw/composition-src
      git -C "$GITHUB_WORKSPACE" worktree remove --force "$composition_src" 2>/dev/null || true
      rm -rf "$composition_src"
      git -C "$GITHUB_WORKSPACE" worktree add --detach "$composition_src" "$GITHUB_SHA"
      install -d /tmp/gh-aw/agent-module
      rm -rf /tmp/gh-aw/agent-module/gh_aw_agent
      cp -R "$composition_src/.github/scripts/gh_aw_agent" /tmp/gh-aw/agent-module/
  # gh-aw's repository checkout happens between pre-steps and pre-agent-steps,
  # so this reads the workspace copy of the scripts.
  - name: Install tools for the AWF sandbox (ripgrep, uv)
    run: bash .github/scripts/install-sandbox-tools.sh
post-steps:
  # gh-aw only allows a custom job before the agent or after detection, so this runs
  # in the agent job. It fetches the forwarder from the default branch and calls the
  # interpreter and curl by absolute path, so a directory prepended to PATH does not
  # redirect these two calls. The rest of the step is not isolated: it declares no
  # `shell:`, so `bash` itself still resolves through PATH, and GITHUB_ENV can set
  # `BASH_ENV`, `LD_PRELOAD` or `PYTHONPATH` for it.
  - name: Forward agent spans to the OTLP endpoint
    if: always()
    continue-on-error: true
    env:
      DEFAULT_BRANCH: ${{ github.event.repository.default_branch }}
    run: |
      set -euo pipefail
      /usr/bin/curl -fsSL "https://raw.githubusercontent.com/${GITHUB_REPOSITORY}/${DEFAULT_BRANCH}/.github/scripts/forward_agent_traces.py" \
        -o "$RUNNER_TEMP/forward_agent_traces.py"
      /usr/bin/python3 "$RUNNER_TEMP/forward_agent_traces.py"
---
