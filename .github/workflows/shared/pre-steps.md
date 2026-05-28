---
# Shared pre-steps for the Pydantic AI gh-aw shim.
#
# Setting engine.command (in engine-minimax.md) makes gh-aw skip ALL engine
# installation steps, which also drops the bundled AWF firewall binary install.
# This step re-runs gh-aw's own installer so the firewall binary is present.
#
# Usage:
#   imports:
#     - shared/pre-steps.md
pre-steps:
  - name: Install AWF firewall binary (skipped by custom engine.command)
    run: bash "${RUNNER_TEMP}/gh-aw/actions/install_awf_binary.sh" v0.25.46
---
