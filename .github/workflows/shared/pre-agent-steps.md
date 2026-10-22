---
# Repository policy and tools for the public Pydantic AI engine.
#
# These steps run after checkout but before the agent container starts, so
# they still have open-network access.
#
# The engine import installs CLAI 2. These steps stage repository policy after
# checkout and install its dependencies before the firewall starts.
#
# Usage:
#   imports:
#     - shared/pre-agent-steps.md
pre-agent-steps:
  - name: Install repository workflow policy
    run: bash .github/scripts/stage-gh-aw-agent.sh
  # Install ripgrep and expose uv+rg inside the AWF chroot.
  # AWF auto-merges /opt/hostedtoolcache/**/bin into the container PATH
  # and also reads $GITHUB_PATH entries added before the engine step.
  - name: Install tools for AWF sandbox (ripgrep)
    run: bash .github/scripts/install-sandbox-tools.sh
  # Workflow investigations can install the checkout's development dependencies.
  # Keep reachable tags so dynamic versioning resolves the checkout's 2.x version.
  # Remove the mutable `clai2-bleeding` checkout tag before fetching its current value.
  - name: Restore full history and tags for dynamic versioning
    run: |
      if git show-ref --verify --quiet refs/tags/clai2-bleeding; then
        git tag -d clai2-bleeding
      fi
      if [ -f "$(git rev-parse --git-dir)/shallow" ]; then
        git fetch --unshallow --tags origin
      else
        git fetch --tags origin
      fi
---
