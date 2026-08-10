---
# Fetch the issue and PR indexes only for workflows that may file an issue.
# The credential is not exposed to dependency installation or to the model.
pre-agent-steps:
  - name: Prefetch issue and PR context
    env:
      GH_TOKEN: ${{ github.token }}
    run: bash .github/scripts/prefetch-github-context.sh
---
