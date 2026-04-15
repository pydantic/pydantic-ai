# `gh` CLI setup required

This repo's workflow depends on an installed, authenticated `gh` CLI — hooks, the `address-feedback` skill, and the post-push review flow all use it.

Before making any code changes or running non-read-only tool calls, walk the user through these three steps:

1. **Install** (if missing): `brew install gh` on macOS, or see <https://github.com/cli/cli#installation> for other platforms.
2. **Authenticate**: `gh auth login`
3. **Verify**: `gh auth status` should print 'Logged in to github.com'.

Do not proceed with the task until all three succeed.
