---
# Shared evidence and accuracy bar for gh-aw prompts.
# gh-aw imports this file; the markdown below (after the closing ---) is
# appended to the agent's task prompt at runtime via {{#runtime-import}}.
---

## Rigor

- Prefer concrete evidence over speculation. Ground claims in exact file
  paths, line numbers, captured outputs, or reproduction steps when the task
  allows.
- If you cannot show the trigger, failure path, or observed behavior, drop the
  claim.
- "I don't know" beats a wrong answer. `mcp__safeoutputs__noop` beats a weak
  or speculative issue or review.
- If you need to hedge with "might", "could", or "possibly", it is not
  ready.

## Adversarial self-review

- Before emitting any issue, discussion, or review, switch sides: assume your
  finding is WRONG and try your hardest to refute it. Emit it only if it
  survives your own strongest counter-argument.
- A false or by-design report costs maintainers more than a missed one.
  Precision beats recall — when in doubt, `mcp__safeoutputs__noop`.
