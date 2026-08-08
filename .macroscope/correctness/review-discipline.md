---
include:
  - "**/*.py"
  - "**/*.md"
  - "**/*.yml"
  - "**/*.yaml"
  - "**/*.ts"
  - "**/*.tsx"
---

Two rules that keep findings precise, applied to every finding.

## Flag what this PR introduces, not what it inherits

Raise a finding only for a problem the PR's own added or changed lines create. A
real bug that already existed in code this PR merely touches, moves, or sits next
to is out of scope: the PR did not introduce it, so flagging it as a blocking
issue here is noise the author dismisses and tracks separately. If a serious
pre-existing defect is genuinely worth surfacing, note it as low-severity and say
it is pre-existing, do not rate it high on this PR.

A line can show as added (`+`) without being new. When code is wrapped in a new
block (a loop, `try`, `with`, `if`) or relocated, its indentation changes and git
records the old line as a delete plus an add. Treat a `+` line as introduced only
when the same text (ignoring leading whitespace) is not also deleted in the same
hunk under unchanged semantics. When in doubt whether the line, and the problem
with it, is genuinely new, prefer not flagging.

## Verify the claim before flagging

A finding must rest on a verified fact, not an inferred one -- the repo's own
"trust but verify" principle applies to review too. Before asserting a mechanism,
confirm it: what a function or stdlib call actually does, what actually failed and
on which commit, how a relative link or path actually resolves from the file it
lives in (`overview.md` referenced from `docs/ui/ag-ui.md` resolves to
`docs/ui/overview.md`, not one directory up). You have code-browsing and web
tools; use them. A finding whose premise is wrong, or a suggested fix that would
itself break the code, costs more trust than a missed nit, so when the mechanism
is unverified, do not raise it.
