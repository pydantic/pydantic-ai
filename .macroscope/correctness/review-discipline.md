Two rules that keep findings precise, applied to every finding.

## Flag what this PR introduces, not what it inherits

Raise a finding only for a problem the PR's own added or changed lines create. A
real bug that already existed in code this PR merely touches, moves, or sits next
to is out of scope: the PR did not introduce it, so flagging it here is noise the
author dismisses and tracks separately. A pre-existing defect worth surfacing
belongs in its own issue, not in this PR's review output at any severity.

A line can show as added (`+`) without being new. When code is wrapped in a new
block (a loop, `try`, `with`, `if`) or relocated, its indentation changes and git
records the old line as a delete plus an add. Treat a `+` line as introduced only
when the same text (ignoring leading whitespace) is not deleted anywhere in the
PR's diff -- including another hunk or another file, since code moves across both
-- under unchanged semantics. When in doubt whether the line, and the problem
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
