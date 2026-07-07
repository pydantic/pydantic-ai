---
# Shared adversarial validity gate for bug/behavior-filing gh-aw sweeps.
# gh-aw imports this file; the markdown below (after the closing ---) is
# appended to the agent's task prompt at runtime via {{#runtime-import}}.
#
# Encodes the concrete failure modes that made past sweep issues get closed as
# not-a-bug / by-design / duplicate. Every create-issue must clear this gate.
---

## Adversarial validity gate — mandatory before `create_issue`

Put the finding through this validity gate before filing, reviewing it as a
skeptical maintainer would — treat **"this is NOT a bug"** as the default and
file **only** if it survives every check below; otherwise call
`mcp__safeoutputs__noop`. Most runs should noop — a false or by-design report
costs more maintainer time than a missed one.

Record the results as an **`## Adversarial review`** section in the issue body.
An issue that omits it is incomplete — noop instead.

1. **Reproduced on current `main`, for real.** You must have *executed* code this
   run — a source-level snippet via `uv run python -c …`, a tiny script, or a
   single `uv run pytest -k …` — and **observed** the failure. Paste the exact
   command and its actual output. A claim you only reasoned about (e.g. "this
   *would* fail") is not a bug and is the most common reason past reports were
   rejected as a false premise.

2. **Existing tests don't already bless the behavior.** Grep the suite for the
   symbol / code path and **read** the nearest tests. If a passing test already
   asserts the current behavior — or your proposed fix would change or break any
   existing test — the behavior is intentional → noop. (Past reports proposed
   one-line "fixes" that broke a dozen adapter/serialization tests asserting the
   opposite on purpose.)

3. **Ruled out "by design."** Check for: a nearby comment/docstring explaining
   the choice, the provider profile, a maintainer decision in a linked issue/PR,
   and whether other providers/adapters deliberately do the same thing.
   Programmatic-only fields (`metadata`, `conversation_id`) excluded from wire /
   UI protocols, and request-only parts absent from a *response* union (or vice
   versa), are intentional — not bugs.

4. **No cross-provider false equivalence** *(provider-specific findings only)*.
   If the finding concerns a provider's request/response payload or SDK shape,
   verify the real type for **that** provider from its own types or docs — never
   infer a bug by analogy to a different provider. For provider-agnostic findings
   (core serialization/round-trip, streaming lifecycle, message plumbing), this
   check does not apply — skip it.

5. **Not already tracked.** Re-confirm the dedup above — label-filtered where
   this sweep has a dedicated label, otherwise the full open-issue scan —
   returned nothing covering this exact finding.

If any *applicable* check fails or is genuinely inconclusive,
`mcp__safeoutputs__noop`. A check that doesn't apply (e.g. the provider check for
a core finding) is not a failure — skip it. One issue that clears every
applicable check beats five that don't.
