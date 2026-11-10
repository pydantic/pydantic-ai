# Verdict rubric

Every finding ends with its own line `Verdict: <do|defer|skip> — <one-line reason>`. Assign the
verdict by **net value (benefit − cost − risk), not effort**. State the reason the first time,
on that line. Never write a fuzzy label such as "low-ROI" or "nice-to-have".

## The four verdicts

- **net-positive → `do`** — actionable now. Either the fix is cheap and clearly correct (it
  removes a real future cost), or the finding is *non-deferrable*: without the fix, the branch
  merges something unverified or stale that could be fixed or verified on the spot (a stray
  `Any` or `cast`, an unverified forward-compatibility assumption, stale version copy). A
  non-deferrable finding is a **required action**, never a mere warning.
- **net-positive but out of scope → `defer`** — valid only when one of these holds:
  - (a) doing it now or later changes nothing about the merged artifact; or
  - (b) it opens enough new work to deserve its own issue and PR.

  In case (b), first **verify the underlying behavior on the spot**, then **name a persisted
  tracking artifact**: a GitHub issue, or a tracked PR review comment. A `defer` without a
  tracking artifact is invalid: make it `do` or `skip`.
- **net-neutral → `skip`** — say so. A deliberate, documented scope boundary is a `skip`;
  name it as one.
- **net-negative → `skip`** — name why the fix is worse: it reintroduces a problem the code
  avoids, deletes useful docs or coverage, or adds risk for no gain.

## Before a `skip`

- Verify the claimed benefit in its observable unit when local reproduction is practical.
  Never price an unmeasured benefit at zero.
- A missing linked issue, benchmark or public report does not show missing value. Treat
  explicit maintainer or product context as evidence. Ask the user only when the value stays
  unknown and would change the verdict.
- Before calling machinery novel or scope asymmetric, inspect repository precedent and trace
  the actual artifact or runtime path. Existing precedent removes a novelty objection; it does
  not erase maintenance cost.

## Severity is not actionability

Severity (BLOCKING, WARNING, INFO) and verdict are orthogonal. A WARNING or INFO with `do` is a
required action. Promote a finding to `do` when several independent reviewers converge on it,
or when the fix is comment- or deletion-only. Demote it to net-negative when the fix undoes
something the code did on purpose.
