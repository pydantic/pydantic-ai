---
name: issue-fix-tdd
description: "TDD bug-fix flow ported from local /mre-bug-workflow. Failing pytest first; abort if not reproducible; fix; verify; emit MRE scripts; push; open PR."
---

---
name: issue-fix-tdd
description: TDD-style fix for a GitHub issue. Read brief context if present, write a failing pytest reproducing the bug, implement the fix, generate MRE scripts (UV inline deps), commit, push, open a PR. Cloud-side cousin of the local /mre-bug-workflow.
---

# Issue fix (TDD)

You produce a PR that fixes the linked GitHub issue. **Tests first.**
If the bug doesn't reproduce in a fresh test, abort instead of guessing.
If the fix doesn't pass tests, abort instead of pushing.

## Output path

`/workspace/.artifact-output/issue-fix.json`

The team-app picks this JSON up on turn-complete and records a
`fix_attempt` row keyed on (owner, repo, number, session_id).

## Inputs

The user message contains a GitHub issue URL. Parse `<owner>/<repo>`
and `<number>`.

If `/workspace/.aica-context/issue-brief.md` exists, that's the **primary
context** — the orchestrator put the latest brief artifact body there.
Read it first; the brief carries the Fix roadmap, complexity rating,
docs-consulted, and affected-surface sections.

If the brief context isn't present (manual standalone dispatch), fall
back to: `gh issue view <number> --repo <owner>/<repo> --json
title,body,comments,labels,state`. Spawn an Explore subagent to map
affected files / tests. The fix won't be as well-grounded as one
chained from a brief, but the workflow still ships value.

## Steps

1. **Branch.** `git checkout -b aica/fix-issue-<number>`. Use the
   issue number verbatim. If the branch already exists locally, that's a
   re-attempt — keep going on it, don't abort.
2. **Write the failing test.** From the brief's `Fix roadmap` (or
   inferred from the issue), pick the most direct test to write. Place
   it at the path the roadmap suggests (or the natural location for
   the affected surface). Run only that test:
   `uv run pytest <path>::<test_name> -x`. **It must fail.** If it
   passes, you can't reproduce — write `outcome: cannot-reproduce`
   to the output JSON and stop. Do not push, do not commit.
3. **Implement the fix.** Follow the brief's roadmap; touch only files
   in the brief's `Affected surface`. Format/lint per repo conventions
   (`make format && make lint` if those targets exist).
4. **Re-run the targeted test.** It must pass. If it still fails after
   one fix attempt, you may try once more. After two failed attempts
   write `outcome: fix-incorrect` and stop.
5. **Run the broader test suite** for the touched module (`uv run
   pytest <module>/`) to catch regressions. If new failures appear in
   files you touched, treat as fix-incorrect.
6. **Generate MRE scripts** at `local-notes/mre/` (the local
   convention from the source skill):
   - `mre_release.py` — UV inline dep on the published `pydantic-ai`
     PyPI release; demonstrates the bug.
   - `mre_branch.py` — UV inline dep on `pydantic-ai @ file:///workspace`
     (or whatever the worktree path is); demonstrates the fix.

   Both scripts are runnable standalone via `uv run <path>`. Commit
   them.
7. **Commit.** Single commit, conventional format:
   `<scope>: <one-line summary> (fix #<number>)`. Body lists the
   touched files and references the brief if present.
8. **Push.** `git push -u origin aica/fix-issue-<number>`.
9. **Open the PR.** `gh pr create --base main --head
   aica/fix-issue-<number> --title "..." --body "..."` with a body
   that embeds:
   - the issue link (closes #<N>)
   - a link to the brief artifact version, when known (the team-app's
     dashboard URL)
   - a link to the playground share, when present (deferred for v1 —
     leave a TODO line if no playground key)
   - the MRE script paths
   - a one-paragraph summary of the change

10. **Write the JSON** to `/workspace/.artifact-output/issue-fix.json`:

```json
{
  "outcome": "pr-opened",
  "repoOwner": "<owner>",
  "repoName": "<repo>",
  "number": <N>,
  "prUrl": "<URL returned by gh pr create>",
  "prBranch": "aica/fix-issue-<N>",
  "briefArtifactId": null,
  "briefVersionId": null,
  "playgroundShareUrl": null,
  "notes": "1-paragraph: which test reproduced, which file(s) you changed, anything risky."
}
```

Or, on abort:

```json
{
  "outcome": "cannot-reproduce" | "fix-incorrect" | "error",
  "repoOwner": "<owner>",
  "repoName": "<repo>",
  "number": <N>,
  "notes": "what you tried, what failed, what next session should start from."
}
```

## Rules

- **Tests first.** Always write the failing test before any fix code.
  No fix without a reproducer.
- Don't touch `.github/`, `docs/`, or `pyproject.toml` unless the
  brief explicitly calls for it.
- One commit per fix attempt. No squash. No rebase. The branch is
  short-lived.
- The PR title must include `fix #<number>` so GitHub auto-closes the
  issue on merge.
- Don't post comments on the issue — the PR's "fix #<N>" closes it; the
  team-app will surface the fix attempt in its UI.
- The output JSON is the team-app's signal. Don't summarise the fix in
  chat; the PR + the JSON are the deliverables.
