---
name: adopt-pr
description: Bootstrap branch-context on an existing PR that wasn't started via /initialize-worktree. Writes issue-brief.md from the linked issue(s) + current PR state, and backfills pr-decisions.md with decision-bearing entries from already-resolved review threads. Use when picking up a PR mid-flight (either yours or someone else's) without prior local context.
---

# Adopt PR — bootstrap branch-context on an existing PR

Populate `issue-brief.md` and `pr-decisions.md` for a worktree whose PR already exists and has history. Companion to `/initialize-worktree`, which only handles fresh starts.

## When to use

- You just checked out an existing PR as a worktree (`pyai-checkout <pr>`) and the branch-context files are empty templates
- You're picking up someone else's PR
- Your own PR predates the branch-context setup and you want to backfill

Do NOT use this for fresh work — `/initialize-worktree` is the entry point there.

## Startup

**Premise gate — adoption presumes the linked issue is real.** If no premise validation (`/assess-readiness`, or an independent issue-validity assessment derived from the problem alone) preceded this adoption, **STOP and flag to the manager that readiness wasn't run** — don't bootstrap + review on an unvalidated premise. Authorship of the issue or PR (bot, contributor, or an automated/sweep issue) is never proof the bug is real. Proceed only once the manager confirms the premise was validated.

1. Read `CLAUDE.md` and `CLAUDE.local.md` for project context
2. Skim the DDD+ protocol + reviewer priority in the autoloaded `CLAUDE.local.md` for shared vocabulary
3. Verify the branch-context files exist and are still the unfilled templates:
   - If `issue-brief.md` already has populated `issues:` frontmatter → ask: "Brief is already populated. Overwrite? Re-seed decisions? Both? Neither?" before proceeding.

## Step 1 — Resolve the PR

Parse `$ARGUMENTS`:
- **PR number/URL** → use it directly
- **Empty** → detect from current branch:
  ```bash
  PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null)
  ```
  If detection fails, ask the user for the PR.

Fetch PR metadata:
```bash
gh pr view $PR_NUMBER --json number,title,url,state,headRefName,body,closingIssuesReferences,createdAt,author
```

## Step 2 — Resolve linked issues

Two sources, in order:
1. `closingIssuesReferences` from the PR metadata (canonical — set via the PR sidebar or `Fixes #N` keywords)
2. Fallback: grep the PR body for `(?:Fixes|Closes|Resolves|Relates to) #\d+` if (1) is empty

For each linked issue:
```bash
gh issue view <N> --json number,url,title,state,body,labels,comments,updatedAt
```

If no linked issue and body has no problem description worth linking, proceed with `issues: []` (free-text problem) — ask the user whether to continue or abort.

## Step 3 — Study the existing diff

The PR already has code. Understand it before synthesizing the brief:

```bash
git fetch upstream main 2>/dev/null || git fetch origin main
MERGE_BASE=$(git merge-base HEAD upstream/main 2>/dev/null || git merge-base HEAD origin/main)
git diff --stat $MERGE_BASE..HEAD
git log --oneline $MERGE_BASE..HEAD
```

Inspect the diff to map the changes:
- What files/modules are touched
- What tests exist on the branch (existing tests = implied success criteria)
- What public symbols changed
- Any notable new abstractions or patterns

## Step 4 — Write `issue-brief.md`

Use the same schema as `/initialize-worktree` Step 3 (see `.claude/skills/initialize-worktree/SKILL.md`). Specific adaptations for adoption:

- `related_pr`: the PR URL (not `TBD` — the PR already exists)
- `branch`: `git rev-parse --abbrev-ref HEAD`
- `issues[].updated_at`: the issue's current `updatedAt` value from GitHub
- `issues[].comments_fingerprint`: run
  `.agents/skills/branch-context/issue-comment-fingerprint <issue-number>`; the helper paginates
  comment `id` and `updatedAt` through GraphQL because `gh issue view` omits comment `updatedAt`.
- **Success criteria** — derive from:
  1. The issue text (as usual)
  2. Tests already on the branch — each existing test is a *de facto* criterion. Cross-reference them explicitly in the table.
- **Affected surface** — extract from the diff, not from a planning pass
- **Constraints** — include anything the reviewers have already emphasized in comments (e.g. "must preserve backwards-compat" from a maintainer reply)

Write to `.claude/skills/branch-context/issue-brief.md`, overwriting the template.

## Step 5 — Backfill `pr-decisions.md` from resolved threads

Fetch all review threads:
```bash
.agents/skills/adopt-pr/fetch-resolved-threads $PR_NUMBER > /tmp/adopt-pr-threads.json
```

The helper returns every resolved thread with its complete comment conversation. Inspect each
thread directly:
```bash
jq '.[] | {id, comments}' /tmp/adopt-pr-threads.json
```

For each resolved thread, read the full conversation. Then classify:

- **Decision-bearing** — the thread debated two or more options, or the reviewer flagged a concern and the author adjusted the code. Examples: "use kwargs over positional", "renamed the public method", "moved the helper to a different module".
- **Noise** — typo fixes, "nit: missing docstring", "good catch thanks", resolved without code change. **Skip these**.

For each decision-bearing thread, append an entry:
```bash
.claude/skills/branch-context/append-pr-decision.sh \
  "thread <N>: <short title>" \
  "<one-line summary of what was decided>" \
  "<one-line reason, quoting reviewer or author if concise>" \
  "<thread URL — use the root comment's html_url>" \
  "-"
```

Iteration is `-` because adoption runs outside the ralph loop.

Decision budget: **aim for ≤10 entries**. If there are more resolved threads than that worth logging, you're probably over-including noise. Re-apply the filter.

## Step 6 — Seed a "adoption" meta-entry

Append one final entry documenting the adoption itself:
```bash
.claude/skills/branch-context/append-pr-decision.sh \
  "adopted PR #<N> at <DATE>" \
  "Branch-context bootstrapped from existing PR + issue(s). Decisions prior to this entry are backfilled from resolved threads." \
  "PR predates the branch-context setup" \
  "<PR URL>" \
  "-"
```

This marks the boundary between backfilled decisions (everything above) and live-logged decisions (everything below going forward).

## Step 7 — Takeover comment (contributor PRs only)

Skip this for your own PRs. When you're taking over **someone else's** PR, post a short takeover comment so the contributor knows you've got it and to avoid churn. Assume the PR is "ready" in the contributor's eyes when it's **non-draft with green CI** — sometimes the author signals explicitly (requests a review, pings you, leaves a comment). The comment should:

- **Thank them** for the contribution.
- **Say you're taking it over** from here.
- **Ask them to hold off on new commits** while you do — every commit they push after this is another diff you have to re-review from scratch, which slows the merge.

GitHub's maintainer-edits metadata is advisory context, not proof of push access:

```bash
gh api repos/pydantic/pydantic-ai/pulls/$PR_NUMBER --jq .maintainer_can_modify
```

Do not call the branch unpushable or ask for maintainer edits from that value. When work later needs
to be pushed, attempt the normal push to the contributor's branch. Only an actual permission error
establishes the restriction; then ask the contributor to enable **Allow edits from maintainers**.

Draft the comment with the repository's required attribution and **confirm with the user before posting** — it's contributor-visible. Post with `gh pr comment $PR_NUMBER --body-file <file>` only on their go.

> Template — "Thanks for this. I'm going to take it over from here to get it across the line. Could you hold off on pushing new commits while we work on it? Each new commit is another change we'd have to re-review, so it's smoother to keep the branch put. Thanks again!"

## Step 8 — Report

Print a concise summary:
```
Adopted PR #<N> — "<title>"
  Issues linked: #<A>, #<B>
  Resolved threads seeded: <count> decisions
  Unresolved threads (for triage): <count>

Next:
  - Inspect unresolved threads with `.agents/skills/pr-review-feedback/sweep-unresolved <PR#>`
  - Then work the feedback (triage → fix → resolve), or hand the PR to `/pr-orchestrator`
```

## Rules

- Don't open a new PR (it already exists). Don't commit anything to the branch during adoption — only read, classify, and write to the two branch-context files.
- Don't re-classify unresolved threads during adoption — just flag the count. Triage happens later when you actually work the feedback.
- Don't log every resolved thread — only decision-bearing ones. The decisions log is meant to reward reading; diluting it with noise defeats the point.
- Every backfilled decision entry must have a thread URL in `Source:`. If you can't find one, you're over-inferring — skip it.
- If the existing `issue-brief.md` / `pr-decisions.md` is already populated (not template), the user's confirmation in Startup step 3 governs whether to overwrite.
- **Fork PRs — push access**: The "pushing to a contributor's fork branch" note under `## PR flow` in `CLAUDE.local.md` applies. Attempt the normal push; if it returns a permission error, flag the exact failure and ask the contributor to enable maintainer edits rather than working around it.
