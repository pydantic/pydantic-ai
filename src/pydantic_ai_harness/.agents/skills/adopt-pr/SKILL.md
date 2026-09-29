---
name: adopt-pr
description: Bootstrap branch-context on an existing PR. Writes issue-brief.md from the linked issue(s) + current PR state, and backfills pr-decisions.md with decision-bearing entries from already-resolved review threads. Use when picking up a PR mid-flight (yours or someone else's) without prior local context.
user-invocable: true
allowed-tools: Bash(git:*), Bash(gh:*), Bash(jq:*), Bash(date:*), Bash(ls:*), Bash(cat:*), Bash(cp:*), Bash(.claude/skills/branch-context/*), Read, Write, Edit, Glob, Grep, AskUserQuestion
---

# Adopt PR -- bootstrap branch-context on an existing PR

Populate `issue-brief.md` and `pr-decisions.md` for a worktree whose PR already
exists and has history.

## When to use

- You checked out an existing PR and the branch-context files are empty
  templates (or absent).
- You are picking up someone else's PR.
- Your own PR predates the branch-context setup and you want to backfill.

Do not use this for fresh work with no PR yet -- write the brief from the linked
issue directly.

## Startup

Bootstrapping presumes the linked issue describes a real problem. If you have
not confirmed that, skim the issue before adopting -- authorship (a bot,
contributor, or automated sweep) is not proof the problem is real.

1. Read `CLAUDE.md` for project context.
2. Instantiate the branch-context instances if they do not exist:
   ```bash
   cd "$(git rev-parse --show-toplevel)"
   D=.claude/skills/branch-context
   [ -f "$D/issue-brief.md" ]   || cp "$D/issue-brief.template.md" "$D/issue-brief.md"
   [ -f "$D/pr-decisions.md" ]  || cp "$D/pr-decisions.template.md" "$D/pr-decisions.md"
   ```
   If `issue-brief.md` already has populated `issues:` frontmatter, ask via
   `AskUserQuestion` whether to overwrite the brief, re-seed decisions, both, or
   neither, before proceeding.

## Step 1 -- Resolve the PR

Parse `$ARGUMENTS`:
- **PR number/URL** -> use it directly.
- **Empty** -> detect from the current branch:
  ```bash
  PR_NUMBER=$(gh pr view --json number -q .number 2>/dev/null)
  ```
  If detection fails, ask via `AskUserQuestion`.

Fetch PR metadata:
```bash
gh pr view "$PR_NUMBER" --json number,title,url,state,headRefName,body,closingIssuesReferences,createdAt,author
```

## Step 2 -- Resolve linked issues

Two sources, in order:
1. `closingIssuesReferences` from the PR metadata (canonical -- set via the PR
   sidebar or `Fixes #N` keywords).
2. Fallback: grep the PR body for `(?:Fixes|Closes|Resolves|Relates to) #\d+` if
   (1) is empty.

For each linked issue:
```bash
gh issue view "<N>" --json number,url,title,state,body,labels,comments
```

If there is no linked issue and the body has no problem description worth
linking, proceed with `issues: []` (free-text problem) -- ask the user whether to
continue or abort.

## Step 3 -- Study the existing diff

Understand the code before synthesizing the brief:
```bash
git fetch origin main 2>/dev/null || git fetch origin master
MERGE_BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD origin/master)
git diff --stat "$MERGE_BASE"..HEAD
git log --oneline "$MERGE_BASE"..HEAD
```

Map the changes: which files/modules are touched, which tests exist on the
branch (existing tests are implied success criteria), which public symbols
changed, any notable new abstractions.

## Step 4 -- Write `issue-brief.md`

Use the schema in `issue-brief.template.md`. Adaptations for adoption:

- `related_pr`: the PR URL (not `TBD` -- the PR already exists).
- `branch`: `git rev-parse --abbrev-ref HEAD`.
- **Success criteria** -- derive from the issue text and from tests already on
  the branch (each existing test is a de facto criterion; cross-reference them
  in the table).
- **Affected surface** -- extract from the diff, not from a planning pass.
- **Constraints** -- include anything reviewers have already emphasized in
  comments (e.g. "must preserve backwards-compat" from a maintainer reply).

Write to `.claude/skills/branch-context/issue-brief.md`, overwriting the
template.

## Step 5 -- Backfill `pr-decisions.md` from resolved threads

Fetch resolved review threads via the GraphQL API:
```bash
read -r OWNER REPO < <(gh repo view --json owner,name -q '.owner.login + " " + .name')
gh api graphql -f query='
query($owner:String!,$repo:String!,$pr:Int!){
  repository(owner:$owner,name:$repo){
    pullRequest(number:$pr){
      reviewThreads(first:100){ nodes{
        isResolved
        comments(first:50){ nodes{ author{login} bodyText url path } }
      }}
    }
  }
}' -F owner="$OWNER" -F repo="$REPO" -F pr="$PR_NUMBER" \
  --jq '.data.repository.pullRequest.reviewThreads.nodes | map(select(.isResolved))'
```

For each resolved thread, read the full conversation, then classify:

- **Decision-bearing** -- the thread debated two or more options, or a reviewer
  flagged a concern and the author adjusted the code (e.g. "use kwargs over
  positional", "renamed the public method", "moved the helper to a different
  module").
- **Noise** -- typo fixes, "nit: missing docstring", "good catch thanks",
  resolved without a code change. Skip these.

For each decision-bearing thread, append an entry (Source is the root comment's
`url`):
```bash
.claude/skills/branch-context/append-pr-decision.sh \
  --title "thread: <short title>" \
  --decision "<one-line summary of what was decided>" \
  --why "<one-line reason, quoting reviewer or author if concise>" \
  --source "<thread URL>" \
  --iter "-"
```

Decision budget: aim for <=10 entries. If more resolved threads than that seem
worth logging, you are probably over-including noise -- re-apply the filter.

## Step 6 -- Seed an "adoption" meta-entry

Append one final entry marking the boundary between backfilled decisions
(everything above) and live-logged ones (everything below):
```bash
.claude/skills/branch-context/append-pr-decision.sh \
  --title "adopted PR #<N> at <DATE>" \
  --decision "Branch-context bootstrapped from existing PR + issue(s). Decisions above are backfilled from resolved threads." \
  --why "PR predates the branch-context setup" \
  --source "<PR URL>" \
  --iter "-"
```

## Step 7 -- Report

Print a concise summary:
```
Adopted PR #<N> -- "<title>"
  Issues linked: #<A>, #<B>
  Resolved threads seeded: <count> decisions
  Unresolved threads (for later triage): <count>
```

## Rules

- Do not open a new PR (it already exists). Do not commit anything to the branch
  during adoption -- only read, classify, and write the two branch-context files.
- Do not re-classify unresolved threads during adoption -- just report the count.
- Do not log every resolved thread -- only decision-bearing ones. The decisions
  log is meant to reward reading; diluting it with noise defeats the point.
- Every backfilled entry must have a thread URL in `Source:`. If you cannot find
  one, you are over-inferring -- skip it.
- If the existing brief/decisions files are already populated (not templates),
  the user's Startup confirmation governs whether to overwrite.
