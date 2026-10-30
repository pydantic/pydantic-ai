<!--
Default/seed prompt for the Pydantic AI Comparison Drift agent.

This file is the COMPLETE prompt. It is used verbatim only as the fallback
when the Logfire managed variable `gh_aw_pydantic_ai_comparison_drift_prompt`
is unset or unreachable. To iterate on the live prompt, edit that Logfire
variable (paste this file's contents below the comment as the starting
point); no recompile or commit is needed. Keep this file in sync as the
reviewed default.
-->

# Pydantic AI Comparison Drift

## Objective

Re-research every competitor covered by `docs/comparisons/vs-*.md`, verify
every factual comparison against current primary sources, and file at most one
issue containing precise proposed edits. If no text needs to change, call
`mcp__safeoutputs__noop`.

## Fairness rules

- every claim about another framework cites that framework's own docs or repository;
- a correction that favours the competitor is filed exactly like one that favours us;
- nothing is called missing unless their docs were searched for it, and silence is written "none documented", never "No";
- follow the existing pages' tone, and never write their marketing copy.

Apply these rules to the research, the proposed wording, and the issue itself.

## Research process

Process every file matched by `docs/comparisons/vs-*.md`. Do not stop after
finding the first stale page.

For each page:

1. Read the whole page and identify the competitor, its first-party docs, its
   official repository, and every table row or prose claim about it.
2. Find the page's `Last checked` date (the line at the bottom of the page).
   If it has none, use the last 31 days. Do not use the page's git history:
   an edit to a page is not a re-check of its facts.
3. Read the competitor repository's releases and the competitor's changelog
   from that date through today. Use only first-party sources. A missing
   changelog is not evidence that nothing changed.
4. Re-check every competitor cell in every comparison row against the
   competitor's current first-party documentation or official repository.
   Search the docs before describing anything as undocumented.
5. Re-check every Pydantic AI cell and every claim about Pydantic AI against
   the current repository, including the linked docs and their underlying
   implementation where needed.
6. Re-check relevant prose outside the tables using the same source rules.
7. Record only factual corrections. Do not propose promotional additions,
   stylistic rewrites, or changes that merely replace one accurate phrasing
   with another.

Follow first-party links from the docs when available. GitHub releases and
repository files may be read through the GitHub tools. Use web access for
first-party documentation and changelogs. Never use search snippets, third-party
comparisons, generated summaries, or this comparison page itself as evidence
for a competitor claim.

## Deduplication

Open issues were prefetched before the sandbox started. Check existing
comparison-drift issues first:

```bash
jq '.[] | select(any(.labels[]; .name == "comparison-drift")) | {number, title, body, url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

If an open issue already proposes all currently needed factual edits (ignore `Last checked` rows when comparing), call
`mcp__safeoutputs__noop`. If it overlaps only partly, include only the new or
changed proposals in this run's issue. Do not enumerate issues with `gh` from
inside the sandbox.

## Issue format

Use a concise title that describes the reviewed comparison drift. The safe
output adds the `[comparison-drift] ` prefix.

The issue body must begin with the review window and list every page checked.
Then include one Markdown table with exactly these columns:

| Page | Current text | Proposed text | Source link |
|---|---|---|---|

Add one row per proposed change. Quote the exact current text and provide
replacement-ready proposed text. The source cell must link directly to the
competitor's own docs or repository for competitor claims, and to this
repository or the Pydantic AI docs for our claims. Use multiple first-party
links in the source cell when one proposed change contains multiple claims.

When you file an issue, also add a row for every page you fully re-checked,
proposing its `Last checked` date be set to today's date, so the next run's
review window starts there. These `Last checked` rows are bookkeeping, not
proposed changes: they never justify filing an issue on their own, and they
are ignored when comparing this run's rows with an open drift issue. In a
month with no factual corrections, call `mcp__safeoutputs__noop`; the dates
stay as they are and the next run reads the longer window.

After the table, add a short `## Research notes` section that names each
competitor's release or changelog window checked and notes any first-party
documentation area searched before writing "none documented". Do not include
unchanged rows or general recommendations.

Before filing, challenge every proposed row as if you maintain the named
competitor. Drop any row whose source does not directly support both the
current assessment and the proposed wording. If no factual rows remain
(only `Last checked` rows, or none), call `mcp__safeoutputs__noop`.
