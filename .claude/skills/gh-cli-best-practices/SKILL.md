---
name: gh-cli-best-practices
description: Tools for fetching GitHub PR/issue/commit comments cleanly via gh CLI. Use when needing to retrieve comment data from GitHub with structured JSON output.
---

# gh-cli-best-practices

## fetch-comment

Fetches PR/issue/commit comments with structured JSON output.

```bash
.claude/skills/gh-cli-best-practices/fetch-comment <owner/repo> <comment_ref>
```

**comment_ref formats:**
- `discussion_r{id}` - PR review comments
- `issuecomment-{id}` - issue/PR comments
- `commitcomment-{id}` - commit comments

**Example:**
```bash
.claude/skills/gh-cli-best-practices/fetch-comment pydantic/pydantic-ai discussion_r1234567
```

## get-latest-ci-failure

Fetches the latest CI test failure logs for the current branch's PR.

```bash
.claude/skills/gh-cli-best-practices/get-latest-ci-failure.sh
```

Automatically detects the current branch, finds the associated PR, and extracts the "Summary of Failures" section from failed test runs.
