---
name: pr-feedback
description: Fetch and organize PR feedback (inline comments, reviews, general comments) from GitHub. Use when reviewing PR feedback or preparing to address review comments.
---

# pr-feedback

## fetch-latest.sh

Fetches all recent PR feedback combining:
- Inline code review comments (`/pulls/{pr}/comments`)
- General PR comments (`/issues/{pr}/comments`)
- Formal review submissions (`/pulls/{pr}/reviews`)

```bash
.claude/skills/pr-feedback/fetch-latest.sh [PR_NUMBER] [DAYS_BACK]
```

**Arguments:**
- `PR_NUMBER` - optional, auto-detects from current branch if omitted
- `DAYS_BACK` - optional, defaults to 7

**Output:** JSON with three arrays: `pr_comments`, `issue_comments`, `reviews`

**Example:**
```bash
# Current branch's PR, last 7 days
.claude/skills/pr-feedback/fetch-latest.sh

# Specific PR, last 3 days
.claude/skills/pr-feedback/fetch-latest.sh 3826 3

# Pipe to jq for filtering
.claude/skills/pr-feedback/fetch-latest.sh | jq '.pr_comments | map(select(.user != "dsfaccini"))'
```

**Useful jq filters:**
```bash
# Only external feedback (exclude self)
jq '.pr_comments | map(select(.user != "YOUR_USERNAME"))'

# Group by file
jq '.pr_comments | group_by(.path) | map({file: .[0].path, comments: .})'

# Latest N comments
jq '.pr_comments | sort_by(.created_at) | reverse | .[0:5]'
```
