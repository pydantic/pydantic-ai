#!/bin/bash
# Fetch latest PR feedback (reviews + inline comments + issue comments)
# Usage: fetch-latest.sh [PR_NUMBER] [DAYS_BACK]
#
# If PR_NUMBER not provided, detects from current branch
# DAYS_BACK defaults to 7

set -e

PR_NUMBER="${1:-}"
DAYS_BACK="${2:-7}"

# Auto-detect PR if not provided
if [[ -z "$PR_NUMBER" ]]; then
    BRANCH=$(git branch --show-current)
    PR_NUMBER=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number' 2>/dev/null || echo "")
    if [[ -z "$PR_NUMBER" ]]; then
        echo "Error: Could not find PR for branch '$BRANCH'. Provide PR number as argument." >&2
        exit 1
    fi
fi

# Calculate cutoff date
if [[ "$(uname)" == "Darwin" ]]; then
    CUTOFF=$(date -v-${DAYS_BACK}d -u +%Y-%m-%dT%H:%M:%SZ)
else
    CUTOFF=$(date -u -d "$DAYS_BACK days ago" +%Y-%m-%dT%H:%M:%SZ)
fi

REPO=$(gh repo view --json nameWithOwner --jq '.nameWithOwner')

echo "Fetching feedback for PR #$PR_NUMBER in $REPO (since $CUTOFF)" >&2

# Fetch all three types and combine with jq
{
    echo '{"pr_comments":'
    gh api "repos/$REPO/pulls/$PR_NUMBER/comments" \
        -X GET -f sort=created -f direction=desc -f per_page=100 \
        --jq "[.[] | select(.created_at > \"$CUTOFF\") | {type: \"inline\", created_at, user: .user.login, path, line: (.line // .original_line), body, in_reply_to_id, html_url}]"

    echo ',"issue_comments":'
    gh api "repos/$REPO/issues/$PR_NUMBER/comments" \
        --jq "[.[] | select(.created_at > \"$CUTOFF\") | {type: \"general\", created_at, user: .user.login, body, html_url}]"

    echo ',"reviews":'
    gh api "repos/$REPO/pulls/$PR_NUMBER/reviews" \
        --jq "[.[] | select(.submitted_at > \"$CUTOFF\") | {type: \"review\", submitted_at, user: .user.login, state, body}]"

    echo '}'
} | jq -s 'add'
