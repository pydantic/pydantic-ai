#!/usr/bin/env bash
# Gather PR context for auto-review into .github/.review-context/
# Usage: scripts/gather-review-context.sh <pr-number> [repo]
#
# Examples:
#   scripts/gather-review-context.sh 4269
#   scripts/gather-review-context.sh 4269 pydantic/pydantic-ai

set -euo pipefail

PR_NUMBER="${1:?Usage: $0 <pr-number> [repo]}"
REPO="${2:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
CTX=".github/.review-context"
mkdir -p "$CTX"

echo "Gathering context for PR #${PR_NUMBER} in ${REPO}..."

# PR details (title, body, author, labels)
echo "  - PR details"
gh pr view "$PR_NUMBER" --repo "$REPO" --json title,body,author,headRefName,baseRefName,additions,deletions,changedFiles,labels,isDraft,reviewDecision,state,createdAt,updatedAt,url > "$CTX/pr-details.json"

# PR comments
echo "  - PR comments"
gh api "repos/${REPO}/issues/${PR_NUMBER}/comments" --paginate --jq '.[] | "### \(.user.login) (\(.author_association)) at \(.created_at)\n\(.body)\n"' > "$CTX/pr-comments.txt"

# Inline review comments (with diff hunks and resolved state via GraphQL)
# Fetch all review threads first, then determine last auto-review timestamp, then format
echo "  - Review comments"
OWNER="${REPO%%/*}"
REPO_NAME="${REPO##*/}"
CURSOR=""
THREADS_JSON=$(mktemp)
echo '[]' > "$THREADS_JSON"
while true; do
  CURSOR_ARG=""
  if [ -n "$CURSOR" ]; then
    CURSOR_ARG=", after: \"$CURSOR\""
  fi
  RESULT=$(gh api graphql -f query="
    query {
      repository(owner: \"$OWNER\", name: \"$REPO_NAME\") {
        pullRequest(number: $PR_NUMBER) {
          reviewThreads(first: 100$CURSOR_ARG) {
            pageInfo { hasNextPage endCursor }
            nodes {
              id
              isResolved
              isOutdated
              comments(first: 50) {
                nodes {
                  author { login }
                  authorAssociation
                  body
                  diffHunk
                  path
                  line
                  createdAt
                  replyTo { id }
                }
              }
            }
          }
        }
      }
    }
  ")
  # Accumulate thread nodes into temp file
  jq -s '.[0] + [.[1].data.repository.pullRequest.reviewThreads.nodes[]]' "$THREADS_JSON" <(echo "$RESULT") > "${THREADS_JSON}.tmp"
  mv "${THREADS_JSON}.tmp" "$THREADS_JSON"
  CURSOR=$(echo "$RESULT" | jq -r '.data.repository.pullRequest.reviewThreads.pageInfo | select(.hasNextPage) | .endCursor')
  if [ -z "$CURSOR" ]; then
    break
  fi
done

# Find timestamp of last auto-review from both issue comments and inline review comments
echo "  - Checking for previous auto-review"
LAST_ISSUE_COMMENT_TS=$(gh api "repos/${REPO}/issues/${PR_NUMBER}/comments" --paginate \
  --jq '[.[] | select(.user.login == "github-actions" or .user.login == "github-actions[bot]") | .created_at] | last // empty')
LAST_REVIEW_COMMENT_TS=$(jq -r '
  [.[] | .comments.nodes[] |
    select(.author.login == "github-actions" or .author.login == "github-actions[bot]") |
    .createdAt
  ] | sort | last // empty
' "$THREADS_JSON")

# Take the later of the two timestamps
if [ -n "$LAST_ISSUE_COMMENT_TS" ] && [ -n "$LAST_REVIEW_COMMENT_TS" ]; then
  if [[ "$LAST_ISSUE_COMMENT_TS" > "$LAST_REVIEW_COMMENT_TS" ]]; then
    LAST_REVIEW_TS="$LAST_ISSUE_COMMENT_TS"
  else
    LAST_REVIEW_TS="$LAST_REVIEW_COMMENT_TS"
  fi
else
  LAST_REVIEW_TS="${LAST_ISSUE_COMMENT_TS:-$LAST_REVIEW_COMMENT_TS}"
fi

if [ -n "$LAST_REVIEW_TS" ]; then
  echo "    Last auto-review: $LAST_REVIEW_TS"
else
  echo "    No previous auto-review found"
fi

# Format review threads with compaction
> "$CTX/review-comments.txt"
jq -r --arg last_review "$LAST_REVIEW_TS" '
  [ .[] |
    {
      id: .id,
      resolved: .isResolved,
      outdated: .isOutdated,
      state: (
        (if .isResolved then "RESOLVED" else "UNRESOLVED" end) +
        (if .isOutdated then ", OUTDATED" else "" end)
      ),
      first: .comments.nodes[0],
      lastCommentAt: (.comments.nodes | last | .createdAt),
      replies: [ .comments.nodes[1:][] | { author: .author.login, body: .body, createdAt: .createdAt } ]
    }
  ] as $arr |
  range($arr | length) as $i |
  $arr[$i] as $t |
  $t.first as $first |

  # Compact if: (resolved AND outdated) OR (all comments predate last auto-review)
  (
    ($t.resolved and $t.outdated) or
    ($last_review != "" and $t.lastCommentAt < $last_review)
  ) as $compact |

  if $compact then
    "- [\($t.state)] \($first.author.login) at \($first.createdAt) on \($first.path)\(if $first.line then ":\($first.line)" else "" end) (thread \($t.id)) — \($first.body | gsub("\n"; "  ") | if length > 200 then .[:200] + "..." else . end)"
  else
    (
      ($first.path + ":" + ($first.diffHunk | split("\n")[0])) as $hunkKey |
      (if $i > 0 then ($arr[$i - 1].first.path + ":" + ($arr[$i - 1].first.diffHunk | split("\n")[0])) else "" end) as $prevKey |
      (if $hunkKey != $prevKey then true else false end) as $showHunk |
      "### \($first.author.login) (\($first.authorAssociation)) at \($first.createdAt) on \($first.path)\(if $first.line then ":\($first.line)" else "" end) [\($t.state)] (thread \($t.id))" +
      (if $showHunk then "\n```diff\n\($first.diffHunk)\n```" else "" end) +
      "\n\($first.body)\n" +
      ([ $t.replies[] | "  > **\(.author)** at \(.createdAt) (reply): \(.body)\n" ] | join(""))
    )
  end
' "$THREADS_JSON" >> "$CTX/review-comments.txt"
rm -f "$THREADS_JSON"

# Related issues: extract issue numbers from PR body
echo "  - Related issues"
PR_BODY=$(gh pr view "$PR_NUMBER" --repo "$REPO" --json body --jq '.body')
{
  echo "$PR_BODY" | grep -oiP '(?:closes|fixes|resolves|close|fix|resolve)\s*#\K\d+' || true
  echo "$PR_BODY" | grep -oiP '(?:closes|fixes|resolves|close|fix|resolve)\s+https://github\.com/[^/]+/[^/]+/issues/\K\d+' || true
} | sort -u | while read -r ISSUE_NUM; do
  echo "=== Issue #${ISSUE_NUM} ==="
  gh issue view "$ISSUE_NUM" --repo "$REPO" --json title,body,author,comments --jq '"## \(.title)\nBy: \(.author.login)\n\(.body)\n\n### Comments:\n\(.comments | map("#### \(.author.login) (\(.authorAssociation))\n\(.body)\n") | join("\n"))"'
done > "$CTX/related-issues.txt"

# List of ALL changed files with change counts (including generated, for awareness)
echo "  - Changed files"
gh api "repos/${REPO}/pulls/${PR_NUMBER}/files" --paginate \
  --jq '.[] | "\(.filename)\t+\(.additions) -\(.deletions)"' > "$CTX/changed-files.txt"

# Diff of non-generated files only
echo "  - Diff (excluding generated files)"
gh pr diff "$PR_NUMBER" --repo "$REPO" | awk '
  /^diff --git/ {
    skip = ($0 ~ /uv\.lock/ || $0 ~ /\/cassettes\//)
  }
  !skip { print }
' > "$CTX/diff.txt"

# Gather directory-specific AGENTS.md files for changed directories
echo "  - Directory AGENTS.md files"
> "$CTX/agents-md.txt"
for agents_file in docs/AGENTS.md pydantic_ai_slim/pydantic_ai/models/AGENTS.md tests/AGENTS.md; do
  dir=$(dirname "$agents_file")
  if grep -q "^${dir}/" "$CTX/changed-files.txt" 2>/dev/null && [ -f "$agents_file" ]; then
    echo "=== ${agents_file} ==="
    cat "$agents_file"
    echo ""
  fi
done >> "$CTX/agents-md.txt"

echo ""
echo "Context gathered in ${CTX}/:"
ls -lh "$CTX/"
