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
[ -s "$CTX/pr-comments.txt" ] || echo "(No PR comments)" > "$CTX/pr-comments.txt"

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
                  id
                  databaseId
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
  | jq -s '[.[][] | select(.user.login == "github-actions" or .user.login == "github-actions[bot]") | .created_at] | sort | last // empty' -r)
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
  def truncate: gsub("[\\r\\n]+"; "  ") | if length > 200 then .[:200] + "..." else . end;

  [ .[] |
    {
      resolved: .isResolved,
      outdated: .isOutdated,
      state: (
        (if .isResolved then "RESOLVED" else "UNRESOLVED" end) +
        (if .isOutdated then ", OUTDATED" else "" end)
      ),
      first: .comments.nodes[0],
      lastCommentAt: (.comments.nodes | last | .createdAt),
      replies: [ .comments.nodes[1:][] | { author: .author.login, databaseId: .databaseId, body: .body, createdAt: .createdAt } ]
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
    "- [\($t.state)] \($first.author.login) at \($first.createdAt) on \($first.path)\(if $first.line then ":\($first.line)" else "" end) (comment \($first.databaseId)) — \($first.body | truncate)" +
    ([ $t.replies[] | "\n  > \(.author) at \(.createdAt) (comment \(.databaseId)): \(.body | truncate)" ] | join(""))
  else
    (
      ($first.path + ":" + ($first.diffHunk | split("\n")[0])) as $hunkKey |
      (if $i > 0 then ($arr[$i - 1].first.path + ":" + ($arr[$i - 1].first.diffHunk | split("\n")[0])) else "" end) as $prevKey |
      (if $hunkKey != $prevKey then true else false end) as $showHunk |
      "### [\($t.state)] \($first.author.login) (\($first.authorAssociation)) at \($first.createdAt) on \($first.path)\(if $first.line then ":\($first.line)" else "" end) (comment \($first.databaseId))" +
      (if $showHunk then "\n```diff\n\($first.diffHunk)\n```" else "" end) +
      "\n\($first.body)\n" +
      ([ $t.replies[] | "  > **\(.author)** at \(.createdAt) (comment \(.databaseId)): \(.body)\n" ] | join(""))
    )
  end
' "$THREADS_JSON" >> "$CTX/review-comments.txt"
rm -f "$THREADS_JSON"
[ -s "$CTX/review-comments.txt" ] || echo "(No review comments)" > "$CTX/review-comments.txt"

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
[ -s "$CTX/related-issues.txt" ] || echo "(No issues referenced in PR description)" > "$CTX/related-issues.txt"

# Compute merge base for function-context diffs
echo "  - Computing merge base for function-context diffs"
BASE_REF=$(jq -r '.baseRefName' "$CTX/pr-details.json")
MERGE_BASE=""
if [ -n "$BASE_REF" ]; then
  # Try origin first (works for same-repo PRs)
  if git fetch origin "$BASE_REF" --quiet 2>/dev/null; then
    MERGE_BASE=$(git merge-base HEAD "origin/$BASE_REF" 2>/dev/null || echo "")
  fi
  # For fork PRs, origin is the fork — fetch from the base repo instead
  if [ -z "$MERGE_BASE" ]; then
    git remote add base-repo "https://github.com/${REPO}.git" 2>/dev/null || true
    if git fetch base-repo "$BASE_REF" --quiet 2>/dev/null; then
      MERGE_BASE=$(git merge-base HEAD "base-repo/$BASE_REF" 2>/dev/null || echo "")
    fi
  fi
fi
if [ -n "$MERGE_BASE" ]; then
  echo "    Merge base: ${MERGE_BASE:0:12} (using function-context diffs)"
else
  echo "    Could not determine merge base (falling back to API diff)"
fi

# Per-file diffs with function context (excluding generated files)
echo "  - Per-file diffs (excluding generated files)"
mkdir -p "$CTX/diff"
if [ -n "$MERGE_BASE" ]; then
  # -W (--function-context) shows the full function body around each change,
  # so the reviewer can see the function signature and surrounding logic without
  # needing to read the full source file separately.
  git diff -W --no-color "$MERGE_BASE" HEAD
else
  gh pr diff "$PR_NUMBER" --repo "$REPO"
fi | awk -v dir="$CTX/diff" '
  /^diff --git/ {
    # Close previous file to avoid running out of file descriptors
    if (outfile) close(outfile)
    outfile = ""

    # Extract new (b/) filename from "diff --git a/path b/path"
    # Uses b/ side so renamed files match the GitHub API .filename field
    fname = $0
    sub(/^.* b\//, "", fname)

    skip = (fname ~ /uv\.lock/ || fname ~ /\/cassettes\//)
    if (!skip) {
      # Sanitize path: replace / with __, strip leading dots to avoid hidden files
      safe = fname
      gsub(/\//, "__", safe)
      sub(/^\.+/, "", safe)
      outfile = dir "/" safe ".diff"
    }
  }
  !skip && outfile { print > outfile }
'

# List of ALL changed files with change counts + diff file paths
# NOTE: This uses the GitHub API (not local git) for accurate per-file addition/deletion counts.
# The diff files above come from local git when the merge base is available. In rare edge cases
# (force pushes, rebases between steps), the two sources could disagree — this is acceptable
# since it only affects the diff file path column, and missing diffs degrade gracefully.
echo "  - Changed files"
gh api "repos/${REPO}/pulls/${PR_NUMBER}/files" --paginate \
  --jq '.[] | "\(.filename)\t+\(.additions) -\(.deletions)\t\(.filename | gsub("/"; "__") | gsub("^\\.+"; "")).diff"' \
  | while IFS=$'\t' read -r fname counts diffname; do
    if echo "$fname" | grep -qE 'uv\.lock|/cassettes/'; then
      printf '%s\t%s\n' "$fname" "$counts"
    else
      printf '%s\t%s\tdiff/%s\n' "$fname" "$counts" "$diffname"
    fi
  done > "$CTX/changed-files.txt"

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
[ -s "$CTX/agents-md.txt" ] || echo "(No directory-specific AGENTS.md files for changed directories)" > "$CTX/agents-md.txt"

echo ""
echo "Context gathered in ${CTX}/:"
ls -lh "$CTX/"
DIFF_COUNT=$(find "$CTX/diff" -name '*.diff' 2>/dev/null | wc -l)
echo "  Per-file diffs: ${DIFF_COUNT} files in ${CTX}/diff/"
