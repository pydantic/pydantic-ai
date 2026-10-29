#!/usr/bin/env bash
# Gather PR context for the CI Review agent into $GITHUB_WORKSPACE/.review-context/.
# Usage: scripts/gather-pydantic-ai-review-context.sh <pr-number> [repo]
#
# Examples:
#   scripts/gather-pydantic-ai-review-context.sh 4269
#   scripts/gather-pydantic-ai-review-context.sh 4269 pydantic/pydantic-ai
#
# Why outputs live at the workspace ROOT, and not under /tmp or `.github/`:
# the agent's `Read` tool refuses any path outside the workspace, so a `/tmp`
# destination makes every instructed read fail (#6766 measured ~15 failed
# `Read` calls per run, in 80 of 80 sampled runs). But gh-aw's pre-agent flow
# also runs a "Save/Restore agent config folders from base branch" step that
# rewrites `.github/` (and `.agents`, `.claude`, `.codex`, …) from the BASE
# branch, so anything written inside those is liable to be wiped or shadowed.
# The workspace root satisfies both: readable by the agent, untouched by the
# restore.
#
# TODO(consolidate): This is a fork of scripts/gather-review-context.sh used
# by the legacy Claude-action workflow (.github/workflows/bots.yml). The two
# will be consolidated once the Claude-action workflow is migrated to the
# Pydantic AI gh-aw shim — until then, keep edits scoped to whichever consumer
# needs them and leave the other script alone.

set -euo pipefail

# The partition mode is also used by local regression tests. It accepts the same
# GitHub files JSON consumed by the full context gatherer and makes no API calls.
if [ "${1:-}" = "--partition-files" ]; then
  FILES_JSON="${2:?Usage: $0 --partition-files <files.json> <output-dir>}"
  ORDERINGS_DIR="${3:?Usage: $0 --partition-files <files.json> <output-dir>}"
  mkdir -p "$ORDERINGS_DIR"

  # Keep the historical ordering files for consumers that use them. Resolve
  # equal-size ties by path so assignments do not depend on GitHub API order.
  jq -r '
    [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
    | sort_by(.filename) | .[].filename
  ' "$FILES_JSON" > "$ORDERINGS_DIR/az.txt"
  jq -r '
    [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
    | sort_by(.filename) | reverse | .[].filename
  ' "$FILES_JSON" > "$ORDERINGS_DIR/za.txt"

  PARTITIONS=$(jq '
    [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
    | sort_by([-( (.additions // 0) + (.deletions // 0) ), .filename]) as $files
    | ($files | length) as $file_count
    | ([$files[] | (.additions // 0) + (.deletions // 0)] | add // 0) as $diff_lines
    | (if $file_count <= 3 and $diff_lines <= 200 then 0
       elif $file_count > 10 or $diff_lines > 1000 then 3
       else 2 end) as $requested
    | ([ $requested, $file_count ] | min) as $delegate_count
    | (reduce range(0; $delegate_count) as $i
         ({bins: [], loads: []}; .bins += [[]] | .loads += [0])) as $empty
    | reduce $files[] as $file ($empty;
        . as $state
        | (reduce range(1; $delegate_count) as $i
             (0; . as $best
              | if $state.loads[$i] < $state.loads[$best]
                   or ($state.loads[$i] == $state.loads[$best]
                       and ($state.bins[$i] | length) < ($state.bins[$best] | length))
                then $i else $best end)) as $best
        | .bins[$best] += [$file.filename]
        | .loads[$best] += (($file.additions // 0) + ($file.deletions // 0)))
    | {delegate_count: $delegate_count, bins: .bins}
  ' "$FILES_JSON")

  printf '%s\n' "$PARTITIONS" | jq -r '.delegate_count' > "$ORDERINGS_DIR/delegate-count.txt"
  DELEGATE_COUNT=$(printf '%s\n' "$PARTITIONS" | jq -r '.delegate_count')
  for ((index = 0; index < DELEGATE_COUNT; index++)); do
    jq -r ".bins[$index][]" <<<"$PARTITIONS" > "$ORDERINGS_DIR/delegate-$((index + 1)).txt"
  done
  # Existing largest.txt remains the ordered source used for first-pass
  # assignments. Assign each next-largest file to the least-loaded delegate.
  jq -r '
    [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
    | sort_by([-( (.additions // 0) + (.deletions // 0) ), .filename])
    | .[].filename
  ' "$FILES_JSON" > "$ORDERINGS_DIR/largest.txt"
  exit 0
fi

PR_NUMBER="${1:?Usage: $0 <pr-number> [repo]}"
REPO="${2:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
CTX="${GITHUB_WORKSPACE:-$PWD}/.review-context"
mkdir -p "$CTX"

# Track every `mktemp` we allocate and unlink them on exit, including the
# `set -e` early-termination path. Callers use `track_tmp <file>` after each
# `mktemp` instead of relying on individual cleanup paths.
_TMP_FILES=()
track_tmp() { _TMP_FILES+=("$1"); }
cleanup_tmp() {
  for f in "${_TMP_FILES[@]:-}"; do
    [ -n "$f" ] && rm -rf "$f"
  done
}
trap cleanup_tmp EXIT

echo "Gathering context for PR #${PR_NUMBER} in ${REPO}..."

# PR details (title, body, author, labels)
echo "  - PR details"
gh pr view "$PR_NUMBER" --repo "$REPO" --json title,body,author,headRefName,baseRefName,additions,deletions,changedFiles,labels,isDraft,reviewDecision,state,createdAt,updatedAt,url > "$CTX/pr-details.json"

# PR comments
echo "  - PR comments"
gh api "repos/${REPO}/issues/${PR_NUMBER}/comments" --paginate --jq '.[] | "### \(.user.login) (\(.author_association)) at \(.created_at)\n\(.body)\n"' > "$CTX/pr-comments.txt"
[ -s "$CTX/pr-comments.txt" ] || echo "(No PR comments)" > "$CTX/pr-comments.txt"

# Inline review comments (with diff hunks and resolved state via GraphQL)
# Fetch all review threads first, then determine last douwebot-review timestamp, then format
echo "  - Review comments"
OWNER="${REPO%%/*}"
REPO_NAME="${REPO##*/}"
CURSOR=""
THREADS_JSON=$(mktemp)
track_tmp "$THREADS_JSON"
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

# Find timestamp of last douwebot review from both issue comments and inline review comments
echo "  - Checking for previous douwebot review"
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
  echo "    Last douwebot review: $LAST_REVIEW_TS"
else
  echo "    No previous douwebot review found"
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

  # Compact if: (resolved AND outdated) OR (all comments predate last douwebot review)
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

# Fetch base branch for function-context diffs
# Always fetch from the target repo URL, not origin — for fork PRs, origin points to
# the fork (which may have an outdated base branch), causing incorrect merge bases
# and diffs that include unrelated changes from the base repo.
echo "  - Fetching base branch for function-context diffs"
BASE_REF=$(jq -r '.baseRefName' "$CTX/pr-details.json")
MERGE_BASE=""
BASE_SHA=""
if [ -n "$BASE_REF" ]; then
  if git fetch "https://github.com/${REPO}.git" "$BASE_REF" --quiet 2>/dev/null; then
    MERGE_BASE=$(git merge-base HEAD FETCH_HEAD 2>/dev/null || echo "")
    BASE_SHA=$(git rev-parse FETCH_HEAD 2>/dev/null || echo "")
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

# Annotate commentable diff lines with source line numbers (NL:/OL: prefixes)
# so the review bot can target inline comments without computing line numbers.
echo "  - Annotating diffs with source line numbers"
for diff_file in "$CTX/diff/"*.diff; do
  [ -f "$diff_file" ] || continue
  awk '
    BEGIN { NEAR = 3 }

    # Diff metadata: flush any buffered hunk, pass through
    /^diff --git/ || /^index / || /^---/ || /^\+\+\+/ ||
    /^new file/ || /^deleted file/ || /^old mode/ || /^new mode/ ||
    /^rename / || /^similarity / || /^dissimilarity / || /^Binary / {
        flush_hunk()
        print
        next
    }

    # Hunk header: flush previous hunk, parse line numbers
    /^@@ / {
        flush_hunk()
        split($2, _o, ","); old_num = substr(_o[1], 2) + 0
        split($3, _n, ","); new_num = substr(_n[1], 2) + 0
        hunk_hdr = $0
        n = 0
        next
    }

    # "\ No newline at end of file"
    /^\\/ {
        n++; lines[n] = $0; types[n] = "\\"; is_chg[n] = 0
        next
    }

    # Hunk body lines
    {
        n++; lines[n] = $0
        c = substr($0, 1, 1)
        if (c == "+") {
            types[n] = "+"; lnums[n] = new_num++; is_chg[n] = 1
        } else if (c == "-") {
            types[n] = "-"; lnums[n] = old_num++; is_chg[n] = 1
        } else {
            types[n] = " "; lnums[n] = new_num++; old_num++; is_chg[n] = 0
        }
    }

    function flush_hunk(    i, dist, min_d) {
        if (n == 0) return

        # Forward pass: context-line distance from nearest preceding change
        dist = NEAR + 1
        for (i = 1; i <= n; i++) {
            if (is_chg[i]) dist = 0
            else if (types[i] != "\\") { dist++; fwd[i] = dist }
        }
        # Backward pass: context-line distance from nearest following change
        dist = NEAR + 1
        for (i = n; i >= 1; i--) {
            if (is_chg[i]) dist = 0
            else if (types[i] != "\\") { dist++; bwd[i] = dist }
        }

        print hunk_hdr
        for (i = 1; i <= n; i++) {
            if (types[i] == "\\") { print lines[i] }
            else if (is_chg[i]) {
                if (types[i] == "+") printf "NL:%d %s\n", lnums[i], lines[i]
                else                printf "OL:%d %s\n", lnums[i], lines[i]
            } else {
                min_d = fwd[i]; if (bwd[i] < min_d) min_d = bwd[i]
                if (min_d <= NEAR) printf "NL:%d %s\n", lnums[i], lines[i]
                else print lines[i]
            }
        }

        delete lines; delete types; delete lnums
        delete is_chg; delete fwd; delete bwd
        n = 0
    }

    END { flush_hunk() }
  ' "$diff_file" > "${diff_file}.tmp" && mv "${diff_file}.tmp" "$diff_file"
done

# List of ALL changed files with change counts + diff file paths.
# Also written as JSON so the orderings below don't have to re-parse the columns.
echo "  - Changed files"
FILES_JSON=$(mktemp)
track_tmp "$FILES_JSON"
gh api "repos/${REPO}/pulls/${PR_NUMBER}/files" --paginate \
  | jq -s 'add // []' > "$FILES_JSON"
jq -r '.[] | [.filename, "+\(.additions) -\(.deletions)", (.filename | gsub("/"; "__") | gsub("^\\.+"; "")) + ".diff"] | @tsv' "$FILES_JSON" \
  | while IFS=$'\t' read -r fname counts diffname; do
    if echo "$fname" | grep -qE 'uv\.lock|/cassettes/'; then
      printf '%s\t%s\n' "$fname" "$counts"
    else
      printf '%s\t%s\tdiff/%s\n' "$fname" "$counts" "$diffname"
    fi
  done > "$CTX/changed-files.txt"

# File orderings and disjoint first-pass assignments. The historical ordering
# files remain available to other consumers; assignments use largest-first
# greedy balancing and exclude generated files.
echo "  - File orderings and delegate assignments"
mkdir -p "$CTX/file-orderings"
"$0" --partition-files "$FILES_JSON" "$CTX/file-orderings"

# PR size summary — file count and total diff lines. The prompt uses this to
# pick a single-pass vs fan-out review strategy.
FILE_COUNT=$(jq '[.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)] | length' "$FILES_JSON")
DIFF_LINES=$(jq '[.[] | select(.filename | test("uv\\.lock|/cassettes/") | not) | (.additions // 0) + (.deletions // 0)] | add // 0' "$FILES_JSON")
printf '%s files, %s diff lines (excluding generated files)\n' "$FILE_COUNT" "$DIFF_LINES" > "$CTX/pr-size.txt"
rm -f "$FILES_JSON"

# Core definitions the durable engines depend on that this PR changes, found as the
# `durable-exec-label` job in `.github/workflows/bots.yml` finds them: with the base branch's
# classifier and manifest, never this checkout's copies, so a PR's edits to either take effect
# only once merged.
# Written only when there are any. Non-fatal: without it the reviewer has less context.
echo "  - Durable-exec contract entries"
DURABLE_EXEC_AGENTS_MD=pydantic_ai_slim/pydantic_ai/durable_exec/AGENTS.md
DURABLE_EXEC_GATE=$(mktemp -d)
track_tmp "$DURABLE_EXEC_GATE"
durable_exec_touched() {
  local path
  mkdir -p "$DURABLE_EXEC_GATE/scripts" || return 1
  for path in scripts/durable_exec_contract.py scripts/durable_exec_contract.json; do
    gh api -H 'Accept: application/vnd.github.raw+json' "repos/${REPO}/contents/${path}?ref=${BASE_SHA}" \
      > "$DURABLE_EXEC_GATE/$path" || return 1
  done
  python3 -I "$DURABLE_EXEC_GATE/scripts/durable_exec_contract.py" \
    --touched "$REPO" "$PR_NUMBER" "$BASE_SHA" "$(git rev-parse HEAD)"
}
if [ -n "$BASE_SHA" ] && TOUCHED=$(durable_exec_touched); then
  if [ "$(jq '[.[] | length] | add // 0' <<<"$TOUCHED")" -gt 0 ]; then
    {
      echo "# Durable-exec contract entries this PR touches"
      echo ""
      echo "The base branch's \`scripts/durable_exec_contract.py --touched\` placed this PR's changed lines in"
      echo "these entries of \`scripts/durable_exec_contract.json\`: core definitions a durable engine"
      echo "depends on, each with its kinds, its engines and the engine sites that depend on it."
      echo "\`agents-md.txt\` carries \`${DURABLE_EXEC_AGENTS_MD}\`, whose section \"Core changes that"
      echo "engines depend on\" says what to check."
      echo ""
      jq -r 'to_entries[] | .key as $file | .value | to_entries[]
        | "- `\($file)::\(.key)`: \(.value.kinds | join(", ")); engines: \(.value.engines | join(", "))",
          (.value.sites[] | "  - `\(.)`")' <<<"$TOUCHED"
    } > "$CTX/durable-exec-contract.md"
  fi
else
  echo "    Could not classify the PR against the durable-exec contract (skipping)"
fi

# Gather AGENTS.md files relevant to the PR — the repo-root file (always,
# when present), any per-directory AGENTS.md whose directory has changed
# files in this PR, and the durable_exec one when the PR touches a contract entry.
echo "  - AGENTS.md files (repo-root + changed directories + touched durable-exec contract)"
> "$CTX/agents-md.txt"
{
  if [ -f ./AGENTS.md ]; then
    echo "=== AGENTS.md ==="
    cat ./AGENTS.md
    echo ""
  fi
  for agents_file in $(find . -name AGENTS.md -not -path './.venv/*' -not -path ./AGENTS.md | sed 's|^\./||' | sort); do
    dir=$(dirname "$agents_file")
    if { grep -q "^${dir}/" "$CTX/changed-files.txt" 2>/dev/null \
         || { [ "$agents_file" = "$DURABLE_EXEC_AGENTS_MD" ] && [ -f "$CTX/durable-exec-contract.md" ]; }; } \
      && [ -f "$agents_file" ]; then
      echo "=== ${agents_file} ==="
      cat "$agents_file"
      echo ""
    fi
  done
} >> "$CTX/agents-md.txt"
[ -s "$CTX/agents-md.txt" ] || echo "(No AGENTS.md files relevant to this PR)" > "$CTX/agents-md.txt"

# Shared review conventions — the severity scale, false-positive catalog,
# and calibration examples. Pre-writing this file (instead of inlining the
# same content into the workflow prompt) means the parent agent doesn't
# have to copy 100+ lines of rules into every Task sub-agent prompt — the
# parent just tells each sub-agent to `Read` this file once. Matches the
# pattern elastic/ai-github-actions uses with `/tmp/pr-context/review-
# instructions.md`. Keep this content in sync with what the PR review
# seed prompt cites.
echo "  - Review instructions for sub-agents"
cat > "$CTX/review-instructions.md" <<'REVIEW_INSTRUCTIONS_EOF'
# Pydantic AI PR Review — Shared Review Conventions

This file is the **single source of truth** for the severity scale, false-
positive catalog, calibration examples, and sub-agent output format. The
parent PR-review agent and every Task sub-agent should `Read` this file
once before reviewing.

## Severity scale

Determine severity AFTER investigating the finding, not before.

- 🔴 **CRITICAL** — must fix before merge. Security vulnerability, data
  corruption, public-API break without deprecation, type-safety hole
  that would silently mistype user code.
- 🟠 **HIGH** — should fix before merge. Logic bug with a concrete
  failure trigger, missing validation at an external boundary, race
  condition, significant perf regression, broken backward compatibility.
- 🟡 **MEDIUM** — address soon, non-blocking. Error-handling gap with
  an unlikely trigger, missing test for a non-trivial code path, subtly
  surprising behavior, docs that contradict the code.
- ⚪ **LOW** — author discretion. Minor improvements, missing docstrings
  on small helpers, narrow refactor opportunities.
- 💬 **NITPICK** — truly optional. Naming preferences, comment polish.

**Verdict mapping:** any verified HIGH or CRITICAL finding requires
`REQUEST_CHANGES`, even when review coverage is incomplete. Otherwise, incomplete
coverage requires `COMMENT`; complete coverage with MEDIUM-only or lower findings
requires `APPROVE` (post comments anyway). Name every unreviewed file in the
review body for either verdict.
Complete coverage with no findings → `APPROVE`.

**Cap inline comments at 30 per run.** If more findings survive, keep the
highest-severity 30 inline and list the rest briefly in the review body.

## What NOT to flag

This repo runs ruff and pyright in CI and has expert maintainers. The
common false-positive patterns below all *look* like real issues — verify
the surrounding code before posting.

- **Style / formatting** — ruff handles it.
- **Type nits already covered by pyright** — `make typecheck` runs in CI.
- **Coverage-gate / `# pragma: no cover` predictions** — coverage is
  deterministic and CI reports it exactly: the `fail_under = 100` job names
  every uncovered `file:line`, and a strict-no-cover audit flags a
  `# pragma: no cover` sitting on a line that was actually covered. Don't
  flag "this drops below 100%", "removing this pragma breaks coverage", or
  "this pragma is wrong/unneeded" — CI catches all of these clearly. (A
  missing test for genuinely *new* behavior or public API is still fair
  game — that's about correctness, not the coverage number. A profile-gated
  branch whose added tests all pin one side of the flag is fair game too
  (Example 4): the branch line can read at 100% while the flag-on and
  flag-off *combination* stays unverified, so it is a value-combination gap,
  not a coverage-percentage prediction.)
- **"Missing tests" for pure refactor** — if the PR moves or renames
  existing code and existing tests still exercise the behavior, no new
  test is needed. Only flag missing tests for new behavior or new public
  API, which includes a newly added or modified `profile.get(...)` branch
  under `models/` (Example 4): that is new behavior, not a move.
- **`None` / `Optional` access guarded upstream** — internal helpers
  often assume a precondition the caller enforces (or a type narrows the
  value via an `assert` / `isinstance` / early-return). Read the caller
  before flagging.
- **Internal renames** — anything with a leading underscore (or in a
  module that starts with `_`) is private. Renaming or removing private
  surface is fine; only flag breakage of *public* API.
- **Provider-specific knobs** — request params, role mappings, finish
  reasons differ deliberately across providers. Check the provider's
  SDK docs (or recent commits in `pydantic_ai/models/<provider>.py`)
  before asserting a "bug".
- **Cassettes / `uv.lock` / generated files** — never review.
- **Theoretical performance** — `O(n²)` is only a problem if `n` is
  realistically large in this use case. Don't flag without evidence of
  real-world impact.
- **Validation already enforced by Pydantic** — if the input is a
  Pydantic model, don't flag missing manual validation of its fields.
- **"This might break some user"** — if you can't name the user, the
  call site, or the scenario, drop the finding.

## Calibration examples

### Example 1 — `None` access

**Flag this (HIGH):**
```python
# PR adds a new public helper
def first_text_message(messages: list[ModelMessage]) -> str:
    for m in messages:
        for p in m.parts:
            if isinstance(p, TextPart):
                return p.content
```
*Why:* The function is typed `-> str` but falls off the end and implicitly
returns `None` when no `TextPart` is found, so any caller that does
`.upper()` on the result silently breaks at runtime. Public API.

**Don't flag this:**
```python
# PR adds this line inside agent.run() after the model call
text = response.parts[-1].content
```
*Why:* Reading the surrounding code shows `_validate_response` runs
before this line and guarantees `parts` is non-empty and the last part
is text-bearing. The "missing None check" is handled at the layer above.

### Example 2 — provider mapping

**Flag this (HIGH):**
```python
# PR adds tool_call mapping for a new provider
return ToolCallPart(tool_name=tc.name, args=tc.input)
```
*Why:* Every other provider sets `tool_call_id=tc.id` for round-trip
identity; the new mapping silently drops it, breaking tool-result
pairing for any agent that uses multi-tool calls.

**Don't flag this:**
```python
# PR adds reasoning-effort mapping
if model_settings.reasoning_effort:
    request['reasoning'] = {'effort': model_settings.reasoning_effort}
```
*Why:* Even though OpenAI uses `reasoning_effort` at the top level,
this provider's SDK docs (check `pydantic_ai/models/<provider>.py`
neighbouring code) show the nested `reasoning.effort` shape is correct
for this provider. Different providers, different shapes — not a bug.

### Example 3 — backward compatibility

**Flag this (CRITICAL):**
```python
# PR renames a public method on Agent
- def run_sync(self, ...): ...
+ def sync_run(self, ...): ...
```
*Why:* `Agent.run_sync` is widely used public API. Removing it without
a deprecation shim breaks every user on upgrade.

**Don't flag this:**
```python
# PR renames an internal helper
- def _build_request(...): ...
+ def _assemble_request(...): ...
```
*Why:* Leading underscore = private. Internal refactors don't need
deprecation.

### Example 4 — profile-flag test pinning

Trigger: the diff adds or modifies a `profile.get(...)` read under
`pydantic_ai_slim/pydantic_ai/models/`, and the tests plus cassettes the same
diff adds pin that flag to a single value. Decide it by calling the provider's
profile function for each pinned model and reading the flag.

**Flag this (MEDIUM):**
```python
# PR adds a background-mode include gated on a profile flag under models/...
if profile.get('openai_supports_encrypted_reasoning_content'):
    include.append('reasoning.encrypted_content')
# ...and every test and cassette the same PR adds pins `gpt-4o`.
```
*Why:* The new branch only runs when the flag is on, but
`openai_model_profile('gpt-4o').get('openai_supports_encrypted_reasoning_content')`
is `False`, so every added test sits on the flag-off side and the flag-on
branch ships unverified. Coverage does not catch this: the line is exercised
by other tests and reads at 100%, yet the (flag on) × (this path) combination
is never visited. Ask for one added test pinning a model on the other side (a
reasoning model, e.g. `gpt-5.6` or `o3`, for which the flag is `True`), or an
in-test note saying why that side is unreachable on this path. MEDIUM and
advisory: post the comment, never `REQUEST_CHANGES`.

**Don't flag this:**
```python
# Same branch, and the PR adds one test pinning `gpt-4o` plus one pinning `o3`.
```
*Why:* The added tests pin a model on each side of the flag
(`openai_model_profile('gpt-4o')` is `False`, `openai_model_profile('o3')` is
`True`), so both paths are covered. Also don't flag when the diff pins one
value but a comment in the added test explains why the other side is
unreachable on this path, or when the touched `profile.get(...)` read is not a
new or modified branch (a pure move or rename).

### Example 5 — test cost

Trigger: the diff adds or changes tests under `tests/`. Every test runs on
every push across the whole CI matrix, and the next test in the file copies the
pattern, so cost is a review concern even when CI is green. The rules are in
"Test cost" in `tests/AGENTS.md`; `tests/cost_guards.py` already fails an
unmarked Python launch and (on one CI leg) a test over the time budget, so the
review's job is the escape hatches and the patterns the guards cannot see.

**Flag this (MEDIUM):**
```python
@pytest.mark.subprocess(reason='runs the CLI')
def test_cli_prints_version(tmp_path: Path):
    result = subprocess.run([sys.executable, '-m', 'pydantic_clai2', '--version'], ...)
```
*Why:* The claim (the version string) does not depend on the process
boundary, so the marker only waves through a second-per-test interpreter
launch; ask for the entry point to be called in-process. The same goes for a
new `@pytest.mark.slow(reason=...)` whose cost could be designed away, and for
a new test that waits out a real `sleep`/timeout/poll interval instead of
injecting it, or proves a complexity bound with one huge input instead of a
growth ratio across small sizes.

**Don't flag this:**
```python
@pytest.mark.subprocess(reason='asserts what a fresh interpreter imports')
def test_import_is_light():
    ...
```
*Why:* What a fresh interpreter imports can only be observed in a fresh
interpreter; the reason names the process-boundary property under test. Also
don't flag a launch in a `session`- or `module`-scoped fixture (shared cost,
the recommended pattern) or a short real wait whose duration is the claim.

### Example 6 — message history and prompt cache

Trigger: the diff enqueues or edits messages, contributes instructions or
tools, renders history in a model adapter, wraps content in tags, or converts
history to or from a UI protocol. The rules, each with its reason, are in
`agent_docs/message-history-and-caching.md`. A violation with a concrete
trigger is HIGH: a moved prefix makes every later request re-process it uncached, with no error,
and content in the wrong channel is a prompt injection.

**Flag this (HIGH):**
```python
async def before_model_request(self, ctx, request_context):
    report = await self._sub_agent_report()
    ctx.enqueue(SystemPromptPart(f'Sub-agent finished:\n{report}'))
```
*Why:* The report is another agent's output, which can repeat anything it
read, and here it is sent in operator voice. It belongs in the tool result
channel, or in a framed and escaped data wrapper on a user turn. The same goes
for a framework or harness hook that rewrites or strips earlier messages by
default or on every request, for
framework guidance enqueued as a plain `str` (it becomes a user message), for
code that finds its own earlier parts by matching their text, and for a new
wrapper with a hand-rolled escaper instead of `_framing.wrap()`, or one that escapes only the closing
`</name>` spelling (`</NAME >`, `< / name>` or an opening `<name>` still gets
through) or escapes only its own tag (a `</context>` or `<system>` in its
body still gets through), uses a `kebab-case` tag name, or puts an
untagged "Tool X completed." line in front of the frame. Likewise a summarizer or judge
that renders a framework data part (a task report, a reminder) as a `user:` turn.

**Don't flag this:**
```python
async def before_model_request(self, ctx, request_context):
    ctx.enqueue(SystemPromptPart('Keep the answer under 100 words.', scope='request'))
```
*Why:* Fixed framework text in system voice, request-scoped, and added to
history. On Anthropic it's sent natively with `clear_at`, so later requests
send it unchanged; elsewhere it's sent at the tail of its own request, behind
the cache breakpoint, and left out of later ones, a deliberate tail move that
keeps the cached prefix. Also don't flag compaction
moving the prefix once when it crosses its threshold, a history processor the
user configures explicitly (`ProcessHistory`) whose docs say it costs the
cached prefix, a request-only change
to `request_context.messages` that every later request applies the same way
(redacting a part from the first request that sends it), or code using today's
mechanism where a rule is marked *(pending #N)*.

### Example 7 — provider adapters

Trigger: the diff changes a model adapter's request mapping, its response or
stream parsing, a capability check in an adapter, or a profile flag an adapter
reads. Read `agent_docs/provider-adapters.md` and apply the rules that govern
what the diff changes; each rule carries its precedent and reason. In
`pydantic_ai_slim/pydantic_ai/models/AGENTS.md`, "Adapter structure"
summarizes the structure rules, and the "Narrow a capability by client class"
bullet holds the gating rule's gateway reasoning. The rules govern the code a
diff adds. A PR that aligns existing code must also meet the guide's scope
rules: flag as MEDIUM a structural alignment whose PR doesn't show the request
bodies captured on `main` and on the branch to be byte-identical, and a
behavioral one that doesn't name each change it makes. A violation is HIGH
when it has a concrete trigger that changes a request or a replayed history,
or that the user's `profile=` can't override. A violation of code shape alone
is MEDIUM.

**Flag this (HIGH):**
```python
# PR adds a beta the adapter needs for a new block type; Vertex doesn't accept it yet
if not isinstance(self.client, AsyncAnthropicVertex):
    betas.add('context-hints-2026-09-01')
```
*Why:* The adapter checks the client class where it decides to send the
feature, so a user can't turn the feature on with `profile=` once Vertex
accepts it, and every other use site needs its own copy of the check. Ask for a
profile flag set in `_instance_profile_defaults()`, which the user's `profile=`
overrides, and read only the flag here. The same goes for a
gate on the model name, the provider name or the `base_url`, a tool return with
files sent as `model_response_str()` alone, `thinking_tags` joined by hand, and
an own native tool part with an unknown `kind` sent in a guessed shape. Flag as
MEDIUM a new part type mapped in a branch nested inside the mapping loop, and a
per-kind branch that only swaps a native tool's wire name.

**Don't flag this:**
```python
# PR adds a service tier to GoogleModel
if self._is_google_cloud:
    headers.update(_google_cloud_service_tier_headers(_resolve_google_cloud_service_tier(model_settings)))
else:
    gla_service_tier = _resolve_gla_service_tier(model_settings)
```
*Why:* Both branches send the feature, each in its transport's own wire form:
Vertex takes headers, the Gemini API a request field. That is mapping, not a
gate, so it may branch on the transport. Also don't flag existing code that the
diff only touches and that already diverges, such as a one-line fix inside a
deeply nested mapper or next to an inline gate on `main`: aligning it is its
own PR.

### Example 8 — durable-exec contract

Trigger: `.review-context/durable-exec-contract.md` exists. Each entry in it is
a core definition a durable engine imports, overrides, reads or projects, with
the engine sites that depend on it. Durable engines run the agent across
activity, step or task boundaries and replay workflow code, so a broken site
fails only inside a durable run, never in a plain `Agent` test. A broken site
with a concrete trigger is HIGH.

**Flag this (HIGH):**
```python
# Entry `models/__init__.py::Model.request`, override sites in TemporalModel,
# DBOSModel, PrefectModel and DurableModel
async def request(self, messages, model_settings, model_request_parameters, *, run_context=None):
```
*Why:* Each listed site overrides `Model.request` with the old signature, so
once core passes `run_context=`, a durable run raises `TypeError` in the
engine's model. Ask for every override to take the new parameter, and for a
workflow-level test that runs a request through an engine. The same goes for a
renamed or retyped field that an engine serializes or reads, a field that a
`projected_field` site copies across a boundary, and a projected field that
becomes a read-only property, which `ModelRequestContextProjection.build_context`
assigns.

**Don't flag this:**
```python
# Entry `models/__init__.py::infer_model`, import sites in durable_exec/_base.py
# and TemporalModel._infer_model
if provider == 'acme':
    return AcmeModel(model_name)
```
*Why:* The sites call `infer_model(name)` and use the `Model` it returns. Its
signature and return contract are unchanged and nothing new crosses a durable
boundary, so the sites still hold and no workflow-level test is needed. Also
don't flag an entry whose sites the diff already updates and tests, or a
missing entry: the manifest is the contract this check reads, not something to
second-guess.

## Sub-agent finding format

When a Task sub-agent returns findings, use this exact format (one block
per finding):

```
- file: path/to/file.py
  line: 42
  severity: HIGH | MEDIUM | LOW | NITPICK | CRITICAL
  title: one-line title
  body: one-paragraph problem statement + concrete failure scenario
  suggestion: (optional) concrete code suggestion
```

Return an empty list if no finding applies.
REVIEW_INSTRUCTIONS_EOF

echo ""
echo "Context gathered in ${CTX}/:"
ls -lh "$CTX/"
DIFF_COUNT=$(find "$CTX/diff" -name '*.diff' 2>/dev/null | wc -l)
echo "  Per-file diffs: ${DIFF_COUNT} files in ${CTX}/diff/"
