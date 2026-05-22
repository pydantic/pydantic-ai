#!/usr/bin/env bash
# Gather PR context for the Pydantic AI PR Review agent into /tmp/gh-aw/.review-context/.
# Usage: scripts/gather-pydantic-ai-review-context.sh <pr-number> [repo]
#
# Examples:
#   scripts/gather-pydantic-ai-review-context.sh 4269
#   scripts/gather-pydantic-ai-review-context.sh 4269 pydantic/pydantic-ai
#
# Why outputs live under /tmp (not in the workspace): gh-aw's pre-agent flow
# runs a "Save/Restore agent config folders from base branch" step that
# touches `.github/` (a security feature to prevent PRs from rewriting the
# agent's own configuration). Writing context inside `.github/` or any other
# managed folder (`.agents`, `.claude`, `.codex`, …) is unreliable because
# those snapshots can wipe or shadow what we wrote. `/tmp/gh-aw/...` is
# under the runner's tmp tree, untouched by gh-aw's restore step.
#
# TODO(consolidate): This is a fork of scripts/gather-review-context.sh used
# by the legacy Claude-action workflow (.github/workflows/bots.yml). The two
# will be consolidated once the Claude-action workflow is migrated to the
# Pydantic AI gh-aw shim — until then, keep edits scoped to whichever consumer
# needs them and leave the other script alone.

set -euo pipefail

PR_NUMBER="${1:?Usage: $0 <pr-number> [repo]}"
REPO="${2:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
CTX="/tmp/gh-aw/.review-context"
mkdir -p "$CTX"

# Track every `mktemp` we allocate and unlink them on exit, including the
# `set -e` early-termination path. Callers use `track_tmp <file>` after each
# `mktemp` instead of relying on individual cleanup paths.
_TMP_FILES=()
track_tmp() { _TMP_FILES+=("$1"); }
cleanup_tmp() {
  for f in "${_TMP_FILES[@]:-}"; do
    [ -n "$f" ] && rm -f "$f"
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
# Fetch all review threads first, then determine last auto-review timestamp, then format
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
if [ -n "$BASE_REF" ]; then
  if git fetch "https://github.com/${REPO}.git" "$BASE_REF" --quiet 2>/dev/null; then
    MERGE_BASE=$(git merge-base HEAD FETCH_HEAD 2>/dev/null || echo "")
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

# File orderings for sub-agent fan-out. Each ordering primes one sub-agent to
# spend its early attention on a different slice of the PR; the parent merges
# findings. Generated files (uv.lock, cassettes) are excluded — they don't
# get reviewed. Each file contains one path per line.
echo "  - File orderings (az / za / largest)"
mkdir -p "$CTX/file-orderings"
jq -r '
  [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
  | sort_by(.filename) | .[].filename
' "$FILES_JSON" > "$CTX/file-orderings/az.txt"
jq -r '
  [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
  | sort_by(.filename) | reverse | .[].filename
' "$FILES_JSON" > "$CTX/file-orderings/za.txt"
jq -r '
  [.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)]
  | sort_by(-((.additions // 0) + (.deletions // 0))) | .[].filename
' "$FILES_JSON" > "$CTX/file-orderings/largest.txt"

# PR size summary — file count and total diff lines. The prompt uses this to
# pick a single-pass vs fan-out review strategy.
FILE_COUNT=$(jq '[.[] | select(.filename | test("uv\\.lock|/cassettes/") | not)] | length' "$FILES_JSON")
DIFF_LINES=$(jq '[.[] | select(.filename | test("uv\\.lock|/cassettes/") | not) | (.additions // 0) + (.deletions // 0)] | add // 0' "$FILES_JSON")
printf '%s files, %s diff lines (excluding generated files)\n' "$FILE_COUNT" "$DIFF_LINES" > "$CTX/pr-size.txt"
rm -f "$FILES_JSON"

# Gather AGENTS.md files relevant to the PR — the repo-root file (always,
# when present) plus any per-directory AGENTS.md whose directory has changed
# files in this PR.
echo "  - AGENTS.md files (repo-root + changed directories)"
> "$CTX/agents-md.txt"
{
  if [ -f ./AGENTS.md ]; then
    echo "=== AGENTS.md ==="
    cat ./AGENTS.md
    echo ""
  fi
  for agents_file in $(find . -name AGENTS.md -not -path './.venv/*' -not -path ./AGENTS.md | sed 's|^\./||' | sort); do
    dir=$(dirname "$agents_file")
    if grep -q "^${dir}/" "$CTX/changed-files.txt" 2>/dev/null && [ -f "$agents_file" ]; then
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

**Verdict mapping:** any HIGH or CRITICAL → `REQUEST_CHANGES`. MEDIUM-only
or below → `APPROVE` (post the comments anyway). No findings → `APPROVE`.

**Cap inline comments at 30 per run.** If more findings survive, keep the
highest-severity 30 inline and list the rest briefly in the review body.

## What NOT to flag

This repo runs ruff and pyright in CI and has expert maintainers. The
common false-positive patterns below all *look* like real issues — verify
the surrounding code before posting.

- **Style / formatting** — ruff handles it.
- **Type nits already covered by pyright** — `make typecheck` runs in CI.
- **"Missing tests" for pure refactor** — if the PR moves or renames
  existing code and existing tests still exercise the behavior, no new
  test is needed. Only flag missing tests for new behavior or new public
  API.
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
