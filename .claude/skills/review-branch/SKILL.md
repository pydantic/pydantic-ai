---
name: review-branch
description: Run all review checks on the branch diff against main. Phase-aware — wave 1 (code-correctness) reviewers run first; wave 2 (test-shape, cassette) gates on wave 1 settling. Dispatches only the subset whose preconditions hold, in parallel within each wave. State persisted in `local-notes/review-branch/review-state.json` for re-runs and dashboard consumption.
user-invocable: true
allowed-tools: Read, Write, Glob, Grep, Bash(git diff:*), Bash(git status:*), Bash(git log:*), Bash(git merge-base:*), Bash(git show:*), Bash(git blame:*), Bash(git fetch:*), Bash(gh pr view:*), Bash(gh issue view:*), Bash(gh api:*), Bash(mkdir:*), Bash(cat:*), Bash(grep:*), Bash(wc:*), Bash(sort:*), Bash(uniq:*), Bash(head:*), Bash(tail:*), Bash(sed:*), Bash(jq:*), Bash(sha256sum:*), Bash(shasum:*), Agent
---

# Review Branch

Run all applicable review checks on the branch in one shot, organized into waves so that test-level reviewers (wave 2) only run after code-correctness reviewers (wave 1) have settled.

## Wave 1 — code-correctness reviewers (parallel)

| Reviewer | Subagent type / tool | Always / conditional |
|---|---|---|
| Pattern | invoke `review-patterns` skill via inlined prompt to a `general-purpose` Agent | always (on code changes) |
| Architecture | invoke `auto-review` skill via inlined prompt to a `general-purpose` Agent | always (on code changes) |
| Public API | `Agent(subagent_type="review-public-api", ...)` | `public_modules_changed` |
| Integration impact | `Agent(subagent_type="review-integration-impact", ...)` | `core_symbols_touched` non-empty |
| Spec coverage | `Agent(subagent_type="review-spec-coverage", ...)` | `spec_url_in_pr_body` |
| Spec conformance | `Agent(subagent_type="review-spec-conformance", ...)` | always on code changes (agent NL-gates itself) |
| Runtime behavior | `Agent(subagent_type="review-runtime-behavior", ...)` | any `pydantic_ai/` `.py` touched |
| Code reuse | `Agent(subagent_type="review-code-reuse", ...)` | `new_py_file` or new top-level def/class |

## Wave 2 — test- / artefact-shape reviewers (gated on wave 1 closing)

| Reviewer | Subagent type / tool | Conditional |
|---|---|---|
| Test shape | `Agent(subagent_type="review-test-shape", ...)` | new wire-protocol module + ≥10 small unit tests on internal helpers |
| Cassette | inlined below (small, cassette-specific) | `cassettes_changed` |

Wave 2 runs only when wave 1's `gate_status == "closed"` (= zero BLOCKING findings across all wave-1 reviewers on the current `diff_hash`).

The split exists because test reviewers should pin behavior the production code is committing to, not behavior the production code is about to fix. Running test-shape against a diff with open BLOCKING findings green-lights tests of buggy behavior.

## Critical: dispatch via dedicated subagent_type

When the corresponding agent definition exists at `.claude/agents/review-<name>.md` (or in the symlinked `.agents/`), dispatch with **`subagent_type="review-<name>"`** — NOT `general-purpose`. This loads the agent's lane-fenced system prompt as the system prompt, so the inline `prompt:` field only carries per-run inputs (paths, merge_base, branch).

If a dispatch with the dedicated `subagent_type` fails ("unknown subagent type"), surface that as a diagnostic — it means the agent file isn't being discovered. Do NOT fall back silently to `general-purpose` with the role inlined; report the discovery failure so the user can fix it.

Pattern and Architecture currently remain skills (consumed by ralph agents too), so they're dispatched as `general-purpose` with the skill body inlined — that's expected.

## State file

Persistent at `local-notes/review-branch/review-state.json`. Read at the start of every run, written at the end. Dashboard consumes the same file (mirrors how `ralph-state.json` powers the Live tab).

```json
{
  "version": 1,
  "branch": "<branch-name>",
  "merge_base": "<sha>",
  "diff_hash": "sha256:<hex>",
  "last_run_at": "<ISO-8601>",
  "current_wave": 1,
  "waves": [
    {
      "wave_number": 1,
      "name": "code-correctness",
      "reviewers": {
        "<reviewer-id>": {
          "status": "pending|running|complete|skipped|failed",
          "blocking": <int>,
          "warnings": <int>,
          "report_path": "<path>",
          "reason": "<for skipped/failed only>"
        }
      },
      "blocking_total": <int>,
      "gate_status": "open|closed"
    },
    { "wave_number": 2, "name": "test-shape-and-cassette", ... same shape ... }
  ]
}
```

Reviewer ids:
- `pattern`, `auto-review` — wave 1
- `review-public-api`, `review-integration-impact`, `review-spec-coverage`, `review-spec-conformance`, `review-runtime-behavior`, `review-code-reuse` — wave 1
- `review-test-shape`, `cassette` — wave 2

`diff_hash` is `sha256` of `/tmp/review-branch/code.diff`. If the new run's `diff_hash` differs from the state's, the diff has changed → reset wave to 1, mark all reviewers `pending`, recompute.

## Steps

### 1. Compute branch-specific diff

Identify branch-specific commits (excluding merges that dragged in main):

```bash
git fetch origin main
git diff --stat "$(git merge-base origin/main HEAD)..HEAD"
git log --oneline --no-merges "$(git merge-base origin/main HEAD)..HEAD"
```

Compute file union and the cassette-stripped code diff:

```bash
mkdir -p /tmp/review-branch

MERGE_BASE="$(git merge-base origin/main HEAD)"

# Files actually changed by branch-specific commits.
git diff --name-only "${MERGE_BASE}..HEAD" \
  > /tmp/review-branch/branch-files.txt

# Code diff (cassette-stripped).
git diff "${MERGE_BASE}..HEAD" \
  -- $(cat /tmp/review-branch/branch-files.txt) \
  ':!**/cassettes/**' ':!**/*.cassette.yaml' \
  > /tmp/review-branch/code.diff

# Hash for state continuity.
shasum -a 256 /tmp/review-branch/code.diff | awk '{print "sha256:"$1}' \
  > /tmp/review-branch/diff-hash.txt

# Cassette-only file list.
git diff --name-status "${MERGE_BASE}..HEAD" \
  -- '**/cassettes/**' '**/*.cassette.yaml' \
  | sort -u > /tmp/review-branch/cassettes.list
```

**Important — branch-files.txt is the source of truth for "what was modified."** Subagent prompts must reference only files that appear in this list. Never seed prompts with claims like "X was modified" unless `grep -q "^X$" /tmp/review-branch/branch-files.txt` succeeds.

Report the two counts (code diff line count, cassette file count) before going further.

### 2. Tag added lines as NEW or PRE-EXISTING

For each `+` line in `code.diff`, run `git blame` to find the introducing commit. If the commit is in `git log --no-merges "${MERGE_BASE}..HEAD"` → **NEW**. Otherwise → **PRE-EXISTING**.

Save as `/tmp/review-branch/code.blame-map.txt` — one entry per added line: `<path>:<new-line-no> [NEW|PRE-EXISTING] <commit-sha>`.

Subagents reference this for finding labels. **Do not exclude PRE-EXISTING lines** — flyby fixes are welcome. The label lets the fixer decide in-scope vs. optional.

### 3. Compute dispatcher preconditions

```bash
# public_modules_changed
grep -E '^[+-]{3} [ab]/pydantic_ai_slim/pydantic_ai/' /tmp/review-branch/code.diff \
  | grep -Ev '/_[^/]*\.py|/tests/|/test_|/__pycache__/' \
  > /tmp/review-branch/public-modules.txt

# core_symbols_touched
CORE_SYMBOLS='Agent|Model|ModelSettings|ModelResponse|ModelRequest|ModelResponsePart|ModelRequestPart|ToolDefinition|ToolCallPart|ToolReturnPart|AbstractCapability|AbstractToolset'
grep -E "^[+-]" /tmp/review-branch/code.diff \
  | grep -vE '^[+-]{3}' \
  | grep -oE "\\b(${CORE_SYMBOLS})\\b" \
  | sort -u > /tmp/review-branch/core-symbols-touched.txt

# new_py_file
git diff --name-status --diff-filter=A "${MERGE_BASE}..HEAD" \
  -- 'pydantic_ai_slim/pydantic_ai/**/*.py' \
  > /tmp/review-branch/new-py-files.txt

# spec_url_in_pr_body
gh pr view --json body -q .body 2>/dev/null \
  | grep -iE '^(Feature spec|Spec|Docs|Reference):\s*https?://' \
  > /tmp/review-branch/spec-url.txt || true

# cassettes_changed
[ -s /tmp/review-branch/cassettes.list ] && echo "yes" > /tmp/review-branch/cassettes-flag

# small-unit-tests-on-internal-helpers (review-test-shape gate)
TEST_FN_COUNT=$(grep -cE '^\+(async )?def test_' /tmp/review-branch/code.diff || echo 0)
echo "$TEST_FN_COUNT" > /tmp/review-branch/test-fn-count.txt
```

Report the precondition table:

```
Preconditions:
- public_modules_changed: yes/no (<N> files)
- core_symbols_touched: <comma list or "none">
- new_py_file: <count>
- spec_url_in_pr_body: yes (<url>) / no
- cassettes_changed: <count>
- new_test_fn_count: <count>
```

### 4. Read / initialize state

```bash
STATE_FILE="local-notes/review-branch/review-state.json"
NEW_HASH=$(cat /tmp/review-branch/diff-hash.txt)

if [ -f "$STATE_FILE" ]; then
  OLD_HASH=$(jq -r .diff_hash "$STATE_FILE")
  if [ "$NEW_HASH" != "$OLD_HASH" ]; then
    # Diff changed → reset to wave 1, clear reviewer statuses.
    # Note the change in the report.
    echo "diff changed since last run — resetting to wave 1"
    # ...regenerate state with all wave-1 reviewers as "pending"
  fi
else
  mkdir -p "$(dirname "$STATE_FILE")"
  # initialize state with wave 1 pending, wave 2 blocked
fi

CURRENT_WAVE=$(jq -r .current_wave "$STATE_FILE")
```

### 5. Dispatch wave reviewers in parallel

Issue all dispatches for the current wave in a **single message** so they run concurrently. Skip reviewers whose preconditions don't hold (record them as `status: skipped` in state with the reason).

Common context every dispatch carries:
- `code_diff_path = /tmp/review-branch/code.diff`
- `blame_map_path = /tmp/review-branch/code.blame-map.txt`
- `merge_base = $(cat /tmp/review-branch/merge-base.txt)` (or compute on the fly)
- `branch_files_path = /tmp/review-branch/branch-files.txt` — the canonical "what was modified" list

#### Prompt-seeding rule (critical)

Dispatch prompts must reference **only files that appear in `branch-files.txt`**. Do not say "X was modified" unless X actually appears there. If you want to highlight a file the reviewer should focus on, first verify with:

```bash
grep -qx "<path>" /tmp/review-branch/branch-files.txt && echo "actually modified"
```

When in doubt, omit. The reviewer has the diff and can find what's changed itself.

#### Wave 1 dispatches

Issue these in one message when wave 1 is current and reviewer status is `pending`:

```
Agent(subagent_type="review-runtime-behavior",
      description="Runtime behavior review",
      prompt="Inputs — code_diff_path: /tmp/review-branch/code.diff, blame_map_path: /tmp/review-branch/code.blame-map.txt, merge_base: <sha>, worktree_root: <path>. Run all three passes (behavior shift, SDK/typing correctness, perf signals).")

Agent(subagent_type="review-spec-conformance",
      description="Spec conformance review",
      prompt="Inputs — code_diff_path: /tmp/review-branch/code.diff, blame_map_path: /tmp/review-branch/code.blame-map.txt, branch: <name>. NL-gate yourself; use 'gh pr view' / 'gh issue view' to find linked spec URLs.")

Agent(subagent_type="review-public-api",
      description="Public API review",
      prompt="Inputs — code_diff_path: ..., blame_map_path: ..., merge_base: <sha>.")
# Skip if /tmp/review-branch/public-modules.txt is empty.

Agent(subagent_type="review-integration-impact",
      description="Integration impact review",
      prompt="Inputs — code_diff_path: ..., core_symbols_path: /tmp/review-branch/core-symbols-touched.txt, blame_map_path: ...")
# Skip if core-symbols-touched.txt is empty.

Agent(subagent_type="review-spec-coverage",
      description="Spec coverage review",
      prompt="Inputs — spec_url: <url>, code_diff_path: ...")
# Skip if /tmp/review-branch/spec-url.txt is empty.

Agent(subagent_type="review-code-reuse",
      description="Code reuse review",
      prompt="Inputs — code_diff_path: ..., blame_map_path: ..., new_py_files_path: /tmp/review-branch/new-py-files.txt.")
# Skip if new-py-files.txt is empty AND no new top-level def/class in existing files.

# Pattern (skill, inlined into general-purpose subagent)
Agent(subagent_type="general-purpose",
      description="Pattern review",
      prompt="Run the review-patterns skill against /tmp/review-branch/code.diff. Read .claude/skills/review-patterns/SKILL.md for instructions. Tag findings using /tmp/review-branch/code.blame-map.txt.")

# Architecture (skill, inlined)
Agent(subagent_type="general-purpose",
      description="Architecture review",
      prompt="Run the auto-review skill against /tmp/review-branch/code.diff. Read .claude/skills/auto-review/SKILL.md for instructions. Tag findings using /tmp/review-branch/code.blame-map.txt.")
```

#### Wave 2 dispatches

Issue these only when wave 1's `gate_status == "closed"`. Otherwise mark wave-2 reviewers as `blocked` with `reason: "wave-1 gate open (<N> blocking)"` and skip.

```
Agent(subagent_type="review-test-shape",
      description="Test shape review",
      prompt="Inputs — code_diff_path: ..., blame_map_path: ..., wave_state_path: local-notes/review-branch/review-state.json. Run only if wave-1 gate is closed; otherwise defer.")
# Skip if test-fn-count.txt is below threshold (10) or no new ui/<protocol>/ module.

# Cassette (inlined here — small, cassette-specific):
# For each new/modified cassette in /tmp/review-branch/cassettes.list, parse via
# the pytest-vcr skill's parse_cassette.py and check:
#   - Secrets that survived filtering
#   - Response model name doesn't match the test's configured model
#   - Unexpected 4xx/5xx in non-error tests
#   - Cassette filename doesn't match the test function path
#   - Cassette > 5000 lines
```

### 6. Collect results, update state, compute gate

For each wave-N reviewer that ran:
- Save its report to `/tmp/review-branch/wave<N>/<reviewer-id>.md`
- Parse `BLOCKING` and `WARNING` counts from the report header (each agent emits a Summary block)
- Update its entry in state: `status: complete`, `blocking: N`, `warnings: M`, `report_path: ...`

Compute wave gate:
- `wave.blocking_total = sum(reviewer.blocking for reviewer in wave.reviewers)`
- `wave.gate_status = "closed"` if `wave.blocking_total == 0` else `"open"`

If `wave 1.gate_status == "closed"` and `state.current_wave == 1`:
- Advance `state.current_wave` to 2
- Suggest the user re-run `/review-branch` to dispatch wave 2 (or auto-dispatch in the same run if wave 2 has anything to do)

Write state file and emit the combined report (next step).

### 7. Combined report

```
## Branch Review: <branch-name>

Code diff: <N> lines across <M> files
Cassettes: <P> new, <Q> modified
Diff hash: sha256:<short>  (changed since last run: yes/no)

Current wave: <N>  ·  Wave gate: <open/closed>

### Wave 1 — code-correctness

| Reviewer | Status | Blocking | Warnings | Report |
|---|---|---|---|---|
| pattern | complete | 0 | 26 | wave1/pattern.md |
| auto-review | complete | 4 | 8 | wave1/auto-review.md |
| ... | | | | |

#### <Reviewer name> — <N findings>
<inlined report>

### Wave 2 — test-shape & cassette

(if gate closed)
| Reviewer | Status | Blocking | Warnings | Report |
| ... | | | | |

(if gate open)
Wave 2 deferred — wave 1 has <N> blocking findings to address first.

### Summary
- New-in-branch issues: <N>
- Pre-existing (optional flyby): <N>
- Blocking: <N>
- Warnings: <N>
- Verdict: [PASS / REVIEW BEFORE PUSHING]
- Next action: <one-line — "address blocking findings, re-run /review-branch" / "push when ready" / "wave 2 ready, re-run /review-branch">
```

`Verdict` is PASS only when both waves are closed and zero NEW blocking findings remain.

## Design constraints

- **Cassette exclusion at orchestration.** Code reviewers see `/tmp/review-branch/code.diff` (cassette-stripped). Per-reviewer prompts never re-filter.
- **Reviewers are read-only.** No reviewer mutates the diff, working tree, or git state.
- **Each reviewer's prompt is self-contained.** No reading prior reviewers' reports, no inferring from branch state — only the orchestrator's inputs. Keeps reviewers parallelizable and prevents cross-contamination.
- **State persistence enables re-runs.** Re-running `/review-branch` with the same diff hash continues from the saved state (skipping completed reviewers within a wave is allowed). Re-running with a different diff hash resets to wave 1.
- **Dashboard reads the same state.** `review-state.json` is the source of truth for the `?tab=review` UI; do not duplicate state in another file.
