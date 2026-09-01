# Skill routing

Pick the canonical entry point for the user's intent. **One arrow per intent.** If two skills look right, the one listed here wins.

| Intent | Entry point |
|--------|-------------|
| Start work on a new issue | `/initialize-worktree` |
| Pick up an existing PR (yours or someone else's) | `/adopt-pr` |
| Re-sync `issue-brief.md` after new issue comments | `/refresh-issue-brief` |
| User asks to hand off / clear and continue (plan + clear if harness allows) | EPEH in `branch-context/SKILL.md` (or `/handoff` for persist-only) |
| User asks for a handoff dump without plan mode | `/handoff` → append-only `branch-context/handoffs/` |
| Explain a PR, issue, task, incident, or proposal without Pydantic AI or agent-framework priors | `/explain-pydantic-ai-work-item` |
| Review the branch diff (default) | `/review-branch` |
| Feed a human review miss back into the review suite (the suite missed what a human caught) | `/review-meta-improver` |
| Mechanical lint pass on the diff before committing | `/review-patterns` |
| Inspect or handle PR review feedback | `pr-review-feedback` helpers (`fetch-latest-review`, `sweep-unresolved`) → fix → react 👍/👎 + reply → resolve-threads or minimize-comment |
| Open or advance a PR through current-head CI, hosted review, threads, and metadata | tracked `pushing-commits-to-the-repo` skill |
| Conversation-paced multi-PR triage (David hands you URLs, you drive per-PR claude agents in tmux) | `/pr-orchestrator` |
| Discovered real but out-of-scope work on a complete/stable PR | `/file-followup-issue` (park it as an issue; don't expand the PR) |
| Run tests / diagnose test failures locally | dispatch the `test-runner` agent |
| Record / re-record / debug VCR cassettes | `/testing-skill` |
| Investigate a CI failure on a PR | `/check-ci-runs` |

## Available but not deployed here — read from the toolkit on demand

These skills are NOT copied into worktrees (rarely needed → kept out of every session's context). When your intent matches one, read its `SKILL.md` straight from the toolkit repo (`pydantic-ai-claude-tools`), then follow it. Locally:

`$PYDANTIC_AI_TOOLS_REPO/pydantic-ai-config/.claude/skills-ondemand/<name>/SKILL.md`

| Intent | Skill |
|--------|-------|
| Fix a reported bug (always — repro before fix) | `mre-bug-workflow` |
| Add support for a newly-released model id | `add-new-model` |
| Query Logfire production telemetry | `analyze-logfire-data` |
| Triage / remediate Dependabot security alerts | `dependabot` |
| Publish a repro/demo snippet as a `play.pydantic.work` share via API | `create-playground` |

The abandoned **Ralph loop** (`ralph-loop`, `manage-ralph`, `advance-prs`) and the **v2-cut** machinery (`merge-main-into-v2-main`, `v2-review`, `version-cut-review-rules`, the `v2-sync-*` agents) are archived under `.claude/skills-archive/` + `.claude/agents-archive/` — dormant, kept for the v3 cut / reference. The live PR-advancement workflow is `/pr-orchestrator`.

## Things that look like skills but aren't user-facing

These exist only as orchestrators' children. Don't invoke them directly:

- `auto-review` is the engine inside `/review-branch`; call `/review-branch` instead.
- `pr-review-feedback` is script packaging for the PR-feedback flow.
- All `review-*` agents (code-reuse, integration-impact, public-api, runtime-behavior, spec-conformance, spec-coverage, test-shape, …) are dispatched by `/review-branch`. The user never names them.

## Reference docs (not skills)

When a skill in the table above tells you to consult `pyai-knowledge/<file>.md`, read only the section it points at. Files live at `.agents/skills/pyai-knowledge/` and are large — don't autoload.

## When two intents collide

- "Look at PR feedback **and** check CI" → run them sequentially: inspect with `sweep-unresolved`, fix and resolve feedback, then use `/check-ci-runs`.
- post-v2 there are no new v2:prep/v2:exec PRs; the v2-cut skills (`v2-review`, `merge-main-into-v2-main`, `version-cut-review-rules`) are archived under `skills-archive/` (kept for the v3 cut). Open PRs needing a v2-port use `/review-branch` + the v2 conflict lens in `CLAUDE.local.md`.
- "Bug fix on existing PR" → `/adopt-pr` (if not yet adopted) → `mre-bug-workflow` (on-demand, see above) → then normal flow.
- "Manage several PRs in one sitting" / "drive these reviews for me" → `/pr-orchestrator` (human-in-loop manager).
