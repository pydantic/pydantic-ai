---
name: complete-partial-pr
description: Evaluate and complete an issue or PR where the submitted patch fixes only a narrow symptom of the reported pain point. Use when a contribution may miss adjacent integration surfaces, provider/spec semantics, roundtrip behavior, tests, docs, or historical maintainer decisions.
user-invocable: true
allowed-tools: Bash(git:*), Bash(gh:*), Bash(jq:*), Bash(rg:*), Bash(sed:*), Bash(ls:*), Bash(cat:*), Bash(mkdir:*), Bash(date:*), Bash(uv:*), Read, Write, Edit, Glob, Grep, WebFetch, AskUserQuestion, Agent
---

# Complete Partial PR

Use this when a PR or issue patch fixes a small visible failure but may not address the full integration contract. The goal is to turn a narrow contribution into a maintainable Pydantic AI change, or to explain precisely why it should stay narrow.

This is not limited to UI or provider integrations. Apply it to any patch that touches one variant of a broader surface: parse/dump, request/response, streaming/non-streaming, static/dynamic, sync/async, native/provider/local tools, metadata, state machines, durable execution, docs, or tests.

## When To Use

- A contributor's PR addresses the immediate error but not the rest of the user's workflow.
- A fix accepts one shape of data but may not preserve roundtrip semantics.
- A provider or external protocol has more states, fields, or variants than the PR covers.
- The user asks "is this enough?", "what did the contributor miss?", "verify this against the spec", or "improve this branch".
- A reviewer suspects the patch contradicts historical decisions or creates future integration debt.

Do not use this as a replacement for `/review-branch` when the task is only a general code review. If the PR has no local context yet, run `/adopt-pr` first.

## Operating Principle

Separate three things before implementing:

1. The contributor's exact patch.
2. The underlying user pain point.
3. The integration contract Pydantic AI should support.

Only (3) determines the final shape. The submitted patch is evidence, not the boundary.

## Delegation

Use subagents for independent research and review lanes. Keep the critical path local: branch selection, final synthesis, implementation, and push decisions.

Good subagent lanes:

- **Spec researcher**: read the linked issue/PR, provider docs, SDK types, protocol docs, and relevant project docs. Return source-backed facts only, with URLs or file paths.
- **Codepath mapper**: map affected code and adjacent surfaces: loaders, dumpers, stream handlers, request builders, response parsers, tool/native-tool paths, model profiles, docs, and tests.
- **History researcher**: inspect `git blame`, `git log`, previous PRs, review comments, and decision logs around the touched code. Return historical decisions with source links.
- **Test-shape reviewer**: decide what coverage proves the full contract without bloating tests. Identify where parametrization, snapshots, VCR, or direct adapter tests are appropriate.

Give each subagent a narrow question, known PR/issue number, relevant file paths, and the exact output you need. Do not ask multiple subagents to answer the same question. Claims that answer the task need a source.

## Workflow

### 1. Startup

Read the local context before scoping:

- `CLAUDE.md` and `CLAUDE.local.md`
- `agent_docs/index.md`
- `.claude/skills/branch-context/issue-brief.md` and `.claude/skills/branch-context/pr-decisions.md`, if present
- `.claude/skills/pyai-knowledge/feature-map.md` for the affected feature group
- `.claude/skills/pyai-knowledge/internals-model.md` for layer ownership
- `.claude/skills/pyai-knowledge/maintainer-mindset.md` for review tells

If the investigation spans multiple surfaces or subagents, create a short working note under `local-notes/`, for example `local-notes/complete-partial-pr.md`. Keep facts, sources, and open questions there. Do not put research prose in `issue-brief.md`.

### 2. Resolve The Work Item

Use `gh` for GitHub context.

```bash
gh pr view <PR> --json number,title,url,state,body,author,headRepository,headRepositoryOwner,headRefName,maintainerCanModify,baseRefName,comments,reviews,closingIssuesReferences
```

Read linked issues, PR comments, review threads, and CI context. If the work item is an issue with no PR, inspect the linked branch or proposed patch if one exists.

For fork PRs, record:

- `headRepository.nameWithOwner`
- `headRefName`
- `maintainerCanModify`
- whether a plain `git push` targets the actual PR branch

Never force push. If updating a contributor PR and `maintainerCanModify` allows it, push to the contributor's PR branch. If not, create a separate branch and report the limitation.

### 3. Reconstruct The Pain Point

Write a short local working note, even if it only lives in the final report:

- What user workflow failed?
- Which exact symptom does the PR fix?
- Which broader contract might users reasonably expect?
- Which variants or states are implied by the same contract?
- What would be a deliberate non-goal?

This note keeps the implementation from being scoped by the first test the contributor happened to write.

### 4. Verify The Spec

Find authoritative sources before changing code:

- Provider docs, SDK types, protocol references, or API schemas.
- Pydantic AI docs and public API contracts.
- Existing tests and snapshots that encode current behavior.

For external technical specs, use primary sources. If the spec is versioned, note the version used by Pydantic AI and whether the PR targets the same version. Distinguish confirmed spec behavior from inference.

### 5. Map The Integration Surface

Search for symmetric and adjacent code paths. Typical pairs and variants:

- load / dump
- parse / serialize
- request / response
- streaming / non-streaming
- initial / intermediate / final states
- success / error / denied / unavailable
- static / dynamic
- function tools / native tools / provider-executed tools
- IDs, provider names, metadata, and roundtrip preservation
- sync / async
- public API / internal adapter / docs / examples / skills

Do not assume a one-line parser fix is sufficient until this map is complete.

### 6. Check History

Use local git history and GitHub history around the touched code:

```bash
git blame -- <path>
git log --oneline -- <path>
gh pr list --search '<symbol-or-file> repo:pydantic/pydantic-ai' --state all --json number,title,url,state
```

Read previous PRs or comments that introduced the relevant behavior. Capture any decision that constrains the new change. If the new approach contradicts history, call that out explicitly before implementing.

### 7. Decide The Correct Scope

Classify missing work:

- **Required**: needed for spec compliance, roundtrip correctness, backwards compatibility, or the reported user workflow.
- **Regression test**: behavior that should remain deliberately unsupported or should not silently regress.
- **Nice to have**: adjacent polish that is not required for this PR.
- **Out of scope**: broader feature work that deserves a separate issue or design discussion.

Push back on expanding the PR only when the broader behavior is genuinely a separate product decision. Otherwise, complete the contract.

If choosing broader scope or explicitly deferring an adjacent surface, append a concise entry to `.claude/skills/branch-context/pr-decisions.md` with the source that justified the decision.

If the broader fix changes public API, provider semantics, durable behavior, or safety posture in a non-obvious way, ask a maintainer before coding or draft a PR comment/proposal instead of silently expanding the branch.

### 8. Implement Conservatively

Prefer small changes that fit existing abstractions. Preserve backwards compatibility and avoid new public API unless the full contract requires it.

For tests:

- Cover the behavior at the same level users rely on it.
- Prefer integration/adapter-level tests over clusters of helper tests.
- Parametrize related variants and states to reduce bloat.
- Use VCR only when the real provider interaction is the behavior under test. Local protocol conversion usually belongs in direct tests.
- Include regression tests for deliberate non-support decisions.

Run targeted checks first, for example:

```bash
uv run pytest tests/<target>.py::<test_name>
PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright <changed-file.py>
```

Escalate to broader checks only when the blast radius warrants it.

After code changes, run `/review-branch` unless the change is documentation-only or the user explicitly asks for a narrower pass.

### 9. Communicate And Push

Before pushing, verify the remote target:

```bash
git remote -v
git branch -vv
git status --short --branch
```

If posting a GitHub comment, summarize:

- What the original PR covered.
- What additional integration surfaces were checked.
- What improvements were added.
- Which spec and historical sources were used.
- Which tests were run.

If the work changes durable project knowledge, update the relevant docs, branch-context, or skill files.

## Output Checklist

End with a concise report containing:

- Work item and branch updated.
- Spec sources checked.
- Historical decisions checked.
- Integration surfaces covered.
- Tests added or changed.
- Commands run.
- Remaining non-goals or follow-up issues.
