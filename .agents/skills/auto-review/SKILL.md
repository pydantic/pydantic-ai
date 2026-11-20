---
name: auto-review
description: INTERNAL — architecture review engine inside /review-branch. Use /review-branch as the user-facing entry. Kept user-invocable as a fallback for standalone architecture-only passes.
user-invocable: true
disable-model-invocation: true
agent: Explore
allowed-tools: Read, Glob, Grep, Bash(git diff:*), Bash(git status:*), Bash(git log:*), Bash(git show:*)
---

# Auto Review — Architecture

Flag the architecture concerns no other review lane owns:

- provider-specific features leaking out of provider classes
- duplicate validation across layers
- feature parity when 2+ providers are affected
- module layout (profiles vs. providers)
- god methods
- dependency-floor blast radius when raising a minimum

Everything else belongs to a peer lane. Skip a finding that fits one:

| Concern | Lane |
|---|---|
| Public-API signature or symbol changes | `review-public-api` |
| Downstream integration updates (durable exec, UI, MCP, gateway, harness) | `review-integration-impact` |
| Behavior shift, SDK literal correctness, perf signals | `review-runtime-behavior` |
| Items a spec names are present in the diff | `review-spec-coverage` |
| Logic and feature semantics from provider docs | `review-spec-conformance` |
| Single-use helpers, new-file shape, duplicate utilities | `review-code-reuse` |
| Line-level mechanical style | `review-patterns` |

Architecture findings never need SDK source or SDK construction. Do not run
`uv run python -c "from <sdk> import X; X(...)"`, and do not flag wire-format mismatches
against SDK models or a `ToolReturnPart.tool_name` value against a provider's wire spec:
those belong to `review-runtime-behavior` or `review-spec-conformance`.

## Scope

Under `/review-branch`, the dispatch prompt names `code_diff_path` and `merge_base`. Review
only that diff. Never run `git diff main`, `git diff origin/main`, `git diff upstream/main` or
`git merge-base` to recompute the base: in a worktree those refs can be stale or point at a fork.

Standalone, review the uncommitted changes:

!git diff --name-only HEAD

## Instructions

1. Read each changed file.
2. Check alignment with the repo's `AGENTS.md` / `CLAUDE.md` standards and `agent_docs/index.md`.
3. Read `.agents/skills/review-branch/maintainer-mindset.md` → *Review Tells* and *Internals:
   Centralize Logic, Respect Layers*. Ask its questions ("Where does this fact belong?", "Is this
   the same abstraction under another name?", "Is this cross-cutting behavior a capability now?")
   alongside the rules below.
4. Flag architecture decisions that need maintainer awareness or approval.

## Rules

- **Provider-specific features stay in provider classes.** A feature for one provider must not
  leak into the generic `Model` / `Agent` interface.
- **No duplicate validation across layers.** If the SDK validates a value, do not re-validate it.
  If a boundary validates, downstream layers trust it.
- **Feature parity.** A new feature that applies to 2+ providers ships for all of them, not as a
  one-provider stub. A bug fix stays narrow: extend it to a sibling only once the defect is
  reproduced there (root `AGENTS.md` → "Requirements of all contributions").
- **Module layout.** Profiles in `profiles/`, routing in `providers/`. New configuration logic
  follows that split.
- **No god methods.** Split methods that mix distinct responsibilities.
- **Dependency floor.** Before raising a dependency's minimum version, check the blast radius. When
  the new requirement is isolated to one file or function, prefer a runtime warning or error on
  use over raising the floor for everyone.
- **Pyright errors on dataclass / `__init__` / `__post_init__`**: compare how equivalent classes in
  the codebase solve it before choosing a fix. Do not reach for `# pyright: ignore`.

## Output Format

End every finding with its own line `Verdict: <do|defer|skip> — <reason>`, assigned by the
net-value rules in `.agents/skills/review-branch/verdict-rubric.md`. Severity is not
actionability: a WARNING or INFO with `do` is a required action. "No behavioral change" is not a
reason to skip.

```
## [BLOCKING] <title>
- File: <path>
- Rule: <rule violated>
- Detail: <explanation>
- Verdict: do — <reason>
- Fix: <suggested fix>

## [WARNING] <title>
- File: <path>
- Detail: <explanation>
- Verdict: <do|defer|skip> — <reason>

## [INFO] <title>
- File: <path>
- Detail: <observation>
- Verdict: <do|defer|skip> — <reason>
```

If no issues found: "Auto-review passed. No architecture issues detected."
