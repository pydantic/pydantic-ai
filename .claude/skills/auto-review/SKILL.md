---
name: auto-review
description: Architecture review — provider-agnostic interfaces, duplicate-validation, multi-provider feature parity, layout, god methods, dependency floor. Use the other review-* skills for line-level, public-API, behavior, and code-reuse concerns.
user-invocable: true
agent: Explore
allowed-tools: Read, Glob, Grep, Bash(git diff:*), Bash(git status:*), Bash(git log:*)
---

# Auto Review — Architecture

Checks the architecture residual that the other review-* skills don't cover. Local equivalent of the design portion of bots.yml CI review.

## Scope

This skill flags the small, high-value set of architecture concerns:

- Provider-specific features leaking out of provider classes
- Duplicate validation across layers
- Feature parity when 2+ providers are affected
- Module layout (profiles vs. providers)
- God methods
- Dependency floor blast-radius when raising minimums

Everything else is routed to a peer reviewer:

| Concern | Where it's reviewed |
|---|---|
| Public-API signature / symbol changes | `/review-public-api` |
| Downstream integration updates (temporal, ui, mcp, gateway) | `/review-integration-impact` |
| Behavior shift, SDK literal correctness, perf signals | `/review-runtime-behavior` |
| Named items from a spec/doc are present in the diff | `/review-spec-coverage` |
| Logic / feature semantics described in provider docs | `/review-spec-conformance` |
| Single-use helpers, new-file shape, duplicate utilities | `/review-code-reuse` |
| Line-level mechanical style (comments, imports, types, tests) | `/review-patterns` |

If a finding fits one of those, skip it here — the peer reviewer will catch it.

**Specific don'ts** (these have leaked from architecture into other lanes in past runs):

- Do not run `uv run python -c "from <sdk> import X; X(...)"` to validate SDK literal types. SDK construct-time validation is `/review-runtime-behavior`'s tool.
- Do not flag wire-format mismatches against external SDK pydantic models. The fact that the OpenAI SDK rejects `type='function_tool_call'` is runtime-behavior's lane.
- Do not flag whether a `ToolReturnPart.tool_name` carries the right value relative to a provider's wire spec — that's `/review-spec-conformance` (logic/semantics) or `/review-runtime-behavior` (SDK construct-time bug), depending on the source of truth.
- Architecture findings should be expressible without reading any SDK source or running any SDK construction. If you find yourself doing that, you've crossed into another lane.

## Changed Files

!git diff --name-only HEAD

## Instructions

1. Read each changed file.
2. If `.claude/skills/start-worktree-loop/triage.json` exists, read it for maintainer guidance context.
3. Check alignment with `AGENTS.md` / `CLAUDE.md` / `CLAUDE.local.md` standards.
4. Flag architecture decisions needing maintainer awareness or approval.

## Rules

- **Provider-specific features stay in provider classes.** A feature that only applies to one provider must not leak into the generic `Model` / `Agent` interface.
- **No duplicate validation across layers.** If the SDK validates a value, we don't re-validate. If we validate at a boundary, downstream layers trust it.
- **Feature parity.** If a feature applies to 2+ providers, implement it for all of them upfront. Don't leave a one-provider stub.
- **Module layout.** Profiles in `profiles/`, routing in `providers/`. New configuration logic follows that split.
- **No god methods.** Break up methods that mix distinct responsibilities.
- **Dependency floor.** Before raising a dependency's minimum version, check blast radius. If the new requirement is isolated (one file/function), prefer a runtime warning/error when the feature is used over raising the floor for everyone.
- **Pyright errors on dataclass / `__init__` / `__post_init__`**: find equivalent classes in the codebase and compare approaches before choosing a fix. Don't blindly add `# pyright: ignore`.

## Iteration Awareness

If `.claude/skills/start-worktree-loop/review-report.md` exists from a previous iteration, read it first. Do not re-flag issues already reported — only flag new issues or issues whose context materially changed since last review.

## Output Format

```
## [BLOCKING] <title>
- File: <path>
- Rule: <rule violated>
- Detail: <explanation>
- Fix: <suggested fix>

## [WARNING] <title>
- File: <path>
- Detail: <explanation>

## [INFO] <title>
- Detail: <observation>
```

Only **BLOCKING** findings trigger a CODE loop-back in the Ralph Loop. Warnings and info are advisory.

If no issues found: "Auto-review passed. No architecture issues detected."
