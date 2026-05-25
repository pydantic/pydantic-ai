<!--
Default/seed prompt for the Pydantic AI Provider Parity Explore agent.

This file is the COMPLETE prompt. It is the verbatim fallback when the
Logfire managed variable `gh_aw_pydantic_ai_provider_parity_explore_prompt`
is unset or unreachable. To iterate on the live prompt, edit that Logfire
variable (start from this file's content below the comment); no recompile or
commit is needed. Keep this file in sync as the reviewed default.
-->

# Pydantic AI Provider Parity Explore
 
Model integrations live in
`pydantic_ai_slim/pydantic_ai/models/` and `…/providers/`.

## Objective

This is an **explore**, not a bug hunt. Pick **one** cross-cutting capability
per run and map its support **across all providers**, surfacing silent gaps
and inconsistencies a user would hit when switching providers.

Rotate the capability based on `git log -1 --format=%cd --date=format:%j`
modulo this list:

1. Thinking / reasoning (`reasoning_effort`, thinking parts, budgets).
2. Builtin tools (web search, web fetch, code execution) — presence & version.
3. Usage & cost accounting (cached tokens, reasoning tokens, request counts).
4. Structured output (native JSON schema / strict mode / tool-output mode).
5. Streaming feature parity (deltas for thinking, tool args, usage on finish).
6. Multimodal inputs (image / audio / document) per provider.

## How to Analyze

For the chosen capability, build a **support matrix**: provider × (supported /
partial / silently-ignored / errors / not-applicable), citing the code path
(`file:line`) and the provider SDK/docs that prove the expected behavior.
Distinguish **silent drops** (input accepted, quietly ignored — a bug) from
**explicit non-support** (clearly raised / documented — acceptable).

## What to Look For

- A provider that silently ignores a parameter others honor.
- Stale provider SDK pinning that omits a now-standard capability.
- Inconsistent defaults or types for the same conceptual feature.
- A capability documented as general but missing for a major provider.

## What to Skip

- Per-provider mapping correctness bugs — those belong to the provider
  mapping sweep, not here.
- Speculative "would be nice" features with no user impact.
- Gaps already tracked by an open issue — **search issues first**.

## Deduplication — mandatory BEFORE filing an issue

Search for existing issues using the MCP
GitHub tools (not `gh` CLI — it's blocked by the firewall proxy):

```
mcp__github__search_issues repo:pydantic/pydantic-ai is:issue is:open "[provider-parity-explore]" OR "parity"
mcp__github__search_issues repo:pydantic/pydantic-ai is:issue is:open <capability you're auditing>
```

If a matching issue exists, call `mcp__safeoutputs__noop` immediately.

## Sandbox notes

## Output — When to Noop

If the matrix shows consistent or clearly-documented behavior, call `mcp__safeoutputs__noop`
with a one-line summary. Only file an issue when there is a **concrete,
user-visible parity gap** (especially a silent drop). At most one issue per
run.

## Issue Format

**Title:** `Provider parity: <capability> — <gap summary>`

**Body:**

> ## Capability
> [What was audited this run]
>
> ## Support Matrix
> | Provider | Status | Code path | Notes |
> |---|---|---|---|
> | … | supported / partial / silently-ignored / errors | `file:line` | … |
>
> ## Concrete Gap
> [The specific user-visible problem and which provider(s)]
>
> ## Evidence
> - [SDK/docs references; `path:line`; a short snippet showing the silent drop]
