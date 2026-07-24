<!--
Default/seed prompt for the Pydantic AI Round-Trip Sweep agent.

This file is the COMPLETE prompt. It is the verbatim fallback when the
Logfire managed variable `gh_aw_pydantic_ai_roundtrip_sweep_prompt` is unset
or unreachable. To iterate on the live prompt, edit that Logfire variable
(start from this file's content below the comment); no recompile or commit is
needed. Keep this file in sync as the reviewed default.
-->

# Pydantic AI Round-Trip Sweep

## Objective

Find one concrete **state-loss bug across a serialize → deserialize
boundary** — the highest-density reproducible cluster in this repo. Pick
**one** boundary per run and audit it deeply:

- `ModelMessagesTypeAdapter` / `to_jsonable_python` ↔ `ModelMessage` round-trip.
- `Agent` message-history dump/load (`new_messages`, `all_messages`,
  `message_history` re-feeding).
- AG-UI adapter and Vercel AI adapter request/response conversion.
- Temporal / durable-exec serialization (`value_to_type`, activity payloads).
- Deferred-tool / tool-approval round-trip across a run boundary.

## How to Verify — mandatory

Construct messages that include the **edge-case parts** most likely to be
lost: thinking/reasoning parts, tool calls + tool returns (with ids),
multimodal/binary content, retry/error parts, builtin-tool calls, usage and
timestamps, custom `result_type`/output objects. Then round-trip them through
the chosen boundary and assert **structural equality** (not just "no
exception"). Write this as a **new** minimal test; do not run and report the
existing suite. The bug must be one you triggered and observed.

## What to Look For

- Fields silently dropped or defaulted (timestamps, ids, part kinds, usage).
- Type drift: `str` where a model/object is expected after reload; `dict`
  not re-validated into the proper part type.
- Ordering changes (tool call/return pairing broken after reload).
- Asymmetric adapters (encode then decode ≠ identity).
- Re-fed `message_history` changing run behavior vs the original run.

## What to Skip

- Speculation without a failing reproduction.
- By-design lossy fields explicitly documented as such.
- Behavior already tracked by an open issue or fixed by an open PR — **search both first**.

## Deduplication — mandatory BEFORE filing an issue

The gap may already be tracked by an open **issue** or already fixed by an
open **PR** — check both using the local corpora prefetched before the sandbox
started.

**(a) Existing issues** — first check this sweep's own prior findings with a
label filter:

```
jq '.[] | select(any(.labels[]; .name == "roundtrip-sweep")) | {number, title, url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

Only if that is inconclusive, widen to a full open-issue scan and grep locally
for "round-trip", "serialize", and the boundary/function you investigated:

```
jq '.[] | {number, title, labels: [.labels[].name], url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

**(b) Existing PRs** — a fix may already be open (and even approved). List
open PRs and scan for one touching the failing symbol or file:

```
jq '.[] | {number, title, labels: [.labels[].name], url}' \
  /tmp/gh-aw/agent/github-context/open-pull-requests.json
```

Do not enumerate issues or PRs with `gh` from inside the sandbox; list
requests can stall until the workflow times out.

If a matching open issue or PR exists, call `mcp__safeoutputs__noop`
immediately instead of filing. If a PR looks related but you cannot confirm it
covers this exact gap, still file but fill in the optional **`Possibly
addressed by #<N>`** row at the top of the body template (see Issue Format),
linking that PR.

## Sandbox notes

- Read files in large ranges (500+ lines per call). Do NOT read in 30–80 line chunks.
- Use the native `Grep` and `Glob` tools for codebase search.

## Quality Gate — When to Noop

`mcp__safeoutputs__noop` is the expected outcome most runs. Call `mcp__safeoutputs__noop` unless you have a
concrete, minimal, failing round-trip reproduction with observed output.

## Issue Format

**Title:** `<boundary>: <what is lost> on round-trip`

**Body:** (include the first row only for an uncertain PR match; omit it otherwise)

> **Possibly addressed by #<N>** — [link the related open PR]
>
> ## Impact
> [Who is affected; e.g. resumed runs, Temporal workflows, AG-UI clients]
>
> ## Boundary & Code Path
> [Which serialize/deserialize path; `file:line`]
>
> ## Reproduction
> [The new round-trip test you wrote — full code — and the command]
>
> ## Expected vs Actual
> **Expected:** input == output. **Actual:** [diff of what changed]
>
> ## Evidence
> - [Captured output / diff; `path:line` references]
>
> ## Adversarial review
> - **Reproduced on `main`:** [exact command + real captured output]
> - **Existing tests checked:** [adapter/serialization tests read; none assert this loss is intentional, and the fix doesn't break them]
> - **Ruled out by-design:** [programmatic-only field / request-vs-response union placement / maintainer decision checked]
> - **Not a duplicate:** [label-filtered dedup returned nothing]
