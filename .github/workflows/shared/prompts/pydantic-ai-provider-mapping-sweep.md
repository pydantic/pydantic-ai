<!--
Default/seed prompt for the Pydantic AI Provider Mapping Sweep agent.

This file is the COMPLETE prompt. It is the verbatim fallback when the
Logfire managed variable `gh_aw_pydantic_ai_provider_mapping_sweep_prompt`
is unset or unreachable. To iterate on the live prompt, edit that Logfire
variable (start from this file's content below the comment); no recompile or
commit is needed. Keep this file in sync as the reviewed default.
-->

# Pydantic AI Provider Mapping Sweep

Model integrations live in
`pydantic_ai_slim/pydantic_ai/models/` and providers in
`pydantic_ai_slim/pydantic_ai/providers/`.

## Objective

This is a **rotating, single-provider sweep**. Audit exactly **one** model
provider per run for request/response **mapping** bugs — the most frequent,
most reproducible bug class in this repo.

1. Pick one provider to focus on this run. Rotate based on the day-of-year so
   coverage spreads over time: compute `git log -1 --format=%cd --date=format:%j`
   modulo the provider list and pick that index. Provider list (skip any not
   present in the tree): `openai`, `anthropic`, `google`, `groq`, `bedrock`,
   `mistral`, `cohere`, `huggingface`, `openrouter`, `vercel`.
2. State the chosen provider up front.

## What to Look For

In that provider's model + profile modules, audit the request builder
(`_map_messages` / `_map_message` / request-param assembly) and the response
parser against the provider SDK's current types:

- Missing or mismatched `tool_call_id` / tool-call-vs-tool-return pairing.
- Role mapping errors (system/developer/user/assistant/tool), multi-system
  handling, instructions placement.
- Dropped or mis-serialized parts: thinking/reasoning, multimodal (image/audio/
  file), citations, builtin-tool calls, retry/error parts.
- Request params silently dropped or sent in the wrong shape (`service_tier`,
  `reasoning_effort`, `extra_body`, structured-output/strict, stop sequences).
- Malformed-arguments / invalid-tool-call retry path producing a wrong message
  shape.
- Finish-reason / usage mapping (missing cached tokens, wrong `finish_reason`).

## How to Verify — mandatory

Write a **new** minimal test (do not run the existing suite and report its
failures). Prefer constructing `ModelRequest`/`ModelResponse` and asserting on
the mapped provider payload, or use `TestModel`/recorded fixtures. The bug must
be triggered by code you wrote and observed to fail.

## What to Skip

- Speculation without a concrete failing reproduction.
- By-design behavior: check nearby comments, the provider profile, and that
  other providers follow the same pattern before assuming a bug.
- Behavior already tracked by an open issue — **search issues first**.
- Pure feature requests (a provider not supporting a capability at all) — that
  belongs to the parity explore agent, not here.

## Deduplication — mandatory BEFORE filing an issue

Open issues were prefetched before the sandbox started. First check this
sweep's own prior findings with a local label filter:

```
jq '.[] | select(any(.labels[]; .name == "provider-mapping-sweep")) | {number, title, url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

Only if that is inconclusive, widen to a full open-issue scan and grep locally
for your chosen provider name (and the `[provider-mapping-sweep]` and
`[bug-hunter]` prefixes — the latter also files provider bugs):

```
jq '.[] | {number, title, labels: [.labels[].name], url}' \
  /tmp/gh-aw/agent/github-context/open-issues.json
```

Do not enumerate issues with `gh` from inside the sandbox; list requests can
stall until the workflow times out.

If a matching issue covers the same mapping bug, call `mcp__safeoutputs__noop`.

## Sandbox notes

- Read the provider file in full (or large ranges). Model files are typically
  800–1500 lines — read them in 1–2 calls.
- Do NOT spend time trying to import provider SDK type stubs (`mypy_boto3_*`,
  etc.) — they are not installed. Instead, grep the raw
  `botocore/data/*/service-2.json` or use `WebFetch` on the provider's API docs.
- Write your reproduction test using source-level assertions (construct the
  input, call the mapping function, assert on output) — this avoids needing
  `pytest` or the full test environment.

## Quality Gate — When to Noop

`mcp__safeoutputs__noop` is the expected outcome most runs. Call `mcp__safeoutputs__noop` if you could not write a
failing reproduction, the evidence is speculative, a similar issue is open, or
the impact is cosmetic. One well-evidenced issue beats several weak ones.

## Issue Format

**Title:** `<provider>: <short mapping bug summary>`

**Body:**

> ## Impact
> [Who is affected and how — wrong output, dropped data, raised error]
>
> ## Provider & Code Path
> [Provider; file:line of the mapping code at fault]
>
> ## Reproduction
> [The new test you wrote — full code — and the exact command]
>
> ## Expected vs Actual
> **Expected:** … **Actual:** … [captured output]
>
> ## Evidence
> - [SDK type / docs reference showing the correct shape; `path:line` refs]
>
> ## Adversarial review
> - **Reproduced on `main`:** [exact command + real captured output]
> - **Existing tests checked:** [tests read; none assert the current behavior, and the fix doesn't break them]
> - **Ruled out by-design:** [nearby comment / profile / maintainer decision / same in other providers]
> - **SDK verified for this provider:** [the real type/shape, not inferred by analogy to another provider]
> - **Not a duplicate:** [label-filtered dedup returned nothing]
