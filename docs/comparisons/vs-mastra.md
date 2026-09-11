# Pydantic AI vs Mastra

Mastra is TypeScript all-in-one: agents, workflows, memory, evals, a playground, Studio, Cloud. If
the product is TypeScript, pick Mastra or the Vercel AI SDK.

Pydantic AI is Python. One extension point, a capability, instead of tools + processors +
guardrails + scorers as separate concepts. Tracing is OpenTelemetry you already run, off by default.
Memory is thinner than theirs.

## Side by side

| | Mastra 1.28 (`@mastra/core` 1.65) | Pydantic AI 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| Extending | Tools, processors, scorers, workflows | One capability |
| Memory | Working, observational, semantic | Deps and history processors |
| Durability | `DurableAgent` in core; Inngest via `@mastra/inngest` | Six engines wrap the agent |
| Tracing | `mastra dev`, Studio, Cloud | OpenTelemetry, when enabled |
| Deploy | Mastra Cloud, or `mastra start` | Anywhere |

## FAQ

**Both?** Mastra in the browser, a Python agent behind HTTP. UI adapters exist.

**Drop-in?** No. Different language.

---

*`mastra` 1.28.0 and `@mastra/core` 1.65.0 tarballs unpacked 2026-09-11 (npm latest that day was
1.29 / 1.66; this page stays on the pins we read). Memory docs in the tarball cover working,
observational, and semantic recall. `./agent/durable` documents `DurableAgent` and
`createInngestAgent`. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
