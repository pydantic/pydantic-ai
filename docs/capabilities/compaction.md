# Compaction

As a conversation grows, its message history can approach the model's context window. *Compaction* keeps it in check by shrinking older messages — trimming, clearing, or summarizing them — while preserving recent context and tool-call integrity. Pydantic AI supports this at several levels, from provider-native APIs to model-agnostic history editing.

## Provider-native compaction

Some providers expose a built-in compaction API that runs on their side. Pydantic AI wraps these as [capabilities](overview.md):

| Provider | Capability | Details |
|----------|-----------|---------|
| OpenAI Responses API | [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] | [OpenAI compaction](../models/openai.md#message-compaction) |
| Anthropic | [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] | [Anthropic compaction](../models/anthropic.md#message-compaction) |

Each uses the corresponding provider API, so it's only available on that provider.

## Model-agnostic compaction

To compact on any model, edit the message history yourself with a [history processor](../message-history.md#processing-message-history) wrapped as a [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] capability — this works with every provider. Common patterns:

- [Keep only recent messages](../message-history.md#keep-only-recent-messages) — a zero-cost sliding window over the most recent turns.
- [Summarize old messages](../message-history.md#summarize-old-messages) — use a (cheaper) model to condense older messages into a summary.

## Pydantic AI Harness

[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) packages a menu of ready-made, model-agnostic [compaction strategies](https://pydantic.dev/docs/ai/harness/compaction/): mostly zero-LLM history editing — sliding-window trimming, clearing old tool results, deduplicating repeated file reads, clamping oversized message parts — plus LLM summarization for when that's not enough, and a `TieredCompaction` orchestrator (the recommended default) that escalates from cheap to expensive strategies only as far as needed to fit the target.
