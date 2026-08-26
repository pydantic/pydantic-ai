# Compaction

As a conversation grows, its message history can approach the model's context window. *Compaction* keeps it in check by shrinking older messages (trimming, clearing, or summarizing them) while preserving recent context and tool-call integrity. Pydantic AI supports this at several levels: [provider-native compaction APIs](#provider-native-compaction), [model-agnostic history editing](#model-agnostic-compaction) you write yourself, and [Pydantic AI Harness](#pydantic-ai-harness)'s menu of ready-made model-agnostic strategies.

## Provider-native compaction

Some providers expose a built-in compaction API that runs on their side. Pydantic AI wraps these as [capabilities](overview.md):

| Provider | Capability | Details |
|----------|-----------|---------|
| OpenAI Responses API | [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] | [OpenAI compaction](../models/openai.md#message-compaction) |
| Anthropic | [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] | [Anthropic compaction](../models/anthropic.md#message-compaction) |

Each uses the corresponding provider API, so it's only available on that provider.

Pydantic AI treats a compaction part as a visibility boundary for state that feeds future requests. Tool discoveries and on-demand capability loads before the boundary reset, so later requests advertise them again. Capability and toolset authors should apply the same rule to their own derived state: compute anything the model needs to have seen — announcements, disclosures, catalogs — from [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] rather than remembering it in instance attributes, so it self-heals when compaction replaces the history that carried it.

When dispatching a tool call, Pydantic AI uses the provider that served that response to determine whether the provider actually honored the boundary on the request wire. A foreign-provider compaction part, an OpenAI part without encrypted content, or an Anthropic part without summary content does not hide earlier callability evidence, because that provider sent the earlier history to the model. A boundary emitted inside the response containing the call is likewise too late to affect what the model saw for that response. If the response has no provider name, dispatch falls back to the provider-agnostic boundary.

### Client-held history

[`CompactionPart`][pydantic_ai.messages.CompactionPart]s round-trip through the [UI adapters](../ui/overview.md), whose protocols have the client transmit the full conversation history on each request, so compacted conversations keep working with such frontends. A client-submitted compaction item is honored — the conversation stays compacted — but it is never trusted to stand in for the agent's [system prompt](../agent.md#system-prompts): that still reaches the model on every request, as described under [Loading untrusted history](../message-history.md#loading-untrusted-history).

If a run also receives its own server-side history — the [server-side persistence pattern](../ui/overview.md#trust-model-for-client-submitted-messages), where stored messages are passed as `message_history` and client messages only supply the latest turn — client-submitted compaction items are ignored instead. A compaction item marks a boundary before which nothing is sent to the model, so honoring one from the client would let it hide the server's own stored history from the model and substitute its summary for that context. Client-submitted compaction items are only honored when the client-transmitted messages are the entire conversation.

Even then, a client can replay any compaction item the server's provider account has ever produced — opaque encrypted state on OpenAI, a plaintext summary on Anthropic. That is equivalent in kind to fabricating plain-text history, which client-transmitted history always permits (see [Trust boundary for client-supplied history](../message-history.md#trust-boundary-for-client-supplied-history)), with one difference: the server cannot inspect what an opaque item contains. If that matters for your deployment, keep the history server-side: persist the full message list keyed by conversation, send the client only display data, and pass the stored messages as `message_history` on each run. Don't trim the stored history around compaction boundaries yourself — each model adapter already omits what its own provider's compaction replaces, while models from other providers, which ignore a foreign compaction item, still get the full earlier history they need.

## Model-agnostic compaction

To compact on any model, edit the message history yourself with a [history processor](../message-history.md#processing-message-history) wrapped as a [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] capability — this works with every provider. Common patterns:

- [Keep only recent messages](../message-history.md#keep-only-recent-messages) — a zero-cost sliding window over the most recent turns.
- [Summarize old messages](../message-history.md#summarize-old-messages) — use a (cheaper) model to condense older messages into a summary.

## Pydantic AI Harness

[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) packages a menu of ready-made, model-agnostic [compaction strategies](https://pydantic.dev/docs/ai/harness/compaction/): mostly zero-LLM history editing (sliding-window trimming, clearing old tool results, deduplicating repeated file reads, clamping oversized message parts) plus LLM summarization for when that's not enough, and a `TieredCompaction` orchestrator (the recommended default) that escalates from cheap to expensive strategies only as far as needed to fit the target.
