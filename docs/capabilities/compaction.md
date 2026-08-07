# Compaction

As a conversation grows, its message history can approach the model's context window. *Compaction* keeps it in check by shrinking older messages — trimming, clearing, or summarizing them — while preserving recent context and tool-call integrity. Pydantic AI supports this at several levels, from provider-native APIs to model-agnostic history editing.

## Provider-native compaction

Some providers expose a built-in compaction API that runs on their side. Pydantic AI wraps these as [capabilities](overview.md):

| Provider | Capability | Details |
|----------|-----------|---------|
| OpenAI Responses API | [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] | [OpenAI compaction](../models/openai.md#message-compaction) |
| Anthropic | [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] | [Anthropic compaction](../models/anthropic.md#message-compaction) |

Each uses the corresponding provider API, so it's only available on that provider.

Pydantic AI treats a compaction part as a visibility boundary: the model starts anew from that point for derived tool state. Tool discoveries and on-demand capability loads before the boundary reset, so their tools are hidden again until searched for or loaded after the boundary. Searchable tools remain in the corpus and all registered tools remain callable if the model emits a valid call, even when their earlier schema or reveal evidence is no longer visible to the model. Capability and toolset authors should apply the same rule to their own derived state: compute anything the model needs to have seen — announcements, disclosures, catalogs — from [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] rather than remembering it in instance attributes, so it self-heals when compaction replaces the history that carried it.

### History custody

[`CompactionPart`][pydantic_ai.messages.CompactionPart]s round-trip through the [UI adapters](../ui/overview.md), so compacted conversations remain usable when a frontend owns the message history. Client-supplied compaction items are honored — the conversation stays compacted — but never trusted to have retained the server's standing system prompt, which is re-inserted on ingest (see [Loading untrusted history](../message-history.md#loading-untrusted-history)).

Server-side custody is the strongest posture for compacted history. When history lives client-side, a caller can replay any compaction item the organization has ever minted — opaque provider state on OpenAI, a plaintext summary on Anthropic — which is equivalent in kind to fabricating plaintext history (see [Trust boundary for client-supplied history](../message-history.md#trust-boundary-for-client-supplied-history)), with the difference that the server cannot inspect an opaque item's contents. To keep custody server-side, store the full history keyed by conversation, send only display data to the client, and derive the model-visible window with [`post_compaction_window`][pydantic_ai.messages.post_compaction_window]:

```python {title="server_side_compaction_custody.py"}
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.messages import CompactionPart, post_compaction_window
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel())
conversation_id = 'conversation-123'
server_store: dict[str, list[ModelMessage]] = {
    conversation_id: [
        ModelResponse(
            parts=[
                CompactionPart(
                    content='The user is planning a trip to Kyoto.',
                    provider_name='anthropic',
                ),
                TextPart('What would you like to plan next?'),
            ]
        )
    ]
}

history = post_compaction_window(server_store[conversation_id])
result = agent.run_sync('Find a quiet neighborhood.', message_history=history)
server_store[conversation_id].extend(result.new_messages())

print(result.output)
#> success (no tool calls)
```

## Model-agnostic compaction

To compact on any model, edit the message history yourself with a [history processor](../message-history.md#processing-message-history) wrapped as a [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] capability — this works with every provider. Common patterns:

- [Keep only recent messages](../message-history.md#keep-only-recent-messages) — a zero-cost sliding window over the most recent turns.
- [Summarize old messages](../message-history.md#summarize-old-messages) — use a (cheaper) model to condense older messages into a summary.

## Pydantic AI Harness

[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) packages a menu of ready-made, model-agnostic [compaction strategies](https://pydantic.dev/docs/ai/harness/compaction/): mostly zero-LLM history editing — sliding-window trimming, clearing old tool results, deduplicating repeated file reads, clamping oversized message parts — plus LLM summarization for when that's not enough, and a `TieredCompaction` orchestrator (the recommended default) that escalates from cheap to expensive strategies only as far as needed to fit the target.
