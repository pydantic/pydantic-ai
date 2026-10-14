---
description: "Choose how to persist Pydantic AI conversations: store a `Conversation` in your database, use the Harness for persistence and memory, or use durable execution."
---

# Persistence

"Persistence", "memory", "sessions": several different problems go by those names, and they have different answers. Start here:

| You want to… | Use | Where it lives |
|---|---|---|
| Save a conversation and pick it up later — a chat thread, a support ticket, an assistant that remembers yesterday | [Store its `Conversation`](#storing-a-conversation-yourself) in a column of your own database | Core |
| Not write the save-and-load code yourself, and get continue-and-fork for free | [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| The agent to remember what it learned about someone *across* conversations, not just within one | [`Memory`](https://pydantic.dev/docs/ai/harness/memory/) | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| A run to survive the process dying mid-tool-call, and resume exactly where it stopped | [Durable execution](durable_execution/overview.md) | Core |

For checkpoint-style persistence, [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) saves a checkpoint after every step of a run, so you can continue the run later or fork it from any step; to resume a run that crashed partway through a step, use [durable execution](durable_execution/overview.md).

The first two rows are also the answer to "how do I give my agent memory?" for most of what people mean by it: an agent's memory of the conversation it is having *is* its message history. There is no separate memory system to add for that — storing the history and passing it back is the whole mechanism. Memory becomes [its own thing](#remembering-across-conversations) only once it has to outlive the thread.

The rows compose: a durable engine keeps one run alive, `StepPersistence` records what each run did, a stored conversation is what you hand to the next run, and `Memory` is what's left when the thread is over. Reaching for a durable engine because you wanted to store a chat thread is the common mistake — a `jsonb` column is enough for that.

## Storing a conversation yourself

Pydantic AI is deliberately unopinionated about your database. It gives you a full-fidelity serialization boundary and leaves the schema to you, because teams' choices here vary more than the framework can usefully guess: which table the history hangs off, which tenant column it needs, how long you keep it.

The thing to store is a [`Conversation`][pydantic_ai.conversation.Conversation]. Every run hands you one as [`result.conversation`][pydantic_ai.agent.AgentRunResult.conversation], and every run takes one back as `conversation=`. It holds the messages and what lives outside them: the running [`usage`][pydantic_ai.conversation.Conversation.usage], the [`conversation_id`][pydantic_ai.conversation.Conversation.conversation_id] to key it by, and any [deferred tool requests](deferred-tools.md#pausing-a-conversation) the last run paused on. Store it as a field on a Pydantic model of your own, or on its own with [`ConversationTypeAdapter`][pydantic_ai.conversation.ConversationTypeAdapter]:

```python {title="storing_a_conversation.py"}
from pydantic_ai import Agent, ConversationTypeAdapter

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

result = agent.run_sync('Tell me a joke.')
stored = ConversationTypeAdapter.dump_json(result.conversation)  # (1)!

conversation = ConversationTypeAdapter.validate_json(stored)
result = agent.run_sync('Explain?', conversation=conversation)
print(result.usage.requests)  # (2)!
#> 2
```

1. JSON bytes for a `jsonb` column or equivalent, keyed by `result.conversation_id`.
2. The second run counts on from the first, so a [`UsageLimits`][pydantic_ai.usage.UsageLimits] budget covers the whole conversation rather than each run.

It round-trips with the same fidelity as the message history's own [`ModelMessagesTypeAdapter`](message-history.md#storing-and-loading-messages-to-json), including fields that are never sent to the model, like a part's application-only `metadata`. Because that field is typed `Any`, values with no JSON form are normalized on the way through: a `tuple` reloads as a `list`, a `datetime` as its ISO string, and raw `bytes` as their base64 string. The ["What survives a round-trip"](message-history.md#storing-and-loading-messages-to-json) note covers the edges. No schema migration is needed when Pydantic AI adds a message part, because a conversation serialized by an older version still deserializes.

Rewriting the whole conversation on every turn keeps the code simple, but makes each write as large as the conversation so far. To keep writes proportional to the turn, store the messages in a table of their own, appending each run's [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages] with `ModelMessagesTypeAdapter`, and keep the rest of the conversation beside them: its `usage`, its `conversation_id`, and its `deferred_tool_requests`, which a paused conversation can't be resumed without. Reassemble it as `Conversation(messages=..., usage=..., conversation_id=..., deferred_tool_requests=...)` to continue. [Persisting sessions](message-history.md#persisting-sessions) walks through both.

To keep a finished run's output alongside its conversation, store the [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] itself: see [Storing complete run results](message-history.md#storing-complete-run-results).

### Storing a history a chat UI sent you

A frontend on [Vercel AI](ui/vercel-ai.md) or [AG-UI](ui/ag-ui.md) keeps a message list of its own, and the adapter that serves it converts in both directions without a request in hand: [`load_messages`][pydantic_ai.ui.UIAdapter.load_messages] turns the protocol's messages into [`ModelMessage`][pydantic_ai.messages.ModelMessage]s, and [`dump_messages`][pydantic_ai.ui.UIAdapter.dump_messages] turns them back.

```python {title="storing_ui_history.py"}
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import TextUIPart, UIMessage

sent_by_the_browser = [
    UIMessage(id='1', role='user', parts=[TextUIPart(text='Tell me a joke.')])
]

history = VercelAIAdapter.load_messages(sent_by_the_browser)  # (1)!
for_the_browser = VercelAIAdapter.dump_messages(history)  # (2)!
```

1. What the browser sent, as a history an agent can run against — and the shape to store.
2. What the browser gets back, converted at the edge rather than on the way into the database.

Store the Pydantic AI side and convert at the edge, rather than storing the protocol's shape. The wire formats have no place for everything a history carries — what each one keeps and drops is spelled out under [Vercel AI message metadata](ui/vercel-ai.md#message-metadata) and [AG-UI preserving files across round-trips](ui/ag-ui.md#preserving-files-across-round-trips) — and the fields they drop are the ones the next model request needs.

!!! note "Client-supplied history is not trusted state"
    If the history you load came from a browser, sanitize it before passing it to an agent. See [Loading untrusted history](message-history.md#loading-untrusted-history) and the [trust boundary](message-history.md#trust-boundary-for-client-supplied-history).

### Why a conversation rather than its messages

Messages carry more than they look like they do: [`run_id` and `conversation_id`](message-history.md#correlating-runs-with-run_id-and-conversation_id) are stamped onto each one, so a history reloaded from storage stays correlated in [Logfire](logfire.md) with no bookkeeping of your own.

What they can't carry is why the [`Conversation`][pydantic_ai.conversation.Conversation] exists. Its [`usage`][pydantic_ai.conversation.Conversation.usage] is the conversation's running total, including [`tool_calls`][pydantic_ai.usage.RunUsage.tool_calls], which no message records; continue from the messages alone and every turn's [`UsageLimits`][pydantic_ai.usage.UsageLimits] budget starts over from zero. And a run that [paused for deferred tools](deferred-tools.md#pausing-a-conversation) leaves calls that the messages show as unanswered without saying which need approval and which an external result, or what metadata they were deferred with. The conversation carries both. Carrying usage changes nothing about what your traces show: each run's span reports that run's own tokens either way, so a conversation's spend is the sum of its runs.

Nor does a history reach past its own conversation. Replaying yesterday's threads to give an agent that continuity works until it doesn't: the prompt grows without bound, every request pays for it, and [compaction](capabilities/compaction.md) drops the parts you were counting on. [Remembering across conversations](#remembering-across-conversations) is a different mechanism.

## Not writing that code yourself

[`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) packages the pattern as a capability you add to an agent, so the load and save calls are not yours to write. It ships in-memory, file, SQLite and MongoDB backends, and its store is a protocol you can implement against your own database.

It records more than the messages: an append-only event log of what the agent did at each boundary, continuable snapshots you can resume or fork a conversation from, and a tool-effect ledger that tells you, after a crash, whether a side effect actually happened.

[`ConversationSearch`](https://pydantic.dev/docs/ai/harness/conversation-search/) builds on that without storing anything of its own: it ranks the history `StepPersistence` already wrote and gives the model a tool to pull earlier turns back into context on demand, including turns [compaction](capabilities/compaction.md) dropped. Pair the two on one store instance and recall needs no extra write path.

## Remembering across conversations

Everything above is scoped to a conversation. What an agent knows about someone *between* conversations — their preferences, a decision from last week, a correction they shouldn't have to repeat — has a different key and a different lifetime.

[`Memory`](https://pydantic.dev/docs/ai/harness/memory/) is the capability for that. It gives the agent a notebook of Markdown files that it writes, reads, and searches through its own tools, and puts a bounded excerpt in each request rather than the whole notebook. It is keyed by a namespace you resolve from your [dependencies](dependencies.md) — usually a user or tenant ID — rather than by `conversation_id`, which is exactly what lets it outlive the thread. Its stores are the durable ones: a file directory, SQLite, PostgreSQL, or one you implement against your own database.

It stays a capability of its own, with a store of its own, rather than something `StepPersistence` writes: a versioned notebook and an append-only run log have little in common. But the two take the same database, so keeping an agent's notes next to its conversations is one connection and one thing to back up.

Anthropic exposes a memory tool on its own side of the API: [`MemoryTool`](native-tools.md#memory-tool) has the model drive a directory of memory files through a tool contract the provider defines, with the storage behind it still yours to supply.

!!! note "Memory is content the model wrote"
    Notes an agent left for itself re-enter later prompts, and a note can be mistaken or planted by whoever the agent was talking to. `Memory` injects them as user-role content rather than as instructions, which lowers their authority, but that is not a hard prompt-injection boundary. The capability's [security and provenance notes](https://pydantic.dev/docs/ai/harness/memory/#security-and-provenance) cover what it does and doesn't guarantee.

## Letting a provider hold it

Some providers keep conversation state on their side and reconstruct earlier turns from it, so each request carries only what is new — on the OpenAI Responses API that is [`openai_conversation_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_conversation_id], covered under [durable conversations](models/openai.md#using-durable-conversations).

Weigh it against a store of your own: it is one provider's feature, OpenAI documents that earlier input tokens in a chain are still billed, and it is unavailable to organizations with Zero Data Retention enabled.

## What isn't here

Nothing here snapshots state in the middle of a step, so there is no "rewind to step 4 of a half-finished run and replay from there" inside a single run. Snapshots are taken at settled boundaries between runs, not mid-node. For a run that must survive a crash *while it is executing*, that is what [durable execution](durable_execution/overview.md) is for.
