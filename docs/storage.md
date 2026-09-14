# Storage

Three different problems get called "persistence", and they have three different answers. Start here:

| You want to… | Use | Where it lives |
|---|---|---|
| Save a conversation and pick it up later — a chat thread, a support ticket, an assistant that remembers yesterday | [Serialize the message history](message-history.md#storing-and-loading-messages-to-json) into a column of your own database | Core |
| Not write the save-and-load code yourself, and get continue-and-fork for free | [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| A run to survive the process dying mid-tool-call, and resume exactly where it stopped | [Durable execution](durable_execution/overview.md) | Core |

The three compose: a durable engine keeps one run alive, `StepPersistence` records what each run did, and a serialized history is what you hand to the next run. Reaching for a durable engine because you wanted to store a chat thread is the common mistake — a `jsonb` column is enough for that.

## Storing a conversation yourself

Pydantic AI is deliberately unopinionated about your database. It gives you a full-fidelity serialization boundary and leaves the schema to you, because teams' choices here vary more than the framework can usefully guess: which table the history hangs off, which tenant column it needs, how long you keep it.

The primitive is [`ModelMessagesTypeAdapter`](message-history.md#storing-and-loading-messages-to-json), which round-trips a message history to JSON and back with no loss — including fields that are never sent to the model. Store the bytes in a `jsonb` column or equivalent; no schema migration is needed when Pydantic AI adds a message part, because a history serialized by an older version still deserializes.

To store a finished run rather than just its messages — keeping the output, usage, and conversation ID alongside — put an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] on a Pydantic model of your own and serialize that: see [Storing complete run results](message-history.md#storing-complete-run-results).

[`conversation_id`](message-history.md#correlating-runs-with-run_id-and-conversation_id) is the key to store it under, and appending each run's [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages] rather than rewriting the whole list keeps each write proportional to the turn. [Persisting sessions](message-history.md#persisting-sessions) walks through the pattern.

!!! note "Client-supplied history is not trusted state"
    If the history you load came from a browser, sanitize it before passing it to an agent. See [Loading untrusted history](message-history.md#loading-untrusted-history) and the [trust boundary](message-history.md#trust-boundary-for-client-supplied-history).

## Not writing that code yourself

[`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) packages the pattern as a capability you add to an agent, so the load and save calls are not yours to write. It ships in-memory, file, SQLite and MongoDB backends, and its store is a protocol you can implement against your own database.

It records more than the messages: an append-only event log of what the agent did at each boundary, continuable snapshots you can resume or fork a conversation from, and a tool-effect ledger that tells you, after a crash, whether a side effect actually happened.

Two related capabilities build on what it stores: [`ConversationSearch`](https://pydantic.dev/docs/ai/harness/conversation-search/) ranks the stored history and gives the model a tool to pull earlier turns back into context, and [`Memory`](https://pydantic.dev/docs/ai/harness/memory/) keeps notes the agent writes for itself, deliberately outliving any single conversation.

## Letting a provider hold it

Some providers keep conversation state on their side and reconstruct earlier turns from it, so each request carries only what is new — on the OpenAI Responses API that is [`openai_conversation_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_conversation_id], covered under [durable conversations](models/openai.md#using-durable-conversations).

Weigh it against a store of your own: it is one provider's feature, OpenAI documents that earlier input tokens in a chain are still billed, and it is unavailable to organizations with Zero Data Retention enabled.

## What isn't here

Pydantic AI does not checkpoint graph execution state, so there is no "rewind to step 4 of a half-finished run and replay from there" inside a single run. Snapshots are taken at settled boundaries between runs, not mid-node. For a run that must survive a crash *while it is executing*, that is what [durable execution](durable_execution/overview.md) is for.
