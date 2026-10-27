---
description: "Keep long Pydantic AI conversations within the context window using OpenAI or Anthropic native compaction, or trim and summarize message history on any model."
---

# Compaction

As a conversation grows, its message history can approach the model's context window. *Compaction* keeps it in check by shrinking older messages (trimming, clearing, or summarizing them) while preserving recent context and tool-call integrity. Pydantic AI supports this at several levels: [provider-native compaction APIs](#provider-native-compaction), [model-agnostic history editing](#model-agnostic-compaction) you write yourself, and [Pydantic AI Harness](#pydantic-ai-harness)'s menu of ready-made model-agnostic strategies.

## Provider-native compaction

Some providers expose a built-in compaction API that runs on their side. Pydantic AI wraps these as [capabilities](overview.md):

| Provider | Capability | Details |
|----------|-----------|---------|
| OpenAI Responses API | [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] | [OpenAI compaction](../models/openai.md#message-compaction) |
| Anthropic | [`AnthropicCompaction`][pydantic_ai.models.anthropic.AnthropicCompaction] | [Anthropic compaction](../models/anthropic.md#message-compaction) |

Each uses the corresponding provider API, so it's only available on that provider, and compacts when the provider's server-side threshold is crossed.

### Compacting on demand

To compact at a moment you choose, add the [`NativeCompaction`][pydantic_ai.capabilities.NativeCompaction] capability. When compaction is requested, with [`ctx.request_compaction()`](#the-compaction-hook) during a run or with [`Agent.compact()`](#compacting-outside-a-run) outside one, it calls the [`compact_messages`][pydantic_ai.models.Model.compact_messages] API of the model that will serve the next request: OpenAI's `/responses/compact` endpoint for [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel.compact_messages], on OpenAI and Azure OpenAI, and Anthropic's [on-demand compaction](../models/anthropic.md#compacting-on-demand) for [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel.compact_messages]. For any other model it does nothing, so the next capability with a [`compact_history`](#the-compaction-hook) hook compacts instead:

```python {title="native_compaction.py" test="skip"}
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeCompaction

agent = Agent('{{ anthropic:medium }}', capabilities=[NativeCompaction()])


@agent.tool
def compact_conversation(ctx: RunContext[Any], focus: str | None = None) -> str:
    """Compact the conversation before your next turn."""
    ctx.request_compaction(reason='manual', focus=focus)
    return 'The conversation will be compacted before your next turn.'
```

The compaction request is the conversation's own request, with its settings, tools and instructions, so it reads the prompt cache the conversation's last request wrote. It summarizes everything before the turn being answered, which it keeps as it was: the latest request, and the response whose tool calls it answers. The compacted history is the provider's [`CompactionPart`][pydantic_ai.messages.CompactionPart], in the `ModelResponse` it arrived in, followed by that turn. Like any `compact_history` hook, it compacts the history as the request shows it, and the run keeps the persistent originals of the turn it kept; see [request-only changes and compaction](custom.md#request-only-changes-and-compaction). Under [`ModelFallback`](model-fallback.md), it compacts for the model that will serve the request, not the run's model; a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] isn't compacted for, as which of its models answers isn't known in advance.

What the providers do with a [`focus`][pydantic_ai.capabilities.CompactionRequest.focus] differs. Anthropic replaces its summarization prompt with the instructions it's given, so the focus is sent inside a full summarization prompt. OpenAI uses them as the system prompt of the compaction request, in place of the agent's instructions, so the request no longer starts with the prefix the conversation cached: by default, `NativeCompaction` leaves a focused request on OpenAI to the next capability, which can keep the cache. Set [`focus_as_system_prompt=True`][pydantic_ai.capabilities.NativeCompaction.focus_as_system_prompt] to send it as the system prompt anyway.

Which models compact on demand, and how, is in their [profile](../models/overview.md#models-and-providers):

| Profile flag | Meaning | OpenAI Responses | Anthropic |
|---|---|---|---|
| [`supports_compaction_on_demand`][pydantic_ai.profiles.ModelProfile.supports_compaction_on_demand] | `compact_messages` works | OpenAI and Azure OpenAI, not through the Pydantic AI Gateway | Claude models that compact on demand |
| [`compaction_instructions`][pydantic_ai.profiles.ModelProfile.compaction_instructions] | What instructions, such as a focus, are | `'system_prompt'` | `'replaces_prompt'` |
| [`compaction_result_readable`][pydantic_ai.profiles.ModelProfile.compaction_result_readable] | Whether the compaction part holds a readable summary | `False`, it's encrypted | `True`, with a signature |

To compact without either, call [`Model.compact_messages`][pydantic_ai.models.Model.compact_messages] directly. It returns a `ModelResponse` holding the compaction part to put in the message history.

### Compacting outside a run

[`Agent.compact()`][pydantic_ai.agent.AbstractAgent.compact] compacts a conversation's message history without a run, for example for a `/compact` command or to compact a stored conversation. The agent's capabilities compact it with their [`compact_history`](#the-compaction-hook) hooks, with the same [events](#compaction-events), for the request a run with this history would make next: with the agent's model, settings, tools and instructions, or the ones you pass. Nothing else is sent. It returns a [`CompactionResult`][pydantic_ai.capabilities.CompactionResult], or `None` when no capability compacted or a listener cancelled the compaction:

```python {title="agent_compact.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeCompaction

agent = Agent('{{ openai:medium }}', capabilities=[NativeCompaction()])


async def main():
    result = await agent.run('Our API uses PostgreSQL 17 behind pgbouncer. Note that down.')

    compaction = await agent.compact(result.all_messages())
    assert compaction is not None

    result = await agent.run('Which database do we use?', message_history=compaction.messages)
    print(result.output)
```

Continue the conversation with the result's [`messages`][pydantic_ai.capabilities.CompactionResult.messages]. If compacting fails, the error is raised: unlike a run, it doesn't move on to a [fallback model](model-fallback.md). Its [`record`][pydantic_ai.capabilities.CompactionResult.record] is the [`CompactionRecord`][pydantic_ai.messages.CompactionRecord] a run would keep on its [`compactions`](#storing-the-full-conversation), so a store saves it the same way. A history that ends with a response whose tool calls haven't been answered can't be compacted: run the agent with it first.

The history is compacted as you pass it, the way a run's own history is: the compacting capability sees the [model request hooks' substitutions](custom.md#request-only-changes-and-compaction) for the request that would follow, but not what they only add to it, and the result keeps the messages you passed as you passed them. `compact()` runs on an agent run that ends before its first request is sent, so the capabilities' `before_run` and `wrap_run` hooks fire as for any run, and set up per-run state, durability and instrumentation. The run then ends the way an [`iter()`][pydantic_ai.agent.AbstractAgent.iter] run you stop iterating does: `after_run` doesn't fire, and the handler `wrap_run` awaits is cancelled.

Under [durable execution](../durable_execution/overview.md), call it from workflow code with an agent that has the durability capability, and the compaction request runs as a durable model operation, as it does during a run. Without one, the request is made inline.

Pydantic AI treats a compaction part as a visibility boundary for state that feeds future requests. Tool discoveries and on-demand capability loads before the boundary reset, so later requests advertise them again. Capability and toolset authors should apply the same rule to their own derived state: compute anything the model needs to have seen — announcements, disclosures, catalogs — from [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] rather than remembering it in instance attributes, so it self-heals when compaction replaces the history that carried it.

When dispatching a tool call, Pydantic AI uses the provider that served that response to determine whether the provider actually honored the boundary on the request wire. A foreign-provider compaction part, an OpenAI part without encrypted content, or an Anthropic part without summary content does not hide earlier callability evidence, because that provider sent the earlier history to the model. A boundary emitted inside the response containing the call is likewise too late to affect what the model saw for that response. If the response has no provider name, dispatch falls back to the provider-agnostic boundary.

### Client-held history

[`CompactionPart`][pydantic_ai.messages.CompactionPart]s round-trip through the [UI adapters](../ui/overview.md), whose protocols have the client transmit the full conversation history on each request, so compacted conversations keep working with such frontends. On AG-UI the compaction item is an activity message, which `@ag-ui/client` doesn't send back; see [Records carried as activity messages](../ui/ag-ui.md#activity-records). A client-submitted compaction item is honored — the conversation stays compacted — but it is never trusted to stand in for the agent's [system prompt](../agent.md#system-prompts): that still reaches the model on every request, as described under [Loading untrusted history](../message-history.md#loading-untrusted-history).

If a run also receives its own server-side history — the [server-side persistence pattern](../ui/overview.md#trust-model-for-client-submitted-messages), where stored messages are passed as `message_history` and client messages only supply the latest turn — client-submitted compaction items are ignored instead. A compaction item marks a boundary before which nothing is sent to the model, so honoring one from the client would let it hide the server's own stored history from the model and substitute its summary for that context. Client-submitted compaction items are only honored when the client-transmitted messages are the entire conversation.

Even then, a client can replay any compaction item the server's provider account has ever produced — opaque encrypted state on OpenAI, a plaintext summary on Anthropic. That is equivalent in kind to fabricating plain-text history, which client-transmitted history always permits (see [Trust boundary for client-supplied history](../message-history.md#trust-boundary-for-client-supplied-history)), with one difference: the server cannot inspect what an opaque item contains. If that matters for your deployment, keep the history server-side: persist the full message list keyed by conversation, send the client only display data, and pass the stored messages as `message_history` on each run. Don't trim the stored history around compaction boundaries yourself — each model adapter already omits what its own provider's compaction replaces, while models from other providers, which ignore a foreign compaction item, still get the full earlier history they need.

## Model-agnostic compaction

To compact on any model, edit the message history yourself with a [history processor](../message-history.md#processing-message-history) wrapped as a [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] capability — this works with every provider. To decide *when* to compact, check [`ctx.context_window_used`][pydantic_ai.tools.RunContext.context_window_used]: the fraction of the model's [`context_window`][pydantic_ai.profiles.ModelProfile.context_window] occupied as of the last response, or `None` when unknown. Common patterns:

- [Keep only recent messages](../message-history.md#keep-only-recent-messages) — a zero-cost sliding window over the most recent turns.
- [Summarize old messages](../message-history.md#summarize-old-messages) — use a (cheaper) model to condense older messages into a summary.

To store a summary you wrote, put it in a [`CompactionPart`][pydantic_ai.messages.CompactionPart] with only `content` set, in a `ModelRequest` that takes the place of the messages it summarizes. That keeps a model-written summary out of the system prompt. A provider's own compaction parts stay in the `ModelResponse` they arrived in; a request-side one can't set `provider_name`, `id` or `provider_details`.

A request-side compaction part wasn't produced by any provider's compaction API, so every provider model sends it as user-voice text inside `<conversation_summary>`, after a line saying it was written by a separate summarization model and is historical context, not new instructions. That's how Claude Code, Codex and OpenCode send their own summaries: the assistant continuing the conversation didn't write it. It's part of a user turn already, so turns still alternate, and the summary can't open or close its tag or a `<system>` tag. A compaction part a provider produced stays in that provider's own role. A [`FunctionModel`][pydantic_ai.models.function.FunctionModel] receives the part itself, and realtime sessions send it as user text. Like any compaction part, it's also a [visibility boundary](#provider-native-compaction) for tool discoveries and on-demand capability loads. [Pydantic AI Harness](#pydantic-ai-harness)'s `SummarizingCompaction` stores its summaries this way, and after each summary records again any capability loads it replaced, so they stay loaded ([details](https://pydantic.dev/docs/ai/harness/compaction/#on-demand-capabilities-across-compaction)).

## The compaction hook

!!! note "The foundation for built-in compaction"
    The hook, [`request_compaction`][pydantic_ai.tools.RunContext.request_compaction] and the compaction events below are the foundation that Pydantic AI's built-in compaction is being built on. [`NativeCompaction`](#compacting-on-demand) compacts with the serving model's own API; a size threshold and a portable summarizer follow in later releases. Until then, you write the capability that decides when to compact, and the one that compacts on other models.

Any capability hook or tool can ask for the history to be compacted with [`ctx.request_compaction()`][pydantic_ai.tools.RunContext.request_compaction], passing a [`reason`][pydantic_ai.messages.CompactionReason] and optionally a `focus`, what the compaction should keep. The next attempt at a model request then calls the capabilities' [`compact_history`][pydantic_ai.capabilities.AbstractCapability.compact_history] hooks with a [`CompactionRequest`][pydantic_ai.capabilities.CompactionRequest]: the history the request would be sent with, and the [`model`][pydantic_ai.capabilities.CompactionRequest.model] that will serve it. Hooks are asked in capability order, through [combined][pydantic_ai.capabilities.CombinedCapability] and [wrapper][pydantic_ai.capabilities.WrapperCapability] capabilities, and the first one to return a [`CompactionResult`][pydantic_ai.capabilities.CompactionResult] wins, so capability order is the fallback chain: a hook that can't compact for this model, or this request, returns `None`.

```python {title="keep_latest_turn.py"}
from typing import Any

from pydantic_ai import Agent, CompactionPart, ModelRequest, RunContext, UserPromptPart
from pydantic_ai.capabilities import AbstractCapability, CompactionRequest, CompactionResult


class KeepLatestTurn(AbstractCapability[Any]):
    async def compact_history(
        self, ctx: RunContext[Any], request: CompactionRequest
    ) -> CompactionResult | None:
        # A turn starts with a user prompt, so cutting there keeps each tool call with its result.
        start = max(
            (
                index
                for index, message in enumerate(request.messages)
                if isinstance(message, ModelRequest)
                and any(isinstance(part, UserPromptPart) for part in message.parts)
            ),
            default=0,
        )
        if start == 0:
            return None
        part = CompactionPart(content=f'{start} earlier messages were left out.')
        return CompactionResult(
            messages=[ModelRequest(parts=[part]), *request.messages[start:]],
            part=part,
            handler='keep_latest_turn',
            kept_from=1,
        )


agent = Agent('{{ openai:medium }}', capabilities=[KeepLatestTurn()])


@agent.tool
def compact_conversation(ctx: RunContext[Any], focus: str | None = None) -> str:
    """Compact the conversation before your next turn."""
    ctx.request_compaction(reason='manual', focus=focus)
    return 'The conversation will be compacted before your next turn.'
```

The result's [`messages`][pydantic_ai.capabilities.CompactionResult.messages] replace the history the request is sent with. Whichever capability compacted, Pydantic AI checks that they contain the result's [`part`][pydantic_ai.capabilities.CompactionResult.part] as their latest [`CompactionPart`][pydantic_ai.messages.CompactionPart], with a summary or a provider's encrypted content, and that they end with the request being answered. Providers also reject a tool result whose tool call was left out, so cut the history at a turn boundary, as the example does.

Set [`kept_from`][pydantic_ai.capabilities.CompactionResult.kept_from] to the index where the messages you kept as they were start, the last messages of the history you were given: `1` above, right after the summary, or the index of the request being answered when you keep nothing else. Pydantic AI checks that those messages are unchanged, and reports the index on the [run result](#storing-the-full-conversation), so a store can tell which messages were summarized. Without it, Pydantic AI uses the longest tail after your `CompactionPart` that matches the end of the history you were given.

The compaction's [`usage`][pydantic_ai.capabilities.CompactionResult.usage] is counted in the run's usage and checked against its [usage limits](../agent.md#usage-limits), without counting as a request.

The hooks are given the run's history as this request shows it, and never commit anything only this request has: a hook's [request-only change](custom.md#request-only-changes-and-compaction) that substitutes for a message, like a file described for a model that can't read it, is what the compaction is made from, but the history keeps the original of every message the compaction keeps; content a hook only adds for this request, like a reminder, isn't given to the compaction at all, and is added again to the request sent after it. A [request-scoped prompt](../message-history.md#request-scoped-system-prompts) is given with its `scope`, so don't summarize it as standing content.

The hooks run for each attempt at the request, after the [`prepare_model_request`](custom.md#model-request-attempts) hooks, so under [`ModelFallback`](model-fallback.md) they compact for the model that will actually serve the request. The compacted history replaces the run's message history only once a response to it is kept. If the attempt fails and the request moves on to another model, the compaction is discarded, its usage is counted like a failed attempt's, and the next attempt compacts again for its own model. A [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] raised by a hook fails the attempt like an error from the model, so `ModelFallback` moves on when the first model's compaction call is rate limited. A request that resumes a [suspended response](model-fallback.md#continuing-suspended-responses) is never compacted: the request waits for the step after it. That includes a step collecting a [batch request](../batch.md)'s result, and the compaction itself is never batched: the hooks get the model before [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests] wraps it.

A capability loaded [on demand](on-demand.md) before the compaction stays loaded: whichever hook compacted, Pydantic AI records each load the compaction dropped again right after it, so the capability's tools and instructions are still delivered. A tool discovered with [tool search](../tools-advanced.md#tool-search) isn't recorded again yet, so the model has to discover it again.

Under [durable execution](../durable_execution/overview.md), the hooks run in workflow code. A request a hook makes through [`request.model`][pydantic_ai.capabilities.CompactionRequest.model] runs as a durable model operation; other I/O belongs in a [`durable_operation`][pydantic_ai.capabilities.durable_operation]. `request_compaction` raises inside a durable unit, such as a tool running in a Temporal activity, whose result is replayed without running it again: request compaction from a capability hook there instead.

### Context overflow

When the model rejects a request for exceeding its context window, with a [`ModelContextWindowExceededError`][pydantic_ai.exceptions.ModelContextWindowExceededError] before any output, and a capability has a `compact_history` hook, Pydantic AI compacts the attempt's history with reason `'context_overflow'` and sends it to the same model once more. That happens before the [`on_model_request_error`](custom.md#error-hooks) hooks see the error, so [`ModelFallback`](model-fallback.md) stays on the model, which can serve the request once it fits. If the compacted request overflows again, or no hook compacts, the overflow reaches the error hooks as it would without compaction (with a note when compacting failed), and `ModelFallback` falls back on it by default. A streamed request is retried the same way: with a `compact_history` hook, the stream is opened before it's handed over, so an overflow the provider reports before any output can still be retried. Under durable execution, the retry is part of the same attempt in workflow code.

### Compaction events

Compaction goes through the run's [event stream](overview.md#capability-events), whichever capability compacts:

- [`CompactionStartEvent`][pydantic_ai.messages.CompactionStartEvent] is emitted before the `compact_history` hooks run, with the reason, the model and the size of the history. It's dispatched immediately: capability listeners run before the hooks do, and any of them can [`cancel()`][pydantic_ai.messages.CompactionStartEvent.cancel] it, which skips compaction for this attempt and drops the request. Under durable execution an `event_stream_handler` runs in its own activity or step, so it only observes the event.
- [`CompactionEndEvent`][pydantic_ai.messages.CompactionEndEvent] is emitted once the history was compacted, with the [`handler`][pydantic_ai.capabilities.CompactionResult.handler] that did it, the history's size before and after, where the kept messages start, the size of the summary, and the compaction's usage.

```python {title="hold_compaction.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import CompactionStartEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.toolsets import FunctionToolset


@dataclass
class HoldCompaction(AbstractCapability[Any]):
    """Holds off compaction while a multi-step plan still needs the full history."""

    planning: bool = False

    async def for_run(self, ctx: RunContext[Any]) -> 'HoldCompaction':
        # A fresh instance per run, so one run's plan doesn't hold off another's compaction.
        return HoldCompaction()

    def get_toolset(self) -> FunctionToolset[Any]:
        toolset = FunctionToolset[Any]()

        @toolset.tool_plain
        def set_planning(planning: bool) -> str:
            """Say whether you're in a multi-step plan that needs the whole conversation."""
            self.planning = planning
            return 'Noted.'

        return toolset

    @on_event(CompactionStartEvent)
    async def _hold(self, ctx: RunContext[Any], event: CompactionStartEvent) -> None:
        if self.planning and event.reason != 'context_overflow':
            event.cancel('a plan step needs the full history')
```

### Storing the full conversation

The message history only shows the compacted history, so a run's [`compactions`][pydantic_ai.agent.AgentRunResult.compactions] say what each compaction it kept replaced, in order; so do the [streamed][pydantic_ai.result.StreamedRunResult.compactions] and [iterated][pydantic_ai.agent.AgentRun.compactions] runs. Each [`CompactionRecord`][pydantic_ai.messages.CompactionRecord] has the [`part`][pydantic_ai.messages.CompactionRecord.part], the handler, reason and model, and the history the compaction was given, as [`replaced`][pydantic_ai.messages.CompactionRecord.replaced]. Its [`summarized`][pydantic_ai.messages.CompactionRecord.summarized] messages are that history without the tail it kept, from [`kept_from`][pydantic_ai.messages.CompactionRecord.kept_from] on in the compacted history. `replaced` is held in memory only and isn't serialized: a record loaded from a serialized result has the rest, but not the messages.

[`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages] keeps its meaning: this run's messages that are still in the history, after any compaction. The run's messages a compaction summarized are only on its record. So a store that never rewrites its log appends them from the records, before each compaction:

```python {title="append_only_log.py"}
from typing import Any

from pydantic_ai import CompactionRecord, ModelMessage
from pydantic_ai.agent import AgentRunResult


def store_run(
    log: list[ModelMessage | CompactionRecord],
    history: list[ModelMessage],
    result: AgentRunResult[Any],
) -> list[ModelMessage]:
    """Append a run to a log that is never rewritten, and return the history for the next run."""
    # Where the messages not stored yet start, in the history each compaction was given.
    position = len(history)
    for record in result.compactions:
        assert record.summarized is not None
        log.extend(record.summarized[position:])
        log.append(record)
        # The messages it kept start at `kept_from` in the compacted history, and some may be stored already.
        position = record.kept_from + max(0, position - len(record.summarized))
    log.extend(result.all_messages()[position:])
    return result.all_messages()
```

## Pydantic AI Harness

[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) packages a menu of ready-made, model-agnostic [compaction strategies](https://pydantic.dev/docs/ai/harness/compaction/): mostly zero-LLM history editing (sliding-window trimming, clearing old tool results, deduplicating repeated file reads, clamping oversized message parts) plus LLM summarization for when that's not enough, and a `TieredCompaction` orchestrator (the recommended default) that escalates from cheap to expensive strategies only as far as needed to fit the target.
