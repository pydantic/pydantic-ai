# Wrap Up At Limits

End a run with a final answer when a [usage limit](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) would otherwise stop it with `UsageLimitExceeded`. Use it where the work done before the limit is worth returning, such as a research or coding agent running on a request, tool-call or token budget.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/wrap_up_at_limits/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import WrapUpAtLimits

agent = Agent('anthropic:claude-opus-5-5', capabilities=[WrapUpAtLimits()])


async def research(topic: str) -> str:
    result = await agent.run(topic, usage_limits=UsageLimits(request_limit=10, tool_calls_limit=30))
    return result.output
```

When the run's `UsageLimits` leave room for one more model request but not for another after it, `WrapUpAtLimits` makes that request the last:

- It sets `tool_choice='none'` and keeps the tool definitions, so the cached tools and instructions still match. Some providers still miss part of the cache: Anthropic caches messages per `tool_choice`, and [some cases](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-choice-caching) send a trimmed tool list. With [tool output](https://pydantic.dev/docs/ai/core-concepts/output/#tool-output), `tool_choice='none'` sends only the output tools.
- It adds a [request-scoped system prompt](https://pydantic.dev/docs/ai/core-concepts/message-history/#request-scoped-system-prompts) asking the model to answer now with what it has. The model's answer is the run's output.

| Field | Default | Meaning |
|---|---|---|
| `prompt` | `DEFAULT_WRAP_UP_PROMPT` | What the model is told on the final request |

## When the final request is made

| Limit | The final request |
|---|---|
| `request_limit` | The last request the limit allows |
| `tool_calls_limit` | The request after the tool-call budget is used up, or after a batch of tool calls that would go over it |
| `input_tokens_limit`, `output_tokens_limit`, `total_tokens_limit` | The first request after which a final request is projected to go over the limit |

- A batch of tool calls that would go over `tool_calls_limit` is all or none, as without the capability: none of its calls run. Each is answered with `NOT_EXECUTED` under its original `tool_call_id`, so the model sees which calls didn't happen. These answers don't go through core's tool pipeline: no `FunctionToolCallEvent` or `FunctionToolResultEvent` is emitted for them and tool hooks don't see them, so a UI that waits for a call's result event shows them as pending. Both the calls and their answers are in the message history.
- Token use is projected from this run's responses: each further request is assumed to grow by as much as the last response grew over the one before, plus the final prompt. A final request that uses more than projected still raises `UsageLimitExceeded`.
- `cost_limit` isn't projected, and raises as before.
- A model that still calls tools on the final request has them answered with `NOT_EXECUTED` when the tool-call budget is used up, and run as usual otherwise. The next request is made the last again, or raises `UsageLimitExceeded` when no request is left.
- A batch you stream yourself with `node.stream()` under `Agent.iter()` runs before the capability can answer it, so one over `tool_calls_limit` raises as before. Under `Agent.run_stream()`, a response whose text or output arrives before its tool calls ends the stream there, and core runs or checks those calls as before. A run resumed with [deferred tool results](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) is checked by core as before too.

## Telling a wrapped-up answer from a finished one

The final prompt is a `SystemPromptPart` whose `metadata` holds `'origin': 'pydantic-ai-harness.wrap_up_at_limits.v1'` and the `'limit'` that triggered it, a `LimitName`:

```python
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai_harness.wrap_up_at_limits import LimitName


def wrapped_up_by(messages: list[ModelMessage]) -> LimitName | None:
    """The limit that cut the run short, given `result.new_messages()`."""
    for message in messages:
        for part in message.parts if isinstance(message, ModelRequest) else ():
            metadata = part.metadata if isinstance(part, SystemPromptPart) else None
            if metadata and metadata.get('origin') == 'pydantic-ai-harness.wrap_up_at_limits.v1':
                return metadata['limit']
    return None
```

## Durable execution

`WrapUpAtLimits` keeps no state of its own: it decides from the run's usage and message history. Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) its hooks run in workflow code, so when Temporal, DBOS or Prefect replays a run, they make the same request the last.

## Telemetry

`WrapUpAtLimits` emits no spans of its own. On an [instrumented](https://pydantic.dev/docs/ai/integrations/logfire/) agent, the final request shows as an ordinary model request, with `tool_choice` in its settings.

## Composition

- [`WarnNearLimits`](../compaction/) warns the model as limits approach. `WrapUpAtLimits` acts on the last request. Use both to have the model wind down first.
- It decides on the usage when its hook runs. List it after capabilities that spend from the run's usage before a request, such as `SummarizingCompaction`, so it sees what they spent.
- [Compaction](../compaction/) still runs on the final request. A strategy that makes a model request of its own, such as `SummarizingCompaction`, has no request left there and raises `UsageLimitExceeded`; wrap it in `FallbackCompaction(fallback_on=(UnexpectedModelBehavior, UsageLimitExceeded))` with a strategy that makes no model requests.
- In [Subagents](../sub_agents/), pass it in `shared_capabilities` so a sub-agent that reaches its budget returns its wrapped-up answer to the parent.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/wrap_up_at_limits/).
