# Wrap Up At Limits

End a run with a final answer when a [usage limit](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) would otherwise stop it with `UsageLimitExceeded`, or when the run's `timeout=` would stop it with `RunTimedOut`. Use it where the work done before the limit is worth returning, such as a research or coding agent running on a request, tool-call, token, cost or time budget.

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
    result = await agent.run(topic, usage_limits=UsageLimits(request_limit=10, tool_calls_limit=30), timeout=600)
    return result.output
```

When the run's `UsageLimits` leave room for one more model request but not for another after it, or the time left before the run's [`timeout=`](https://pydantic.dev/docs/ai/core-concepts/timeouts/#bounding-a-whole-run) deadline is projected to fit only one more, `WrapUpAtLimits` makes that request the last:

- It keeps the tool definitions and lets the model call none of the function tools, so the cached tools and instructions still match. Some providers still miss part of the cache: Anthropic caches messages per `tool_choice`, and [some cases](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-choice-caching) send a trimmed tool list.
- The run's output stays available. With [tool output](https://pydantic.dev/docs/ai/core-concepts/output/#tool-output), the default for a structured `output_type`, it sets `tool_choice` to `ToolOrOutput(function_tools=[])`, which sends only the output tools, so the model can still give the structured result. Otherwise it sets `tool_choice='none'`: text output is answered in text, and [native](https://pydantic.dev/docs/ai/core-concepts/output/#native-output) and [prompted](https://pydantic.dev/docs/ai/core-concepts/output/#prompted-output) output answer in text that is validated against the schema as usual.
- It adds a request-scoped [instruction capability part](https://pydantic.dev/docs/ai/capabilities/custom/#capability-parts), a `WrapUpPart`, asking the model to answer now with what it has. The model's answer is the run's output.

| Field | Default | Meaning |
|---|---|---|
| `prompt` | `DEFAULT_WRAP_UP_PROMPT` | What the model is told on the final request |
| `timeout_margin` | `0.0` | Seconds to keep in reserve before the run's `timeout=` deadline, on top of the projected time; `None` doesn't wrap up for time |

`WrapUpAtLimits` is opt-in: a run without it raises at its limits as before.

## When the final request is made

| Limit | The final request |
|---|---|
| `request_limit` | The last request the limit allows |
| `tool_calls_limit` | The first request after which the tool calls left are fewer than the run's largest batch so far, or none are left |
| `input_tokens_limit`, `output_tokens_limit`, `total_tokens_limit`, `cost_limit` | The first request after which a final request is projected to go over the limit |
| The run's `timeout=` | The first request after which a final response is projected not to finish before the deadline |

- A batch of tool calls larger than any before it that would go over `tool_calls_limit` still raises `UsageLimitExceeded`, before any of its calls run. Answering such a batch as not executed and wrapping up after it needs core to stop counting denied tool results against the limit ([#8840](https://github.com/pydantic/pydantic-ai/pull/8840)); the request stays open in [#11846](https://github.com/pydantic/pydantic-ai/issues/11846).
- Token use is projected from this run's responses. Input and total tokens are assumed to grow on each further request by as much as the last response grew over the one before, plus the final prompt; output tokens to be at most the largest response's. Cost is projected the same way as input tokens, from the cost of this run's priced responses, without the final prompt's; a response with no known cost isn't counted. A final request that uses more than projected still raises `UsageLimitExceeded`.
- `per_request_input_tokens_limit` isn't projected, and raises as before.
- If the final answer fails output validation, no request is left for the retry and the run raises `UsageLimitExceeded`.
- The run's deadline is read with `RunContext.remaining_time()`. The time left has to fit this request's step, from this request to the next with its tool calls, assumed to take as long as the run's longest step so far, and a final response, assumed to take as long as the run's slowest response, since it makes no tool calls. A final request that takes longer than projected still raises `RunTimedOut`; set `timeout_margin` to keep more time in reserve, such as for a final answer that is longer than the run's other responses. The first request has no step to project from, so it is only made the last when less than `timeout_margin` is left.
- A run given no `usage_limits` gets the default [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits), whose `request_limit` is 50, so it wraps up on its 50th request instead of raising.
- A model that still calls tools on the final request, as a provider that doesn't enforce `tool_choice='none'` allows, has them run as usual, unless `tool_calls_limit` is used up, in which case they raise `UsageLimitExceeded`. After calls that ran, the next request is made the last again, or raises when no request is left.

## Telling a wrapped-up answer from a finished one

The final prompt is a request-scoped `WrapUpPart`, an instruction capability part whose `limit` is the `LimitName` that triggered it. `wrapped_up_by()` reads it back:

```python
from pydantic_ai import AgentRunResult
from pydantic_ai_harness import wrapped_up_by


def report(result: AgentRunResult[str]) -> str:
    if limit := wrapped_up_by(result.new_messages()):
        return f'{result.output}\n\n(cut short by `{limit}`)'
    return result.output
```

## Durable execution

`WrapUpAtLimits` decides from the run's usage and message history, and from the time left before the run's deadline at each of its requests, which it keeps for the run. Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) its hooks run in workflow code, so when Temporal, DBOS or Prefect replays a run, they make the same request the last. On Temporal, the time left is read from the workflow's replay-safe clock, so this holds for `timeout=` too; DBOS and Prefect measure it against the system clock from the run's recorded start, so a recovered run counts the time that passed before recovery, and its time trigger can fire at a different request.

## Telemetry

`WrapUpAtLimits` marks its decision with a zero-duration span on an [instrumented](https://pydantic.dev/docs/ai/integrations/logfire/) agent:

| Span | Attribute | Meaning |
|---|---|---|
| `run wrapped up` | `wrap_up_at_limits.limit` | The `LimitName` that made the request the last |

The span carries no prompt text. The final request itself shows as an ordinary model request.

## Composition

- [`WarnNearLimits`](../compaction/) warns the model as limits approach. `WrapUpAtLimits` acts on the last request. Use both to have the model wind down first.
- It decides on the usage when its hook runs. List it after capabilities that spend from the run's usage before a request, such as `SummarizingCompaction`, so it sees what they spent.
- [Compaction](../compaction/) still runs on the final request. A strategy that makes a model request of its own, such as `SummarizingCompaction`, has no request left there and raises `UsageLimitExceeded`; wrap it in `FallbackCompaction(fallback_on=(UnexpectedModelBehavior, UsageLimitExceeded))` with a strategy that makes no model requests. Skipping compaction on the final request needs a core ordering hook ([#11454](https://github.com/pydantic/pydantic-ai/pull/11454)).
- In [Subagents](../sub_agents/), give a sub-agent its own `SubAgent.usage_limits` and add `WrapUpAtLimits` to its capabilities: it returns a wrapped-up answer to the parent instead of reaching its budget. A sub-agent that shares the parent's budget (no `usage_limits` of its own) would spend the parent's last request on its answer, so the parent then raises.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/wrap_up_at_limits/).
