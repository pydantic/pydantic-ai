# Compact Retry History

[`CompactRetryHistory`][pydantic_ai.capabilities.CompactRetryHistory] is an opt-in [capability](overview.md) that bounds how much retry context is sent to the model. By default, every failed output or tool attempt stays in the [message history](../message-history.md) — the failed response plus its [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] — so tokens grow with `retries=N`. Add this capability to keep only the last failed attempt and its retry prompt:

```python {title="compact_retry_history.py"}
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.capabilities import CompactRetryHistory
from pydantic_ai.models.function import AgentInfo, FunctionModel


class Item(BaseModel):
    name: str
    qty: int


calls = 0
request_lengths: list[int] = []


def flaky_item(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    global calls
    assert info.output_tools is not None
    calls += 1
    request_lengths.append(len(messages))
    if calls < 3:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": 1, "qty": "nope"}')])
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "widget", "qty": 2}')])


agent = Agent(
    FunctionModel(flaky_item),
    output_type=Item,
    retries=3,
    capabilities=[CompactRetryHistory()],
)

result = agent.run_sync('Give me an item.')
print(request_lengths)
#> [1, 3, 3]
print(result.output)
#> name='widget' qty=2
```

_(This example is complete, it can be run "as is")_

`request_lengths` is the history the model actually receives: `1` on the first request, then `3` on every retry. Without the capability the third request would be `5` messages — the original prompt plus two failed attempts.

A trailing streak of (`ModelResponse`, retry-only `ModelRequest`) pairs is collapsed to the most recent pair before each model request. A request is retry-only when every part is a `RetryPromptPart`. That covers [output retries](../retries.md#output-retries) (text path and tool path) and a streak of function-tool retries where every call in the step failed. Mixed requests — a tool return alongside a retry prompt — are left untouched, as are earlier turns and successful tool exchanges.

Like other [history processors](../message-history.md#processing-message-history), this replaces the run's message history, so [`all_messages()`][pydantic_ai.agent.AgentRunResult.all_messages] after the run also reflects the compacted streak. The default (no capability) is unchanged: every attempt stays in history.

Use a [`ProcessHistory`](process-history.md) processor if you need a different window, such as keeping the last two failures.
