---
description: "Route each Pydantic AI run or step to a different model, chosen from dependencies, message history or usage, with the SelectModel capability."
---

# Select Model

[`SelectModel`][pydantic_ai.capabilities.SelectModel] is a [capability](overview.md) that chooses a model from run dependencies, the run's prompt, message history, usage, the current step, or the run's IDs, deadline and usage limits. The selector is first evaluated during run setup, so the agent does not need a constructor model:

```python {title="adaptive_model.py"}
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel


@dataclass
class Deps:
    """Dependencies that influence model selection."""

    task_complexity: Literal['standard', 'complex']


def select_model(ctx: ModelSelectionContext[Deps]) -> str:
    """Use the larger model for complex tasks."""
    return '{{ openai:medium }}' if ctx.deps.task_complexity == 'complex' else '{{ openai:small }}'


agent = Agent(deps_type=Deps, capabilities=[SelectModel(select_model)])
```

`SelectModel` always receives a callable, which is evaluated before each new logical model request step. The callable may be synchronous or asynchronous. When it returns the same model ID on multiple steps, the resolved model/provider instance is reused for the rest of that run. Provider-side continuation polling within the same step remains pinned to the selected model. The context also holds the run's output retry count and budget, as [`retry`][pydantic_ai.models.ModelSelectionContext.retry] and [`max_retries`][pydantic_ai.models.ModelSelectionContext.max_retries]: the same values output validators see on [`RunContext`][pydantic_ai.tools.RunContext]. The count goes up each time the model's output is rejected and isn't reset by the function tool calls in between, so a selector can switch to a stronger model once `ctx.retry` reaches a threshold.

See [Selecting the model](custom.md#selecting-the-model) to implement the hook in a custom capability and for precedence and lifecycle details.

## Route an output retry to another model

The selector sees how many times the run's output has been rejected, as `ctx.retry`, and the run's output retry budget, as `ctx.max_retries`. It can send the request that follows a rejected output to a stronger model, while the run keeps its history and output retry budget:

```python {title="select_model_output_retry.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    ModelSelectionContext,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models.function import AgentInfo, FunctionModel


def primary_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('draft')])


def backup_response(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    last_request = messages[-1]
    if isinstance(last_request, ModelRequest) and any(isinstance(part, ToolReturnPart) for part in last_request.parts):
        return ModelResponse(parts=[TextPart('ready')])
    return ModelResponse(parts=[ToolCallPart('look_up_answer')])


primary = FunctionModel(primary_response, model_name='primary')
backup = FunctionModel(backup_response, model_name='backup')
# These deterministic responses stand in for calls to the primary and backup models.


def select_model(ctx: ModelSelectionContext[object]) -> FunctionModel:
    return backup if ctx.retry >= 1 else primary


agent = Agent(
    None,
    output_type=str,
    retries={'output': 1},
    capabilities=[SelectModel(select_model)],
)


@agent.tool_plain
def look_up_answer() -> str:
    return 'The answer is ready.'


retry_states: list[tuple[int, int]] = []


@agent.output_validator
def validate_output(ctx: RunContext[object], value: str) -> str:
    retry_states.append((ctx.retry, ctx.max_retries))
    if value == 'draft':
        raise ModelRetry('This is still a draft.')
    return value


result = agent.run_sync('Return the final answer.')
print(result.output)
#> ready
print([message.model_name for message in result.new_messages() if isinstance(message, ModelResponse)])
#> ['primary', 'backup', 'backup']
print(retry_states)
#> [(0, 1), (1, 1)]
```

The primary model's draft is rejected, so the selector routes the retry to the backup model. The backup calls a function tool before it answers, and the step after the tool call stays on the backup: `ctx.retry` counts the run's rejected outputs and isn't reset by the steps in between. The validator sees the same count and budget. With `retries={'output': 1}`, the backup gets one attempt; if its output is also rejected, the run raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].
