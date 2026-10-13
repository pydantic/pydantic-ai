---
description: "Route each Pydantic AI run or step to a different model, chosen from dependencies, message history or usage, with the SelectModel capability."
---

# Select Model

[`SelectModel`][pydantic_ai.capabilities.SelectModel] is a [capability](overview.md) that chooses a model from run dependencies, the run's prompt, message history, usage, or the current step. The selector is first evaluated during run setup, so the agent does not need a constructor model:

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
    return 'openai:gpt-5.6-sol' if ctx.deps.task_complexity == 'complex' else 'openai:gpt-5.6-luna'


agent = Agent(deps_type=Deps, capabilities=[SelectModel(select_model)])
```

`SelectModel` always receives a callable, which is evaluated before each new logical model request step. The callable may be synchronous or asynchronous. When it returns the same model ID on multiple steps, the resolved model/provider instance is reused for the rest of that run. Provider-side continuation polling within the same step remains pinned to the selected model. See [Selecting the model](custom.md#selecting-the-model) to implement the hook in a custom capability and for precedence and lifecycle details.

## Route an output retry to another model

The selector sees the message history before each request. It can use a retry prompt from a known output validator to choose a backup model for the retry, while the run keeps its history and output retry budget:

```python {title="select_model_output_retry.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    ModelSelectionContext,
    RetryPromptPart,
    RunContext,
    TextPart,
)
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

retry_reason = 'Use the backup model for this output.'


def primary_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('draft')])


def backup_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('ready')])


primary = FunctionModel(primary_response, model_name='primary')
backup = FunctionModel(backup_response, model_name='backup')
# These deterministic responses stand in for calls to the primary and backup models.


def select_model(ctx: ModelSelectionContext[object]) -> FunctionModel:
    current_request = ctx.messages[-1]
    has_validator_retry = isinstance(current_request, ModelRequest) and any(
        isinstance(part, RetryPromptPart) and part.content == retry_reason for part in current_request.parts
    )
    return backup if has_validator_retry else primary


agent = Agent(
    None,
    output_type=str,
    retries={'output': 1},
    capabilities=[SelectModel(select_model)],
)

retry_states: list[tuple[int, int]] = []


@agent.output_validator
def validate_output(ctx: RunContext[object], value: str) -> str:
    retry_states.append((ctx.retry, ctx.max_retries))
    if value == 'draft':
        raise ModelRetry(retry_reason)
    return value


result = agent.run_sync('Return the final answer.')
print(result.output)
#> ready
print([message.model_name for message in result.new_messages() if isinstance(message, ModelResponse)])
#> ['primary', 'backup']
print(retry_states)
#> [(0, 1), (1, 1)]
```

The selector matches the validator's exact retry message in the current request, so retry prompts from other turns do not trigger this route. The response history shows both models in the same run, and the validator sees the single global output retry budget advance from `0` to `1`. With `retries={'output': 1}`, the backup gets one attempt; if that response also fails validation, the run raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].

This recipe uses text output and no function tools. The selector checks only the current request for the retry prompt. If the backup responds with a function call, the next step is selected from a request containing the tool result, so the selector returns to the primary model. Keep this pattern to agents without function tools.
