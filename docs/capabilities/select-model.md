# Select Model

[`SelectModel`][pydantic_ai.capabilities.SelectModel] is a [capability](overview.md) that chooses a model from run dependencies, message history, usage, or the current step. The selector is first evaluated during run setup, so the agent does not need a constructor model:

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
