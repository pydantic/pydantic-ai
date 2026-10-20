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
    return '{{ openai:medium }}' if ctx.deps.task_complexity == 'complex' else '{{ openai:small }}'


agent = Agent(deps_type=Deps, capabilities=[SelectModel(select_model)])
```

`SelectModel` always receives a callable, which is evaluated before each new logical model request step. The callable may be synchronous or asynchronous. When it returns the same model ID on multiple steps, the resolved model/provider instance is reused for the rest of that run. Provider-side continuation polling within the same step remains pinned to the selected model. The context also holds the run's output retry count and budget, as [`retry`][pydantic_ai.models.ModelSelectionContext.retry] and [`max_retries`][pydantic_ai.models.ModelSelectionContext.max_retries]: the same values output validators see on [`RunContext`][pydantic_ai.tools.RunContext]. The count goes up each time the model's output is rejected and isn't reset by the function tool calls in between, so a selector can switch to a stronger model once `ctx.retry` reaches a threshold.

See [Selecting the model](custom.md#selecting-the-model) to implement the hook in a custom capability and for precedence and lifecycle details.
