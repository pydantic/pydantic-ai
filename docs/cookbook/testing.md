---
title: Test an agent without model requests
description: Exercise agent tools and orchestration deterministically with TestModel and Agent.override.
---

# Test an agent without model requests

Use `TestModel` through `Agent.override()` to exercise the real agent and tools without changing production configuration or calling a provider.

```bash
pip/uv-add pydantic-ai-slim pytest
```

```python {title="test_inventory_agent.py" call_name="test_inventory_agent"}
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5.6-sol')


@agent.tool_plain
def inventory(sku: str) -> int:
    """Return the number of units available for a SKU."""
    return 12


def test_inventory_agent() -> None:
    models.ALLOW_MODEL_REQUESTS = False
    test_model = TestModel(call_tools=['inventory'])

    with agent.override(model=test_model):
        result = agent.run_sync('How many units of SKU-123 are available?')

    assert result.output == '{"inventory":12}'
    assert test_model.last_model_request_parameters is not None
    assert test_model.last_model_request_parameters.function_tools[0].name == 'inventory'
```

`ALLOW_MODEL_REQUESTS = False` prevents an accidental provider call anywhere in the test process. The override is scoped to the `with` block, while the production agent keeps its real model configuration.

## Related

See [Unit testing](../testing.md) for `FunctionModel`, pytest fixtures, and testing model requests directly.
