# Set Tool Metadata

[`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] is a [capability](overview.md) that merges metadata key-value pairs onto selected tools. This is useful for tagging tools with configuration that other capabilities or custom logic can inspect:

```python {title="set_tool_metadata.py" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata
from pydantic_ai.models.test import TestModel


test_model = TestModel()
agent = Agent(
    test_model,
    capabilities=[SetToolMetadata(tools=['search'], sensitive=True)],
)


@agent.tool_plain
def search(query: str) -> str:
    """Search for information."""
    return f'Results for: {query}'


@agent.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'


result = agent.run_sync('Search for pydantic')
params = test_model.last_model_request_parameters
assert params is not None
search_tool = next(t for t in params.function_tools if t.name == 'search')
greet_tool = next(t for t in params.function_tools if t.name == 'greet')
assert search_tool.metadata is not None and search_tool.metadata.get('sensitive') is True
assert greet_tool.metadata is None or greet_tool.metadata.get('sensitive') is None
```

_(This example is complete, it can be run "as is")_

The same effect can be achieved at the toolset level using [`.with_metadata()`][pydantic_ai.toolsets.AbstractToolset.with_metadata] — see [toolset composition](../toolsets.md#setting-tool-metadata).
