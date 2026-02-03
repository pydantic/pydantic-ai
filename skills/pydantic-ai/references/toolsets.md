# Toolsets Reference

Source: `pydantic_ai_slim/pydantic_ai/toolsets/`

Toolsets are reusable collections of tools that can be registered with agents, composed, filtered, and modified at runtime.

## FunctionToolset

Create a toolset from decorated functions:

```python {title="function_toolset.py"}
from datetime import datetime

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.models.test import TestModel


def temperature_celsius(city: str) -> float:
    return 21.0


def temperature_fahrenheit(city: str) -> float:
    return 69.8


weather_toolset = FunctionToolset(tools=[temperature_celsius, temperature_fahrenheit])


@weather_toolset.tool
def conditions(ctx: RunContext, city: str) -> str:
    if ctx.run_step % 2 == 0:
        return "It's sunny"
    else:
        return "It's raining"


datetime_toolset = FunctionToolset()
datetime_toolset.add_function(lambda: datetime.now(), name='now')

test_model = TestModel()
agent = Agent(test_model)

result = agent.run_sync('What tools are available?', toolsets=[weather_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['temperature_celsius', 'temperature_fahrenheit', 'conditions']

result = agent.run_sync('What tools are available?', toolsets=[datetime_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['now']
```

## Toolset Registration

Toolsets can be registered in multiple ways:

```python {title="toolsets.py"}
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.models.test import TestModel


def agent_tool():
    return "I'm registered directly on the agent"


def extra_tool():
    return "I'm passed as an extra tool for a specific run"


def override_tool():
    return 'I override all other tools'


agent_toolset = FunctionToolset(tools=[agent_tool])
extra_toolset = FunctionToolset(tools=[extra_tool])
override_toolset = FunctionToolset(tools=[override_tool])

test_model = TestModel()
agent = Agent(test_model, toolsets=[agent_toolset])

result = agent.run_sync('What tools are available?')
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool']

result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['agent_tool', 'extra_tool']

with agent.override(toolsets=[override_toolset]):
    result = agent.run_sync('What tools are available?', toolsets=[extra_toolset])
    print([t.name for t in test_model.last_model_request_parameters.function_tools])
    #> ['override_tool']
```

## CombinedToolset

Combine multiple toolsets into one:

```python
from pydantic_ai import CombinedToolset

combined = CombinedToolset([weather_toolset, datetime_toolset])
agent = Agent('openai:gpt-5', toolsets=[combined])
```

## FilteredToolset

Filter tools based on context:

```python
# Filter out fahrenheit tools
filtered = combined_toolset.filtered(
    lambda ctx, tool_def: 'fahrenheit' not in tool_def.name
)
```

## PrefixedToolset

Add prefixes to prevent name conflicts:

```python
combined = CombinedToolset([
    weather_toolset.prefixed('weather'),
    datetime_toolset.prefixed('datetime')
])
# Results in: weather_temperature_celsius, datetime_now, etc.
```

## RenamedToolset

Rename specific tools:

```python
renamed = combined_toolset.renamed({
    'current_time': 'datetime_now',
    'temp_c': 'weather_temperature_celsius',
})
```

## PreparedToolset

Modify tool definitions dynamically before each step:

```python
from dataclasses import replace

from pydantic_ai import RunContext, ToolDefinition

descriptions = {
    'temperature_celsius': 'Get temperature in Celsius',
    'temperature_fahrenheit': 'Get temperature in Fahrenheit',
}

async def add_descriptions(
    ctx: RunContext, tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    return [
        replace(tool_def, description=descriptions.get(tool_def.name, tool_def.description))
        for tool_def in tool_defs
    ]

prepared = renamed_toolset.prepared(add_descriptions)
```

## ApprovalRequiredToolset

Require human approval for tool calls:

```python
from pydantic_ai import DeferredToolRequests, DeferredToolResults

# Require approval for temperature tools
approval_required = toolset.approval_required(
    lambda ctx, tool_def, tool_args: tool_def.name.startswith('temperature')
)

agent = Agent(
    'openai:gpt-5',
    toolsets=[approval_required],
    output_type=[str, DeferredToolRequests],
)

result = agent.run_sync('Get the temperature')
if isinstance(result.output, DeferredToolRequests):
    # Handle approval flow
    for approval in result.output.approvals:
        print(f'Approve {approval.tool_name}? Args: {approval.args}')
```

## WrapperToolset

Subclass to customize tool execution:

```python
from typing import Any

from pydantic_ai import RunContext, WrapperToolset
from pydantic_ai.toolsets import ToolsetTool


class LoggingToolset(WrapperToolset):
    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext, tool: ToolsetTool
    ) -> Any:
        print(f'Calling {name} with {tool_args}')
        result = await super().call_tool(name, tool_args, ctx, tool)
        print(f'Result: {result}')
        return result

logging_toolset = LoggingToolset(my_toolset)
```

## ExternalToolset

Define tools executed by external services (frontend, upstream API):

```python
from pydantic_ai import ExternalToolset, ToolDefinition, DeferredToolRequests

frontend_tools = [
    ToolDefinition(
        name='get_user_location',
        parameters_json_schema={'type': 'object', 'properties': {}},
        description='Get location from browser',
    )
]

external = ExternalToolset(frontend_tools)
agent = Agent(
    'openai:gpt-5',
    toolsets=[external],
    output_type=[str, DeferredToolRequests],
)

result = agent.run_sync('Where am I?')
# result.output contains DeferredToolRequests with calls to execute externally
```

## Dynamic Toolset Building

Build toolsets dynamically based on run context:

```python {title="dynamic_toolset.py", requires="function_toolset.py"}
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel

from function_toolset import datetime_toolset, weather_toolset


@dataclass
class ToggleableDeps:
    active: Literal['weather', 'datetime']

    def toggle(self):
        if self.active == 'weather':
            self.active = 'datetime'
        else:
            self.active = 'weather'

test_model = TestModel()
agent = Agent(
    test_model,
    deps_type=ToggleableDeps
)

@agent.toolset
def toggleable_toolset(ctx: RunContext[ToggleableDeps]):
    if ctx.deps.active == 'weather':
        return weather_toolset
    else:
        return datetime_toolset

@agent.tool
def toggle(ctx: RunContext[ToggleableDeps]):
    ctx.deps.toggle()

deps = ToggleableDeps('weather')

result = agent.run_sync('Toggle the toolset', deps=deps)
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['toggle', 'now']

result = agent.run_sync('Toggle the toolset', deps=deps)
print([t.name for t in test_model.last_model_request_parameters.function_tools])
#> ['toggle', 'temperature_celsius', 'temperature_fahrenheit', 'conditions']
```

## Building Custom Toolsets

Subclass `AbstractToolset` for fully custom implementations:

```python
from pydantic_ai import AbstractToolset, RunContext, ToolDefinition
from pydantic_ai.toolsets import ToolsetTool

class MyToolset(AbstractToolset):
    async def get_tools(self, ctx: RunContext) -> list[ToolsetTool]:
        # Return list of available tools
        ...

    async def call_tool(
        self, name: str, tool_args: dict, ctx: RunContext, tool: ToolsetTool
    ):
        # Execute the tool
        ...

    # Optional: implement __aenter__/__aexit__ for resource management
```

## Third-Party Toolsets

| Toolset | Description |
|---------|-------------|
| `MCPServer` | MCP SDK-based client for MCP servers |
| `FastMCPToolset` | FastMCP-based client with OAuth support |
| `LangChainToolset` | Wrap LangChain tools |
| `ACIToolset` | ACI.dev tools integration |

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `FunctionToolset` | `pydantic_ai.FunctionToolset` | Toolset from functions |
| `CombinedToolset` | `pydantic_ai.CombinedToolset` | Combine multiple toolsets |
| `FilteredToolset` | `pydantic_ai.FilteredToolset` | Filter tools by predicate |
| `PrefixedToolset` | `pydantic_ai.PrefixedToolset` | Add name prefixes |
| `RenamedToolset` | `pydantic_ai.RenamedToolset` | Rename tools |
| `PreparedToolset` | `pydantic_ai.PreparedToolset` | Modify definitions dynamically |
| `ApprovalRequiredToolset` | `pydantic_ai.ApprovalRequiredToolset` | Require human approval |
| `WrapperToolset` | `pydantic_ai.WrapperToolset` | Base for custom wrappers |
| `ExternalToolset` | `pydantic_ai.ExternalToolset` | External/deferred tools |
| `AbstractToolset` | `pydantic_ai.AbstractToolset` | Base class for custom toolsets |

## See Also

- [tools.md](tools.md) — Individual tool registration
- [deferred-tools.md](deferred-tools.md) — Tool approval workflows
- [mcp.md](mcp.md) — MCP server toolsets
- [third-party-tools.md](third-party-tools.md) — LangChain and ACI.dev toolsets
