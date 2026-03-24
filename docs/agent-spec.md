
# Agent Specs

Agent specs let you define agents declaratively in YAML or JSON — model, instructions, capabilities, and all. One line to load, no Python agent construction code required.

This is useful for:

- Separating agent configuration from application code
- Letting non-developers (prompt engineers, domain experts) configure agents
- Storing agent definitions alongside other config files
- Sharing agent configurations across teams or projects

## Defining a spec

A spec file defines the agent's configuration in YAML or JSON:

```yaml {title="agent.yaml" test="skip"}
model: anthropic:claude-opus-4-6
instructions: You are a helpful research assistant.
model_settings:
  max_tokens: 8192
capabilities:
  - WebSearch
  - Thinking:
      effort: high
```

## Loading specs

[`Agent.from_file`][pydantic_ai.Agent.from_file] loads a spec from a YAML or JSON file and constructs an agent:

```python {title="from_file_example.py" test="skip"}
from pydantic_ai import Agent

agent = Agent.from_file('agent.yaml')
```

[`Agent.from_spec`][pydantic_ai.Agent.from_spec] accepts a dict or [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] instance and supports additional keyword arguments that supplement or override the spec:

```python {title="from_spec_example.py"}
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class UserContext:
    user_name: str


agent = Agent.from_spec(
    {
        'model': 'anthropic:claude-opus-4-6',
        'instructions': 'You are helping {{user_name}}.',
        'capabilities': ['WebSearch'],
    },
    deps_type=UserContext,
)
```

Keyword arguments interact with spec fields as follows:

* **Scalar fields** (`model`, `name`, `retries`, `end_strategy`, etc.) — the keyword argument overrides the spec value when provided.
* **`instructions`** — merged: spec instructions come first, then keyword argument instructions.
* **`capabilities`** — merged: spec capabilities come first, then keyword argument capabilities.
* **`model_settings`** — merged additively: keyword argument settings override matching spec settings.
* **`output_type`** — takes precedence over `output_schema` from the spec.

When `deps_type` is passed, [template strings](#template-strings) in the spec's `instructions`, `description`, and capability arguments are compiled and validated against the deps type at construction time.

For more control over spec loading, use [`AgentSpec.from_file`][pydantic_ai.agent.spec.AgentSpec.from_file] to load the spec separately before passing it to `Agent.from_spec`.

## Template strings

[`TemplateStr`][pydantic_ai.TemplateStr] provides Handlebars-style templates (`{{variable}}`) that are rendered against the agent's [dependencies](dependencies.md) at runtime. Template strings work anywhere instructions or descriptions are accepted — in Python code, in [capabilities](capabilities.md), and in agent specs.

```python {title="template_instructions.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, TemplateStr


@dataclass
class UserProfile:
    name: str
    role: str


agent = Agent(
    'openai:gpt-5.2',
    deps_type=UserProfile,
    instructions=TemplateStr('You are assisting {{name}}, who is a {{role}}.'),
)
result = agent.run_sync('hello', deps=UserProfile(name='Alice', role='engineer'))
print(result.output)
#> Hello! How can I help you today?
```

Template variables are resolved from the fields of the `deps` object. When a `deps_type` is provided, template variable names are validated at construction time.

In agent specs, strings containing `{{` are automatically converted to template strings — no explicit `TemplateStr` wrapper is needed. This applies to the `instructions` and `description` fields.

!!! tip
    In Python code, a callable with [`RunContext`][pydantic_ai.tools.RunContext] is generally preferred over `TemplateStr` for IDE autocomplete and type checking. Template strings shine in spec files where callables aren't available.

## Capability spec syntax

Capabilities in specs support three forms:

* `'MyCapability'` — no arguments, calls `MyCapability.from_spec()`
* `{'MyCapability': value}` — single positional argument, calls `MyCapability.from_spec(value)`
* `{'MyCapability': {key: value, ...}}` — keyword arguments, calls `MyCapability.from_spec(**kwargs)`

## Custom capabilities in specs

To make a custom capability work with specs, it needs a [`get_serialization_name`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name] (defaults to the class name) and a constructor that accepts serializable arguments. The default [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] implementation calls `cls(*args, **kwargs)`, so for simple dataclasses no override is needed:

```python {title="custom_spec_capability.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class RateLimit(AbstractCapability[Any]):
    """Limits requests per minute."""

    rpm: int = 60


# In YAML: `- RateLimit: {rpm: 30}`
# In Python:
agent = Agent.from_spec(
    AgentSpec(model='test', capabilities=[{'RateLimit': {'rpm': 30}}]),
    custom_capability_types=[RateLimit],
)
```

Override [`from_spec`][pydantic_ai.capabilities.AbstractCapability.from_spec] when the constructor takes types that can't be represented in YAML/JSON. The spec fields should mirror the dataclass fields, but with serializable types:

```python {title="from_spec_override_example.py" test="skip" lint="skip"}
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import ToolDefinition


@dataclass
class ConditionalTools(AbstractCapability[Any]):
    """Hides tools unless a condition is met."""

    condition: Callable[[RunContext[Any]], bool]  # not serializable
    hidden_tools: list[str] = field(default_factory=list)

    @classmethod
    def from_spec(cls, hidden_tools: list[str]) -> 'ConditionalTools[Any]':
        # In the spec, there's no condition callable — always hide
        return cls(condition=lambda ctx: True, hidden_tools=hidden_tools)

    async def prepare_tools(
        self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        if self.condition(ctx):
            return [td for td in tool_defs if td.name not in self.hidden_tools]
        return tool_defs
```

In YAML this would be `- ConditionalTools: {hidden_tools: [dangerous_tool]}`. In Python code, the full constructor is available: `ConditionalTools(condition=my_check, hidden_tools=['dangerous_tool'])`.

Pass custom capability types via the `custom_capability_types` parameter so the spec resolver can find them.

## `AgentSpec` reference

The [`AgentSpec`][pydantic_ai.agent.spec.AgentSpec] model represents the full spec structure:

| Field | Type | Description |
|---|---|---|
| `model` | `str` | Model name (required) |
| `name` | `str \| None` | Agent name |
| `description` | `str \| None` | Agent description (supports [templates](#template-strings)) |
| `instructions` | `str \| list[str] \| None` | System prompt instructions (supports [templates](#template-strings)) |
| `model_settings` | `dict \| None` | Model settings |
| `capabilities` | `list` | Capabilities (see [spec syntax](#capability-spec-syntax)) |
| `deps_schema` | `dict \| None` | JSON Schema for [template string](#template-strings) validation (see below) |
| `output_schema` | `dict \| None` | JSON Schema for [structured output](output.md) (see below) |
| `retries` | `int` | Default tool retries (default: `1`) |
| `output_retries` | `int \| None` | Output validation retries |
| `end_strategy` | `EndStrategy` | When to stop (`'early'` or `'exhaustive'`) |
| `tool_timeout` | `float \| None` | Default tool timeout in seconds |
| `instrument` | `bool \| None` | Enable [Logfire](logfire.md) instrumentation |
| `metadata` | `dict \| None` | Agent metadata |

### `deps_schema`

When loading a spec without a Python `deps_type`, `deps_schema` provides a JSON Schema that is used to validate [template string](#template-strings) variable names at construction time. It does **not** validate the actual deps object at runtime — it only ensures that template variables like `{{user_name}}` correspond to properties defined in the schema.

### `output_schema`

When provided (and no `output_type` keyword argument is passed to `from_spec`), `output_schema` defines the structure the model should produce as its final output. Under the hood, it creates a [`StructuredDict`][pydantic_ai.output.StructuredDict] output type: the JSON Schema is sent to the model API so the model knows what structure to produce, and the response is returned as a `dict[str, Any]`.

!!! note
    The model's response is not validated against the schema's `properties` or `required` fields — it is accepted as a plain dict. The schema serves as an instruction to the model, not a runtime validation constraint.

```yaml {title="agent_with_schema.yaml" test="skip"}
model: anthropic:claude-opus-4-6
deps_schema:
  type: object
  properties:
    user_name:
      type: string
  required: [user_name]
output_schema:
  type: object
  properties:
    answer:
      type: string
    confidence:
      type: number
  required: [answer, confidence]
instructions: "You are helping {{user_name}}. Always include a confidence score."
capabilities:
  - WebSearch
```

## Saving specs

[`AgentSpec.to_file`][pydantic_ai.agent.spec.AgentSpec.to_file] saves a spec to YAML or JSON and optionally generates a companion JSON Schema file for editor autocompletion:

```python {title="save_spec_example.py"}
from pydantic_ai.agent.spec import AgentSpec

spec = AgentSpec(
    model='anthropic:claude-opus-4-6',
    instructions='You are a helpful assistant.',
    capabilities=['WebSearch'],
)
spec.to_file('agent.yaml')
# Also generates ./agent_schema.json for editor autocompletion
```

The generated JSON Schema file enables autocompletion and validation in editors that support the [YAML Language Server](https://github.com/redhat-developer/yaml-language-server) protocol. Pass `schema_path=None` to skip schema generation.
