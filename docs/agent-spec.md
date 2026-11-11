---
description: "Define Pydantic AI agents declaratively in YAML or JSON (model, instructions, capabilities) and load them with one line, without agent construction code."
---

# Agent Specs

Agent specs let you define agents declaratively in YAML or JSON — [model](models/overview.md), [instructions](agent.md#instructions), [capabilities](capabilities/overview.md), and all. One line to load, no Python agent construction code required.

This is useful for:

- Separating agent configuration from application code
- Letting non-developers (prompt engineers, domain experts) configure agents
- Storing agent definitions alongside other config files
- Sharing agent configurations across teams or projects

## Defining a spec

A spec file defines the agent's configuration in YAML or JSON:

```yaml {title="agent.yaml" test="skip"}
model: {{ anthropic:large }}
instructions: You are a helpful research assistant.
model_settings:
  max_tokens: 8192
capabilities:
  - WebSearch:
      local: duckduckgo
  - Thinking:
      effort: high
```

### Models with a configured provider

The `model` field takes a [model name](models/overview.md) like the `model` argument to [`Agent`][pydantic_ai.Agent]. To configure the provider the model uses, such as an OpenAI-compatible service with its own base URL, `model` can instead be a [`ModelSpec`][pydantic_ai.agent.ModelSpec] with a `name` and a `provider`:

```yaml {title="gateway_agent.yaml" test="skip"}
model:
  name: openai-chat:qwen/qwen3-coder
  provider:
    OpenAICompatible:
      name: my-gateway
      base_url: https://gateway.example.com/v1
      api_key: {env: MY_GATEWAY_API_KEY}
instructions: You are a helpful coding assistant.
```

The prefix of `name` picks the model class, as it does for a model name: `openai-chat` builds an [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]. `provider` builds the provider the model uses, here an [`OpenAICompatibleProvider`][pydantic_ai.providers.openai_compatible.OpenAICompatibleProvider] as described in [Other endpoints](models/compatible-apis.md#other-endpoints). It takes the same forms as a [capability](#capability-spec-syntax), with the arguments given by name, and the provider can be named in two ways:

* The class-style name of a provider class, which is the class name without its `Provider` suffix, like `OpenAICompatible` or `Azure`. It builds that class with its own defaults.
* A provider name from a model name, like `azure`, `openai-responses`, or `gateway/openai`. It builds the provider that model name would use, with your arguments, so `gateway/openai` builds the [Pydantic AI Gateway](gateway.md) provider for OpenAI. `openai-compatible` also builds an `OpenAICompatibleProvider`.

The arguments are the provider's constructor arguments that YAML and JSON can hold, and are validated against their types: client objects such as `openai_client` and `http_client` can't be set. The [schema](#saving-specs) that `to_file` generates lists the arguments of each provider. The provider name only picks the provider, never the model class, and a model class that can't use the provider raises a [`UserError`][pydantic_ai.exceptions.UserError]: `name: anthropic:...` with an `OpenAICompatible` provider fails, for example.

Secret arguments, like `api_key`, can only reference an environment variable, as `{env: VAR_NAME}`, so a spec never contains a secret, and a value written in the spec raises a `UserError`. These are the arguments a provider types as [`SecretStr`][pydantic.SecretStr]. The environment variables are read when the model is built, and a missing one raises a `UserError`. `Agent.from_spec` and `Agent.from_file` build the model when the agent is constructed, even with `defer_model_check=True`. A spec passed to a run method or to [`override`][pydantic_ai.agent.Agent.override] builds it when the run starts or the override is entered. An explicit `model` argument replaces the spec's model without building it, and so does an active `override(model=...)` around a run that passes the spec.

A provider without a secret argument in the spec still uses its usual environment variable: an `OpenAI` provider with a `base_url` sends `OPENAI_API_KEY` to that URL. Like everything else in a spec, the provider is trusted configuration; see [Where specs come from](#where-specs-come-from).

## Loading specs

[`Agent.from_file`][pydantic_ai.agent.Agent.from_file] loads a spec from a YAML or JSON file and constructs an agent:

```python {title="from_file_example.py" test="skip"}
from pydantic_ai import Agent

agent = Agent.from_file('agent.yaml')
```

[`Agent.from_spec`][pydantic_ai.agent.Agent.from_spec] accepts a dict or [`AgentSpec`][pydantic_ai.agent.AgentSpec] instance and supports additional keyword arguments that supplement or override the spec:

```python {title="from_spec_example.py"}
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class UserContext:
    user_name: str


agent = Agent.from_spec(
    {
        'model': '{{ anthropic:large }}',
        'instructions': 'You are helping {{user_name}}.',
        'capabilities': [{'WebSearch': {'local': 'duckduckgo'}}],
    },
    deps_type=UserContext,
)
```

Keyword arguments interact with spec fields as follows:

* **Scalar fields** (`model`, `name`, `end_strategy`, etc.) — the keyword argument overrides the spec value when provided. For retry budgets, the `retries` keyword argument overrides the spec's `retries` value.
* **`instructions`** — merged: spec instructions come first, then keyword argument instructions.
* **`capabilities`** — merged: spec capabilities come first, then keyword argument capabilities.
* **`model_settings`** — merged additively: keyword argument settings override matching spec settings.
* **`output_type`** — takes precedence over `output_schema` from the spec.

The spec may omit `model`; it can instead be supplied to `Agent.from_spec` or when running the agent.

When `deps_type` is passed, [template strings](#template-strings) in the spec's `instructions`, `description`, and capability arguments are compiled and validated against the deps type at construction time.

For more control over spec loading, use [`AgentSpec.from_file`][pydantic_ai.agent.AgentSpec.from_file] to load the spec separately before passing it to `Agent.from_spec`.

### Where specs come from

A spec has the same authority as the code that builds the agent. It chooses the model and provider, the instructions, and every capability with its arguments, including the endpoints the agent connects to. For example, an [`MCP`](capabilities/mcp.md) capability connects to the server URL the spec names, `localhost` and private addresses included, with the network access of the machine the agent runs on.

Load specs only from sources you trust. If your application accepts specs from its users, load each one with [`AgentSpec.from_file`][pydantic_ai.agent.AgentSpec.from_file] or `AgentSpec.model_validate`, and check every entry in `spec.capabilities` (its `name`, and its `kwargs` such as an `MCP` `url`), and the `provider` of a [model spec](#models-with-a-configured-provider) (such as its `base_url`), against what that user may use before passing it to `Agent.from_spec`.

## Template strings

[`TemplateStr`][pydantic_ai.template.TemplateStr] provides Handlebars-style templates (`{{variable}}`) that are rendered against the agent's [dependencies](dependencies.md) at runtime. In spec files, strings containing `{{` are automatically converted to template strings:

```yaml {test="skip"}
instructions: "You are assisting {{name}}, who is a {{role}}."
```

Template variables are resolved from the fields of the `deps` object. When a `deps_type` (or [`deps_schema`](#deps_schema)) is provided, template variable names are validated at construction time.

In Python code, [`TemplateStr`][pydantic_ai.template.TemplateStr] can be used explicitly, but a callable with [`RunContext`][pydantic_ai.tools.RunContext] is generally preferred for IDE autocomplete and type checking:

```python {title="template_instructions.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, TemplateStr


@dataclass
class UserProfile:
    name: str
    role: str


agent = Agent(
    '{{ openai:medium }}',
    deps_type=UserProfile,
    instructions=TemplateStr('You are assisting {{name}}, who is a {{role}}.'),
)
result = agent.run_sync('hello', deps=UserProfile(name='Alice', role='engineer'))
print(result.output)
#> Hello! How can I help you today?
```

## Capability spec syntax

Capabilities in specs support three forms:

* `'MyCapability'` — no arguments, calls `MyCapability.from_spec()`
* `{'MyCapability': value}` — single positional argument, calls `MyCapability.from_spec(value)`
* `{'MyCapability': {key: value, ...}}` — keyword arguments, calls `MyCapability.from_spec(**kwargs)`

These built-in capabilities can be declared in specs: [`Thinking`][pydantic_ai.capabilities.Thinking], [`Caching`][pydantic_ai.capabilities.Caching], [`Instrumentation`][pydantic_ai.capabilities.Instrumentation], [`WebSearch`][pydantic_ai.capabilities.WebSearch], [`WebFetch`][pydantic_ai.capabilities.WebFetch], [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration], [`XSearch`][pydantic_ai.capabilities.XSearch], [`MCP`][pydantic_ai.capabilities.MCP], [`ToolSearch`][pydantic_ai.capabilities.ToolSearch], [`PrefixTools`][pydantic_ai.capabilities.PrefixTools], [`NativeTool`][pydantic_ai.capabilities.NativeTool], [`IncludeToolReturnSchemas`][pydantic_ai.capabilities.IncludeToolReturnSchemas], [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata], [`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError], [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt], [`LocalWorkspace`][pydantic_ai.capabilities.LocalWorkspace], [`ModelFallback`][pydantic_ai.capabilities.ModelFallback] (see [Model Fallback](capabilities/model-fallback.md#agent-specs)), and [`BatchModelRequests`][pydantic_ai.capabilities.BatchModelRequests] (see [Batch processing](batch.md#agent-specs)). The other [built-in capabilities](capabilities/overview.md#available-capabilities) take non-serializable arguments (callables, toolset objects) and can only be used in Python code.

## Custom capabilities in specs

See [Publishing capabilities](capabilities/custom.md#publishing-capabilities) for how to make custom capabilities work with agent specs.

## Custom providers in specs

A custom [provider](models/overview.md#models-and-providers) class can be used in a [model spec](#models-with-a-configured-provider) by passing it to `custom_provider_types` on [`Agent.from_spec`][pydantic_ai.agent.Agent.from_spec] or [`Agent.from_file`][pydantic_ai.agent.Agent.from_file], and to [`AgentSpec.to_file`][pydantic_ai.agent.AgentSpec.to_file] to include it in the schema. Its name in a spec comes from [`get_serialization_name`][pydantic_ai.providers.Provider.get_serialization_name], the class name without its `Provider` suffix by default. Its spec arguments are those of `__init__`, or of [`from_spec`][pydantic_ai.providers.Provider.from_spec] if you override it to choose what a spec can set:

```python {title="custom_provider_spec.py" test="skip"}
from typing import Self

from pydantic import SecretStr

from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider


class OpenAIProxyProvider(OpenAIProvider):
    """A provider for a company proxy to OpenAI, with an endpoint per region."""

    @classmethod
    def from_spec(cls, *, region: str, api_key: SecretStr) -> Self:
        return cls(base_url=f'https://openai-proxy.{region}.example.com/v1', api_key=api_key)


agent = Agent.from_spec(
    {
        'model': {
            'name': '{{ openai:medium }}',
            'provider': {'OpenAIProxy': {'region': 'eu', 'api_key': {'env': 'OPENAI_PROXY_API_KEY'}}},
        }
    },
    custom_provider_types=[OpenAIProxyProvider],
)
```

Custom providers need class-style names, so a spec's provider names always mean the same built-in providers.

## `AgentSpec` reference

The [`AgentSpec`][pydantic_ai.agent.AgentSpec] model represents the full spec structure:

| Field | Type | Description |
|---|---|---|
| `model` | `str \| ModelSpec \| None` | [Model](models/overview.md) name, or a [model with a configured provider](#models-with-a-configured-provider) |
| `name` | `str \| None` | Agent name |
| `description` | `str \| None` | Agent description (supports [templates](#template-strings)) |
| `instructions` | `str \| list[str] \| None` | [Instructions](agent.md#instructions) (supports [templates](#template-strings)) |
| `model_settings` | `dict \| None` | [Model settings](agent.md#model-run-settings) |
| `capabilities` | `list` | [Capabilities](capabilities/overview.md) (see [spec syntax](#capability-spec-syntax)) |
| `deps_schema` | `dict \| None` | JSON Schema for [template string](#template-strings) validation (see below) |
| `output_schema` | `dict \| None` | JSON Schema for [structured output](output.md) (see below) |
| `retries` | `int \| AgentRetries \| None` | Retry budgets for [tools](tools-advanced.md#tool-retries) and [output validation](output.md#output-validator-functions). Pass an integer to use the same budget for both, or [`AgentRetries`][pydantic_ai.agent.AgentRetries] to configure them separately. |
| `end_strategy` | `EndStrategy` | When to stop (`'early'`, `'graceful'`, or `'exhaustive'`) |
| `tool_timeout` | `float \| None` | Default [tool](tools.md) timeout in seconds |
| `instrument` | `bool \| None` | Enable [Logfire](logfire.md) instrumentation |
| `metadata` | `dict \| None` | Agent [metadata](agent.md#run-metadata) |

### `deps_schema`

When loading a spec file without a Python `deps_type`, `deps_schema` provides a JSON Schema that validates [template string](#template-strings) variable names at construction time. It does **not** validate the actual deps object at runtime — it only ensures that template variables like `{{user_name}}` correspond to properties defined in the schema.

### `output_schema`

When provided (and no `output_type` keyword argument is passed to `from_spec`), `output_schema` defines the structure the model should produce as its final output. Under the hood, it creates a [`StructuredDict`][pydantic_ai.output.StructuredDict] output type: the JSON Schema is sent to the model API so the model knows what structure to produce, and the response is returned as a `dict[str, Any]`.

!!! note
    The model's response is not validated against the schema's `properties` or `required` fields — it is accepted as a plain dict. The schema serves as an instruction to the model, not a runtime validation constraint.

```yaml {title="agent_with_schema.yaml" test="skip"}
model: {{ anthropic:large }}
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
  - WebSearch:
      local: duckduckgo
```

## Saving specs

[`AgentSpec.to_file`][pydantic_ai.agent.AgentSpec.to_file] saves a spec to YAML or JSON and optionally generates a companion JSON Schema file for editor autocompletion:

```python {title="save_spec_example.py"}
from pydantic_ai import AgentSpec

spec = AgentSpec.model_validate(
    {
        'model': '{{ anthropic:large }}',
        'instructions': 'You are a helpful assistant.',
        'capabilities': [{'WebSearch': {'local': 'duckduckgo'}}],
    }
)
spec.to_file('agent.yaml')
# Also generates ./agent_schema.json for editor autocompletion
```

The generated JSON Schema file enables autocompletion and validation in editors that support the [YAML Language Server](https://github.com/redhat-developer/yaml-language-server) protocol. Pass `schema_path=None` to skip schema generation.
