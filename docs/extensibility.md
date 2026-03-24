
# Extensibility

Pydantic AI is designed to be extended. [Capabilities](capabilities.md) are the primary extension point — they bundle tools, lifecycle hooks, instructions, and model settings into reusable units that can be shared across agents, packaged as libraries, and loaded from [spec files](agent-spec.md).

Beyond capabilities, Pydantic AI provides several other extension mechanisms for specialized needs.

## Capabilities

Capabilities are the recommended way to extend Pydantic AI. They are useful for:

- **Teams** building reusable internal agent components (guardrails, audit logging, authentication)
- **Package authors** shipping extensions that work across models and agents
- **Community contributors** sharing solutions to common problems

See [Capabilities](capabilities.md) for using and building capabilities, and [Hooks](hooks.md) for the lightweight decorator-based approach.

## Publishing capability packages

To make a capability installable and usable in [agent specs](agent-spec.md):

1. **Implement [`get_serialization_name()`][pydantic_ai.capabilities.AbstractCapability.get_serialization_name]** — defaults to the class name. Return `None` to opt out of spec support.

2. **Implement [`from_spec()`][pydantic_ai.capabilities.AbstractCapability.from_spec]** — defaults to `cls(*args, **kwargs)`. Override when your constructor takes non-serializable types.

3. **Package naming** — use the `pydantic-ai-` prefix (e.g. `pydantic-ai-guardrails`) so users can find your package.

4. **Registration** — users pass custom capability types via `custom_capability_types` on [`Agent.from_spec`][pydantic_ai.Agent.from_spec] or [`Agent.from_file`][pydantic_ai.Agent.from_file].

```python {test="skip" lint="skip"}
from pydantic_ai import Agent

from my_package import MyCapability

agent = Agent.from_file('agent.yaml', custom_capability_types=[MyCapability])
```

See [Custom capabilities in specs](agent-spec.md#custom-capabilities-in-specs) for implementation details.

## Third-party ecosystem

### Capabilities

[Capabilities](capabilities.md) are the recommended extension mechanism for packages that need to bundle tools with hooks, instructions, or model settings. See [Third-party capabilities](capabilities.md#third-party-capabilities) for community packages.

### Toolsets

Many third-party extensions are available as [toolsets](toolsets.md), which can also be wrapped as [capabilities](capabilities.md) to take advantage of hooks, instructions, and model settings:

- [MCP servers](mcp/overview.md) — access thousands of tools via the Model Context Protocol
- [LangChain tools](third-party-tools.md) — use LangChain tools via [`tool_from_langchain`][pydantic_ai.ext.langchain.tool_from_langchain]
- [ACI.dev tools](third-party-tools.md) — use ACI tools via [`tool_from_aci`][pydantic_ai.ext.aci.tool_from_aci]

See [Third-party tools](third-party-tools.md) for the full list.

## Other extension points

### Custom toolsets

For specialized tool execution needs (custom transport, tool filtering, execution wrapping), implement [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] or subclass [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset]:

- [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] — full control over tool definitions and execution
- [`WrapperToolset`][pydantic_ai.toolsets.WrapperToolset] — delegates to a wrapped toolset, override specific methods

See [Building a Custom Toolset](toolsets.md#building-a-custom-toolset) for details.

!!! tip
    If your toolset also needs to provide instructions, model settings, or hooks, consider building a [custom capability](capabilities.md#building-custom-capabilities) instead.

### Custom models

For connecting to model providers not yet supported by Pydantic AI, implement [`Model`][pydantic_ai.models.Model]:

- [`Model`][pydantic_ai.models.Model] — the base interface for model implementations
- [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] — delegates to a wrapped model, useful for adding instrumentation or transformations

See [Custom Models](models/overview.md#custom-models) for details.

### Custom agents

For advanced multi-agent patterns or custom agent behavior, subclass [`AbstractAgent`][pydantic_ai.agent.AbstractAgent]:

- [`AbstractAgent`][pydantic_ai.agent.AbstractAgent] — the base interface for agent implementations, providing `run`, `run_sync`, and `run_stream`

See [Multi-Agent Patterns](multi-agent-applications.md) for details.
