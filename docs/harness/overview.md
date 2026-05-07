# Pydantic AI Harness

[**Pydantic AI Harness**](https://github.com/pydantic/pydantic-ai-harness) is the official [capability](../capabilities.md) library for Pydantic AI -- the batteries for your agent.

```bash
uv add pydantic-ai-harness
```

## What goes where?

Pydantic AI core ships the agent loop, model providers, the capabilities/hooks abstraction, and two kinds of capabilities:

- **Capabilities that require model or framework support** -- anything backed by provider built-in tools (like [image generation](../capabilities.md#provider-adaptive-tools)), provider-specific APIs (like compaction via the OpenAI or Anthropic APIs), or deep agent graph integration. These go hand-in-hand with model class code and need to ship together.
- **Capabilities that are fundamental to the agent experience** -- things nearly every agent benefits from, like [web search](../capabilities.md#provider-adaptive-tools), [tool search](../tools-advanced.md#tool-search), and [thinking](../capabilities.md#thinking). These feel like qualities of the agent itself, not accessories.

**Pydantic AI Harness** is where everything else lives: standalone capabilities that make specific categories of agents powerful, or that are still finding their final shape. Context management, memory, guardrails, file system access, code execution, multi-agent orchestration -- these are the building blocks you pick and choose based on what your agent needs to do.

The harness is also where new capabilities *start*. It ships as a separate package so capabilities can iterate faster without the strict backward-compatibility requirements of core. As a capability stabilizes and proves itself broadly essential, it can graduate into core -- [code mode](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/code_mode) is an early candidate.

Many capabilities benefit from a "fall up" pattern: they typically start as a local implementation that works with every model, then gain provider-native support that uses the provider's built-in API when available -- auto-switching between the two. This is how [web search](../capabilities.md#provider-adaptive-tools), [web fetch](../capabilities.md#provider-adaptive-tools), and [image generation](../capabilities.md#provider-adaptive-tools) already work in core, and the same approach is coming for skills, code mode, and context compaction.

## What's in it?

See the [capability matrix](https://github.com/pydantic/pydantic-ai-harness#capability-matrix) on the harness README for the full list of what's available and what's coming, along with community alternatives.

## Contributing capabilities

Capability contributions should go to the [harness repo](https://github.com/pydantic/pydantic-ai-harness), not to pydantic-ai. The capabilities abstraction gives contributions clear boundaries, which makes them easier to review. See [Contributing](https://github.com/pydantic/pydantic-ai-harness#contributing) for details.

## Build your own

You can also publish capabilities as standalone packages. See [Building custom capabilities](../capabilities.md#building-custom-capabilities) for the API and [Publishing capability packages](../extensibility.md#publishing-capability-packages) for packaging guidance.

## Links

- [GitHub](https://github.com/pydantic/pydantic-ai-harness)
- [PyPI](https://pypi.org/project/pydantic-ai-harness/)
- [Capability matrix](https://github.com/pydantic/pydantic-ai-harness#capability-matrix)
- [Version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy)
- [Contributing](https://github.com/pydantic/pydantic-ai-harness#contributing)
