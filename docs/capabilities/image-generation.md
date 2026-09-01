# Image Generation

The [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] [capability](overview.md) lets your agent generate images. Like all [provider-adaptive tools](overview.md#provider-adaptive-tools), it uses the provider's native image generation when available, with an optional subagent fallback for other models.

[`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] defaults to native-only. Backed by [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] on the native side (see [Image Generation Tool](../native-tools.md#image-generation-tool) for provider support and configuration) — pass `native=ImageGenerationTool(...)` directly for full control, or a callable taking [`RunContext`][pydantic_ai.tools.RunContext] that returns an [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] or `None`. A callable resolves on each model request, and again with the same outer context when the `fallback_model` subagent runs — so keep it free of one-shot side effects. On the subagent, capability-level fields override the factory result. See [Dynamic Configuration](../native-tools.md#dynamic-configuration).

!!! note "A factory returning `None`"
    On the native path, `None` omits the tool for that request. If the request goes through `fallback_model` instead, the subagent has already been invoked and cannot omit anything, so it raises [`UserError`][pydantic_ai.exceptions.UserError] rather than generating with default settings.

For the local side, pass `fallback_model='…'` to delegate unsupported requests to a subagent running an image-generation-capable model (e.g. `openai-responses:gpt-5.4`), or `local=` with any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset] for a custom generator.

```python {title="image_generation.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import ImageGeneration

# Native-only — raises on models without native image generation
ImageGeneration()

# Native preferred; subagent fallback for unsupported models
ImageGeneration(fallback_model='openai-responses:gpt-5.4')

# Native preferred; custom callable as fallback
def my_generator(prompt: str) -> bytes: ...
ImageGeneration(local=my_generator)
```

!!! warning "Durable execution with Temporal"
    Generated images have to cross Temporal's activity boundary, where the payload size limit leaves roughly 1.5MB for raw image bytes. A larger image fails with a `UserError` — naming the tool when it came from a local generator (the subagent fallback or your own `local=` callable or toolset), or naming the model when the native tool put it on the response. See [Large Payloads](../durable_execution/temporal.md#large-payloads) for the options.
