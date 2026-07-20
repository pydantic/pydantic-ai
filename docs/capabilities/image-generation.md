# Image Generation

The [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] [capability](overview.md) lets your agent generate images. Like all [provider-adaptive tools](overview.md#provider-adaptive-tools), it uses the provider's native image generation when available, with an optional direct image-model fallback for other models.

[`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] defaults to native-only. Backed by [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] on the native side (see [Image Generation Tool](../native-tools.md#image-generation-tool) for provider support and configuration) — pass `native=ImageGenerationTool(...)` directly for full control.

For the local side, pass a direct image model name, `ImageGenerator`, or `ImageGenerationModel` to `local`. This calls the direct image generation API without creating a subagent. The capability forwards its normalized image settings to the selected provider adapter; unsupported settings produce a warning, while provider-prefixed settings configured on an explicit generator take precedence. The fallback tool always requests and returns one [`BinaryImage`][pydantic_ai.messages.BinaryImage]; use `ImageGenerator` directly when you need multiple images or reference-image editing.

```python {title="image_generation.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

# Native-only — raises on models without native image generation
ImageGeneration()

# Native preferred; direct image API fallback for unsupported models
ImageGeneration(local='openai:gpt-image-1.5')

# Always use the direct image API
image_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        ImageGeneration(
            native=False,
            local='openai:gpt-image-1.5',
            output_format='png',
            quality='low',
            size='1024x1024',
        )
    ],
)

# Native preferred; custom callable as fallback
def my_generator(prompt: str) -> bytes: ...
ImageGeneration(local=my_generator)
```

As an alternative, `fallback_model='…'` remains available for a conversational model that generates images through its own [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool]. This path creates a subagent; new image-only fallback integrations should use a direct model name such as `local='openai:gpt-image-1.5'`, or an explicit `ImageGenerator` when provider-specific defaults are needed.
