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
            dimensions=(1024, 1024),
        )
    ],
)

# Native preferred; custom callable as fallback
def my_generator(prompt: str) -> bytes: ...
ImageGeneration(local=my_generator)
```

For a direct image model, use `dimensions=(width, height)` when the exact pixel shape matters. Each provider adapter maps it
using the selected model's documented constraints or verified output table, and raises
[`UserError`][pydantic_ai.exceptions.UserError] when that exact shape is unavailable. Use `aspect_ratio` when a canonical
model-specific size is sufficient. These fields are mutually exclusive; `size` remains a provider-dependent compatibility
setting for OpenAI pixel strings and Google/xAI resolution tiers.

The portable ratios supported by each direct model family are:

| Ratio | Typical use | GPT Image 1.x | GPT Image 2 | Gemini 2.5 Flash, 3 Pro, 3.1 Flash Lite | Gemini 3.1 Flash | Grok Imagine |
| --- | --- | --- | --- | --- | --- | --- |
| `1:1` | Social media, thumbnails | ✅ | ✅ | ✅ | ✅ | ✅ |
| `16:9` / `9:16` | Widescreen, mobile, stories | ❌ | ✅ | ✅ | ✅ | ✅ |
| `4:3` / `3:4` | Presentations, portraits | ❌ | ✅ | ✅ | ✅ | ✅ |
| `3:2` / `2:3` | Photography | ✅ | ✅ | ✅ | ✅ | ✅ |
| `4:5` / `5:4` | Social posts, portraits | ❌ | ✅ | ✅ | ✅ | ❌ |
| `21:9` | Cinematic and ultrawide images | ❌ | ✅ | ✅ | ✅ | ❌ |
| `2:1` / `1:2` | Banners, headers | ❌ | ✅ | ❌ | ❌ | ✅ |
| `19.5:9` / `9:19.5` | Modern smartphone displays | ❌ | ✅ | ❌ | ❌ | ✅ |
| `20:9` / `9:20` | Ultra-wide displays | ❌ | ✅ | ❌ | ❌ | ✅ |
| `4:1` / `1:4` | Panoramas and vertical banners | ❌ | ❌ | ❌ | ✅ | ❌ |
| `8:1` / `1:8` | Extreme panoramic strips | ❌ | ❌ | ❌ | ✅ | ❌ |

This matrix follows the current [OpenAI](https://developers.openai.com/api/docs/guides/image-generation#customize-image-output),
[Gemini](https://ai.google.dev/gemini-api/docs/image-generation), and
[xAI](https://docs.x.ai/developers/model-capabilities/images/generation) model documentation. Exact xAI output dimensions
were additionally verified against both current Grok Imagine models because xAI does not publish the complete pixel table.

As an alternative, `fallback_model='…'` remains available for a conversational model that generates images through its own [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool]. This legacy path creates a subagent and retains the native tool's existing geometry surface. Direct-only values such as `dimensions`, arbitrary GPT Image 2 sizes, and the additional ratios in the table above are ignored with a warning. Use `native=False` with a direct model name such as `local='openai:gpt-image-1.5'`, or an explicit `ImageGenerator`, to use the new geometry API.
