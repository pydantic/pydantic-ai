# Image Generation

The [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] [capability](overview.md) lets an agent decide when to
generate an image. It prefers the conversational model provider's native image-generation tool and can fall back to a
dedicated image model through the [direct image-generation API](../image-generation.md).

```python {title="image_generation_capability.py"}
from pydantic_ai import Agent, ImageGenerator
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.images.openai import OpenAIImageGenerationSettings

image_generator = ImageGenerator(
    'openai:gpt-image-2',
    settings=OpenAIImageGenerationSettings(
        openai_output_format='png',
        openai_quality='low',
    ),
)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        ImageGeneration(
            native=False,
            local=image_generator,
            dimensions=(1024, 1024),
        )
    ],
)
```

`ImageGeneration()` is native-only by default. Set `local` to a direct image model name,
[`ImageGenerator`][pydantic_ai.images.ImageGenerator], or
[`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel] to add a fallback without creating another agent:

```python {title="image_generation_routing.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.images.openai import OpenAIImageGenerationSettings

# Native preferred; use the direct model when native generation is unavailable
ImageGeneration(local='openai:gpt-image-1.5')

# Always use the direct image API
ImageGeneration(native=False, local='google:gemini-3.1-flash-lite-image')

# Supply reusable direct settings through an explicit generator
generator = ImageGenerator(
    'openai:gpt-image-2',
    settings=OpenAIImageGenerationSettings(
        openai_output_format='jpeg',
        openai_quality='low',
    ),
)
ImageGeneration(native=False, local=generator, dimensions=(1280, 720))


# A custom callable or Tool can still implement the local side
def my_generator(prompt: str) -> bytes: ...


ImageGeneration(local=my_generator)
```

The portable `dimensions` and `aspect_ratio` capability settings override defaults on an explicit generator.
Native-tool-only settings such as `quality` and `output_format` do not apply to a direct fallback; configure their
provider-prefixed equivalents on the generator. The local tool requires exactly one generated
[`BinaryImage`][pydantic_ai.messages.BinaryImage]; use
[`ImageGenerator`][pydantic_ai.images.ImageGenerator] directly for multiple images or reference-image editing.

[`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] remains the native implementation. Pass an
explicit instance through `native=ImageGenerationTool(...)` when you need its full provider-native configuration.

## Compatibility Fallback

`fallback_model='…'` remains available for applications that delegate to an image-capable conversational subagent. It
creates an additional agent run and uses that model's native `ImageGenerationTool`, so the direct fallback above is the
recommended option for new code.

The compatibility path preserves the native tool's existing geometry vocabulary. Direct-only values such as
`dimensions`, arbitrary GPT Image 2 sizes, and additional aspect ratios are ignored with a warning. Use `native=False`
with `local='provider:image-model'` to apply the [direct geometry settings](../image-generation.md#output-geometry).

Both local paths treat a generation failure the same way: a recoverable failure becomes a retry prompt for the model,
while a provider content block raises
[`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] and ends the run, so rephrasing stays the
application's decision. See [error handling](../image-generation.md#error-handling) for the direct API's exceptions.

## Agent Specs

Direct model names such as `local='openai:gpt-image-1.5'` can be represented in JSON or YAML agent specs. Runtime
objects accepted by the Python constructor — `ImageGenerator`, `ImageGenerationModel`, `Tool`, and callables — are not
serializable and must be configured in Python. [`from_spec()`][pydantic_ai.capabilities.ImageGeneration.from_spec]
keeps that serializable subset explicit while exposing the same setting names. Write `dimensions` as the two-item array
used by JSON and YAML; Pydantic AI converts it to the `(width, height)` tuple used by the Python API:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - ImageGeneration:
      native: false
      local: openai:gpt-image-2
      dimensions: [1280, 720]
```
