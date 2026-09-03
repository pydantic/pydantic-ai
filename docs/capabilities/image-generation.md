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
Only the direct generator can apply `dimensions`, and only it can apply the aspect ratios the native tool does not
share, so pass `native=False` when you need either to be guaranteed: with the default `native=True` a model that
generates images natively takes the native path, which has no equivalent for them, and the request warns that the
settings went unapplied. Native-tool-only settings such as
`quality` and `output_format` do not apply to a direct fallback; configure their
provider-prefixed equivalents on the generator. `action='edit'` and `image_model` do not apply either: the direct
fallback raises [`UserError`][pydantic_ai.exceptions.UserError] for `action='edit'`, because the `generate_image` tool
receives no reference images, and ignores `image_model` with a warning, because `local` already names the image model.
`native=False` makes the direct generator the only path, so both of those land at construction — the dropped settings as
a warning and `action='edit'` as the error; with native enabled the native tool still carries them, and only a request
that routes to the direct generator raises. The direct `local=` generator must return exactly one generated
[`BinaryImage`][pydantic_ai.messages.BinaryImage]; use
[`ImageGenerator`][pydantic_ai.images.ImageGenerator] directly for multiple images or reference-image editing.

[`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] remains the native implementation (see
[Image Generation Tool](../native-tools.md#image-generation-tool) for provider support and configuration). Pass an
explicit instance through `native=ImageGenerationTool(...)` when you need its full provider-native configuration. Its
`aspect_ratio` reaches both fallbacks — the `fallback_model` subagent and the direct `local=` generator — while a
capability-level `aspect_ratio` takes precedence over it.

Instrumentation is per generator, not per agent: the agent-level
[`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability does not reach the direct `local=` generator,
so a run records no `image_generation` span unless the generator carries its own. Pass `instrument=` when you construct
it, or switch it on globally with
[`ImageGenerator.instrument_all()`][pydantic_ai.images.ImageGenerator.instrument_all]:

```python {title="instrumented_image_generation_capability.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.capabilities import ImageGeneration

ImageGeneration(native=False, local=ImageGenerator('openai:gpt-image-2', instrument=True))
```

See [Instrumentation](../image-generation.md#instrumentation) for what those spans carry.

!!! warning "Durable execution with Temporal"
    Generated images have to cross Temporal's activity boundary, where the payload size limit leaves roughly 1.5MB for raw image bytes. A larger image fails with a `UserError` — naming the tool when it came from a local generator (a direct `local=` image model or [`ImageGenerator`][pydantic_ai.images.ImageGenerator], the subagent fallback, or your own `local=` callable or toolset), or naming the model when the native tool put it on the response. See [Large Payloads](../durable_execution/temporal.md#large-payloads) for the options.

## Compatibility Fallback

`fallback_model='…'` remains available for applications that delegate to an image-capable conversational subagent. It
creates an additional agent run and uses that model's native `ImageGenerationTool`, so the direct fallback above is the
recommended option for new code.

The compatibility path preserves the native tool's existing geometry vocabulary. Direct-only values such as
`dimensions`, arbitrary GPT Image 2 sizes, and additional aspect ratios are ignored with a warning. Use `native=False`
with `local='provider:image-model'` to apply the [direct geometry settings](../image-generation.md#output-geometry).

A provider content block becomes a retry prompt on either local path, so the model that wrote the prompt gets to
rephrase it rather than failing the run. Other generation failures still differ between the two: the compatibility path
turns them into a retry prompt as well, while the direct path raises them. Using
[`ImageGenerator`][pydantic_ai.images.ImageGenerator] on its own is unaffected — it raises
[`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError]. See
[error handling](../image-generation.md#error-handling) for the direct API's exceptions.

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
