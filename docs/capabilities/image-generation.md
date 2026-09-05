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

`ImageGeneration()` is native-only by default. Add a fallback that generates through the image API — without creating
another agent — with one of two fields, and which one you use follows what you have: an
[`ImageGenerator`][pydantic_ai.images.ImageGenerator] carries settings of its own, so it goes on `local` beside the
other implementations you supply; a bare [`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel] or a
`'provider:model'` name goes on `fallback_image_model`. Either way the direct model generates through the image API,
where `fallback_model` runs a conversational model in a subagent, and it is reached when the agent's model has no
native image generation — or on every call, with `native=False`. `ImageGeneration` has no bundled local strategy, so
`local` otherwise takes a tool of your own.

Two of the fields name an image model and they are not interchangeable: `image_model` configures the provider's native
tool and is unprefixed (`image_model='gpt-image-2'`), while `fallback_image_model` selects the direct model and carries
its provider:

```python {title="image_generation_routing.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.capabilities import ImageGeneration
from pydantic_ai.images.openai import OpenAIImageGenerationSettings

# Native preferred; use the direct model when native generation is unavailable
ImageGeneration(fallback_image_model='openai:gpt-image-1.5')

# Always use the direct image API
ImageGeneration(native=False, fallback_image_model='google:gemini-3.1-flash-lite-image')

# Supply reusable direct settings through an explicit generator, which goes on `local`
generator = ImageGenerator(
    'openai:gpt-image-2',
    settings=OpenAIImageGenerationSettings(
        openai_output_format='jpeg',
        openai_quality='low',
    ),
)
ImageGeneration(native=False, local=generator, dimensions=(1280, 720))


# A custom callable or Tool of your own also goes on `local`
def my_image_tool(prompt: str) -> bytes: ...


ImageGeneration(local=my_image_tool)
```

The portable `dimensions` and `aspect_ratio` capability settings override defaults on an explicit generator.
Only the direct generator can apply `dimensions`, and only it can apply the aspect ratios the native tool does not
share, so pass `native=False` when you need either to be guaranteed: with the default `native=True` a model that
generates images natively takes the native path, which has no equivalent for them, and the request warns that the
settings went unapplied. Native-tool-only settings such as
`quality` and `output_format` do not apply to a direct fallback; configure their
provider-prefixed equivalents on the generator. `action='edit'` and `image_model` do not apply either: the direct
fallback raises [`UserError`][pydantic_ai.exceptions.UserError] for `action='edit'`, because the `generate_image` tool
receives no reference images, and ignores `image_model` with a warning, because the generator already names the image
model it generates with.
`native=False` makes the direct generator the only path, so both of those land at construction — the dropped settings
as a warning and `action='edit'` as the error; with native enabled the native tool still carries them, so a request
that routes to the direct generator instead is what warns that they went unapplied, and what raises for
`action='edit'`. The direct generator must return exactly one generated
[`BinaryImage`][pydantic_ai.messages.BinaryImage]; use
[`ImageGenerator`][pydantic_ai.images.ImageGenerator] directly for multiple images or reference-image editing.

[`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] is the native implementation (see
[Image Generation Tool](../native-tools.md#image-generation-tool) for provider support and configuration). Pass an
explicit instance through `native=ImageGenerationTool(...)` when you need its full provider-native configuration, or a
callable taking [`RunContext`][pydantic_ai.tools.RunContext] that returns an `ImageGenerationTool` or `None` for
[dynamic configuration](../native-tools.md#dynamic-configuration). A callable resolves on each model request and again
when the `fallback_model` subagent runs. Both resolutions receive the same `deps`, but the subagent has its own
`RunContext`; use `ctx.deps` for configuration that must match across both. Capability-level fields override the
factory result on the subagent.

A static native instance's `aspect_ratio` reaches whichever fallback you configured — the `fallback_model` subagent or
the direct generator — while a capability-level `aspect_ratio` takes precedence over it, as does a
capability-level `dimensions`, which is the same geometry spelled differently and cannot be combined with it. Only
inheritance yields that way: setting both `dimensions` and `aspect_ratio` on the capability itself raises
[`UserError`][pydantic_ai.exceptions.UserError] at construction, once a direct generator is configured to apply
them.

Instrumentation is per generator, not per agent: the agent-level
[`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability does not reach the direct generator,
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
    Generated images have to cross Temporal's activity boundary, where the payload size limit leaves roughly 1.5MB for raw image bytes. A larger image fails with a `UserError` — naming the tool when it came from a local implementation (a `local=` [`ImageGenerator`][pydantic_ai.images.ImageGenerator], a `fallback_image_model`, the subagent fallback, or your own `local=` callable or toolset), or naming the model when the native tool put it on the response. See [Large Payloads](../durable_execution/temporal.md#large-payloads) for the options.

## Fallback Options

Two built-in mechanisms cover a model that does not generate images natively:

- **Direct model**: `local=` with an [`ImageGenerator`][pydantic_ai.images.ImageGenerator], or `fallback_image_model=`
  with an image model name or [`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel]. The tool call is a
  single image API call, with no extra agent run, and it applies the portable `dimensions` and `aspect_ratio` settings.
  Reach for it when the geometry or the choice of image model is yours to make.
- **Subagent**: `fallback_model=` with a conversational model that generates images natively. The tool call runs an
  additional agent whose native [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] produces the
  image. Reach for it when you want that model's native tool semantics and the settings the native tool carries.

A `local=` callable, `Tool`, or toolset of your own replaces both with an implementation you write. The three fields are
alternatives: stating more than one raises [`UserError`][pydantic_ai.exceptions.UserError].

!!! note "A factory returning `None`"
    Without `fallback_model`, `None` omits the native tool for that request. With `fallback_model`, the subagent tool
    stays available, and calling it raises [`UserError`][pydantic_ai.exceptions.UserError] instead of silently using
    default image settings.

!!! note "Dynamic configuration under Temporal"
    The subagent tool call runs inside a Temporal activity, so its `native=` factory receives the limited
    [`TemporalRunContext`][pydantic_ai.durable_exec.temporal.TemporalRunContext]. `ctx.deps` crosses that boundary;
    fields such as `ctx.messages` do not. See
    [Agent Run Context and Dependencies](../durable_execution/temporal.md#agent-run-context-and-dependencies).

The subagent speaks the native tool's geometry vocabulary. Direct-only values such as `dimensions`, arbitrary GPT
Image 2 sizes, and additional aspect ratios are ignored with a warning. Use `native=False` with a direct generator —
`local=ImageGenerator(...)` or `fallback_image_model='provider:image-model'` — to apply the
[direct geometry settings](../image-generation.md#output-geometry).

A provider content block becomes a retry prompt on either local path, so the model that wrote the prompt gets to
rephrase it rather than failing the run. Other failures differ between the two: the subagent also turns an
[`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] from its run into a retry prompt, while the
direct model raises it. Using
[`ImageGenerator`][pydantic_ai.images.ImageGenerator] on its own is unaffected — it raises
[`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError]. See
[error handling](../image-generation.md#error-handling) for the direct API's exceptions.

## Agent Specs

Direct model names such as `fallback_image_model='openai:gpt-image-1.5'` can be represented in JSON or YAML agent
specs. Runtime objects accepted by the Python constructor — `ImageGenerationModel` on `fallback_image_model`, and the
`ImageGenerator`, `Tool`, toolsets and callables `local` takes — are not serializable and must be configured in Python.
[`from_spec()`][pydantic_ai.capabilities.ImageGeneration.from_spec] keeps that serializable subset explicit while
exposing the same setting names. Write `dimensions` as the two-item array used by JSON and YAML; Pydantic AI converts
it to the `(width, height)` tuple used by the Python API:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - ImageGeneration:
      native: false
      fallback_image_model: openai:gpt-image-2
      dimensions: [1280, 720]
```
