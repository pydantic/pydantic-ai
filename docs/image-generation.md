# Image Generation

Pydantic AI provides a provider-agnostic API for generating and editing images with dedicated image models.
Use [`ImageGenerator`][pydantic_ai.images.ImageGenerator] when your application, rather than an agent, decides when to
create an image.

## Quick Start

Pass a provider-prefixed model name to [`ImageGenerator`][pydantic_ai.images.ImageGenerator], then call
[`generate()`][pydantic_ai.images.ImageGenerator.generate]:

```python {title="image_generation_quickstart.py" test="skip"}
from pathlib import Path

from pydantic_ai import ImageGenerator

generator = ImageGenerator('openai:gpt-image-1.5')


async def main():
    result = await generator.generate('A watercolor map of a floating city.')
    image = result.images[0].content
    Path('floating-city.png').write_bytes(image.data)
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`.)_

[`generate_sync()`][pydantic_ai.images.ImageGenerator.generate_sync] provides the same interface for synchronous code.

## Editing Images

Pass reference images through `images` to edit or transform them. The input can contain
[`BinaryImage`][pydantic_ai.messages.BinaryImage], [`ImageUrl`][pydantic_ai.messages.ImageUrl], or
[`UploadedFile`][pydantic_ai.messages.UploadedFile] objects:

```python {title="image_edit.py" test="skip"}
from pydantic_ai import BinaryImage, ImageGenerator

generator = ImageGenerator('google:gemini-3.1-flash-lite-image')


async def replace_subject(source: BinaryImage) -> BinaryImage:
    result = await generator.generate(
        'Replace the cat with a dog while preserving the composition.',
        images=[source],
    )
    return result.images[0].content
```

The order of multiple reference images is preserved. Provider-hosted files are supported by Google and xAI, and the
[`UploadedFile.provider_name`][pydantic_ai.messages.UploadedFile] must match the selected provider. OpenAI's image-edit
endpoint requires file content, so use `BinaryImage` or `ImageUrl` with OpenAI.

| Provider | Generation | Reference editing | `UploadedFile` | Multiple outputs |
| --- | --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ❌ | ✅ |
| Google Gemini API | ✅ | ✅ | ✅ | ❌ |
| xAI | ✅ | ✅ | ✅ | ✅ |

The `google:` shorthand currently covers the Gemini Developer API. Google Cloud is not advertised for the direct API
until direct generation and reference editing have provider-specific integration recordings. The Google adapter asks
Gemini for image-only output, matching the `ImageGenerator` result contract and avoiding unused text output.

## Settings

[`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings] provides normalized settings. Defaults can be
set on the generator and overridden for one call:

```python {title="image_generation_settings.py" test="skip"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images import ImageGenerationSettings

generator = ImageGenerator(
    'openai:gpt-image-2',
    settings=ImageGenerationSettings(quality='low', output_format='jpeg'),
)


async def main():
    result = await generator.generate(
        'A cinematic desert observatory at dusk.',
        settings=ImageGenerationSettings(dimensions=(1280, 720)),
    )
    assert result.images[0].content.media_type == 'image/jpeg'
```

Settings are applied on a best-effort basis. A provider adapter warns when it cannot apply an explicit normalized
setting. Provider-specific settings take precedence over their normalized equivalent and also produce a warning when
the values conflict.

### Output Geometry

Use one of these settings to control output geometry:

- `dimensions=(width, height)` requests an exact pixel shape. It raises
  [`UserError`][pydantic_ai.exceptions.UserError] when the selected model cannot produce that exact shape.
- `aspect_ratio='16:9'` requests a ratio and lets Pydantic AI select a canonical model-specific shape.
- `size='1024x1024'` is a provider-dependent compatibility setting: OpenAI interprets pixel dimensions, while Google
  and xAI interpret resolution tiers such as `1K` and `2K`.

`dimensions` is mutually exclusive with both `aspect_ratio` and `size`. `aspect_ratio` and `size` can be combined when
they describe a geometry supported by the selected model.

### Canonical Dimensions for `aspect_ratio`

When only `aspect_ratio` is provided, Pydantic AI selects the following canonical exact dimensions. A dash means that
the model family cannot represent that ratio through the normalized setting.

| Ratio | GPT Image 1.x | GPT Image 2 | Gemini 2.5 Flash | Gemini 3 Pro / 3.1 Flash Lite | Gemini 3.1 Flash | Grok Imagine |
| --- | --- | --- | --- | --- | --- | --- |
| `1:1` | `1024×1024` | `1024×1024` | `1024×1024` | `1024×1024` | `1024×1024` | `1024×1024` |
| `1:2` | — | `704×1408` | — | — | — | `704×1408` |
| `1:4` | — | — | — | — | `512×2048` | — |
| `1:8` | — | — | — | — | `384×3072` | — |
| `2:1` | — | `1408×704` | — | — | — | `1408×704` |
| `2:3` | `1024×1536` | `832×1248` | `832×1248` | `848×1264` | `848×1264` | `832×1248` |
| `3:2` | `1536×1024` | `1248×832` | `1248×832` | `1264×848` | `1264×848` | `1248×832` |
| `3:4` | — | `864×1152` | `864×1184` | `896×1200` | `896×1200` | `864×1152` |
| `4:1` | — | — | — | — | `2048×512` | — |
| `4:3` | — | `1152×864` | `1184×864` | `1200×896` | `1200×896` | `1152×864` |
| `4:5` | — | `896×1120` | `896×1152` | `928×1152` | `928×1152` | — |
| `5:4` | — | `1120×896` | `1152×896` | `1152×928` | `1152×928` | — |
| `8:1` | — | — | — | — | `3072×384` | — |
| `9:16` | — | `720×1280` | `768×1344` | `768×1376` | `768×1376` | `720×1280` |
| `9:19.5` | — | `672×1456` | — | — | — | `576×1248` |
| `9:20` | — | `720×1600` | — | — | — | `576×1280` |
| `16:9` | — | `1280×720` | `1344×768` | `1376×768` | `1376×768` | `1280×720` |
| `19.5:9` | — | `1456×672` | — | — | — | `1248×576` |
| `20:9` | — | `1600×720` | — | — | — | `1280×576` |
| `21:9` | — | `1568×672` | `1536×672` | `1584×672` | `1584×672` | — |

### Supported Exact `dimensions`

`dimensions` also accepts non-canonical geometries when the selected model documents or has been verified to produce
them exactly:

| Model family | Exact dimensions accepted |
| --- | --- |
| GPT Image 1.x | `1024×1024`, `1024×1536`, or `1536×1024`. |
| GPT Image 2 | Any positive dimensions where both sides are multiples of 16, the longest edge is at most 3840, the aspect ratio does not exceed 3:1, and the total area is between 655,360 and 8,294,400 pixels. |
| Gemini 2.5 Flash Image | The ten dimensions shown in its canonical column above. This model has no separate resolution tier. |
| Gemini 3.1 Flash Lite Image | The ten `1K` dimensions shown in the Gemini 3 Pro / Flash Lite column above. |
| Gemini 3 Pro Image | The ten `1K` dimensions shown above, plus `2K` and `4K` variants obtained by multiplying both sides by 2 or 4. |
| Gemini 3.1 Flash Image | The fourteen `1K` dimensions shown above, their `2K` and `4K` variants obtained by multiplying both sides by 2 or 4, and their `512` variants obtained by halving both sides. The documented `21:9` row is the exception: `792×168`, `1584×672`, `3168×1344`, and `6336×2688`. |
| Grok Imagine | The verified `1k` and `2k` dimensions in the table below. |

xAI documents the ratios and resolution tiers but not their complete exact pixel mapping. These dimensions were verified
against both `grok-imagine-image` and `grok-imagine-image-quality`:

| Ratio | `1k` | `2k` |
| --- | --- | --- |
| `1:1` | `1024×1024` | `2048×2048` |
| `1:2` | `704×1408` | `1456×2912` |
| `2:1` | `1408×704` | `2912×1456` |
| `2:3` | `832×1248` | `1664×2496` |
| `3:2` | `1248×832` | `2496×1664` |
| `3:4` | `864×1152` | `1776×2368` |
| `4:3` | `1152×864` | `2368×1776` |
| `9:16` | `720×1280` | `1584×2816` |
| `16:9` | `1280×720` | `2816×1584` |
| `9:19.5` | `576×1248` | `1344×2912` |
| `19.5:9` | `1248×576` | `2912×1344` |
| `9:20` | `576×1280` | `1440×3200` |
| `20:9` | `1280×576` | `3200×1440` |

See the current [OpenAI](https://developers.openai.com/api/docs/guides/image-generation#customize-image-output),
[Gemini](https://ai.google.dev/gemini-api/docs/image-generation), and
[xAI](https://docs.x.ai/developers/model-capabilities/images/generation) documentation for provider limits and newly
released models.

### Provider-Specific Settings

Use the provider settings types when you need an option that is not portable:

- [`OpenAIImageGenerationSettings`][pydantic_ai.images.openai.OpenAIImageGenerationSettings]
- [`GoogleImageGenerationSettings`][pydantic_ai.images.google.GoogleImageGenerationSettings]
- [`XaiImageGenerationSettings`][pydantic_ai.images.xai.XaiImageGenerationSettings]

These types extend `ImageGenerationSettings`. Their provider-prefixed fields use public types from the corresponding
provider SDK where those types are available. See the [OpenAI](models/openai.md#image-generation),
[Google](models/google.md#image-generation), and [xAI](models/xai.md#image-generation) pages for provider-specific setup
and limitations.

## Results and Usage

[`ImageGenerationResult`][pydantic_ai.images.ImageGenerationResult] contains normalized
[`GeneratedImage`][pydantic_ai.images.GeneratedImage] objects, request usage, model and provider identity, and any
provider-specific response details. Image bytes are always available as a
[`BinaryImage`][pydantic_ai.messages.BinaryImage] through `result.images[n].content`.

xAI's `provider_details` can contain `cost_usd` reported by xAI. This is provider metadata, not a portable cost
calculation, and is kept separate from [`cost()`][pydantic_ai.images.ImageGenerationResult.cost].

!!! note "Image pricing"
    [`ImageGenerationResult.cost()`][pydantic_ai.images.ImageGenerationResult.cost] currently raises `LookupError`.
    Image-token and per-image pricing need to be represented correctly in
    [`genai-prices`](https://github.com/pydantic/genai-prices) before Pydantic AI can calculate a portable cost. Usage
    details and provider-reported metadata are still preserved on the result.

## Instrumentation

Enable OpenTelemetry instrumentation for one generator or for all generators:

```python {title="instrumented_image_generation.py" test="skip"}
import logfire

from pydantic_ai import ImageGenerator

logfire.configure()

generator = ImageGenerator('openai:gpt-image-1.5', instrument=True)

# Or instrument all image generators globally
ImageGenerator.instrument_all()
```

Pydantic AI image-generation spans include model identity, usage, image count, and non-binary output metadata. They do
not include reference-image contents, generated bytes, URLs, or provider file IDs. Provider SDKs can emit their own
independent spans and must be configured separately.

## Testing

Use [`TestImageGenerationModel`][pydantic_ai.images.TestImageGenerationModel] for deterministic tests without API calls:

```python {title="test_image_generation.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images import TestImageGenerationModel


async def test_image_workflow():
    test_model = TestImageGenerationModel()
    generator = ImageGenerator(test_model)

    result = await generator.generate('A test image', settings={'n': 2})

    assert len(result.images) == 2
    assert test_model.last_settings == {'n': 2}
```

## Using Image Generation with an Agent

The direct API and agent image generation serve different use cases:

| API | Use it when |
| --- | --- |
| [`ImageGenerator`][pydantic_ai.images.ImageGenerator] | Your application explicitly generates or edits images, needs multiple outputs, or supplies reference images. |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | An agent should decide when to generate an image, with native execution when available and a direct image-model fallback otherwise. |
| [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] | You need direct control over a conversational model provider's native image-generation tool. |

See the [`ImageGeneration` capability](capabilities/image-generation.md) for provider-adaptive agent usage.
