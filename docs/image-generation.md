# Image Generation

Pydantic AI provides a provider-agnostic API for generating and editing images with dedicated image models.
Use [`ImageGenerator`][pydantic_ai.images.ImageGenerator] when your application, rather than an agent, decides when to
create an image. When the agent should make that call, use the
[`ImageGeneration` capability](capabilities/image-generation.md) instead.

## Quick Start

Install the optional group for the provider you want to use, for example OpenAI:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

Set its API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

Then pass a provider-prefixed model name to [`ImageGenerator`][pydantic_ai.images.ImageGenerator], and call
[`generate()`][pydantic_ai.images.ImageGenerator.generate]:

```python {title="image_generation_quickstart.py"}
from pathlib import Path

from pydantic_ai import ImageGenerator

generator = ImageGenerator('openai:gpt-image-2')


async def main():
    result = await generator.generate('A watercolor map of a floating city.')
    image = result.image
    Path('floating-city.png').write_bytes(image.data)
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`.)_

[`generate_sync()`][pydantic_ai.images.ImageGenerator.generate_sync] provides the same interface for synchronous code.

See [Providers](#providers) for the install group and environment variable each provider uses.

## Choosing a model

Three model families generate images:

- OpenAI's GPT Image family through the
  [Images API](https://developers.openai.com/api/docs/guides/image-generation), under the `openai:` prefix. `dall-e-2`
  and `dall-e-3` are the exception: they are [rejected](#output-geometry) with
  [`UserError`][pydantic_ai.exceptions.UserError] as soon as the model is resolved.
- Google's [Gemini image models](https://ai.google.dev/gemini-api/docs/image-generation), under `google:` on the Gemini
  Developer API and under `google-cloud:` on Vertex AI, or `gateway/google:` through the
  [Pydantic AI Gateway](gateway.md).
- xAI's [Grok Imagine image models](https://docs.x.ai/developers/model-capabilities/images/generation), under the `xai:`
  prefix.

The provider validates the model name, so any current model in one of those families works, including one released after
the Pydantic AI version you are on. [`KnownImageGenerationModelName`][pydantic_ai.images.KnownImageGenerationModelName]
carries the names Pydantic AI recognizes for autocompletion; any other name is passed through unchanged. Check each
provider's documentation for what its models cost and do best.

The exact shapes each family can produce differ, and the portable geometry settings are mapped per model, so check
[Output Geometry](#output-geometry) and [Canonical Dimensions for `aspect_ratio`](#canonical-dimensions-for-aspect_ratio)
before committing to a model for a fixed layout.

## Editing Images

Pass reference images through `images` to edit or transform them. The input can contain
[`BinaryImage`][pydantic_ai.messages.BinaryImage], [`ImageUrl`][pydantic_ai.messages.ImageUrl], or
[`UploadedFile`][pydantic_ai.messages.UploadedFile] objects:

```python {title="image_edit.py"}
from pydantic_ai import BinaryImage, ImageGenerator

generator = ImageGenerator('google:gemini-3.1-flash-lite-image')


async def replace_subject(source: BinaryImage) -> BinaryImage:
    result = await generator.generate(
        'Replace the cat with a dog while preserving the composition.',
        images=[source],
    )
    return result.image
```

The order of multiple reference images is preserved. Provider-hosted files are supported by Google and xAI, and the
[`UploadedFile.provider_name`][pydantic_ai.messages.UploadedFile] must name the provider the file was uploaded to. On
xAI that is exactly the name of the provider you selected; Google additionally accepts `google-gla`, its pre-v2 name,
and reads whether the Gemini Files API is available off the client's transport rather than off the provider name, as
covered under [Google image generation](models/google.md#image-generation). OpenAI's image-edit endpoint requires file
content, so use `BinaryImage` or `ImageUrl` with OpenAI. Image URLs downloaded by Pydantic AI are limited to 50 MiB.

Editing applies to the whole image: masked editing, where a mask restricts the edit to a region, is not supported.
Of the providers below only OpenAI exposes that primitive, so there is nothing portable to map it onto yet.

| Provider | Generation | Reference editing | `UploadedFile` | Multiple outputs | Notes |
| --- | --- | --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ❌ | ✅ | Reference images must be PNG, JPEG, or WebP; any other media type raises [`UserError`][pydantic_ai.exceptions.UserError]. |
| Google Gemini API | ✅ | ✅ | ✅ | ❌ | [`UploadedFile.file_id`][pydantic_ai.messages.UploadedFile] must be the Files API URI (`file.uri`, which starts with `https://`), not the `files/...` resource name; any other value raises [`UserError`][pydantic_ai.exceptions.UserError]. A Files API URL passed as an `ImageUrl` instead needs an explicit `media_type`, since those URLs carry no file extension. See [Uploaded Files](input.md#uploaded-files). |
| Google Cloud (Vertex AI) | ✅ | ✅ | ❌ | ❌ | The Gemini Files API is not available on Vertex AI, and the adapter does not accept the `gs://` URIs Vertex uses instead, so pass reference images as `BinaryImage` or `ImageUrl`. Whether a client targets Vertex is read off the client, not the provider name. |
| xAI | ✅ | ✅ | ✅ | ✅ | xAI documents up to five reference images and enforces the limit itself: six references to `grok-imagine-image` come back as `INVALID_ARGUMENT` with `This model supports at most 5 input image(s), but 6 were provided.`, which surfaces as a 400 [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. Every `UploadedFile` must come before any `ImageUrl` or `BinaryImage`, because xAI sends file IDs ahead of URL and binary inputs; another order raises [`UserError`][pydantic_ai.exceptions.UserError] rather than silently resequencing them. `extra_headers` and `extra_body` are ignored with a warning: the transport is gRPC, which has no per-request header or body escape hatch. |

`google:` is the Gemini Developer API (Google AI Studio) and `google-cloud:` is Vertex AI, exactly as for
conversational models. `gateway/google:` routes Gemini through the [Pydantic AI Gateway](gateway.md), which serves it
over Vertex. `gateway/openai:` and `gateway/xai:` raise [`UserError`][pydantic_ai.exceptions.UserError]: the gateway
reports OpenAI's image endpoints as unsupported, and it has no xAI upstream. The Google adapter asks Gemini for
image-only output, matching the `ImageGenerator` result contract and avoiding unused text output.

## Providers

A `'provider:model-name'` string configures the provider from its usual environment variables.

### OpenAI

[`OpenAIImageGenerationModel`][pydantic_ai.images.openai.OpenAIImageGenerationModel] works with OpenAI's Images API and
the GPT Image model family.

#### Install

To use OpenAI image models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai`
optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

#### Configuration

Go to [platform.openai.com](https://platform.openai.com/) and generate an API key, then set it as an environment
variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

See the [OpenAI image-generation notes](models/openai.md#image-generation) for provider-specific behavior.

### Google

[`GoogleImageGenerationModel`][pydantic_ai.images.google.GoogleImageGenerationModel] works with the Gemini image models
through the Gemini API (Google AI Studio) or Google Cloud (formerly known as Vertex AI).

#### Install

To use Google image models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google`
optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```

#### Configuration

Go to [aistudio.google.com](https://aistudio.google.com/) and generate an API key, then set it as an environment
variable:

```bash
export GOOGLE_API_KEY='your-api-key'
```

The `google-cloud:` prefix uses Google Cloud instead, which authenticates with Application Default Credentials rather
than an API key. See the [Google image-generation notes](models/google.md#image-generation) for provider-specific
behavior and [Google Cloud configuration](models/google.md#google-cloud-enterprise) for the credential options.

### xAI

[`XaiImageGenerationModel`][pydantic_ai.images.xai.XaiImageGenerationModel] works with the Grok Imagine models through
the official xAI SDK, which connects over gRPC.

#### Install

To use xAI image models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `xai`
optional group:

```bash
pip/uv-add "pydantic-ai-slim[xai]"
```

#### Configuration

Go to [console.x.ai](https://console.x.ai/team/default/api-keys) and create an API key, then set it as an environment
variable:

```bash
export XAI_API_KEY='your-api-key'
```

See the [xAI image-generation notes](models/xai.md#image-generation) for provider-specific behavior.

### Customizing the provider

To customize authentication, the base URL, or the underlying SDK client, construct the provider's image model class
yourself and pass it to [`ImageGenerator`][pydantic_ai.images.ImageGenerator]. Each takes the
[`Provider`][pydantic_ai.providers.Provider] its SDK uses, so an OpenAI-compatible gateway or a pre-configured client
works the same way it does for conversational models:

```python {title="image_generation_provider.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images.openai import OpenAIImageGenerationModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIImageGenerationModel(
    'gpt-image-2',
    provider=OpenAIProvider(base_url='https://my-provider.com/v1', api_key='your-api-key'),
)
generator = ImageGenerator(model)
```

## Settings

[`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings] provides portable settings, while provider
settings classes add provider-prefixed controls.

Settings can be specified on the model, at the generator level (applied to all calls), or per call.
They are merged in that order: later settings override earlier values for the same key, while values set only in
earlier layers are preserved. The example below shows generator defaults extended for one call:

```python {title="image_generation_settings.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images import ImageGenerationSettings
from pydantic_ai.images.openai import OpenAIImageGenerationSettings

generator = ImageGenerator(
    'openai:gpt-image-2',
    settings=OpenAIImageGenerationSettings(openai_quality='low', openai_output_format='jpeg'),
)


async def main():
    result = await generator.generate(
        'A cinematic desert observatory at dusk.',
        settings=ImageGenerationSettings(dimensions=(1280, 720)),
    )
    assert result.image.media_type.startswith('image/')
```

Four settings can be dropped with a warning, because the selected request has no field for them: `openai_moderation`
on an edit and `openai_input_fidelity` on a generation, and `extra_headers` and `extra_body` on xAI, whose gRPC
transport has no per-request header or body escape hatch. Google drops `extra_body` with the same warning when it is
not a string-keyed mapping, since only a mapping can be merged into the JSON request body. Everything else is either
forwarded to the provider or, for geometry, rejected before the request — see [Output Geometry](#output-geometry).

OpenAI transparent backgrounds require `openai_output_format='png'` or `'webp'`, and model support varies.
Provider-specific settings are forwarded so the provider remains the authority on current model support; see the
[OpenAI image-generation notes](models/openai.md#image-generation).

### Output Geometry

Use one of these settings to control output geometry:

- `dimensions=(width, height)` requests an exact pixel shape. It raises
  [`UserError`][pydantic_ai.exceptions.UserError] when the selected model cannot produce that exact shape.
- `aspect_ratio='16:9'` requests a ratio and lets Pydantic AI select a canonical model-specific shape.

`dimensions` and `aspect_ratio` are mutually exclusive. Provider-specific geometry controls — `openai_size`,
`google_image_config.aspect_ratio`, `google_image_config.image_size`, `xai_aspect_ratio`, and `xai_resolution` — remain
prefixed because the providers use different concepts and value ranges. An explicit provider-specific geometry setting
takes precedence over the value a portable setting maps to, and warns only when the two disagree.

Gemini takes the aspect ratio as a native request field, so the ratio you ask for is sent as-is and Gemini decides
whether it can honor it; a rejection arrives as a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. OpenAI and
xAI cannot carry every ratio: OpenAI has no ratio field at all, so Pydantic AI maps the ratio to one of the model
family's enumerated sizes, and xAI takes an enumeration with no member for some portable values. Both raise
[`UserError`][pydantic_ai.exceptions.UserError] for a ratio they cannot express, rather than dropping it and billing you
for the model's default shape.

`dimensions` splits along the same wire shapes: OpenAI's `size` is a plain pixel string, so a shape for a model
Pydantic AI has no table for still travels for OpenAI to judge, while Google and xAI send a ratio plus a size tier, so a
shape outside the selected model's table has no wire representation at all and raises
[`UserError`][pydantic_ai.exceptions.UserError] before the request.

An OpenAI model Pydantic AI does not recognize — a GPT Image release newer than your Pydantic AI version — accepts any
structurally valid `dimensions`, which travel to OpenAI as the plain `size` string for it to validate. `aspect_ratio`
still raises [`UserError`][pydantic_ai.exceptions.UserError] for such a model, because Pydantic AI has no canonical
shapes to map the ratio onto; use `dimensions` or `openai_size` instead.

`dall-e-2` and `dall-e-3` are the exception to that fallthrough:
[`OpenAIImageGenerationModel`][pydantic_ai.images.openai.OpenAIImageGenerationModel] raises
[`UserError`][pydantic_ai.exceptions.UserError] on construction for both, because they diverge from the GPT Image
contract in response format, size set, image count, and quality vocabulary.

### Canonical Dimensions for `aspect_ratio`

When only `aspect_ratio` is provided, these are the canonical exact dimensions. Pydantic AI picks the shape for OpenAI,
which has no ratio field to carry one; Gemini and Grok Imagine take the ratio and a size tier as native request fields,
and the table records the shape they return for it. A dash means the model family names no canonical shape for that
ratio: OpenAI and Grok Imagine raise [`UserError`][pydantic_ai.exceptions.UserError], while Gemini still receives the
ratio and answers for itself. Grok Imagine's `21:9` dash is the transport rather than the model — the gRPC
`ImageAspectRatio` enum `xai-sdk` generates has no member for that ratio, so the request cannot carry it.

| Ratio | GPT Image 1.x | GPT Image 2 | Gemini 2.5 Flash | Gemini 3 Pro | Gemini 3.1 Flash / Flash Lite | Grok Imagine |
| --- | --- | --- | --- | --- | --- | --- |
| `1:1` | `1024×1024` | `1024×1024` | `1024×1024` | `1024×1024` | `1024×1024` | `1024×1024` |
| `1:2` | — | `704×1408` | — | — | — | `704×1408` |
| `1:4` | — | — | — | — | `512×2064` | — |
| `1:8` | — | — | — | — | `352×2928` | — |
| `2:1` | — | `1408×704` | — | — | — | `1408×704` |
| `2:3` | `1024×1536` | `832×1248` | `832×1248` | `848×1264` | `848×1264` | `832×1248` |
| `3:2` | `1536×1024` | `1248×832` | `1248×832` | `1264×848` | `1264×848` | `1248×832` |
| `3:4` | — | `864×1152` | `864×1184` | `896×1200` | `896×1200` | `864×1152` |
| `4:1` | — | — | — | — | `2064×512` | — |
| `4:3` | — | `1152×864` | `1184×864` | `1200×896` | `1200×896` | `1152×864` |
| `4:5` | — | `896×1120` | `896×1152` | `928×1152` | `928×1152` | — |
| `5:4` | — | `1120×896` | `1152×896` | `1152×928` | `1152×928` | — |
| `8:1` | — | — | — | — | `2928×352` | — |
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
| GPT Image 1.x (`gpt-image-1`, `gpt-image-1-mini`, `gpt-image-1.5`) | `1024×1024`, `1024×1536`, or `1536×1024`. |
| GPT Image 2 | Any positive dimensions where both sides are multiples of 16, the longest edge is at most 3840, the aspect ratio does not exceed 3:1, and the total area is between 655,360 and 8,294,400 pixels. |
| Any other OpenAI model except DALL·E | Any positive dimensions, forwarded as `size` for OpenAI to accept or reject. |
| Gemini 2.5 Flash Image | The ten dimensions shown in its canonical column above. This model has no separate resolution tier. |
| Gemini 3.1 Flash Lite Image | The fourteen `1K` dimensions shown in its column above. This model serves no other tier. |
| Gemini 3 Pro Image | The ten `1K` dimensions shown above, plus `2K` and `4K` variants obtained by multiplying both sides by 2 or 4. |
| Gemini 3.1 Flash Image | The ten standard `1K` dimensions shown above, their `2K` and `4K` variants obtained by multiplying both sides by 2 or 4, and their `512` variants obtained by halving both sides — plus the five rows in the table below, whose tiers do not scale uniformly. |
| Grok Imagine (`grok-imagine-image`, `grok-imagine-image-quality`) | The verified `1k` and `2k` dimensions in the table below. |

These Gemini 3.1 rows were verified against the live API, which returns shapes different from Google's published table
for the four extended ratios. Flash Lite serves only their `1K` column:

| Ratio | `512` | `1K` | `2K` | `4K` |
| --- | --- | --- | --- | --- |
| `1:4` | `256×1024` | `512×2064` | `1024×4128` | `2048×8256` |
| `1:8` | `176×1456` | `352×2928` | `704×5856` | `1408×11712` |
| `4:1` | `1024×256` | `2064×512` | `4128×1024` | `8256×2048` |
| `8:1` | `1456×176` | `2928×352` | `5856×704` | `11712×1408` |
| `21:9` | `784×336` | `1584×672` | `3168×1344` | `6336×2688` |

xAI documents the ratios and resolution tiers but not their complete exact pixel mapping. These dimensions were verified
against both `grok-imagine-image` and `grok-imagine-image-quality`. `grok-imagine-image-2.0` is a separate model that
nobody has probed, so `dimensions` raises [`UserError`][pydantic_ai.exceptions.UserError] there; use `aspect_ratio` or
the `xai_`-prefixed settings, which xAI validates itself:

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

Image count, output format, quality, background, moderation, input fidelity, compression, and provider resolution are
not portable settings, so OpenAI and xAI expose them as prefixed fields. Google is the exception: the Gemini request
carries all of its image options in one native object, so `GoogleImageGenerationSettings` adds only
`google_image_config`.

Asking for more than one image is the prefixed setting readers reach for most: `openai_n` on OpenAI and `xai_n` on
xAI. Each provider validates its own upper bound and reports an over-limit request as a
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]:

```python {title="image_generation_count.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images.openai import OpenAIImageGenerationSettings
from pydantic_ai.images.xai import XaiImageGenerationSettings

openai_generator = ImageGenerator('openai:gpt-image-2', settings=OpenAIImageGenerationSettings(openai_n=3))
xai_generator = ImageGenerator('xai:grok-imagine-image', settings=XaiImageGenerationSettings(xai_n=3))
```

Gemini returns one image per request, so there is no Google equivalent.

## Results and Usage

[`ImageGenerationResult`][pydantic_ai.images.ImageGenerationResult] contains normalized
[`GeneratedImage`][pydantic_ai.images.GeneratedImage] objects, request usage, model and provider identity, and any
provider-specific response details. Image bytes are always available as a
[`BinaryImage`][pydantic_ai.messages.BinaryImage] through `result.images[n].content`.

A result always holds at least one image, so [`result.image`][pydantic_ai.images.ImageGenerationResult.image] returns
the first one's [`BinaryImage`][pydantic_ai.messages.BinaryImage] directly. Use `result.images` when you asked for more
than one image or need per-image metadata such as `revised_prompt`.

```python {title="image_generation_result.py"}
from pydantic_ai import ImageGenerator

generator = ImageGenerator('openai:gpt-image-2')


async def main():
    result = await generator.generate('A watercolor map of a floating city.')

    print(result.image.media_type)
    #> image/png
    print(len(result.images))
    #> 1
    print(result.images[0].output_format)
    #> png
    print(result.usage.input_tokens)
    #> 8
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`.)_

OpenAI's `provider_details` carries the `size`, `quality`, and `background` values the API echoes back. Those are the
request parameters, not measurements of the returned bytes, so the only geometry-adjacent field on
[`GeneratedImage`][pydantic_ai.images.GeneratedImage] is `output_format`, which is derived from the bytes themselves.

xAI's `provider_details` can contain `cost_usd` reported by xAI. This is provider metadata, not a portable cost
calculation, and is kept separate from [`cost()`][pydantic_ai.images.ImageGenerationResult.cost].

!!! note "Image pricing"
    [`ImageGenerationResult.cost()`][pydantic_ai.images.ImageGenerationResult.cost] covers models priced per token,
    such as the GPT Image and Gemini image families. Models priced per generated image are not yet represented in
    [`genai-prices`](https://github.com/pydantic/genai-prices) and raise `LookupError`; usage details and
    provider-reported metadata are preserved on the result either way.

## Error Handling

Image generation raises the same exceptions as the rest of Pydantic AI:

- [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] when a provider blocks a request or its output for
  content moderation. OpenAI raises it for a `moderation_blocked` response, Google for a safety, recitation,
  prohibited-content, or [Model Armor](models/google.md#model-armor-google-cloud-only) block, and xAI when every image
  in a batch is flagged.
- [`UserError`][pydantic_ai.exceptions.UserError] when the request cannot be built: an empty prompt, a reference-image
  type the selected provider does not accept, or `dimensions` the selected model cannot produce exactly.
- [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] for other 4xx and 5xx provider responses, and
  [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] when the provider cannot be reached. xAI's gRPC status codes
  are mapped onto these same two exceptions.

Because a block is reported as an exception rather than an empty result, you can retry a rejected prompt explicitly:

```python {title="image_generation_content_filter.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.exceptions import ContentFilterError

generator = ImageGenerator('openai:gpt-image-2')


async def main():
    try:
        result = await generator.generate('A watercolor map of a floating city.')
    except ContentFilterError:
        result = await generator.generate('A watercolor map of a quiet harbor.')
    print(result.image.media_type)
    #> image/png
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`.)_

xAI is the exception to the all-or-nothing rule: it moderates silently, so a partially blocked batch returns the clean
images and reports the blocked positions instead of raising. See the
[xAI image-generation notes](models/xai.md#image-generation).

Image generation is slow: complex prompts can take minutes, and fronting proxies often cut connections at 60-180
seconds, so keep client and proxy timeouts above the worst case.

## Instrumentation

Enable OpenTelemetry instrumentation for one generator or for all generators:

```python {title="instrumented_image_generation.py"}
import logfire

from pydantic_ai import ImageGenerator

logfire.configure()

generator = ImageGenerator('openai:gpt-image-2', instrument=True)

# Or instrument all image generators globally
ImageGenerator.instrument_all()
```

Pydantic AI image-generation spans include model identity, usage, image count, and non-binary output metadata. They do
not include reference-image contents, generated bytes, URLs, or provider file IDs. Provider SDKs can emit their own
independent spans and must be configured separately. The `extra_headers` and `extra_body` request escape hatches are
also excluded, matching core model instrumentation.

Each call opens a span named `image_generation {model}` carrying `gen_ai.operation.name='image_generation'` and
`gen_ai.output.type='image'`. `image_generation` is a custom operation name: the OpenTelemetry GenAI conventions
enumerate no value for image generation, while `image` is one of their standard output types.

See the [Debugging and Monitoring guide](logfire.md) for more details on using Logfire with Pydantic AI.

## Testing

Use [`TestImageGenerationModel`][pydantic_ai.images.TestImageGenerationModel] for deterministic tests without API calls:

```python {title="test_image_generation.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images import ImageGenerationSettings, TestImageGenerationModel


async def test_image_workflow():
    generator = ImageGenerator('openai:gpt-image-2')
    test_model = TestImageGenerationModel()

    with generator.override(model=test_model):
        result = await generator.generate(
            'A test image',
            settings=ImageGenerationSettings(dimensions=(1024, 1024)),
        )

        # TestImageGenerationModel returns a single 1x1 PNG
        assert len(result.images) == 1

        # Check what settings were used
        assert test_model.last_settings == {'dimensions': (1024, 1024)}
```

Setting [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] to `False` also blocks image generation
requests, so a generator you forgot to override raises instead of quietly calling the provider.
[`TestImageGenerationModel`][pydantic_ai.images.TestImageGenerationModel] is unaffected, as it never reaches a provider.

## Building Custom Image Generation Models

To integrate an image provider Pydantic AI does not ship, subclass
[`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel]:

```python {title="custom_image_generation_model.py"}
from collections.abc import Sequence

from pydantic_ai import BinaryImage
from pydantic_ai.images import (
    GeneratedImage,
    ImageGenerationInput,
    ImageGenerationModel,
    ImageGenerationResult,
    ImageGenerationSettings,
)


class MyCustomImageGenerationModel(ImageGenerationModel):
    @property
    def model_name(self) -> str:
        return 'my-custom-model'

    @property
    def system(self) -> str:
        return 'my-provider'

    async def generate(
        self,
        prompt: str,
        *,
        images: Sequence[ImageGenerationInput] | None = None,
        settings: ImageGenerationSettings | None = None,
    ) -> ImageGenerationResult:
        prompt, images, settings = self.prepare_generate(prompt, images=images, settings=settings)

        # Call your image generation API here
        data = b'...'  # Placeholder

        return ImageGenerationResult(
            images=[GeneratedImage(content=BinaryImage(data=data, media_type='image/png'))],
            prompt=prompt,
            model_name=self.model_name,
            provider_name=self.system,
            settings=settings,
        )
```

`prepare_generate()` validates the prompt and reference inputs and merges the model's own default settings under the
ones passed in, so a subclass gets the same portable behavior as the built-in adapters. Return at least one image:
[`generate()`][pydantic_ai.images.ImageGenerationModel.generate] promises a non-empty result, and
[`result.image`][pydantic_ai.images.ImageGenerationResult.image] relies on it.

Use [`WrapperImageGenerationModel`][pydantic_ai.images.WrapperImageGenerationModel] if you want to wrap an existing
model to add custom behavior like caching or logging.

## Using Image Generation with an Agent

The direct API and agent image generation serve different use cases:

| API | Use it when |
| --- | --- |
| [`ImageGenerator`][pydantic_ai.images.ImageGenerator] | Your application explicitly generates or edits images, needs multiple outputs, or supplies reference images. |
| [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] | An agent should decide when to generate an image, with native execution when available and a direct image-model fallback otherwise. |
| [`ImageGenerationTool`][pydantic_ai.native_tools.ImageGenerationTool] | You need direct control over a conversational model provider's native image-generation tool. |

See the [`ImageGeneration` capability](capabilities/image-generation.md) for provider-adaptive agent usage.
