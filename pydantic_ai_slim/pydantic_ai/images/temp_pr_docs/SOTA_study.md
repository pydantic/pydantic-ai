# SOTA Study: Direct Image Generation APIs

Date: 2026-05-09

This study surveys the current state of direct image generation APIs that are relevant to a
Pydantic AI image-generation primitive. It focuses on request/response shape, metadata, pricing
signals, and observability implications rather than visual preference alone.

This is supporting research for the draft, not a reviewer contract, implementation plan, or provider
roadmap. A provider appearing here does not imply that Pydantic AI should support it, or that the
first public API should cover every capability exposed by that provider.

The main design question for Pydantic AI is not "which provider looks best?", but what normalized
contract can preserve enough output data, usage, cost, provider details, and safety metadata without
overfitting to one provider or prematurely committing to editing/video/audio abstractions.

## Executive Summary

### Current SOTA Signals

These are the provider/model families that matter most for API design:

| Provider | SOTA / relevant models | Why it matters |
| --- | --- | --- |
| OpenAI | `gpt-image-2`, `gpt-image-1.5`, `gpt-image-1-mini`, `gpt-image-1` | Direct image generation/editing API, base64 output for GPT image models, token-based multimodal usage, strong fit for first implementation. |
| Google | `gemini-3-pro-image-preview`, `gemini-2.5-flash-image` | Gemini-native image models use the existing Gemini content API, return mixed modal parts, and expose token-equivalent image pricing. We intentionally exclude Imagen because it is being phased out for our purposes. |
| xAI | `grok-imagine-image`, `grok-imagine-image-quality` | OpenAI-compatible images endpoint, URL by default, base64 optional, provider-reported cost in usage. Useful second adapter candidate. |
| Black Forest Labs | FLUX.2 family, FLUX.1 Kontext | High-end generation/editing but async-by-default with polling and signed result URLs. Forces us to think about job lifecycle vs simple request/response. |
| Ideogram | Ideogram 3.0 | Strong text/layout/image design use cases, multipart input, ephemeral URLs, per-output-image pricing, rich safety/seed metadata. |
| Stability AI | Stable Image Core/Ultra, SD 3.5 | Multipart API, can return raw image or JSON base64, credit/per-image pricing, safety finish reasons. |
| Amazon Bedrock | Nova Canvas, Titan Image Generator | Enterprise provider path, native `InvokeModel`, base64 images in JSON, no OpenAI-style API, provider-specific pricing tiers. |
| Adobe Firefly | Firefly Services | Async job API, storage upload IDs, result URLs, seed/content metadata, credit/subscription pricing. Less likely first adapter, useful design pressure. |
| Recraft | Recraft V4/V4 Pro, vector models | Production design/vector API, URL outputs, flat API-unit pricing, important if we ever support vector outputs. |

Visual example links and galleries:

- OpenAI GPT Image 2 model page and playground entry: https://developers.openai.com/api/docs/models/gpt-image-2
- Google Gemini image generation documentation: https://ai.google.dev/gemini-api/docs/image-generation
- xAI Grok Imagine docs/examples: https://docs.x.ai/developers/model-capabilities/images/generation
- BFL FLUX.2 text-to-image examples: https://docs.bfl.ai/flux_2/flux2_text_to_image
- BFL FLUX.2 image editing examples: https://docs.bfl.ai/flux_2/flux2_image_editing
- Ideogram 3.0 API docs: https://developer.ideogram.ai/api-reference/api-reference/generate-v3
- Adobe Firefly style reference examples: https://developer.adobe.com/firefly-services/docs/firefly-api/guides/concepts/style-image-reference/
- Adobe Firefly seed examples: https://developer.adobe.com/firefly-services/docs/firefly-api/guides/concepts/seeds/
- Runway Gen-4 Image UI docs/examples: https://help.runwayml.com/hc/en-us/articles/37053594806419-Creating-with-Gen-4-Image

We should link to examples rather than embed provider images in the repo.

## Design Implications For Pydantic AI

### Normalize Generated Output To `BinaryImage`

The safest user-facing default is to normalize generated image outputs to `BinaryImage`:

- OpenAI GPT image models already return base64 image data.
- xAI can return base64 via `response_format='b64_json'`.
- Google Gemini image models return image bytes as inline data parts.
- Stability can return JSON base64 or raw image bytes.
- Providers that return temporary URLs, such as BFL, Ideogram, Adobe, Luma, and xAI default mode, can be downloaded by the adapter before returning.

This avoids leaking temporary URL expiry semantics into the public API.

This recommendation is about generated outputs. Input images for editing, reference images, and
image-to-image flows should be designed separately when those features are added. The first OpenAI
generation PR should accept only a prompt plus settings.

### Keep Input Settings Small

The common input settings should stay narrow and include only concepts that mean roughly the same
thing across providers:

- `n`
- `output_format`
- `extra_headers`
- `extra_body`

The first draft should not expose common `size` or `aspect_ratio` settings. Providers use different
layout concepts: OpenAI direct Images uses pixel sizes and model-specific constraints, Gemini-native
models use aspect ratios plus image-size tiers, and xAI uses aspect ratio plus resolution. A common
field would either mean different things per provider or require Pydantic AI to own a cross-provider
mapping policy. Provider-prefixed settings such as `openai_size` are safer until maintainers decide
whether a higher-level layout abstraction is worth adding.

Provider-specific controls such as OpenAI `quality`, `background`, `moderation`, and
`output_compression` should start as provider-prefixed settings unless multiple providers converge
on the same semantics.

### Preserve Provider Metadata

A normalized `GeneratedImage` should still preserve:

- `size` / `width` / `height`
- `quality`
- `output_format` / `mime_type`
- `background`
- `seed`
- `revised_prompt` / `enhanced_prompt`
- `safety` / moderation result
- `provider_image_id`
- provider-specific metadata where useful and safe

These fields are relevant for repeatability, pricing, debugging, dashboards, and future provider-specific features.

### Usage And Cost Need More Than Tokens

`RequestUsage` is still useful, especially for OpenAI and Gemini image models with token-equivalent pricing. It is not sufficient for every provider:

- OpenAI GPT Image pricing is token-based by text/image input and image output token classes.
- Google Gemini image models use token-equivalent image pricing.
- xAI exposes provider-reported `usage.cost_in_usd_ticks`.
- BFL exposes credits, megapixels, and per-image/megapixel pricing.
- Ideogram/Recraft use per-output-image/API-unit pricing.
- Adobe/Firefly uses generative credits/subscription accounting.

Therefore `ImageGenerationResult` should preserve both:

- `usage: RequestUsage`
- structured image/pricing metadata in result/images/provider details

`cost()` can follow embeddings where `genai-prices` has enough information. It should degrade naturally when prices are unknown.

### Logfire Should Never Record Image Bytes By Default

Instrumentation should record prompt/settings/usage/cost/metadata, but not generated image bytes/base64. Even when `include_content=True`, binary output should be excluded unless a future design explicitly opts into `include_binary_content`.

## Provider Notes

## OpenAI

### State Of The Art

OpenAI describes `gpt-image-2` as its state-of-the-art image generation model. The GPT Image family
supports text and image inputs, image output, generation, and edits through the Images API. The GPT
Image family is the best first implementation target because the direct API is synchronous enough
for a minimal primitive and exposes detailed usage metadata. `gpt-image-2` may require organization
verification, so the draft should not depend on it for the basic demo path.

Sources:

- Model page: https://developers.openai.com/api/docs/models/gpt-image-2
- Images API reference: https://developers.openai.com/api/reference/resources/images
- Pricing: https://openai.com/api/pricing/

### Endpoint Shape

Primary endpoints:

- `POST /v1/images/generations`
- `POST /v1/images/edits`

SDK shape:

```python {test="skip" lint="skip"}
response = await client.images.generate(
    model='gpt-image-1.5',
    prompt='...',
    size='1024x1024',
    quality='low',
    output_format='png',
)
```

For the first Pydantic AI implementation:

- use generations only;
- consume the base64 image payload returned by GPT Image models;
- do not use the older DALL-E `response_format` parameter;
- do not include edits or streaming yet.

### Inputs

Generation inputs:

- `prompt`
- `model`
- `n`
- `size`
- `quality`
- `background`
- `output_format`
- `output_compression`
- `moderation`
- `user`

Editing adds uploaded image inputs, optional masks, and streaming/partial image options.

### Outputs

GPT image models return base64 image data. The image API also exposes normalized response-level fields:

- `background`
- `output_format`
- `quality`
- `size`
- `usage`

The API reference documents `b64_json` as base64 image data and usage with input/output tokens and input token details.

### Metadata

Important fields:

- response `created`
- response `usage`
- response `background`, `output_format`, `quality`, `size`
- per-image `b64_json`
- per-image `revised_prompt` where applicable

### Cost

OpenAI GPT Image 2 pricing is token based:

- image input tokens
- cached image input tokens
- image output tokens
- text input tokens
- cached text input tokens

This maps well to `RequestUsage` plus `details`, e.g.:

- `input_text_tokens`
- `input_image_tokens`
- `output_image_tokens`
- cached variants if present/extractable

OpenAI is therefore the strongest candidate for a first `cost()` implementation once `genai-prices` supports the image API flavor.

### Pydantic AI Takeaways

- First implementation should target OpenAI.
- Output can be `BinaryImage` directly.
- `RequestUsage` is enough for the token part of pricing.
- Preserve `size`, `quality`, `background`, and `output_format` on `GeneratedImage` or result.

## Google: Gemini-Native Image Models

### State Of The Art

For this study we focus only on Gemini-native image models, such as `gemini-3-pro-image-preview`
and `gemini-2.5-flash-image`. Imagen/Vertex image generation is intentionally out of scope for
this design pass because it is a separate API shape and is not the direction we want to optimize for.

Sources:

- Gemini image generation docs: https://ai.google.dev/gemini-api/docs/image-generation
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing

### Endpoint Shape

Gemini-native image generation uses `generateContent` through the Gemini API. Requests can contain
text parts and image parts, and responses can contain mixed text/image parts. For an image-generation
primitive, the adapter should request image output where the SDK/API supports response modality
configuration, then normalize only image parts into the public result.

### Inputs

Inputs are Gemini content parts and generation config:

- prompt text
- optional input images as inline data or file/URI parts
- response modality configuration where available
- output image settings where supported by the SDK/API
- safety settings
- model-specific config such as resolution/quality where supported

### Outputs

Gemini-native image models return generated image parts as inline data mixed with possible text
parts. A direct image primitive should:

- collect inline image parts;
- convert each image part to `BinaryImage`;
- preserve MIME type;
- preserve any text output, finish reason, and safety metadata in provider details rather than
  exposing text as the primary normalized output;
- fail clearly if the provider returns no generated image.

### Metadata

Important fields:

- `usageMetadata`
- `modelVersion` / response model details where available
- `responseId`
- prompt feedback / finish reasons / safety ratings
- SynthID/watermark behavior

### Cost

Gemini image models have token-equivalent image pricing. The Gemini pricing page currently documents:

- `gemini-3-pro-image-preview`: image input token-equivalent pricing and image output token-equivalent pricing; 1K/2K and 4K outputs have different token-equivalent costs.
- `gemini-2.5-flash-image`: image generation priced per image with token-equivalent explanation.

### Pydantic AI Takeaways

- Gemini-native output must filter image parts and decide what to do if text-only is returned.
- Pricing requires both token usage and image count/resolution metadata.
- Provider details should preserve any text parts that came back alongside images without making
  mixed text output part of the normalized image API.

## xAI

### State Of The Art

xAI's Grok Imagine API supports image generation, image editing, multi-image editing, and image understanding. The current docs note `grok-imagine-image-pro` deprecation in favor of `grok-imagine-image-quality` for new image generation requests.

Sources:

- Image generation guide: https://docs.x.ai/developers/model-capabilities/images/generation
- Image overview: https://docs.x.ai/developers/model-capabilities/images
- Model/pricing page: https://docs.x.ai/developers/models/grok-imagine-image
- Cost tracking: https://docs.x.ai/developers/cost-tracking
- REST images reference: https://docs.x.ai/developers/rest-api-reference/inference/images

### Endpoint Shape

OpenAI-compatible endpoints:

- `POST https://api.x.ai/v1/images/generations`
- `POST https://api.x.ai/v1/images/edits`

The OpenAI SDK can be used with:

```python {test="skip" lint="skip"}
client = OpenAI(base_url='https://api.x.ai/v1', api_key='...')
response = client.images.generate(
    model='grok-imagine-image',
    prompt='...',
)
```

### Inputs

Generation inputs:

- `model`
- `prompt`
- `n`
- `aspect_ratio`
- `resolution` (`1k`, `2k`)
- `response_format`

Editing inputs:

- one source image as public URL or base64 data URI;
- multi-image editing with up to three images according to the overview examples;
- prompt and aspect ratio.

### Outputs

Default output is a temporary URL. Base64 can be requested with:

```python {test="skip" lint="skip"}
response_format='b64_json'
```

The xAI SDK exposes extra metadata such as moderation status and actual model used.

### Metadata

Important fields:

- image URL or base64
- moderation status
- actual model
- usage cost ticks
- aspect ratio
- resolution

### Cost

xAI is the clearest example that pricing metadata should not be token-only:

- image generation is flat per generated image;
- image input has a separate per-image charge;
- responses include `usage.cost_in_usd_ticks`;
- docs show `200000000` ticks as `$0.02`.

### Pydantic AI Takeaways

- Good second adapter candidate because it is OpenAI-compatible.
- Need provider-specific settings for `aspect_ratio` and `resolution`.
- If provider-reported cost exists, preserve it in `provider_details` and possibly a normalized `provider_reported_cost`.
- Normalize temporary URLs to `BinaryImage`.

## Black Forest Labs: FLUX

### State Of The Art

BFL's FLUX.2 is the current recommended family for text-to-image and editing. FLUX.1 Kontext remains relevant for generation plus context-aware editing, but docs recommend FLUX.2 for new projects.

Sources:

- Pricing: https://docs.bfl.ai/quick_start/pricing
- Image generation overview: https://docs.bfl.ai/quick_start/generating_images
- FLUX.2 text-to-image: https://docs.bfl.ai/flux_2/flux2_text_to_image
- FLUX.2 image editing: https://docs.bfl.ai/flux_2/flux2_image_editing
- Get result API: https://docs.bfl.ai/api-reference/utility/get-result

### Endpoint Shape

BFL is async by design:

1. Submit a generation request.
2. Receive `id` and `polling_url`.
3. Poll until `Ready`, `Error`, or `Failed`.
4. Download the result from `result.sample`, a signed URL.

Example endpoints:

- `/v1/flux-2-pro`
- `/v1/flux-2-flex`
- `/v1/flux-2-max`
- `/v1/flux-pro-1.1`
- `/v1/flux-kontext-pro`
- `/v1/get_result`

### Inputs

Common inputs:

- `prompt`
- `width`
- `height`
- `seed`
- `prompt_upsampling`
- `safety_tolerance`
- `output_format`
- webhook URL/secret

Editing adds:

- input image(s), often base64
- width/height override
- multi-reference support for FLUX.2

### Outputs

Initial response:

- `id`
- `polling_url`
- sometimes `cost`
- sometimes `input_mp` and `output_mp`

Final result:

- status
- progress
- result object
- `result.sample` signed URL
- details/preview

Signed URLs are short-lived.

### Metadata

Important fields:

- task id
- polling URL
- status/progress
- signed output URL
- cost credits
- input/output megapixels
- seed
- safety/moderation status

### Cost

BFL pricing is credit-based:

- 1 credit = $0.01;
- FLUX.2 pricing scales by model and megapixels;
- FLUX.1 pricing is often per image;
- batch requests multiply base cost by image count.

### Pydantic AI Takeaways

- BFL does not fit a purely synchronous `generate()` implementation unless the adapter polls internally.
- This provider may justify future async job primitives, or adapter-level polling with timeout settings.
- Pricing metadata needs credits and megapixels, not just tokens.
- Output should be downloaded and returned as `BinaryImage`.

## Ideogram

### State Of The Art

Ideogram 3.0 is strong for text rendering, layout, style control, and character consistency workflows. It has a production API with generation, transparent generation, inpainting, remix, reframe, replace background, remove background, upscale, describe, and edit endpoints.

Sources:

- API overview: https://developer.ideogram.ai/
- Generate v3 reference: https://developer.ideogram.ai/api-reference/api-reference/generate-v3
- API pricing: https://ideogram.ai/features/api-pricing

### Endpoint Shape

Primary generation endpoint:

```text
POST https://api.ideogram.ai/v1/ideogram-v3/generate
```

The request is multipart form data.

### Inputs

Generation inputs include:

- `prompt`
- `seed`
- `resolution`
- `aspect_ratio`
- `rendering_speed`
- `magic_prompt`
- `style_type`
- `custom_model_uri`
- style reference images
- character reference images and masks

### Outputs

Response includes:

- `created`
- `data[]`
- per-image `prompt`
- `resolution`
- `is_image_safe`
- `seed`
- `url`
- `style_type`

Image links are temporary and must be downloaded if users want to keep them.

### Metadata

Important fields:

- original/resolved prompt
- resolution
- safety boolean
- seed
- style type
- output URL
- generation speed/quality
- reference image usage

### Cost

Ideogram API pricing is flat per output image for generation/remix/edit/reframe/background replacement. Character reference images have separate pricing. Some endpoints are billed per input/request.

### Pydantic AI Takeaways

- Good example of flat per-image pricing.
- The adapter must download temporary URLs.
- `GeneratedImage.seed`, `is_safe`, `style_type`, and `resolution` metadata matter.
- Character/style reference inputs add pressure to the image editing API design.

## Stability AI

### State Of The Art

Stability's API covers Stable Image Core/Ultra, Stable Diffusion 3.5 variants, image-to-image, inpainting, upscaling, and other editing workflows. It is relevant because it exposes both raw image responses and JSON/base64 responses.

Sources:

- Stable Image Core API example: https://www.postman.com/galactic-spaceship-826546/team-workspace/request/xmf9eiu/stable-image-core
- Stability pricing/plans: https://stability.ai/pricing

### Endpoint Shape

Stable Image Core endpoint:

```text
POST https://api.stability.ai/v2beta/stable-image/generate/core
```

The request is multipart form data. The `Accept` header controls output mode:

- `image/*` to receive image bytes;
- `application/json` to receive base64 JSON.

### Inputs

Common inputs:

- `prompt`
- `aspect_ratio`
- `negative_prompt`
- `seed`
- `output_format`

Other Stability endpoints add image/mask/upscale-specific parameters.

### Outputs

JSON response includes:

- `image`
- `finish_reason`
- `seed`

Raw image response can be requested via `Accept: image/png`, etc.

### Metadata

Important fields:

- seed
- finish reason
- content filtering status
- output format/content type
- operation type

### Cost

Stability uses credits and per-operation pricing. Public plan pages explain credits, but exact per-model API prices are best treated as provider pricing data rather than embedded in Pydantic AI.

### Pydantic AI Takeaways

- Adapter can normalize either raw bytes or JSON base64 to `BinaryImage`.
- `finish_reason` and seed should be preserved.
- Cost probably needs provider-specific price metadata until `genai-prices` supports Stability's image pricing.

## Amazon Bedrock: Nova Canvas And Titan Image

### State Of The Art

Nova Canvas is Amazon's image generation model on Bedrock. It supports text and image inputs, image output, watermarking, moderation, and customization controls through `bedrock-runtime`.

Sources:

- Nova Canvas model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-canvas.html
- Nova Canvas invoke example: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AmazonNovaImageGeneration_section.html
- Titan Image response docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
- Nova pricing overview: https://aws.amazon.com/nova/pricing/

### Endpoint Shape

Bedrock uses provider-native JSON through `InvokeModel`, not an OpenAI-compatible endpoint.

Model ID:

```text
amazon.nova-canvas-v1:0
```

Endpoint pattern:

```text
https://bedrock-runtime.{region}.amazonaws.com
```

### Inputs

Nova Canvas request body:

- `taskType = TEXT_IMAGE`
- `textToImageParams.text`
- optional `textToImageParams.negativeText`
- `imageGenerationConfig`
  - `seed`
  - `quality`
  - `height`
  - `width`
  - `numberOfImages`
  - `cfgScale`

### Outputs

The response body includes:

- `images`: list of base64 encoded images

Titan Image similarly returns `images` or an `error` field.

### Metadata

Important fields:

- seed
- quality
- width/height
- number of images
- model id
- region
- moderation/error information

### Cost

Bedrock pricing depends on provider/model/service tier. Nova pricing pages point to Bedrock pricing and service tiers. The adapter should preserve model id, region, image count, size, quality, and service tier where available.

### Pydantic AI Takeaways

- Bedrock is easy to normalize at the image payload level because output is base64.
- It is not first-PR scope because auth/client patterns differ from OpenAI.
- Cost requires provider-specific pricing data beyond `RequestUsage`.

## Adobe Firefly Services

### State Of The Art

Firefly is important less because of one simple image model and more because of creative workflow controls: style references, structure references, seeds, async jobs, content class, alt text, and storage upload IDs.

Sources:

- Firefly API overview: https://developer.adobe.com/firefly-services/docs/firefly-api/
- Style reference docs: https://developer.adobe.com/firefly-services/docs/firefly-api/guides/concepts/style-image-reference/
- Seed docs: https://developer.adobe.com/firefly-services/docs/firefly-api/guides/concepts/seeds/
- Firefly plan/credit pricing: https://www.adobe.com/products/firefly/plans.html

### Endpoint Shape

Common generation endpoint:

```text
POST https://firefly-api.adobe.io/v3/images/generate-async
```

Firefly uses async jobs:

- request returns `jobId`, `statusUrl`, and `cancelUrl`;
- polling returns final outputs.

Reference images often require an upload step:

```text
POST https://firefly-api.adobe.io/v2/storage/image
```

### Inputs

Inputs include:

- prompt
- `numVariations`
- seeds
- style presets
- style image reference via upload ID
- structure reference
- output size/options

### Outputs

Final status response includes:

- status
- job id
- result size
- outputs
- per-output seed
- image URL
- `contentClass`
- `altText`

### Metadata

Important fields:

- job id
- status URL/cancel URL
- output URL
- seed
- size
- content class
- alt text
- reference upload IDs

### Cost

Firefly uses generative credits and plan-based pricing. Cost is not a simple token/per-image API value in public docs. Preserve enough provider metadata; do not attempt generic cost calculation until `genai-prices` has Firefly-specific support.

### Pydantic AI Takeaways

- Firefly strongly suggests not coupling the core API to only synchronous calls.
- It also supports the argument for future separate editing/reference-image API decisions.
- Metadata preservation is critical.

## Recraft

### State Of The Art

Recraft is relevant for design-production workflows because it supports raster and vector generation, inpainting, background replacement, vectorization, and high-resolution outputs. Vector output is outside the immediate Pydantic AI image primitive, but it is a useful future pressure point.

Sources:

- API overview: https://www.recraft.ai/api
- API endpoints: https://www.recraft.ai/docs/api-reference/endpoints
- API pricing: https://www.recraft.ai/docs/api-reference/pricing

### Endpoint Shape

Example endpoints:

- `/v1/images/explore`
- `/v1/images/imageToImage`

Outputs are URL-based in examples.

### Inputs

Generation inputs include:

- prompt
- model
- size
- response format
- style controls

Editing inputs include multipart image upload and prompt/strength.

### Outputs

Responses include `data[]` objects with URLs.

### Cost

Recraft prices API units:

- API units are prepaid;
- raster generation varies by model, e.g. V4 vs V4 Pro;
- vector generation has separate pricing;
- editing/background/upscale operations have per-request or per-image unit costs.

### Pydantic AI Takeaways

- Recraft is not first implementation scope.
- It suggests that `GeneratedImage` might eventually need to handle vector/document-like outputs if the package expands beyond raster images.
- Per-operation pricing is important.

## Runway And Luma

### Runway

Runway Gen-4 Image is a high-end creative image model, but the public docs are more product/UI oriented than straightforward API reference.

Source:

- Gen-4 Image docs: https://help.runwayml.com/hc/en-us/articles/37053594806419-Creating-with-Gen-4-Image

Relevant signals:

- inputs can include text and image references;
- outputs are fixed aspect ratios/resolutions;
- pricing is credit-based by resolution in the UI docs.

### Luma

Luma Photon image generation uses async generation resources with states and output URLs.

Sources:

- Image generation docs: https://docs.lumalabs.ai/docs/image-generation
- Agents image generation docs: https://docs.agents.lumalabs.ai/guides/image-generation

Relevant signals:

- `POST` creates a generation object;
- states include queued, processing, completed, failed;
- final output contains presigned URLs;
- URLs expire and should be downloaded.

### Pydantic AI Takeaways

- Both reinforce the need to avoid exposing temporary URL semantics by default.
- Both suggest that a future job-aware API may be useful, but first OpenAI implementation can remain simple.

## Provider Integration Plan

This plan ranks integration candidates by practical value for Pydantic AI, not by image quality alone.
The ranking weighs adoption/ecosystem demand, fit with existing Pydantic AI providers, implementation
risk, usage/pricing fit, and whether the provider validates the proposed abstraction without forcing
premature scope.

Adoption signals are necessarily approximate. Providers rarely publish direct API usage for image
generation, so this uses public proxy signals: official usage disclosures, ecosystem presence,
existing Pydantic AI provider support, SDK/API maturity, and whether the model family is widely used
in production image workflows.

| Rank | Provider | Adoption / ecosystem signal | Pros | Cons | Integration stance |
| --- | --- | --- | --- | --- | --- |
| 1 | OpenAI GPT Image family | Very high. OpenAI reported 130M users and 700M generated images in the first week of ChatGPT image generation, and named major product integrations such as Adobe, Airtable, Figma, Canva, Wix, Photoroom, and others. | Existing `OpenAIProvider`, familiar auth/client/error handling, direct Images API, base64 output, detailed usage, token-based pricing closest to current `RequestUsage`/`genai-prices` flow. | Image editing and streaming should remain out of the first PR; newer model names may move faster than SDK type aliases. | First implementation. This is the API-shape validator. |
| 2 | Google Gemini-native image models | Very high. Gemini has broad consumer/product reach and public reports from Alphabet earnings cite large Gemini app usage and high direct API token volume. Pydantic AI already has Google providers/models. | Strategically important, existing provider surface, token-equivalent pricing, strong test of whether an image-only facade can sit on top of a multimodal content API. | `generateContent` may return mixed text/image parts; image generation may need explicit prompting/configuration; not an OpenAI-style Images API. | Strong second/third candidate. Public Pydantic AI API should remain image-only and normalize only generated image parts. |
| 3 | xAI Grok Imagine | Medium/unknown public adoption, but xAI has current image-specific docs, OpenAI-compatible endpoints, documented flat per-image pricing, and existing Pydantic AI xAI provider support. | Low implementation risk after OpenAI, base64 can be requested, similar endpoint shape, useful to prove provider abstraction beyond OpenAI. Provider-reported cost metadata is a useful pricing test. | Less public evidence of API adoption than OpenAI/Google/AWS; pricing is not token-based; current docs include deprecations/migrations in the image model family. | Good low-risk second adapter if maintainers want a quick multi-provider proof. |
| 4 | Amazon Bedrock Nova Canvas | High enterprise relevance through AWS/Bedrock. Pydantic AI already supports Bedrock models and Bedrock embeddings. | Existing `BedrockProvider`, AWS auth/retry/config precedent, enterprise demand, Nova Canvas returns base64 image data through `InvokeModel`, good test for provider-specific request/response shapes. | AWS auth and region setup are heavier; request/response is model-specific rather than a shared Images API; pricing and usage are not as naturally tokenized; likely needs model-family-specific handlers. | Enterprise follow-up candidate, not early unless AWS demand is explicit. Reuse the Bedrock precedent: manual usage mapping, `bedrock_*` settings, provider details. |
| 5 | Stability AI | High image-generation ecosystem relevance due to Stable Diffusion/open-weight workflows and an official image API. | Image-native API, raw bytes/base64 paths are easy to normalize, strong editing/upscaling ecosystem, useful for non-OpenAI image-first workflows. | No existing Pydantic AI provider, multipart inputs, credit-based pricing, direct hosted API may be less central than self-hosted/partner-hosted Stable Diffusion usage. | Good later provider if users ask for image-first APIs. Do not let its editing breadth expand first-release scope. |
| 6 | Black Forest Labs / FLUX | High quality and strong developer mindshare, especially through FLUX/open-weight and hosted inference ecosystems. | Important SOTA pressure point for photorealistic generation, aspect-ratio/resolution controls, seed metadata, strong production image use cases. | Async job/polling and signed URLs complicate the simple request/response primitive; direct BFL API shape may push toward job lifecycle support. | Design pressure first. Implement only after deciding whether adapter-level polling is acceptable. |
| 7 | Ideogram | Medium/high in design and typography-heavy image workflows. API pricing and docs are straightforward. | Excellent stress test for text-in-image, character/style references, seed/safety metadata, and per-output-image pricing. | URL outputs, richer editing/reference concepts, narrower ecosystem than OpenAI/Google/AWS; pricing is per image rather than token-based. | Later candidate if design/typography workflows are a priority. |
| 8 | Recraft | Medium/niche but strong for production design, brand, and vector workflows. | Clear API/pricing, raster plus vector generation, useful for design-production use cases. | Vector/SVG/PDF/Lottie outputs are outside a raster `BinaryImage` primitive; integrating too early would pull the API beyond image bytes. | Mostly design pressure. Raster-only support is possible later; vector output likely needs a separate decision. |
| 9 | Adobe Firefly | High enterprise creative relevance, but less natural for a lightweight first primitive. | Strong creative-suite/workflow relevance, enterprise governance, rich style/reference workflows. | Async jobs, upload/storage IDs, credit/subscription accounting, and Adobe-specific workflow concepts. | Not an early adapter. Keep as evidence against overfitting the first API to synchronous providers only. |
| 10 | Runway / Luma | Strong media-generation ecosystems, but often video/job-first rather than simple image generation. | Useful pressure for async job lifecycle, result URLs, and future media APIs. | Pushes hard toward video/audio/media abstraction and job polling; less aligned with a minimal image-generation primitive. | Out of scope for now, maybe never part of `pydantic_ai.images` directly. |

### Bedrock And Non-Token Pricing Lessons From The Existing Repo

Existing Bedrock integrations are useful precedent for providers with non-OpenAI shapes:

- `pydantic_ai.models.bedrock.BedrockConverseModel` maps Bedrock `inputTokens`,
  `outputTokens`, `cacheReadInputTokens`, and `cacheWriteInputTokens` manually into
  `RequestUsage` instead of relying on a generic extractor.
- `pydantic_ai.embeddings.bedrock.BedrockEmbeddingModel` uses `invoke_model`, model-family handlers
  for Titan/Cohere/Nova, and manual token extraction from Bedrock response headers.
- Bedrock settings are provider-prefixed (`bedrock_*`) and unsupported model-specific settings are
  either ignored deliberately or handled by the model-family handler.

For image generation this suggests:

- use `RequestUsage` when a provider exposes token-equivalent usage;
- preserve image count, size, aspect ratio, quality, credits, provider-reported cost, and output
  format in structured result/provider metadata;
- avoid inventing a universal pricing type in the first PR;
- let `genai-prices` calculate cost only when it has enough provider/model/API-flavor data, and
  otherwise degrade with `LookupError` as existing `cost()` APIs do.

## Cross-Provider Field Matrix

| Field | OpenAI | Google | xAI | BFL | Ideogram | Stability | Bedrock | Firefly |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prompt | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |
| Reference image | Edits | Gemini/edit paths | Edits | Edits | Style/character/edit | Image-to-image/inpaint | Image input | Style/structure refs |
| Base64 output | Yes | Yes | Optional | No, signed URL | No, URL | Yes/raw bytes | Yes | No, URL |
| URL output | Legacy/OpenAI non-GPT | Optional/storage | Default | Signed URL | Temporary URL | Raw/image possible | No | Presigned URL |
| Usage tokens | Yes | Gemini yes | No token focus | No | No | No | No | No |
| Provider cost returned | No direct cost | No direct cost | Yes, ticks | Sometimes credits | No | No | No | Credit accounting |
| Safety metadata | Some | Yes | Moderation | Status/details | `is_image_safe` | finish reason | error/moderation | content class |
| Seed | Limited | Model/config dependent | Not central | Yes | Yes | Yes | Yes | Yes |
| Async job | No for basic generation | No for direct Gemini calls | No for basic | Yes | No | No | No | Yes |
| Streaming/partials | OpenAI supports image streaming for GPT image models | Provider-specific | Not core | No | No | No | No | Async status |

## Recommended First Pydantic AI Scope

### Implement Now

- `ImageGenerator`
- `ImageGenerationModel`
- `ImageGenerationSettings`
- `ImageGenerationResult`
- `GeneratedImage`
- `OpenAIImageGenerationModel`
- `TestImageGenerationModel`
- Logfire instrumentation with no binary output logging
- OpenAI direct `images.generate`
- normalized `BinaryImage` output
- `RequestUsage` with OpenAI image token details
- normalized metadata: size, quality, output format, background, revised/enhanced prompt where available

### Preserve For Later

- provider-reported cost
- image count
- width/height/aspect ratio
- seed
- safety/moderation status
- reference image metadata
- provider-specific metadata where useful and safe

### Defer

- image editing API
- async job API
- streaming/partial images
- multi-provider support
- vector outputs
- video/audio generation
- full `genai-prices` image pricing support if the upstream package is not ready

The deferred list is not a commitment to implement every item. Some provider shapes, especially
async job APIs, vector outputs, and broader media abstractions, may remain outside Pydantic AI if the
extra abstraction cost is higher than the value for the core library.

## Source Index

- OpenAI GPT Image 2 model: https://developers.openai.com/api/docs/models/gpt-image-2
- OpenAI Images API reference: https://developers.openai.com/api/reference/resources/images
- OpenAI API pricing: https://openai.com/api/pricing/
- OpenAI GPT Image API adoption/launch: https://openai.com/index/image-generation-api/
- Google Gemini image generation: https://ai.google.dev/gemini-api/docs/image-generation
- Google Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- Google Gemini adoption/API usage reporting: https://techcrunch.com/2026/02/04/googles-gemini-app-has-surpassed-750m-monthly-active-users/
- xAI image generation: https://docs.x.ai/developers/model-capabilities/images/generation
- xAI image overview: https://docs.x.ai/developers/model-capabilities/images
- xAI image model/pricing: https://docs.x.ai/developers/models/grok-imagine-image
- xAI cost tracking: https://docs.x.ai/developers/cost-tracking
- xAI images REST reference: https://docs.x.ai/developers/rest-api-reference/inference/images
- Black Forest Labs pricing: https://docs.bfl.ai/quick_start/pricing
- Black Forest Labs image generation: https://docs.bfl.ai/quick_start/generating_images
- Black Forest Labs FLUX.2 text-to-image: https://docs.bfl.ai/flux_2/flux2_text_to_image
- Black Forest Labs FLUX.2 image editing: https://docs.bfl.ai/flux_2/flux2_image_editing
- Black Forest Labs result polling: https://docs.bfl.ai/api-reference/utility/get-result
- Ideogram API overview: https://developer.ideogram.ai/
- Ideogram 3.0 generate API: https://developer.ideogram.ai/api-reference/api-reference/generate-v3
- Ideogram API pricing: https://ideogram.ai/features/api-pricing
- Stability Stable Image Core example: https://www.postman.com/galactic-spaceship-826546/team-workspace/request/xmf9eiu/stable-image-core
- Stability image models: https://stability.ai/stable-image
- Stability pricing/plans: https://stability.ai/pricing
- Amazon Nova announcement: https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws/
- Amazon Nova Canvas model card: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-amazon-nova-canvas.html
- Amazon Nova Canvas invoke example: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AmazonNovaImageGeneration_section.html
- Amazon Titan Image response docs: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
- Amazon Nova pricing overview: https://aws.amazon.com/nova/pricing/
- Adobe Firefly API overview: https://developer.adobe.com/firefly-services/docs/firefly-api/
- Adobe Firefly style reference: https://developer.adobe.com/firefly-services/docs/firefly-api/guides/concepts/style-image-reference/
- Adobe Firefly seeds: https://developer.adobe.com/firefly-services/docs/firefly-api/guides/concepts/seeds/
- Adobe Firefly plans: https://www.adobe.com/products/firefly/plans.html
- Recraft API overview: https://www.recraft.ai/api
- Recraft V4 docs: https://www.recraft.ai/docs/recraft-models/recraft-V4
- Recraft API endpoints: https://www.recraft.ai/docs/api-reference/endpoints
- Recraft API pricing: https://www.recraft.ai/docs/api-reference/pricing
- Runway Gen-4 Image docs: https://help.runwayml.com/hc/en-us/articles/37053594806419-Creating-with-Gen-4-Image
- Luma image generation docs: https://docs.lumalabs.ai/docs/image-generation
- Luma Agents image generation docs: https://docs.agents.lumalabs.ai/guides/image-generation
