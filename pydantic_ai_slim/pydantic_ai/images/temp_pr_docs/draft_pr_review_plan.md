# Direct Image Generation Draft Plan

This draft supports issue #3898. It proposes a dedicated image-generation primitive, following the
shape of `pydantic_ai.embeddings`, rather than fitting image-only models into the `Agent` /
`ModelRequest` / `ModelResponse` chat flow.

The implementation is intentionally narrow so reviewers can evaluate the public API direction before
we broaden provider support.

## What This Draft Implements

- `ImageGenerator`, a high-level facade similar to `Embedder`.
- `ImageGenerationModel`, the provider model abstraction.
- `ImageGenerationSettings`, `ImageGenerationResult`, and `GeneratedImage`.
- `TestImageGenerationModel` for deterministic tests.
- `OpenAIImageGenerationModel` using the direct OpenAI Images API for the GPT Image family.
- Text-to-image generation only.
- Generated outputs normalized to `BinaryImage`.
- `RequestUsage` mapping with OpenAI image token breakdowns in `usage.details`.
- Pricing-ready metadata on results/images.
- Logfire/OpenTelemetry instrumentation without generated image bytes/base64, including safe image
  metadata attributes for display/querying.
- Focused tests for settings, model inference, OpenAI response mapping, errors, and instrumentation.

## Non-Goals For This Draft

- No image editing.
- No reference images, masks, image-to-image inputs, or provider file IDs.
- No variations.
- No streaming or partial images.
- No Gemini, xAI, Bedrock, Stability, or other providers.
- No DALL-E compatibility surface.
- No direct video/audio generation.
- No final commitment to the long-term public API before maintainer feedback.
- No commitment that every provider studied in `SOTA_study.md` should eventually be supported.

## Current API Direction

```python {test="skip" lint="skip"}
from pydantic_ai import ImageGenerator

generator = ImageGenerator('openai:gpt-image-1')
result = await generator.generate('A tiny robot watering a basil plant')

image = result.images[0].content
```

Generated images are normalized to `BinaryImage` even when a provider supports URLs or other output
forms. Future image inputs for editing/reference-image flows should be designed separately.

Common semantic settings are intentionally small:

- `n`
- `output_format`

This draft intentionally does not introduce common `size` or `aspect_ratio` settings. Providers use
different concepts here: OpenAI direct Images uses pixel sizes and model-specific constraints,
Gemini-native image models use aspect ratios plus image-size tiers, and xAI uses aspect ratio plus
resolution. A common field would either mean different things per provider or require Pydantic AI to
own a cross-provider layout mapping policy. For the first draft, layout controls should stay
provider-prefixed, e.g. `openai_size`.

The draft also keeps `extra_headers` and `extra_body` as provider/SDK escape hatches, following the
existing `ModelSettings` and `EmbeddingSettings` pattern. They are not semantic cross-provider
settings, and they should not replace provider-prefixed, typed settings for features we intentionally
support.

Provider/SDK escape hatches:

- `extra_headers`
- `extra_body`

OpenAI-specific settings are provider-prefixed and scoped to GPT Image models, e.g. `openai_size`,
`openai_quality`, `openai_background`, `openai_moderation`, `openai_output_compression`, and
`openai_user`. The draft targets the GPT Image family rather than the older DALL-E image API shape.

## Running The Local Demo

A small local demo is available at:

`pydantic_ai_slim/pydantic_ai/images/_demo.py`

It exercises the instrumented path for embeddings, an agent run, and direct image generation with
OpenAI GPT Image models. To run it locally:

```bash
OPENAI_API_KEY=... uv run python -m pydantic_ai.images._demo
```

If `LOGFIRE_TOKEN` is configured, the demo also sends spans to Logfire. Generated image files are
written next to the demo script as `_demo_output_<model>.png`.

This demo is for maintainer review and API/instrumentation inspection. It is not intended as final
user-facing documentation.

## Provider Prioritization

This is prioritization guidance, not a support roadmap.

| Order | Provider | Why it is sensible | Main drawbacks |
| --- | --- | --- | --- |
| 1 | OpenAI GPT Image family | Highest-confidence first target: strong public adoption signals, existing `OpenAIProvider`, direct Images API, base64 output, detailed usage, token-based pricing closest to embeddings. | Editing/streaming should stay out of this draft; model names may outpace SDK literal types. |
| 2 | Google Gemini-native image models | Very high adoption and strategic importance; existing Google provider surface; validates image-only API over a multimodal provider. | Uses `generateContent`, may return mixed text/image parts, and needs explicit image-output handling. |
| 3 | xAI Grok Imagine | Low implementation risk after OpenAI: OpenAI-compatible image endpoint, base64 output option, existing xAI provider, simple flat per-image pricing. | Public API adoption is less proven; pricing is not token-based; docs/model names may change quickly. |
| 4 | Amazon Bedrock Nova Canvas | Enterprise relevance and existing Bedrock provider/embedding precedent in this repo; good test for `InvokeModel` provider-specific shapes. | AWS auth/regions are heavier; request shape and pricing are provider/model-specific; likely needs Bedrock-style handlers. |
| 5 | Stability AI | Strong Stable Diffusion ecosystem and image-native API; output can be normalized from raw bytes/base64. | No existing provider here; multipart inputs and credit pricing; broad editing/upscaling surface could expand scope. |
| 6 | BFL / FLUX | Strong SOTA image quality and developer mindshare; useful pressure for aspect ratio, resolution, seed, and signed URL handling. | Async polling and signed URLs may require job lifecycle decisions. |
| 7 | Ideogram | Strong design/typography use cases, clear API pricing, useful metadata pressure. | URL outputs, per-image pricing, and reference/style flows make it less minimal. |
| 8 | Recraft | Useful for production design and vector workflows. | Vector/SVG/PDF/Lottie outputs are outside a raster `BinaryImage` primitive. |
| 9 | Adobe Firefly | Enterprise creative relevance and rich reference/style workflows. | Async jobs, upload IDs, and credit/subscription accounting make it poor early scope. |
| 10 | Runway / Luma | Useful pressure for future media/job APIs. | More video/media-job oriented than a minimal image-generation primitive; likely out of scope for `pydantic_ai.images`. |

## Open Questions

1. Should image editing live on `ImageGenerator.generate(..., reference_images=...)`, or should it use
   a separate primitive such as `ImageEditor`?
2. What exact image-generation contract should `genai-prices` consume: `RequestUsage.details`,
   provider metadata, a new media usage type, or a combination?
3. Which provider/model/feature matrix is required for the first stable release?
4. Should images remain separate from future video/audio packages rather than introducing a premature
   `MediaGenerator` abstraction?
5. Confirm output policy: generated images should be normalized to `BinaryImage`, without deciding
   future accepted input types for editing/reference-image APIs.
6. Confirm Logfire policy: generated image bytes/base64 should not be logged by default.
7. Decide whether Logfire needs first-class UI support for `image_generation` spans, similar to the
   existing embeddings details tab. The library can expose provider/model/server/usage/cost and
   safe image metadata attributes, but a dedicated Logfire renderer may be needed for the same
   polished table view.
8. Decide whether a future provider-independent image layout abstraction is worth adding, or whether
   size/resolution/aspect-ratio controls should remain provider-specific.
9. Evaluate the strict `resolution='WIDTHxHEIGHT'` proposal in `resolution_mapping_proposal.md`.
   That document is a speculative design sketch; it should not be treated as part of this draft's
   implemented API.

## Proposed Next Steps After Review

- Adjust API naming and module layout based on maintainer feedback.
- Add VCR-backed OpenAI tests.
- Add docs and API docs.
- Decide whether `cost()` should land now or after `genai-prices` adds image-specific support.
- Plan image editing and additional providers as separate PRs.
