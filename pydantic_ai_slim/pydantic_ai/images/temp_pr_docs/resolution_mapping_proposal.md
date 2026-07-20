# Exact Image Geometry Mapping

This document records the implemented working proposal for portable image geometry. It remains a
maintainer-review artifact rather than the normative user documentation.

## Public Contract

The common API has two model-aware controls:

```python {test="skip" lint="skip"}
settings={'dimensions': (1280, 720)}
settings={'aspect_ratio': '16:9'}
```

- `dimensions=(width, height)` requests exact pixels. The adapter raises `UserError` if the selected
  model cannot produce that exact shape; it never rounds or chooses a nearest value.
- `aspect_ratio` requests a ratio and lets Pydantic AI select one documented canonical shape for the
  selected model.
- `dimensions` is mutually exclusive with `aspect_ratio` and the compatibility `size` setting.
- `size: str` remains provider-dependent: an OpenAI pixel string or a Google/xAI resolution tier.
  It is retained as a direct-API compatibility field, not as a portable pixel abstraction.
- Provider-specific settings remain typed from their SDK and take precedence with a warning.

These controls belong to `ImageGenerationSettings` and the direct image adapters. The legacy
`ImageGenerationTool` surface is unchanged. When `ImageGeneration` is used through its native or
`fallback_model` path, direct-only geometry values are ignored with a warning that points users to
`native=False, local='provider:image-model'`.

The name is `dimensions`, not `resolution`, because xAI already uses “resolution” for its `1k`/`2k`
quality tiers and Google uses “image size” for `512`/`1K`/`2K`/`4K` tiers. A `(width, height)` tuple
also makes the unit and ordering explicit.

## Provider Ownership

The capability forwards only common settings. Each provider owns two private operations:

1. validate or reverse-map an exact `(width, height)`;
2. map a supported common ratio to a canonical provider request.

This keeps model tables out of `capabilities/image_generation.py`. OpenAI, Google, and xAI each use
a private `_..._geometry.py` module; the legacy native model implementations remain untouched.

## OpenAI

GPT Image 1.x accepts three exact shapes and maps ratios accordingly:

| Ratio | Canonical dimensions |
| --- | --- |
| `1:1` | `1024x1024` |
| `2:3` | `1024x1536` |
| `3:2` | `1536x1024` |

GPT Image 2 accepts arbitrary `WIDTHxHEIGHT` strings subject to the documented constraints:

- both edges are multiples of 16;
- longest edge at most 3840 pixels;
- aspect ratio at most 3:1;
- total area from 655,360 to 8,294,400 pixels.

The common ratio mapping chooses an exact-ratio output around one megapixel. This table is an
opinionated Pydantic AI policy and needs explicit maintainer validation, because OpenAI documents
constraints but does not publish one canonical shape per ratio. Passing `dimensions` bypasses that
policy and expresses the exact request directly.

The direct common `size` is widened to `str` so GPT Image 2 values are not rejected before reaching
model-aware validation. `ImageGenerationTool.size` retains its existing literal vocabulary.

Source: <https://developers.openai.com/api/docs/guides/image-generation#customize-image-output>

## Google Gemini

Google publishes exact tables relating aspect ratio and image-size tier. The adapter keeps profiles
for the known image families and reverse-maps exact dimensions to `ImageConfig.aspect_ratio` plus
`ImageConfig.image_size`:

- Gemini 2.5 Flash Image: ten ratios with fixed output dimensions and no image-size tier;
- Gemini 3 Pro Image: ten ratios at `1K`, `2K`, and `4K`;
- Gemini 3.1 Flash Image: fourteen ratios at `512`, `1K`, `2K`, and `4K`;
- Gemini 3.1 Flash Lite Image: conservatively the ten ratios and `1K` tier stated by the current
  prose documentation.

The published Flash `21:9` row is not a uniform scaling sequence (`792x168` at `512`, then
`1584x672` at `1K`), so those four values are stored explicitly. Flash Lite has no complete table;
until Google publishes one or a real request proves otherwise, reusing the documented `1K` values
from the supported standard ratios is the conservative interpretation.

Source: <https://ai.google.dev/gemini-api/docs/image-generation>

## xAI Grok Imagine

xAI documents the supported ratios and `1k`/`2k` tiers but not the complete exact output table. The
SDK protobuf only says that aspect ratio is preserved, the final area is capped around `2048^2`, and
dimensions are rounded to multiples of 16. That is insufficient to reproduce the actual rounding.

We therefore sampled every combination on both current models (`grok-imagine-image` and
`grok-imagine-image-quality`): 13 ratios x 2 tiers x 2 models = 52 real requests. Both models returned
identical dimensions for all combinations:

| Ratio | `1k` | `2k` |
| --- | --- | --- |
| `1:1` | `1024x1024` | `2048x2048` |
| `3:4` | `864x1152` | `1776x2368` |
| `4:3` | `1152x864` | `2368x1776` |
| `9:16` | `720x1280` | `1584x2816` |
| `16:9` | `1280x720` | `2816x1584` |
| `2:3` | `832x1248` | `1664x2496` |
| `3:2` | `1248x832` | `2496x1664` |
| `9:19.5` | `576x1248` | `1344x2912` |
| `19.5:9` | `1248x576` | `2912x1344` |
| `9:20` | `576x1280` | `1440x3200` |
| `20:9` | `1280x576` | `3200x1440` |
| `1:2` | `704x1408` | `1456x2912` |
| `2:1` | `1408x704` | `2912x1456` |

Observed request cost was $2.08 in total: $0.52 for the standard model and $1.56 for the quality
model. This proves cross-model consistency for one response per combination; it does not claim a
statistical repeatability study. Unknown future xAI model IDs are rejected for exact dimensions
until their geometry has been verified.

For common `aspect_ratio`, the adapter pins `1k` as the canonical tier. Provider-specific
`xai_aspect_ratio` retains the SDK's native default behavior.

Sources:

- <https://docs.x.ai/developers/model-capabilities/images/generation>
- <https://docs.x.ai/developers/models/grok-imagine-image>
- <https://docs.x.ai/developers/models/grok-imagine-image-quality>

## Precedence and Errors

- Exact common dimensions are validated before the provider request.
- A provider-specific geometry setting wins over a conflicting common value and emits one aggregate
  warning, consistent with the other normalized settings.
- Finite-table providers report unsupported exact dimensions as `UserError`.
- Constraint-based GPT Image 2 errors name each violated constraint.
- An explicit `size` may still be combined with `aspect_ratio` where the pre-existing API supports
  tier plus ratio. For OpenAI, a pixel size is accepted with a mathematically equivalent ratio.

## Maintainer Decisions Still Required

1. Confirm `dimensions` as the public name and exact `(width, height)` tuple contract.
2. Confirm that maintaining model-specific geometry profiles is worth the portable API.
3. Confirm the approximately-one-megapixel canonical ratio policy for GPT Image 2.
4. Confirm conservative Gemini 3.1 Flash Lite support until Google's documentation is unambiguous.
5. Confirm that `size` stays provider-dependent for direct-API compatibility rather than being
   deprecated in this PR.
6. Confirm that new geometry remains direct-only and that warning-and-migrate is preferable to
   expanding `ImageGenerationTool` and the native model implementations.

## Regression Coverage

- common validation and mutual exclusion;
- direct OpenAI GPT Image 1.x/GPT Image 2 mapping and constraints;
- direct Google reverse mapping and model-specific ratio/tier checks;
- all 26 verified xAI geometries on both current models;
- capability direct forwarding, legacy warning behavior, and Agent Spec schema parity;
- real OpenAI, Google, and xAI generation cassettes exercising the mapped request offline.
