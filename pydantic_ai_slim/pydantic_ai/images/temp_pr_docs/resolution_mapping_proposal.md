# Strict Resolution Mapping Proposal

This proposal explores a future common image-generation setting:

```python {test="skip" lint="skip"}
settings={'resolution': '1920x1080'}
```

The setting would mean exact pixel dimensions. It would not be a best-effort target, a quality tier,
or a provider-native size field.

This is a speculative design sketch, not part of the current draft implementation. The concrete
provider tables and constraints below must be verified against current provider docs and recorded
responses before this becomes an API proposal.

## Goal

Give users a provider-independent way to ask for a concrete output resolution while avoiding silent
or approximate layout changes.

The proposed contract:

- `resolution` accepts only `WIDTHxHEIGHT` strings.
- The requested dimensions are exact.
- Provider adapters may map the request only when the selected provider/model can express that exact
  output resolution.
- If no exact mapping exists, adapters raise `UserError`.
- Error messages should explain the provider/model constraints in terms of supported exact
  resolutions whenever possible.
- Provider-native controls such as `openai_size` remain available.
- Common `resolution` must not be combined with provider-native dimension controls that affect the
  same output dimension.

This keeps portability explicit without pretending that all providers expose the same image layout
API.

## Why Not `size`

`size` is ambiguous across providers:

- OpenAI direct Images uses pixel-size strings and model-specific constraints.
- Gemini-native image models use aspect ratios plus image-size tiers.
- xAI uses aspect ratios plus `1k`/`2k` resolution tiers.

A common `size` field would either mean different things depending on the provider or force
Pydantic AI to own a broad layout policy too early. `resolution='WIDTHxHEIGHT'` is narrower and more
honest because it describes the user's desired output dimensions directly.

## Common API Shape

Potential future common setting:

```python {test="skip" lint="skip"}
class ImageGenerationSettings(TypedDict, total=False):
    n: int
    resolution: str
    output_format: ImageOutputFormat
```

Provider-specific settings would still be available:

```python {test="skip" lint="skip"}
class OpenAIImageGenerationSettings(ImageGenerationSettings, total=False):
    openai_size: str
```

Invalid combination:

```python {test="skip" lint="skip"}
settings={'resolution': '1024x1024', 'openai_size': '1536x1024'}
```

This should raise `UserError` because both settings control output dimensions.

## Parsing

The common parser should accept only strict pixel resolution strings:

```text
WIDTHxHEIGHT
```

Examples:

- `1024x1024`
- `1376x768`
- `1920x1080`
- `2048x1152`

Invalid examples:

- `1K`
- `2k`
- `16:9`
- `1024`
- `auto`

Those are provider-native concepts and should remain provider-prefixed.

## Provider Mapping

### OpenAI GPT Image 1.x

OpenAI GPT Image 1.x models support a small fixed set of sizes.

Illustrative mapping:

| `resolution` | OpenAI native setting |
| --- | --- |
| `1024x1024` | `size='1024x1024'` |
| `1536x1024` | `size='1536x1024'` |
| `1024x1536` | `size='1024x1536'` |

Unsupported example:

```python {test="skip" lint="skip"}
settings={'resolution': '1920x1080'}
```

Error should list supported exact resolutions:

```text
OpenAI model 'gpt-image-1.5' does not support resolution '1920x1080'.
Supported resolutions are: 1024x1024, 1536x1024, 1024x1536.
Use `openai_size` to pass a provider-native OpenAI size directly.
```

### OpenAI GPT Image 2

OpenAI GPT Image 2 appears to support many pixel resolutions subject to model constraints rather
than a small enumerated table.

The adapter should validate exact dimensions against OpenAI's documented constraints, once those
constraints have been verified for the specific model version.

Example:

```python {test="skip" lint="skip"}
settings={'resolution': '1920x1080'}
```

If the height must be a multiple of 16, this should raise rather than rounding to a nearby value:

```text
OpenAI model 'gpt-image-2' does not support resolution '1920x1080':
height must be a multiple of 16.
Supported constraints: width and height must be multiples of 16, max edge <= 3840,
aspect ratio <= 3:1, and total pixels must be within the model's allowed range.
```

The adapter should not suggest closest resolutions unless we can do so unambiguously and decide to
support that as part of the API. The initial version should only report constraints.

### Gemini-Native Image Models

Gemini-native image models expose aspect ratio plus image-size tiers, but the docs publish exact
resolution tables for model families.

For Gemini, the adapter can reverse-map exact `WIDTHxHEIGHT` values to native settings.

Illustrative mapping for Gemini 3 Pro Image, to be verified before implementation:

| `resolution` | Gemini native settings |
| --- | --- |
| `1024x1024` | `aspect_ratio='1:1'`, `image_size='1K'` |
| `2048x2048` | `aspect_ratio='1:1'`, `image_size='2K'` |
| `4096x4096` | `aspect_ratio='1:1'`, `image_size='4K'` |
| `1376x768` | `aspect_ratio='16:9'`, `image_size='1K'` |
| `2752x1536` | `aspect_ratio='16:9'`, `image_size='2K'` |
| `5504x3072` | `aspect_ratio='16:9'`, `image_size='4K'` |

Unsupported example:

```python {test="skip" lint="skip"}
settings={'resolution': '1920x1080'}
```

Error should list supported exact resolutions, not just aspect ratios:

```text
Gemini model 'gemini-3-pro-image-preview' does not support resolution '1920x1080'.
Supported 16:9 resolutions are: 1376x768, 2752x1536, 5504x3072.
Supported 1:1 resolutions are: 1024x1024, 2048x2048, 4096x4096.
```

If the full resolution list is long, the error can group by aspect ratio or truncate with a pointer
to provider docs.

### xAI Grok Imagine

xAI exposes aspect ratio plus resolution tier (`1k`, `2k`). The official docs list supported aspect
ratios and resolution tiers. Model/pricing pages describe `1K` as `1024x1024` and `2K` as
`2048x2048` for square output.

The mapping appears to be area-preserving:

```text
target_area = tier_edge * tier_edge
width = sqrt(target_area * aspect_ratio)
height = sqrt(target_area / aspect_ratio)
```

For example:

```text
16:9 + 1k => about 1365x768
16:9 + 2k => about 2731x1536
```

Before implementing xAI resolution mapping, we should confirm the exact rounding rules from
provider docs or recorded real responses. Once confirmed, the adapter can build an exact
resolution-to-native-settings table for supported ratios and tiers.

Illustrative mapping, to be generated from verified provider constraints:

| `resolution` | xAI native settings |
| --- | --- |
| `1024x1024` | `aspect_ratio='1:1'`, `resolution='1k'` |
| `2048x2048` | `aspect_ratio='1:1'`, `resolution='2k'` |
| `1365x768` | `aspect_ratio='16:9'`, `resolution='1k'` |
| `2731x1536` | `aspect_ratio='16:9'`, `resolution='2k'` |

Unsupported example:

```python {test="skip" lint="skip"}
settings={'resolution': '1920x1080'}
```

Error should list supported exact resolutions:

```text
xAI model 'grok-imagine-image-quality' does not support resolution '1920x1080'.
Supported 16:9 resolutions are: 1365x768, 2731x1536.
Supported 1:1 resolutions are: 1024x1024, 2048x2048.
```

The error can include provider-native details after the supported resolutions:

```text
xAI maps output dimensions from aspect_ratio plus resolution tier. Supported tiers are: 1k, 2k.
```

## Error Message Principles

Errors should be written for the common API, not the provider-native API.

Good:

```text
Supported 16:9 resolutions are: 1376x768, 2752x1536, 5504x3072.
```

Less useful:

```text
Supported aspect ratios are: 16:9, 1:1, 3:2.
```

Provider-native ratios and tiers are useful secondary context, but users asked for a concrete
resolution. The primary error should explain which concrete resolutions are valid.

Recommended behavior:

- For finite provider/model tables, list supported exact resolutions.
- For long lists, group by aspect ratio or tier.
- For constraint-based providers such as OpenAI GPT Image 2, report constraints instead of listing
  every possible resolution.
- Do not silently round or choose nearest values.
- Do not suggest closest values unless that becomes an explicit API feature.

## Testing Requirements

If implemented, this feature should include:

- parser tests for valid and invalid `WIDTHxHEIGHT` strings.
- conflict tests for `resolution` plus provider-native dimension settings.
- OpenAI GPT Image 1.x table mapping tests.
- OpenAI GPT Image 2 constraint validation tests.
- Gemini reverse-table mapping tests for at least one Gemini 2.5 image model and one Gemini 3 image
  model.
- xAI mapping tests only after exact rounding rules are confirmed.
- error snapshot tests showing supported resolutions or constraints.

## Open Questions

1. Should `resolution` land as a common setting, or should provider-native controls remain the only
   supported first-release API?
2. Should OpenAI GPT Image 2 constraints be maintained locally, or should invalid resolutions be
   delegated to the provider API?
3. How much of each provider's supported resolution table should be shown in error messages before
   truncating?
4. Should xAI mapping be based on documented constraints only, or are recorded responses sufficient
   to establish the table?
5. Should a future non-strict mode exist, for example `resolution_mode='closest'`, or should the
   common API remain exact-only?
