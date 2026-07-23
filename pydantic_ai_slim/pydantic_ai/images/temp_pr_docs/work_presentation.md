# Direct Image Generation API: Work Presentation

## Status and Purpose

This document presents the implementation currently proposed in
[pydantic/pydantic-ai#5357](https://github.com/pydantic/pydantic-ai/pull/5357), following the maintainer discussion around
[pydantic/pydantic-ai#3898](https://github.com/pydantic/pydantic-ai/issues/3898).

It is review material, not user documentation. It explains the implemented shape, the evidence behind it, and the
decisions that still benefit from maintainer confirmation. The implementation is intentionally concrete and tested, but
the opinionated parts remain reversible.

The review-only assets in `pydantic_ai/images/temp_pr_docs/` and the manual paid demo in
`pydantic_ai/images/_demo.py` should be removed after maintainer alignment. Durable behavior is documented separately in
`docs/image-generation.md`, the capability guide, provider pages, API reference, docstrings, and tests.

## What the PR Adds

The PR introduces a direct, provider-agnostic image API centered on:

- [`ImageGenerator`][pydantic_ai.images.ImageGenerator], the high-level application interface;
- [`ImageGenerationModel`][pydantic_ai.images.ImageGenerationModel], the provider adapter abstraction;
- [`ImageGenerationSettings`][pydantic_ai.images.ImageGenerationSettings], normalized best-effort settings;
- [`ImageGenerationResult`][pydantic_ai.images.ImageGenerationResult] and
  [`GeneratedImage`][pydantic_ai.images.GeneratedImage], normalized results;
- direct adapters for OpenAI, Google Gemini, and xAI;
- generation and reference-image editing through one `generate()` method;
- synchronous and asynchronous entry points;
- deterministic test models, wrappers, overrides, instrumentation, usage, and provider metadata;
- integration with the existing [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] capability.

The high-level call is intentionally small:

```python {test="skip" lint="skip"}
generator = ImageGenerator('openai:gpt-image-1.5')
result = await generator.generate(prompt, images=[reference], settings=settings)
```

`images` accepts the existing Pydantic AI media primitives `BinaryImage`, `ImageUrl`, and `UploadedFile`. Generated
content is always normalized to `BinaryImage`.

## Public Surface and Ownership

### `ImageGenerator`

`ImageGenerator` owns high-level lifecycle behavior analogous to `Embedder`:

- provider-prefixed model inference;
- deferred or eager model validation;
- defaults at generator level and per-call overrides;
- `override()` for tests and temporary model replacement;
- `generate()` and `generate_sync()`;
- local and global instrumentation switches.

Settings precedence is:

1. model defaults;
2. generator defaults;
3. per-call settings.

### `ImageGenerationModel`

The abstract model owns provider-independent invariants only:

- a non-empty prompt;
- image-only reference inputs;
- settings merge and common validation;
- stable provider/model identity.

It deliberately does not define an abstract geometry resolver. OpenAI maps geometry to one pixel-size string, Google
maps it to an image config containing aspect ratio and tier, and xAI maps it to SDK enums. A common abstract result would
either expose provider details in the base class or force a second mapping step. Private provider helpers instead use a
consistent `resolve_<provider>_geometry()` convention without creating a public extension requirement.

### Results

`ImageGenerationResult` preserves:

- normalized images;
- original prompt;
- model and provider identity;
- timestamp;
- normalized usage and per-modality usage details where available;
- normalized settings;
- provider details, response ID, and URL.

Each `GeneratedImage` preserves the binary content plus reported size, quality, output format, background, revised
prompt, provider image ID, and provider details when available.

## Generation and Editing Contract

The portable contract is one textual prompt plus an ordered sequence of image references. It does not expose interleaved
text and image fragments because OpenAI and xAI receive one prompt separately from image inputs, while Google accepts
ordered content parts. Pretending that interleaved text is portable would require silently concatenating it for two
providers and would lose the positional relationship promised by the API.

The presence of `images` selects editing or reference-conditioned generation:

- OpenAI dispatches to `images.generate` without references and `images.edit` with references;
- Google uses `generate_content` for both and emits ordered prompt/image parts;
- xAI uses the official image client's single or batch methods and supplies reference URLs or file IDs.

No separate `ImageEditor` abstraction is introduced. Masks, partial images, and streaming remain outside this PR.

### `UploadedFile`

`UploadedFile` is included in the common input union because it has real semantics on Google and xAI:

- Google accepts Google Files API URIs;
- xAI accepts Files API IDs;
- OpenAI's Images edit endpoint requires file content, so the adapter raises an explicit `UserError` and recommends
  `BinaryImage` or `ImageUrl`.

Every adapter validates `provider_name`. The adapter consumes an already uploaded file; upload and delete lifecycle
remain owned by the caller and provider SDK. The demo shows that lifecycle, while adapter cassettes stay scoped to image
generation/editing and deterministic wire tests cover `UploadedFile` mapping.

## Provider Implementations

### OpenAI

- Uses the official async OpenAI Images client.
- Imports public SDK model and response types.
- Uses concrete multipart tuples rather than the private SDK `FileTypes` alias.
- Routes generate/edit based on reference-image presence.
- Preserves revised prompts, output metadata, response ID, and multimodal usage details.
- Rejects provider file IDs because the endpoint does not accept them.
- Rejects transparent backgrounds on GPT Image 2 and transparent JPEG output before sending a paid request.
- Requires Base64 image output and does not add an implicit output-URL download path.

### Google

- Uses the existing official `google-genai` client from `GoogleProvider`.
- Calls the typed `client.aio.models.generate_content` surface.
- Uses public `GenerateContentConfigDict`, `ImageConfigDict`, `PartDict`, and response types.
- Preserves reference-image `media_resolution` metadata.
- Requests image-only output to match the public `ImageGenerator` result contract and avoid billed text that the result
  cannot expose.
- Filters thought parts and keeps finish and safety metadata without duplicating image bytes.

`generate_content` is an intentional compatibility choice. Interactions is the forward-looking Gemini API, but the
repository's supported SDK surface and Google Cloud path do not yet provide the same public typing and support shape.
The mapping is isolated inside the adapter so a later transport migration need not change the core API.

The shorthand `google:` path is documented for the Gemini Developer API. Google Cloud is not advertised by the direct
API until direct generation and editing are recorded with Cloud credentials and its mixed-modality requirement has an
explicit transport policy.

### xAI

- Uses the official xAI SDK image client over its native transport.
- Raises the SDK floor to `xai-sdk>=1.16.0`, the first version with image file-ID inputs.
- Reuses public SDK model, aspect-ratio, resolution, response, and usage types.
- Supports single and batch generation, remote/inline references, and xAI file IDs.
- Forces Base64 transport internally to produce the common local `BinaryImage` result.
- Preserves moderation metadata and provider-reported cost without converting it into synthetic tokens.

The xAI SDK groups file IDs and URL/data references separately. If a mixed input order would be changed by the SDK, the
adapter raises instead of silently violating the common ordered-input contract.

## Normalized Settings

The normalized surface contains portable user intent plus compatibility fields already present on the capability:

- `n`;
- `output_format`;
- `background`;
- `input_fidelity`;
- `moderation`;
- `output_compression`;
- `quality`;
- `size`;
- `dimensions`;
- `aspect_ratio`;
- `extra_headers` and `extra_body` escape hatches.

Support is best-effort. Unsupported explicit values produce one aggregated `UserWarning` per request. Provider-prefixed
settings remain the typed escape hatch, take precedence over normalized equivalents, and make conflicts visible through
the same warning mechanism. Provider-specific fields use public SDK types whenever the SDK exposes them.

`action` is absent because the presence of reference images determines generate/edit. `image_model` is absent because
the direct model is already selected by `ImageGenerator` or `local='provider:model'`.

## Geometry: Compatibility Plus Two Cross-Provider Controls

Geometry is the most opinionated part of the proposal.

### Compatibility field: `size`

`size: str` is retained because the existing capability already exposes a compact `size` argument and GPT Image 2
requires arbitrary strings. It remains explicitly provider-dependent:

- OpenAI: pixel dimensions or `auto`;
- Google: resolution tiers such as `512`, `1K`, `2K`, and `4K`;
- xAI: `1K` and `2K` tiers.

It is not presented as a cross-provider resolution abstraction.

### Exact control: `dimensions`

`dimensions=(width, height)` means exact pixels and nothing else. It never rounds or selects a nearby shape. The
selected model must be known to support the exact dimensions or the adapter raises `UserError`.

This gives applications one portable way to require an exact output shape while keeping model-specific constraints and
tables inside provider helpers.

### Intent control: `aspect_ratio`

`aspect_ratio` expresses shape without choosing an output pixel count. Each model family maps it to a documented or
verified canonical geometry:

- GPT Image 1.x maps only its three fixed shapes;
- GPT Image 2 uses valid exact-ratio dimensions around one megapixel;
- Gemini uses fixed output tables and a canonical `1K` tier for current Gemini 3 families;
- xAI uses the documented `1k` tier and dimensions verified by real calls.

`dimensions` is mutually exclusive with `size` and `aspect_ratio`. `size` and `aspect_ratio` may be combined when they
are compatible.

### Where the geometry data lives

Each provider has a private geometry helper:

- `_openai_geometry.py`;
- `_google_geometry.py`;
- `_xai_geometry.py`.

These helpers own model-family identification, supported dimensions, canonical ratio mapping, precedence against
provider-specific settings, and ignored/conflicting setting collection. The capability contains no provider branches.

xAI does not publish the complete pixel table. The implementation is backed by 52 real generations covering all 13
documented aspect ratios at `1k` and `2k` on both current Grok Imagine models. Deterministic tests retain all 52 mappings;
the repository does not store 52 additional generated images.

## Relationship with Existing Agent Image Generation

Three surfaces remain intentionally distinct:

| Surface | Responsibility |
| --- | --- |
| `ImageGenerator` | Application-controlled direct generation and editing, including multiple outputs and references. |
| `ImageGeneration` capability | Lets an agent choose when to generate, preferring native execution and using a direct fallback when needed. |
| `ImageGenerationTool` | Provider-native tool configured on a conversational model. |

### Direct fallback is the recommended bridge

`ImageGeneration(local='openai:gpt-image-1.5')` or an explicit `ImageGenerator` now creates a private local tool backed
by the direct API. It avoids the extra agent loop used by the historical fallback and forwards normalized capability
settings to the selected provider adapter. The tool forces `n=1` because the capability contract returns one
`BinaryImage`; a multi-image result is rejected rather than truncated.

### Backward compatibility

`fallback_model`, custom callables/tools, the native path, and `ImageGenerationTool` remain available. No deprecation
warning is introduced before maintainer agreement.

The direct API adds `dimensions`, arbitrary GPT Image 2 sizes, and ratios outside the historical native vocabulary.
Those values are not added to `ImageGenerationTool` or the conversational model implementations in this PR. If supplied
to the native or `fallback_model` path, they are removed with a migration warning recommending the direct fallback.

This is deliberate: adding the new geometry to the historical path would create two public contracts and require
unrelated changes in `models/openai.py`, `models/google.py`, native tool schemas, and conversational model tests.

### Runtime objects and Agent Spec

The Python constructor accepts runtime `ImageGenerator`, `ImageGenerationModel`, `Tool`, and callable objects. Agent
specs cannot serialize those objects, so `ImageGeneration.from_spec()` exposes the same parameter names but restricts
`local` to a serializable direct model name, `False`, or `None`. A test enforces parameter-name parity and snapshots the
schema. JSON and YAML decode `dimensions` as a two-item list, so `from_spec()` normalizes it to the tuple required by the
runtime settings contract. This follows the existing `MCP.from_spec()` separation and avoids a one-off JSON-schema
suppression.

## Instrumentation and Privacy

The Pydantic AI wrapper creates an `image_generation <model>` span with:

- `gen_ai.output.type='image'`;
- provider/model identity and server address;
- prompt length and input-image count;
- usage, response ID, image count, media type, size, format, quality, and background when available;
- prompt only when `include_content` is enabled;
- normalized settings only when `include_model_request_parameters` is enabled.

Pydantic AI spans never include input bytes, Base64, data URLs, remote URLs, provider file IDs, or generated bytes, even
when binary-content instrumentation is enabled.

Provider SDKs may create independent spans. In particular, xAI's SDK can record a large Base64 image attribute. Its
controls are global environment variables initialized by the SDK and already affect xAI chat calls. This PR does not
mutate application environment variables or selectively suppress third-party spans. Maintainer confirmation is needed
on whether the no-binary guarantee should be scoped to Pydantic AI spans or expanded into a cross-provider SDK policy.

## Usage and Pricing

Adapters preserve provider usage and per-modality details where available. xAI's provider-reported cost remains in
`provider_details`; it is not converted into a normalized token or public cost field.

`ImageGenerationResult.cost()` currently raises `LookupError`. `genai-prices` cannot yet represent the combination of
image input/output token tariffs and per-image/per-resolution pricing correctly. Keeping a plausible but wrong number was
considered more harmful than an explicit unavailable result. The implementation links:

- [pydantic/genai-prices#351](https://github.com/pydantic/genai-prices/pull/351);
- [pydantic/genai-prices#185](https://github.com/pydantic/genai-prices/issues/185);
- [pydantic/genai-prices#410](https://github.com/pydantic/genai-prices/issues/410).

Instrumentation still records usage and token metrics, but omits `operation.cost` and the cost metric. Logfire can
currently display its own estimate from usage attributes; correcting that estimate is a separate presentation/pricing
decision rather than an adapter-specific workaround.

## Testing Evidence

The implementation includes:

- core lifecycle, sync/async, inference, override, settings precedence, validation, result, and instrumentation tests;
- provider wire tests for request mapping, settings, errors, usage, metadata, and response normalization;
- exhaustive deterministic geometry tables and boundary tests;
- capability routing, direct/compatibility behavior, warning, schema, and output-cardinality tests;
- downloader tests at the shared infrastructure level rather than duplicated in each adapter;
- six real generation/edit recordings across OpenAI, Google, and xAI, replayable without credentials;
- one GPT Image 2 exact-dimensions recording;
- strict 100% coverage for shipped runtime code, with the manual paid demo locally excluded by coverage pragmas;
- successful repository CI at the final Phase 9 checkpoint.

The recordings deliberately separate adapter behavior from Files API upload/delete lifecycle. Wire tests cover the
`UploadedFile` mapping and the demo proves the full provider lifecycle.

## Documentation and Contributor Workflow

The final documentation is organized around one canonical user guide:

- `docs/image-generation.md`: direct generation/editing, settings, geometry, results, instrumentation, testing, and the
  decision table for the three public surfaces; its geometry section shows both the canonical exact dimensions selected
  by every supported `aspect_ratio` and the complete family-level rules for explicit `dimensions`;
- `docs/capabilities/image-generation.md`: agent routing and compatibility behavior;
- `docs/native-tools.md`: native tool behavior and cross-links;
- provider pages: setup and provider-specific limitations;
- `docs/api/images.md`: generated API reference.

The repository `add-new-model` skill has an image-model section so future model additions update the dedicated known
list, private geometry helper, support matrix, and targeted tests without automatically widening conversational/native
surfaces.

## Main Decisions to Confirm

The implementation is ready for concrete review, but these choices should be explicitly accepted or corrected:

1. Is `ImageGenerator` the desired direct primitive and `ImageGeneration.local` the right capability bridge?
2. Should the direct fallback become the recommended path while `fallback_model` remains compatible but not yet
   deprecated?
3. Are normalized settings with aggregated best-effort warnings preferable to strict provider capability errors?
4. Should `size` remain as a provider-dependent compatibility field alongside the new exact `dimensions` and canonical
   `aspect_ratio` controls?
5. Is it acceptable for Pydantic AI to own and maintain the model-specific geometry tables?
6. Are the canonical GPT Image 2 and Gemini mappings, and the measured xAI table, acceptable?
7. Should direct-only geometry remain outside `ImageGenerationTool` and conversational model implementations?
8. Is `generate_content` the right typed cross-flavor Google transport for now, despite Interactions being the future
   direction?
9. Is `xai-sdk>=1.16.0` an acceptable dependency floor, and should mixed inputs that the SDK would reorder be rejected?
10. Should `ImageGenerationResult.cost()` remain present but unavailable until `genai-prices` supports image units?
11. Does the no-binary guarantee cover Pydantic AI spans only, or must Pydantic AI also control autonomous provider SDK
    telemetry?
12. Should Google Cloud remain unadvertised until direct Cloud recordings are available?

## How to Review the Rationale

The companion `implementation_decisions.md` is an English rendition of the complete working decision register. It
records evidence, alternatives, superseded approaches, tests, and unresolved questions rather than presenting the
current code as necessarily correct. If any implementation choice looks surprising, that document is intended as the
fastest way for a maintainer or an agent reviewing on their behalf to reconstruct why it was made and propose a better
alternative with the same evidence in view.

## Temporary Review Assets

Keep these files only through maintainer alignment:

- `pydantic_ai/images/_demo.py` — paid, manually executed end-to-end comparison with extensive comments;
- `pydantic_ai/images/temp_pr_docs/SOTA_study.md` — initial provider/API research;
- `pydantic_ai/images/temp_pr_docs/resolution_mapping_proposal.md` — geometry research and exact mapping evidence;
- `pydantic_ai/images/temp_pr_docs/implementation_decisions.md` — translated decision register;
- this presentation.

After the maintainers have resolved the open decisions, remove the demo and `temp_pr_docs` in a dedicated cleanup commit,
leaving only durable public docs, docstrings, tests, and source comments.
