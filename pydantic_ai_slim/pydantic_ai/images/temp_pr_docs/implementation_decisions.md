# Implementation Decision Register

## Purpose

This is the English review edition of `test_tmp/phase2/decisioni_implementative.md`, the working register maintained while
implementing the direct Image Generation API. It is intentionally detailed. It separates verified facts, maintainer
guidance, working choices, rejected alternatives, and questions that still need maintainer confirmation.

It is not a public specification and does not assert that every choice is correct. Its purpose is to make the current
implementation discussable without losing the evidence and trade-offs that led to it. Superseded entries are retained
because they explain why the implementation changed.

## References

- PR: [pydantic/pydantic-ai#5357](https://github.com/pydantic/pydantic-ai/pull/5357)
- Issue: [pydantic/pydantic-ai#3898](https://github.com/pydantic/pydantic-ai/issues/3898)
- Full implementation presentation: `work_presentation.md`
- Provider research: `SOTA_study.md`
- Geometry research: `resolution_mapping_proposal.md`

## Status Vocabulary

- **Confirmed by maintainers**: explicitly requested or accepted in the recorded maintainer discussion.
- **Verified fact**: demonstrated by official documentation, public SDK types, real requests, or repository tests.
- **Working choice**: selected to make progress and kept reversible.
- **Needs maintainer validation**: public or architectural choice to review on the concrete implementation.
- **Superseded**: retained for history but replaced by a later decision.

## Current Decision Gates

| Gate | Topic | Current outcome |
| --- | --- | --- |
| D1 | `generate()` input | One `prompt: str` plus ordered `images: Sequence[ImageGenerationInput] | None`. |
| D2 | Interleaved input | Not exposed as portable; image order is preserved. |
| D3 | `UploadedFile` | Supported by Google and xAI; explicit error on OpenAI. |
| D4 | Google transport | Official `google-genai` SDK with typed `generate_content`; revisit Interactions with maintainers. |
| D5 | xAI transport | Official SDK image client; minimum `xai-sdk>=1.16.0`. |
| D6 | Common settings | Best-effort normalized settings plus exact `dimensions`, canonical `aspect_ratio`, and provider-dependent `size`. |
| D7 | Capability bridge | Direct image model through `local`; capability contains no provider mapping. |
| D8 | Portable cost | Public method retained but raises `LookupError` pending correct `genai-prices` support. |
| D9 | xAI cost | Provider-reported value remains in `provider_details`. |
| D10 | Binary telemetry | No image payloads or references in Pydantic AI spans. |
| D11 | Mask and streaming | Outside this PR. |
| D12 | Deprecation | No immediate deprecation; direct fallback is recommended while compatibility remains. |

## Decisions

### DEC-000: Implement before another abstract maintainer round

- **Date / status:** 2026-07-18; superseded procedurally by DEC-014.
- **Decision:** do not send another pre-implementation message. Produce concrete signatures, payloads, tests, and a
  reversible decision register, then review the opinionated parts with David.
- **Reason:** the existing transcript already authorized iteration, and concrete behavior is easier to assess than an
  abstract proposal.
- **Rejected:** blocking implementation on approval of every D1-D7 choice.

### DEC-001: Use SDK types for provider-specific settings

- **Status:** confirmed by maintainers.
- **Decision:** import public provider SDK aliases and `TypedDict` types for provider-specific surfaces. Define Pydantic
  AI types only for standardized cross-provider concepts or when no public SDK type exists, documenting why.
- **Boundary:** do not expose an SDK's whole request object when prompt, model, stream, or response format are owned by
  Pydantic AI.
- **Regression evidence:** provider settings typecheck and exact SDK request mapping tests.

### DEC-002: Pricing must not block the API, but must not be invented

- **Status:** principle confirmed; concrete behavior implemented and still needs validation.
- **Decision:** preserve `ImageGenerationResult.cost()` but make it raise `LookupError` until image-token and per-image
  prices are represented correctly. Preserve all usage and provider-reported cost data.
- **Evidence:** current `genai-prices` data misses or incorrectly classifies OpenAI image prices, falsely matches the
  Google image model to a text entry, and has no Grok Imagine per-image pricing. PR `genai-prices#351` is the intended
  data-driven foundation; issues `#185` and `#410` cover image and non-token units.
- **Rejected:** local price tables, best-effort false matches, blocking the feature, or deleting the method entirely.

### DEC-003: Keep xAI's reported cost in provider details

- **Status:** working choice; needs maintainer validation.
- **Decision:** retain xAI `cost_in_usd_ticks` / `cost_usd` in `provider_details` until there is a normalized field with
  shared units and precision.
- **Reason:** avoids synthetic tokens and avoids stabilizing a core field for one provider prematurely.

### DEC-004: No binary content in Pydantic AI traces

- **Status:** output rule confirmed by maintainers; extension to inputs is a working choice.
- **Decision:** never record bytes, Base64, data URLs, remote/signed URLs, or provider file IDs for input or output.
  Record counts, media type, and safe metadata only.
- **Reason:** input images have the same privacy, size, retention, and telemetry-cost risks as outputs.
- **Scope clarification:** DEC-041 separates autonomous provider SDK spans from Pydantic AI spans.

### DEC-005: Masks and streaming are out of scope

- **Status:** working choice.
- **Decision:** cover non-streaming generation and editing through reference images. Leave masks, partial image events,
  and streaming to follow-ups.
- **Reason:** they require additional request, result, and event models but are not necessary to validate the common
  primitive.

### DEC-006: No immediate deprecation of the existing capability

- **Status:** working choice; needs validation.
- **Decision:** preserve `fallback_model`, `ImageGenerationTool`, and current capability fields. Discuss v3 deprecation
  separately after the direct bridge is accepted.
- **Evidence:** the direct path can be added without breaking existing users; tests keep the historical behavior intact.

### DEC-007: Do not promise portable interleaved text/image input

- **Status:** verified fact plus working choice.
- **Decision:** expose one prompt and an ordered image sequence, not `Sequence[str | ImageGenerationInput]`.
- **Evidence:** Google supports ordered text/image parts; OpenAI Images and xAI Images accept one prompt separately from
  images. They preserve image order but cannot preserve multiple interleaved text fragments.
- **Rejected:** silently concatenating text fragments for OpenAI and xAI.

### DEC-008: `UploadedFile` belongs in the common input with declared limitations

- **Status:** verified and implemented; needs validation.
- **Decision:** accept `UploadedFile` because Google and xAI preserve its provider-hosted-file semantics. Always validate
  `provider_name`; raise on OpenAI rather than treating an ID as a URL or downloading implicitly.
- **Evidence:** Google accepts Files API/GCS URIs, xAI edits accept file IDs, and OpenAI image edit accepts file content.

### DEC-009: Concrete `generate()` signature

- **Status:** implemented; needs validation.
- **Decision:** `ImageGenerationInput = ImageUrl | BinaryImage | UploadedFile`; all surfaces use
  `generate(prompt, *, images=None, settings=None)` and an equivalent sync wrapper.
- **Invariants:** non-empty prompt, image-only inputs, preserved sequence, prompt retained in the result, and no duplicate
  binary inputs in the result.
- **Rejected:** broad `UserContent`, which contains unrelated media and marker types.

### DEC-010: Start Google on `generate_content`

- **Status:** evidence refreshed by DEC-030; technical choice unchanged.
- **Decision:** use `client.aio.models.generate_content`, isolating request and response conversion for a future local
  migration to Interactions.
- **Reason:** it is typed, supported by the repository's minimum SDK, works with the existing Google providers, and the
  stateless direct primitive does not use Interactions-specific state/background features.

### DEC-011: Initial xAI REST design

- **Status:** superseded by DEC-037.
- **Initial decision:** use xAI's documented JSON Images endpoints through `httpx` because the first SDK inspection did
  not find Images on the top-level async client.
- **Why retained:** it records the evidence error. Later inspection found the runtime `image` sub-client and DEC-037
  replaced the entire REST design with the official SDK.

### DEC-012: Initial settings aligned to `ImageGenerationTool`

- **Status:** superseded by DEC-015, DEC-016, DEC-049, and DEC-054.
- **Initial decision:** copy `n`, `size`, `aspect_ratio`, `output_format`, `output_compression`, `extra_headers`, and
  `extra_body`, failing on unsupported settings.
- **Why superseded:** real provider semantics showed that copying the historical surface was not sufficient and that the
  project convention favors best-effort normalized settings plus provider escape hatches.

### DEC-013: Initial separate `fallback_generator` argument

- **Status:** superseded by DEC-017 and the final DEC-044/049 design.
- **Initial decision:** keep `fallback_model` for the conversational subagent and add a distinct direct-generator
  argument.
- **Why superseded:** `ImageGeneration.local` already owns the non-native implementation role, so another public
  fallback argument would duplicate the capability abstraction.

### DEC-014: Review decisions with the user, then implement incrementally

- **Status:** working process decision.
- **Decision:** resolve public/architectural choices step by step with the contributor, without another preventive
  maintainer message. Keep every opinionated choice and its rationale for the concrete review.
- **Reason:** an early partial implementation exposed dependencies on recently merged repository conventions, making
  explicit gates safer than letting API shape emerge accidentally.

### DEC-015: Decide settings convergence last

- **Status:** completed by DEC-046/047 and ultimately DEC-049/054.
- **Decision:** first implement core and typed provider adapters, then compare real semantics before converging the
  direct API, capability, and native tool.
- **Reason:** `ImageGenerationTool` is a consistency constraint, not proof that its existing field set is the correct new
  abstraction.

### DEC-016: Unsupported normalized settings use best-effort warnings

- **Status:** implemented.
- **Decision:** omit valid but unsupported normalized settings and warn once. Provider-specific settings win conflicts
  and warn. Reserve `UserError` for structurally invalid or impossible input.
- **Evidence:** this follows the repository's provider capability-gating convention and preserves configuration
  portability without silent behavior.

### DEC-017: Use `local` for the direct capability bridge

- **Status:** superseded in concrete form by DEC-025 and DEC-044, principle retained.
- **Decision:** do not add `fallback_generator`; integrate the direct generator through `local`. Do not route image-only
  models through conversational model-selection hooks.
- **Reason:** this preserves the existing native/local lifecycle and keeps conversational and image model contracts
  distinct.

### DEC-018: Normalize usage through existing `genai-prices` extraction

- **Status:** verified; cost behavior updated by DEC-042.
- **Decision:** use `RequestUsage.extract()` when compatible, retain per-modality details, map xAI protobuf usage, and
  keep provider cost in details. Do not synthesize image pricing.
- **Reason:** reuse the project-maintained normalization path and retain the data a future unit registry will consume.

### DEC-019: Use the current OpenAI SDK's public image types

- **Status:** verified and maintainer-aligned.
- **Decision:** target the repository's `openai>=2.45.0` floor and remove casts/workarounds that current public SDK types
  make unnecessary.
- **Evidence:** provider adapter Pyright and request mapping tests.

### DEC-020: Preserve Google reference-image metadata

- **Status:** verified and implemented.
- **Decision:** carry `vendor_metadata['media_resolution']` onto Google image parts for binary, URL, and uploaded inputs.
- **Reason:** the direct API should not lose a known quality/configuration signal already preserved by the conversational
  Google adapter.

### DEC-021: Reuse shared instrumentation flags

- **Status:** implemented and completed by DEC-057.
- **Decision:** prompt follows `include_content`; request settings follow `include_model_request_parameters`; no image
  payload or reference is recorded. Do not add `include_image_content`.
- **Reason:** one instrumentation configuration should govern all Pydantic AI surfaces.

### DEC-022: First checkpoint limited to the core contract

- **Status:** completed and superseded operationally.
- **Decision:** initially propagate and validate `images` across core/wrappers/instrumentation while OpenAI explicitly
  rejected references until its edit routing landed.
- **Reason:** prevented a partially updated adapter from silently discarding user inputs. The temporary guard was later
  replaced by real generate/edit dispatch.

### DEC-023: Update the demo with every user-visible checkpoint

- **Status:** implemented; final disposition still needs review.
- **Decision:** keep one functioning end-to-end demo current with each provider, editing, Files API, capability, and
  instrumentation change. Generation uses the agreed cat prompt; mutation replaces the cat with a dog.
- **Current outcome:** four separately selectable sections, one async workflow, minimal stdout, and Logfire telemetry.

### DEC-024: Close core tests independently from provider phases

- **Status:** completed.
- **Decision:** treat provider inference/match as adapter responsibilities and close the core phase with lifecycle,
  override, deferred inference, sync/async propagation, and global instrumentation tests.
- **Reason:** avoids fake provider branches merely to satisfy phase ordering.

### DEC-025: Initial public `common_tools` helper bridge

- **Status:** superseded by DEC-044.
- **Initial decision:** expose `image_generator_tool(...)` that converts a direct generator into the local capability
  tool.
- **Why superseded:** accepting `ImageGenerator` / `ImageGenerationModel` directly in `local` is clearer and avoids a
  public helper almost duplicating the existing historical `image_generation_tool` name.

### DEC-026: Add provider types with each adapter

- **Status:** implemented; xAI part corrected by DEC-037.
- **Decision:** do not predefine unused request types. OpenAI, Google, and xAI types enter with their exercised adapter;
  xAI ultimately needs no local REST request `TypedDict` because the official SDK types exist.
- **Reason:** keeps every provider checkpoint typechecked and reviewable without speculative abstractions.

### DEC-027: Do not import OpenAI's private `FileTypes`

- **Status:** implemented; needs validation.
- **Decision:** type the multipart values as their concrete public tuple shape `(filename, bytes, content_type)` and let
  the public SDK method check them structurally.
- **Reason:** follows the request to reuse SDK types without binding Pydantic AI to a private alias or copying its entire
  union.

### DEC-028: OpenAI routing follows reference-image presence

- **Status:** implemented; needs validation.
- **Decision:** no references calls `images.generate`; references call `images.edit` in order. Endpoint-only settings are
  omitted on the other endpoint and warned as needed.
- **Reason:** matches the common API without introducing a separate editor or ignoring references.

### DEC-029: Record one real generation/edit path per provider phase

- **Status:** implemented for all providers; needs validation.
- **Decision:** pair precise deterministic wire tests with representative real recordings. Do not rely on the manual
  demo or VCR alone for multipart/protobuf correctness.
- **Evidence:** OpenAI, Google, and xAI generation/edit cassettes replay without credentials; large image payloads remain
  only where needed for real response parsing.

### DEC-030: Keep typed `generate_content` after Interactions GA

- **Status:** verified working choice; needs maintainer validation.
- **Facts:** Interactions is GA and recommended for new Gemini Developer API projects, but requires a newer SDK and still
  exposes weaker public request typing; Google Cloud Interactions remains experimental. `generate_content` stays fully
  supported and typed across the repository's existing client/provider shape.
- **Decision:** do not bump the SDK or narrow to Developer API merely to adopt Interactions. Isolate the adapter so this
  transport choice can change later.
- **Open question:** prefer typed cross-flavor compatibility now, or a newer untyped Developer-only transport?

### DEC-031: Advertise Google Cloud only after direct evidence

- **Status:** Cloud gate retained; shared mixed-modality request superseded by DEC-061.
- **Decision:** share a transport-compatible implementation but do not advertise Google Cloud for the direct API until
  direct generation and reference editing have their own Cloud recordings.
- **Evidence:** existing conversational-model recordings and SDK probes establish prior art but do not exercise the new
  adapter contract. Cloud requires mixed `TEXT`/`IMAGE` response modalities, while the Developer API can return images
  only.
- **Rejected:** claiming Cloud support from mocks or from unrelated conversational cassettes alone.

### DEC-032: Use current Google image model IDs for new recordings

- **Status:** verified working choice.
- **Decision:** add only current GA image IDs needed by the direct API and do not create new recordings against retired
  preview IDs. Keep any repository-wide cleanup of older conversational recordings separate if it expands scope.
- **Evidence:** provider model listing, release notes, deprecation docs, and current model pages.
- **Open question:** whether maintainers want the broader stale-preview cleanup in this PR.

### DEC-033: Initial Google checkpoint on Gemini 2.5 Flash Image

- **Status:** superseded by DEC-034/035.
- **Initial decision:** use `gemini-2.5-flash-image` as a familiar inexpensive first implementation target, separating
  API validation from global model-list cleanup.
- **Outcome:** an initial reference request reached the correct backend but quota was zero. After billing was enabled,
  current Gemini 3.1 models became the definitive live evidence.

### DEC-034: Separate known model support from demo/recording selection

- **Status:** implemented and refined by DEC-035.
- **Decision:** the known list describes supported current model IDs; the demo and recordings independently choose a
  low-cost representative. Do not reduce public support merely because one model is cheaper to call repeatedly.
- **Evidence:** the live model endpoint exposed Gemini 2.5 Flash Image, Gemini 3.1 Flash Lite Image, Gemini 3.1 Flash
  Image, and Gemini 3 Pro Image.

### DEC-035: Use Gemini 3.1 Flash Lite Image for demo and recordings

- **Status:** contributor choice, implemented and verified.
- **Decision:** use `gemini-3.1-flash-lite-image` for generation and reference editing while retaining all supported
  current IDs in the known list.
- **Reason/evidence:** it is the cheapest current representative supporting both operations. Real calls and offline
  recordings proved generation, Files upload, edit, and cleanup.

### DEC-036: Keep Files API lifecycle out of Google adapter cassettes

- **Status:** implemented and verified.
- **Decision:** the real edit cassette passes `BinaryImage`; deterministic wire tests cover `UploadedFile`; the manual
  demo owns upload, provider-file edit, and delete.
- **Reason:** upload/delete are SDK Files responsibilities, not adapter behavior. Recording them required an unrelated
  VCR serializer change for binary data incorrectly marked JSON, so that serializer change was removed.

### DEC-037: Use the official xAI image client and require SDK 1.16+

- **Status:** implemented and live-verified; needs validation.
- **Correction:** the xAI SDK already had a runtime `image` sub-client. Version 1.16.0 is the first with image file-ID
  arguments needed for the common `UploadedFile` contract.
- **Decision:** replace the REST design completely with the official gRPC image client; use public SDK model, ratio,
  resolution, response, and usage types; do not modify `XaiProvider` or add `httpx`.
- **Other behavior:** force Base64 transport for normalized local bytes; preserve moderation and cost details; reject
  mixed inputs whose order the SDK would change; include current standard/quality models but not the retired Pro model.
- **Evidence:** live generation/edit/upload/delete, protobuf cassette replay, provider regression suite, lock/type checks.

### DEC-038: Keep xAI Files lifecycle separate and avoid duplicate cassette payloads

- **Status:** implemented; needs validation as a shared testing strategy.
- **Decision:** xAI recordings cover generation and binary-reference editing; wire tests cover file IDs; the demo proves
  upload/delete. Redact images only from diagnostic JSON while retaining the lossless protobuf response used for replay.
- **Reason:** avoids two copies of the same Base64 image and keeps adapter recordings scoped to adapter behavior.

### DEC-039: Demonstrate sync independently from the async xAI Files workflow

- **Status:** implemented.
- **Decision:** keep the runtime adapter and end-to-end Files demo async with one `AsyncClient`; use the real generation
  cassette to prove `ImageGenerator.generate_sync()` separately.
- **Reason:** Pydantic AI sync methods wrap individual operations; they are not facades over provider-specific Files
  clients. A second synchronous xAI client would duplicate configuration and diverge from repository examples.

### DEC-040: Use one async workflow for the multi-provider demo

- **Status:** implemented.
- **Decision:** one `asyncio.run()` drives embedding, agent, OpenAI, Google, xAI, Files operations, and capability
  sections. Sync remains independently tested.
- **Reason:** one event loop avoids provider-specific style differences and duplicated clients, and matches existing
  `UploadedFile` examples.

### DEC-041: Do not suppress autonomous xAI SDK spans from the adapter

- **Status:** explicit contributor decision after inspecting real traces; needs maintainer validation.
- **Facts:** all adapters create the same small Pydantic AI span. xAI additionally creates SDK spans, and its image span
  can contain hundreds of kilobytes of Base64 because the adapter requests Base64 output. The shared Pydantic AI
  instrumentation settings do not control third-party spans.
- **Decision:** do not mutate `XAI_SDK_DISABLE_TRACING` or `XAI_SDK_DISABLE_SENSITIVE_TELEMETRY_ATTRIBUTES`. These are
  process-global, initialization-sensitive controls also affecting xAI chat and concurrent calls.
- **Open question:** scope the no-binary guarantee to Pydantic AI spans, document the SDK flags, request a per-client xAI
  control, or establish a repository-wide provider-SDK policy.

### DEC-042: Disable portable cost until `genai-prices` supports image units

- **Status:** implemented contributor decision; needs validation.
- **Facts:** current OpenAI image entries are absent or use text-output rates, Google's Flash Lite image model falsely
  matches a text model prefix, and Grok Imagine per-image prices are absent. Usage breakdowns and xAI-reported prices are
  still preserved.
- **Decision:** retain the method and return type, link the upstream work in a TODO, and always raise `LookupError`.
  Instrumentation catches it and continues to emit usage without `operation.cost`.
- **Rejected:** a provider allowlist, local price tables, zero cost, removing image spans, or plausible-but-wrong
  best-effort numbers.

### DEC-043: Preserve standard usage; treat Logfire's UI estimate separately

- **Status:** study and recommendation; no runtime suppression applied.
- **Observation:** Logfire can show a cost even when `operation.cost` is absent because its presentation layer estimates
  from standard token usage. Pydantic AI correctly retains provider totals and modality details.
- **Recommendation:** keep `gen_ai.usage.*`, detailed usage, token metrics, and `gen_ai.output.type='image'`; omit cost
  until complete. Logfire should avoid generic text-token pricing for image spans when modalities have different rates.
- **Reason:** deleting standard telemetry to influence one UI would violate OpenTelemetry semantics and hide useful data.

### DEC-044: Make the direct API the canonical capability fallback

- **Status:** direct-path principle retained; concrete settings behavior completed by DEC-049.
- **Decision:** accept `ImageGenerator`, `ImageGenerationModel`, and direct model names through `ImageGeneration.local`;
  create a private `generate_image` tool; force one image and reject extra results.
- **Routing:** native remains preferred unless `native=False`. `fallback_model` and custom tools remain compatible.
- **Reason:** removes an unnecessary subagent/model request, reuses the provider-agnostic primitive, and avoids a new
  public helper.
- **Spec:** Python runtime objects stay constructor-only; direct model strings are serializable via `from_spec()`.

### DEC-045: Separate subagent deprecation from native-tool configuration

- **Status:** deprecation strategy still open; earlier `size` conclusion superseded by DEC-049/054.
- **Decision:** recommend the direct fallback now, but do not emit a deprecation warning for `fallback_model` without
  maintainer agreement. Do not deprecate `ImageGenerationTool`, the capability, or native settings that still have no
  replacement on the conversational path.
- **Reason:** the direct API replaces the extra subagent, not the provider-native tool.

### DEC-046: Interim minimal common settings (`n` and `output_format`)

- **Status:** superseded by DEC-049.
- **Initial decision:** keep only `n` and `output_format` portable; leave compression OpenAI-specific because Google/xAI
  direct APIs have no equivalent.
- **Value retained:** `n=1` remains enforced by the capability tool, and unsupported normalized fields never become inert
  silent payload.

### DEC-047: Interim rejection of common `size`; separate ratio from exact resolution

- **Status:** superseded by DEC-049/054.
- **Initial decision:** do not copy ambiguous historical `size`; consider `aspect_ratio` separately; defer exact pixels
  until model tables and xAI output were verified.
- **Value retained:** exact pixels, shape intent, and provider tiers are distinct concepts. DEC-054 implements that
  separation while retaining `size` only as explicit compatibility.

### DEC-048: Separate runtime API, serializable spec, and settings ownership

- **Status:** runtime/spec separation retained; concrete mapping superseded by DEC-049.
- **Decision:** use `ImageGeneration.from_spec()` rather than a repository-unique `SkipJsonSchema`. Keep reusable direct
  defaults/provider settings on `ImageGenerator` and capability-level normalized overrides on the individual agent use.
- **Evidence:** `MCP.from_spec()` is the existing precedent; tests enforce signature-name parity and schema stability.

### DEC-049: Forward normalized settings and let providers map them

- **Status:** implemented and extended by DEC-054.
- **Decision:** expose the compact capability intents in `ImageGenerationSettings`; `local='provider:model'` is accepted;
  the capability sends all non-`None` normalized values plus `n=1`; provider adapters own translation, ignored values,
  conflicts, and one aggregated warning.
- **Exclusions:** `action` derives from reference presence and cannot be honored by the prompt-only capability tool when
  forced to edit; `image_model` duplicates the model selected by `local` and warns.
- **Precedence:** explicit provider settings on a model/generator win over normalized values.
- **Reason:** keeps the capability readable without embedding OpenAI/Google/xAI branches and preserves typed escape
  hatches.

### DEC-050: Uniform provider flow and separate capability demo section

- **Status:** implemented internal refactor.
- **Decision:** every adapter reads as prepare → resolve settings → warn → map inputs → SDK call → map response, while
  retaining provider-local resolved dataclasses/helpers. The demo has independent OpenAI, Google, xAI, and capability
  sections selected by `--provider` and `--capability`.
- **Reason:** consistent review shape without pretending wire contracts are identical or triggering hidden paid calls.

### DEC-051: Let Logfire carry demo telemetry

- **Status:** implemented, demo-local decision.
- **Decision:** stdout shows only section headings and saved artifact paths. Model, usage, costs, provider details,
  embedding size, and agent output remain in Logfire.
- **Reason:** avoids a second incomplete telemetry representation and avoids emphasizing intentionally unavailable cost.

### DEC-052: Collect normalized capability settings once

- **Status:** implemented internal refactor.
- **Decision:** `_image_settings()` is the single collection point. Native adds only `action`/`image_model`; direct adds
  only `n=1`. Keep the deliberate `from_spec()` signature duplication.
- **Reason:** prevents drift between native and direct forwarding while keeping their semantic differences explicit.

### DEC-053: Recognize GPT Image 2's wider size contract

- **Status:** completed by DEC-054 and narrowed by DEC-055.
- **Facts:** GPT Image 2 accepts arbitrary valid `WIDTHxHEIGHT` strings with multiple-of-16, edge, area, and ratio bounds;
  the SDK types size as `str`. It does not expose a separate aspect-ratio field.
- **Direction:** compare ratios mathematically and support valid direct sizes without constraining them to the GPT Image
  1.x allowlist. Do not add the dated snapshot to the public known list.

### DEC-054: Separate exact dimensions, canonical ratio, and compatibility size

- **Status:** implemented for the direct API; needs maintainer validation.
- **Decision:** add exact `dimensions: tuple[int, int]`; map `aspect_ratio` to a model-specific canonical geometry; widen
  direct `size` to `str` while documenting its provider-dependent meaning.
- **Model evidence:** GPT Image 1.x fixed shapes; GPT Image 2 documented numeric bounds; Gemini documented ratio/tier
  tables including a non-uniform Flash `21:9` row; 52 live xAI generations covering 13 ratios × 2 tiers × 2 models.
- **Unknown models:** do not infer exact-dimension support from similar names. Strong exact promises require a known
  family; weaker native ratio pass-through may remain available.
- **Ownership:** private provider helpers own tables and constraints. Provider settings win conflicts with one warning.
- **Open questions:** public name/tuple shape, maintenance cost of tables, canonical GPT Image 2 dimensions, conservative
  Flash Lite mapping, and adequacy of the xAI measurement.

### DEC-055: Keep new geometry out of the historical native tool

- **Status:** implemented contributor choice; must be presented as intentional.
- **Decision:** do not change `ImageGenerationTool`, `models/openai.py`, or `models/google.py`. Direct geometry types live
  in `pydantic_ai.images`; the capability removes direct-only values from native/`fallback_model` calls and emits a
  migration warning.
- **Reason:** duplicating the direct geometry would expand native schemas and create two maintained contracts. The direct
  API is the intended convergence path; compatibility remains without pretending new support.
- **Organization:** xAI also receives a private geometry module for provider symmetry. Image-model contributor rules
  belong in the repository `add-new-model` skill, not generic end-user capability/native-tool skill references.

### DEC-056: Geometry resolution stays in provider helpers, not the base class

- **Status:** implemented internal design; needs validation with the public geometry choice.
- **Decision:** do not add an abstract geometry method to `ImageGenerationModel`. Keep common merge/validation in the
  base and use `resolve_<provider>_geometry()` privately in each adapter.
- **Reason:** an abstract return would have three incompatible wire shapes and would break wrappers, test models, and
  user-defined models that do not need provider geometry mapping.

### DEC-057: Image instrumentation uses shared flags and excludes payloads

- **Status:** implemented for Pydantic AI spans; SDK boundary needs validation.
- **Decision:** emit `gen_ai.output.type='image'`; prompt only with `include_content`; normalized settings only with
  `include_model_request_parameters`; always emit safe counts/identity/usage/metadata; never serialize image content or
  references.
- **Reason:** aligns with OpenTelemetry output modality and shared Pydantic AI instrumentation configuration without an
  image-specific flag.
- **Tests:** combined `ImageUrl`, `BinaryImage`, and `UploadedFile` inputs prove absence of URL, Base64, and file ID;
  disabled flags remove prompt/settings; `operation.cost` remains absent.

### DEC-058: Regression coverage follows behavior ownership

- **Status:** completed locally and in CI.
- **Decision:** each adapter tests dispatch, provider mapping, response normalization, and errors; the shared downloader
  tests network failures once. Do not duplicate downloader tests per provider or introduce implicit output URL downloads
  not implemented by the direct API.
- **Evidence:** image tests replay six real generation/edit recordings; capability and downloader suites cover bridge and
  shared infrastructure; coverage is 100% for shipped runtime code; the branch's required CI jobs passed.
- **Reason:** prove the real contract at the layer that owns each behavior without turning checklist history into scope
  expansion.

### DEC-059: Document ratios and dimensions as one geometry contract

- **Status:** implemented in documentation; validate together with DEC-054.
- **Problem:** the previous public matrix showed only ratio availability. It did not show the exact shape selected by
  `aspect_ratio` or the additional values accepted by `dimensions`, forcing users to inspect private helpers or probe.
- **Decision:** replace boolean ratio support with a ratio-to-canonical-dimensions matrix for every family. Add exact
  dimension rules covering GPT Image 1.x fixed values, GPT Image 2's constrained range, Gemini tiers and its Flash
  `21:9` exception, and the verified xAI `1k`/`2k` table.
- **Docstrings:** describe family-level possibilities and unsupported behavior on the aliases, settings, and capability,
  while linking to the canonical guide rather than copying every table into API reference output.
- **Ownership:** private provider geometry helpers remain executable truth; the guide is the human-facing truth; table
  tests expose drift. No new mapping is introduced by the documentation.
- **Reason:** `aspect_ratio` promises a canonical choice while `dimensions` promises exact pixels. Showing both makes the
  distinction usable without making every signature unmanageably long.
- **Validation:** documentation examples, Ruff, Pyright, MkDocs build, and generated anchor links pass.

### DEC-060: Normalize Agent Spec dimensions at the serialization boundary

- **Status:** implemented after edge-case review.
- **Problem:** JSON and YAML represent `(width, height)` as an array. `Agent.from_spec()` passed that decoded list through
  `ImageGeneration.from_spec()` unchanged, while the direct settings validator requires the public Python tuple shape.
- **Decision:** keep `ImageDimensions = tuple[int, int]` and the fixed two-item Agent Spec schema. Normalize decoded lists
  to tuples inside `ImageGeneration.from_spec()` before constructing the runtime capability.
- **Reason:** `from_spec()` owns the serialized/runtime boundary. Accepting lists throughout the direct image API would
  widen a public contract only to accommodate the wire representation of Agent Specs.
- **Validation:** a public `Agent.from_spec()` regression test supplies JSON/YAML-shaped `[1280, 720]` and asserts that
  the resulting capability stores `(1280, 720)` and can resolve its direct image toolset.

### DEC-061: Request image-only output from Google

- **Status:** implemented after edge-case review; supersedes the mixed-modality part of DEC-031.
- **Problem:** the adapter requested `TEXT` and `IMAGE` but normalized only image parts. Gemini could generate and bill
  text that callers could not read through `ImageGenerationResult`.
- **Decision:** send `response_modalities=['IMAGE']` for both generation and reference editing on the advertised Gemini
  Developer API adapter. Keep thought filtering and non-content response metadata unchanged.
- **Reason:** `ImageGenerator` promises generated images, not a mixed conversational response. Requesting only the public
  result type avoids hidden output and cost instead of adding Google text to provider metadata.
- **Cloud boundary:** Google Cloud remains unadvertised. Its mixed-modality requirement needs an explicit provider path
  and direct recordings before Cloud support can be claimed.
- **Validation:** the deterministic wire test asserts `responseModalities: ['IMAGE']`; both real Google recordings are
  re-recorded against the image-only request and replay without credentials.

## Questions to Resolve with Maintainers

The current implementation is strongest where provider contracts are verified and intentionally conservative. Review is
most valuable on public policy:

1. direct `local` fallback versus a separate public bridge;
2. no immediate `fallback_model` deprecation;
3. best-effort normalized settings and warning behavior;
4. `dimensions` / `aspect_ratio` / compatibility `size` and ownership of geometry tables;
5. keeping new geometry outside the historical native tool;
6. typed Google `generate_content` now versus Interactions and the timing of Cloud support;
7. the xAI SDK floor, ordered mixed-input rejection, and third-party telemetry boundary;
8. retaining an unavailable `cost()` method until upstream pricing units are complete.

The implementation presentation provides a shorter architecture-level review path. This register should be used when a
specific choice needs its evidence, rejected alternatives, or superseded history reconstructed.
