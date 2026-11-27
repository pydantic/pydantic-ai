---
description: "Use Google Gemini models with Pydantic AI through the Gemini API or Vertex AI, with video and file input, image generation, thinking, safety settings and the Decisions API."
---

# Google

The `GoogleModel` is a model that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to
access Google's Gemini models via both the Gemini API and Google Cloud (formerly known as Vertex AI).

Two providers wrap those endpoints:

- [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] — the Gemini API (Google AI Studio), surfaced under the `'google:'` prefix.
- [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] — [Google Cloud](google-cloud.md) (formerly known as Vertex AI), surfaced under the `'google-cloud:'` prefix.

Both [`gateway/google:<model>`](../gateway.md) and [`gateway/google-cloud:<model>`](../gateway.md) use [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] through Pydantic AI Gateway. Gemini API-only features, such as [`FileSearchTool`][pydantic_ai.native_tools.FileSearchTool], require the direct Gemini API through [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider].

## Install

To use `GoogleModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```


## Configuration

`GoogleModel` lets you use Google's Gemini models through their [Gemini API](https://ai.google.dev/api/all-methods) (`generativelanguage.googleapis.com`) or [Google Cloud](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) (`*-aiplatform.googleapis.com`, formerly known as Vertex AI).

### API Key (Gemini API)

To use Gemini via the Gemini API, go to [aistudio.google.com](https://aistudio.google.com/apikey) and create an API key.

Once you have the API key, set it as an environment variable:

```bash
export GOOGLE_API_KEY=your-api-key
```

You can then use `GoogleModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('{{ google:medium }}')
...
```

!!! tip
    For voice agents, use a Gemini Live model with a [realtime session](../realtime/gemini.md) instead.

Or you can explicitly create the provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-api-key')
model = GoogleModel('{{ google:medium.name }}', provider=provider)
agent = Agent(model)
...
```

### Google Cloud (Enterprise)

See [Google Cloud](google-cloud.md) for Gemini and Claude access, authentication, and Model Garden setup.

- <span id="application-default-credentials"></span>[Application Default Credentials](google-cloud.md#application-default-credentials)
- <span id="service-account"></span>[Service Account](google-cloud.md#service-account)
- <span id="api-key"></span>[API Key](google-cloud.md#api-key)
- <span id="customizing-location-or-project"></span>[Customizing Location or Project](google-cloud.md#customizing-location-or-project)

### Switching between the Gemini API and Google Cloud

The Gemini API and Google Cloud reject each other's [thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures), even though a [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] is named `google` on either. So when history from one continues on the other, for example in a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] that falls back from Google Cloud to the Gemini API, [`GoogleModel`][pydantic_ai.models.google.GoogleModel] sends the documented `skip_thought_signature_validator` signature in place of the other transport's, which both accept, and thinking as text. It tells the transports apart by the response's [`canonical_system`][pydantic_ai.messages.ModelResponse.canonical_system], which `GoogleModel` always sets: `google` for a Gemini API response and `google-cloud` for a Google Cloud one, whatever the base URL. For history recorded before Pydantic AI stored it, a first-party Gemini API or Google Cloud `provider_url` decides, and one through a proxy URL is still sent with its signatures to a model with the same provider name. Google Cloud and the Pydantic AI Gateway's Google route, which uses Google Cloud, accept each other's signatures. See [Sending history to another provider](../message-history.md#sending-history-to-another-provider).

### Service tier (`service_tier`, `google_cloud_service_tier`)

The unified [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] field works on both Google subsystems, with [`google_cloud_service_tier`][pydantic_ai.models.google.GoogleModelSettings.google_cloud_service_tier] available for finer Google Cloud routing control. The provider-specific field wins when both are set.

**Gemini API** — sent as the request's `service_tier` field:

| `service_tier` | Sent to Gemini API |
|---|---|
| `'auto'` | _(omitted — server default)_ |
| `'default'` | `'standard'` |
| `'flex'` | `'flex'` |
| `'priority'` | `'priority'` |

**Google Cloud** — sent as HTTP routing headers; `'flex'` and `'priority'` always pick the **PT-with-spillover** variant, so customers with [Provisioned Throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput/use-provisioned-throughput) (PT) keep using their reserved capacity first:

| `service_tier` | Google Cloud routing headers | Effective behavior |
|---|---|---|
| `'auto'` / `'default'` | _(none)_ | PT first, then standard on-demand spillover |
| `'flex'` | `X-Vertex-AI-LLM-Shared-Request-Type: flex` | PT first, then [Flex PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo) spillover |
| `'priority'` | `X-Vertex-AI-LLM-Shared-Request-Type: priority` | PT first, then [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo) spillover |

To bypass PT entirely (or use it exclusively, or any of the other Google Cloud-specific routing combinations) set [`google_cloud_service_tier`][pydantic_ai.models.google.GoogleModelSettings.google_cloud_service_tier] directly — the unified field is intentionally limited to the safe PT-with-spillover variants.

**Google Cloud — full set of routing values**

The full [`google_cloud_service_tier`][pydantic_ai.models.google.GoogleModelSettings.google_cloud_service_tier] values map to these HTTP headers:

- `'pt_only'`: PT only (`X-Vertex-AI-LLM-Request-Type: dedicated`).
- `'pt_then_flex'`: PT when quota allows, then [Flex PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo) spillover (`X-Vertex-AI-LLM-Shared-Request-Type: flex`).
- `'pt_then_priority'`: PT when quota allows, then [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo) spillover (`X-Vertex-AI-LLM-Shared-Request-Type: priority`).
- `'on_demand'`: Standard on-demand only (`X-Vertex-AI-LLM-Request-Type: shared`).
- `'flex_only'`: [Flex PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/flex-paygo) only (`X-Vertex-AI-LLM-Request-Type: shared` and `X-Vertex-AI-LLM-Shared-Request-Type: flex`).
- `'priority_only'`: [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo) only (`X-Vertex-AI-LLM-Request-Type: shared` and `X-Vertex-AI-LLM-Shared-Request-Type: priority`).

**Example**

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(location='global')
model = GoogleModel('{{ google:medium.name }}', provider=provider)
agent = Agent(model)

result = agent.run_sync(
    'Hello!',
    model_settings=GoogleModelSettings(google_cloud_service_tier='pt_then_flex'),
)
```

Swap `'pt_then_flex'` for any [`GoogleCloudServiceTier`][pydantic_ai.models.google.GoogleCloudServiceTier] value — e.g. `'pt_then_priority'` for [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo) spillover, or `'flex_only'` / `'priority_only'` to bypass PT entirely.

After the request, inspect [`ModelResponse`][pydantic_ai.messages.ModelResponse] `provider_details.get('traffic_type')` (e.g. `ON_DEMAND_FLEX`, `ON_DEMAND_PRIORITY`) to see which tier served it, when the API returns it.

#### Model Garden

See [Google Cloud Model Garden](google-cloud.md#model-garden).

## Gemini 2.5 retirement {#gemini-25-retirement}

Requests to `gemini-2.5-pro` and `gemini-2.5-flash` emit a [`DeprecatedModelWarning`][pydantic_ai.exceptions.DeprecatedModelWarning]
(see [Deprecated models](overview.md#deprecated-models)).
From December 1, 2026, the Gemini API [routes them](https://aistudio.google.com/docs/deprecations#gemini-2.5-pro-models)
to other models, billed at those models' rates: `gemini-2.5-pro` to `gemini-3.8-flash`, and `gemini-2.5-flash` to
`gemini-3.5-flash-lite`. Google Cloud [lists October 20, 2026](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-versions#gemini-models)
as the retirement date of both, and lists `gemini-3.8-flash` first among their replacements.

`GoogleModel` keeps building Gemini 2.5 requests for these model names, so `thinking=False` and each thinking level
still send a `thinking_budget`, and `thinking=True` only asks for the thoughts. Both new models accept these, along
with the function tools, structured output and native tools `GoogleModel` sends for Gemini 2.5. Google's migration guides for
[Gemini 3.8 Flash](https://aistudio.google.com/docs/whats-new-gemini-3.8#migrate-to-gemini-3-8-flash) and
[Gemini 3.5 Flash-Lite](https://aistudio.google.com/docs/whats-new-gemini-3.6#migrate-to-gemini-3-5-flash-lite) list
what changes:

- `temperature`, `top_p` and `top_k` are [ignored](https://aistudio.google.com/docs/whats-new-gemini-3.6#sampling-parameter-deprecation).
- A request that ends with a model turn, which only the [direct API](../direct.md) sends, fails with a 400 error.

`gemini-3.8-flash` also rejects `google_logprobs` with `Logprobs is not enabled for this model`, so once the Gemini API
routes `gemini-2.5-pro` to it, that setting fails there.

`NativeOutput` with function tools, and native tools with function tools, stay unavailable on these model names, as on
Gemini 2.5. To use the new models' features, change the model name.

## Interactions API

Choose Google's [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview) when you need
remote MCP servers through [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool], Google Maps,
or music and video generation. It also exposes remote execution environments through `extra_body`
when the chosen model supports them. For ordinary text, function tools and structured output,
use `google:` and [`GoogleModel`][pydantic_ai.models.google.GoogleModel]. Interactions adds no caching or
conversation-history benefit, and lacks explicit caching, token counting and batch requests.

Select [`GoogleInteractionsModel`][pydantic_ai.models.google_interactions.GoogleInteractionsModel]
explicitly with `google-interactions:<model>`:

```python {gateway_toggle="skip"}
from pydantic_ai import Agent

agent = Agent('google-interactions:{{ google:medium.name }}')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

It uses the same credentials as `GoogleModel`. To use Google Cloud or a configured proxy, pass
an existing provider:

```python {gateway_toggle="skip"}
from pydantic_ai.models.google_interactions import GoogleInteractionsModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

model = GoogleInteractionsModel(
    '{{ google:medium.name }}',
    provider=GoogleCloudProvider(project='your-project', location='global'),
)
```

!!! note "Gateway support"
    The managed Pydantic AI Gateway Google route does not serve `/interactions`. A customer-configured
    Google provider on the Gateway may expose it. Use a direct Gemini API or Google Cloud provider
    unless your Gateway configuration supports this endpoint.

The adapter always sends `store=False` and the full conversation. It preserves separate thought
steps, their signatures, native-tool calls and results, and provider-generated function-call IDs.
Persist and restore the [complete message history](../message-history.md#storing-and-loading-messages-to-json)
when continuing a conversation or resuming a deferred tool call. Removing signed steps can make
Google reject later requests.

Function tools, tool retries, deferred approvals, native and prompted structured output, multimodal
inputs, and streaming use the ordinary Pydantic AI APIs.

### Interactions native tools

Supported native tools include Google Search, URL context, code execution, file search, and remote MCP
servers; availability depends on the model and endpoint. Image-capable models support `BinaryImage` output
and `ImageGenerationTool`. [Native tools](../native-tools.md) lists the parameters each tool supports on
Interactions.

Connect a remote MCP server with [`MCPServerTool`][pydantic_ai.native_tools.MCPServerTool]. Google calls the server
directly, so it must be reachable at a public URL:

```python {title="google_interactions_mcp.py" gateway_toggle="skip"}
from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'google-interactions:{{ google:medium.name }}',
    capabilities=[
        NativeTool(MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'))
    ],
)
result = agent.run_sync('Tell me about the pydantic/pydantic-ai repo.')
print(result.output)
"""
The pydantic/pydantic-ai repo is a Python agent framework for building Generative AI applications.
"""
```

[`google_interactions_native_tools`][pydantic_ai.models.google_interactions.GoogleInteractionsModelSettings.google_interactions_native_tools]
exposes additional native tools, such as Google Maps and Vertex retrieval, that have no shared Pydantic AI tool type.
Each entry is a tool dictionary in the Interactions SDK's shape. Google Maps needs only its type:

```python {title="google_interactions_maps.py" gateway_toggle="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google_interactions import GoogleInteractionsModelSettings

settings = GoogleInteractionsModelSettings(
    google_interactions_native_tools=[{'type': 'google_maps'}],
)
agent = Agent('google-interactions:{{ google:medium.name }}', model_settings=settings)
result = agent.run_sync('Use Google Maps to find the address of the Eiffel Tower.')
print(result.output)
#> The Eiffel Tower is at Av. Gustave Eiffel, 75007 Paris, France.
```

The Maps lookup appears in the response as a `google_maps`
[`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] and
[`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart].

Grounding data includes more than citations: search queries are in `NativeToolCallPart.args`, and
search suggestions, fetched URL status and Maps places are in `NativeToolReturnPart.content`.
Additional raw step metadata remains in each native part's `provider_details`.
An MCP result's content is the server's MCP `CallToolResult` (`content`, `structuredContent`, `isError`).
Inline images, audio and embedded binary resources in it appear as
[`BinaryContent`][pydantic_ai.messages.BinaryContent] values. They respect `include_binary_content` in
telemetry and are converted back to the MCP format when replayed from serialized message history.
The rest of each such MCP content block, such as a resource's `uri`, is kept for that conversion in the value's
`vendor_metadata['google_interactions_mcp_content']`.

### Interactions settings

[`GoogleInteractionsModelSettings`][pydantic_ai.models.google_interactions.GoogleInteractionsModelSettings]
provides typed Interactions settings. Use the shared `thinking`, `tool_choice`, and `service_tier`
settings for portable configuration. `google_interactions_thinking_level` overrides the shared `thinking` setting with
a native level, and `google_interactions_thinking_summaries` controls whether the API includes thought summaries. On
Google Cloud, the shared `service_tier` is ignored so Provisioned Throughput is never skipped silently; set
`google_interactions_service_tier` to choose a tier there. `google_labels` and `google_video_resolution` use the same types
as `GoogleModelSettings`. Safety settings have a different API shape: use
`google_interactions_safety_settings` for Interactions and `google_safety_settings` for GenerateContent.
For native audio or video output, configure
`google_interactions_response_format` and, when needed, `google_interactions_generation_config`, and read the returned
[`FilePart`][pydantic_ai.messages.FilePart]s through the [direct model API](../direct.md).
The typed generation configuration also exposes voice selection, transcription and video task controls.

### Interactions file inputs and outputs

Video inputs support YouTube URLs, `vendor_metadata` keys `start_offset`, `end_offset` and `fps`, and
the shared `google_video_resolution` setting. Per-file `vendor_metadata['media_resolution']` takes
precedence over that default. It uses the same `{'level': 'MEDIA_RESOLUTION_HIGH'}` shape as
`GoogleModel`, including `MEDIA_RESOLUTION_ULTRA_HIGH` where supported by the chosen model.
Explicit `vendor_metadata['media_processing']` replaces the static processing inferred from
`start_offset`, `end_offset` and `fps`.

Input file URLs are resolved by Google on both transports. Set `force_download=True` on an HTTP(S)
file URL to download it locally with the library's address and response-size checks instead.
Google Cloud can reject URLs blocked by the host's `robots.txt`; local downloading also handles that case.

Native tool results accept PDF, MP3 audio and MP4 video for the Gemini 3.8 Flash, Gemini 4 Argon, Gemini 4 Carbon
and Gemini 4 Hydrogen families, as defined by [`google_interactions_model_profile`][pydantic_ai.profiles.google.google_interactions_model_profile].
This support has expanded since the [initial PDF/audio probes](https://github.com/pydantic/pydantic-ai/issues/6192).
Other models retain the conservative GenerateContent MIME profile; unsupported result files are
sent in a framed user turn that identifies their originating tool call.

Use inline delivery for generated media. Video URI delivery requires `store=True`, so it is outside
this stateless adapter's scope. URI-only output raises an error.

### Unsupported Interactions features

Interactions has no token-counting endpoint, batch support, or explicit prompt caching. Implicit caching is
provider-managed. With `google-genai` earlier than
[2.34](https://github.com/googleapis/python-genai/releases/tag/v2.34.0), which doesn't continue an Interactions [long
decode](#long-decode), a request that reaches the model's per-request output cap returns its partial output with finish
reason `length` and `provider_details['status']` set to `requires_continuation`. From 2.34, the SDK continues it in
further requests, and the response holds every request's steps and their summed usage. Unlike on `GoogleModel`, a
failure after the first request raises the SDK's own error, and a non-streamed response doesn't record the first
request's input size for context-window checks ([#13938](https://github.com/pydantic/pydantic-ai/issues/13938)).
Sampling settings absent from this API are ignored; its settings class describes them. Finish reasons are derived from
interaction status, which has less detail than GenerateContent's finish reasons. Raw status and usage remain available
in `provider_details`.

The public Interactions contract has no equivalent for `google_logprobs`, `google_top_logprobs`,
or Google Cloud's `google_model_armor_config`; use `GoogleModel` when you need those features.
Native provider compaction is unavailable. Application-written `CompactionPart` summaries are
sent as portable user text; another provider's opaque response compaction is omitted.
The SDK has no per-request `max_retries` option. Use a [retrying HTTP transport](../retries.md#transport-retries)
to configure retries for this adapter. Empty prompts need sendable content, as tracked in
[#11123](https://github.com/pydantic/pydantic-ai/issues/11123).


## Custom HTTP Client

Google providers use `httpx2` by default. You can pass a custom `httpx2.AsyncClient` to control transport settings:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

custom_http_client = AsyncClient(timeout=30)
model = GoogleModel(
    '{{ google:medium.name }}',
    provider=GoogleProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

The client the Google providers create, including through the [Gateway](../gateway.md), has a 20-minute timeout, which
[long decode](#long-decode) needs. For a client you pass, `google-genai` applies the client's read timeout, or 600
seconds if it has none, to every phase of each request, so give it a read timeout of at least 20 minutes for long-decode
models. A client created without a `timeout` has httpx's default of 5 seconds, which is too short for most model
requests, so the providers warn about it.

`GoogleModel` and `GoogleInteractionsModel` also tell Google how long each whole request may take, with the
`X-Server-Timeout` header: the client's read timeout, or 20 minutes if that's longer. A read timeout only bounds the wait between bytes, so a response that
streams for longer than it isn't cut off by the server. A per-request [`timeout`][pydantic_ai.settings.ModelSettings.timeout]
doesn't change the header. To set a different server deadline, send the header yourself in
[`extra_headers`][pydantic_ai.settings.ModelSettings.extra_headers].

A user-agent you send in `extra_headers`, in any spelling, replaces Pydantic AI's on
[`GoogleModel`][pydantic_ai.models.google.GoogleModel] and
[`GoogleImageGenerationModel`][pydantic_ai.images.google.GoogleImageGenerationModel] requests. `google-genai` puts its
own label in front of it, so it goes out as one `user-agent: google-genai-sdk/… gl-python/… <your value>` header. It
also replaces a user-agent set on a client you pass. If that client spells the header other than `user-agent`, as in
`HttpOptions(headers={'User-Agent': ...})`, `google-genai` keeps its own lowercase `user-agent` beside it, so your value
goes out on both lines.

The Google providers also accept a legacy `httpx.AsyncClient` during Pydantic AI v2, but emit a deprecation warning. Use `httpx2.AsyncClient` for new code; legacy HTTPX client support will be removed in Pydantic AI v3.

## HTTP Retries

!!! note
    For most use cases, the model-agnostic [transport retries](../retries.md#transport-retries) approach is preferable, as it works the same way across all providers. The `retry_options` argument below is a Google-specific alternative that delegates retrying to the `google-genai` SDK's own HTTP layer. See [The layers](../retries.md#the-layers) in the retries guide for where these SDK-level retries sit, and [Retry multiplication](../retries.md#retry-multiplication) for how they compound with the agent's own retry budgets.

By default, the `google-genai` SDK does not retry requests that fail with a transient HTTP error. You can enable retries by passing a [`HttpRetryOptions`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpRetryOptions) instance to the `retry_options` argument of `GoogleProvider` or `GoogleCloudProvider`:

```python
from google.genai.types import HttpRetryOptions

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

retry_options = HttpRetryOptions(
    attempts=4,
    initial_delay=1.0,
    max_delay=60.0,
    http_status_codes=[408, 429, 500, 502, 503, 504],
)
model = GoogleModel(
    '{{ google:medium.name }}',
    provider=GoogleProvider(api_key='your-api-key', retry_options=retry_options),
)
agent = Agent(model)
...
```

This passes the options through to the SDK's [`HttpOptions.retry_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.retry_options). [`ModelSettings['max_retries']`](../retries.md#setting-the-sdk-retry-count-per-request) overrides the number of attempts for a model, agent or run, keeping the delays and status codes set here, or the SDK's defaults (a first delay of one second; status 408, 429, 500, 502, 503 and 504) when `retry_options` isn't set. See the [Vertex AI retry strategy documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy) for guidance on choosing values.

When a Google error carries no `retry-after-ms` or `Retry-After` header, [`ModelAPIError.retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] is read from the `google.rpc.RetryInfo` detail in its body, including for an error chunk inside a stream and for a status a [retrying transport](../retries.md#error-handling) gave up on. [Temporal](../durable_execution/temporal.md) waits that long before retrying a model activity, and you can use it in your own backoff ([Retries](../retries.md#transport-retries)). The SDK's `retry_options` above and [`wait_retry_after`](../retries.md#wait_retry_after) read only headers.

## Document, Image, Audio, and Video Input

`GoogleModel` supports multi-modal input, including documents, images, audio, and video.

Gemini reads JPEG, PNG, GIF, WebP, HEIC, HEIF, AVIF and BMP images. On Google Cloud it also reads TIFF, which the Gemini API rejects. Gemini 2.5 models don't read Word or Excel documents; Gemini 3 models do. A file of another type that a tool returns is [sent as a note naming it](../tools-advanced.md#tool-return-file-provenance).

YouTube video URLs can be passed directly to Google models:

```py {title="youtube_input.py" test="skip" lint="skip"}
from pydantic_ai import Agent, VideoUrl
from pydantic_ai.models.google import GoogleModel

agent = Agent(GoogleModel('{{ google:medium.name }}'))
result = agent.run_sync(
    [
        'What is this video about?',
        VideoUrl(url='https://www.youtube.com/watch?v=dQw4w9WgXcQ'),
    ]
)
print(result.output)
```

Files can be uploaded via the [Files API](https://ai.google.dev/gemini-api/docs/files) and passed as URLs:

```py {title="file_upload.py" test="skip"}
from pydantic_ai import Agent, DocumentUrl
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider()
file = provider.client.files.upload(file='pydantic-ai-logo.png')
assert file.uri is not None

agent = Agent(GoogleModel('{{ google:medium.name }}', provider=provider))
result = agent.run_sync(
    [
        'What company is this logo from?',
        DocumentUrl(url=file.uri, media_type=file.mime_type),
    ]
)
print(result.output)
```

See the [input documentation](../input.md) for more details and examples.

### Agentic video processing

Agentic video processing lets Gemini inspect relevant video segments on demand. Without `media_processing`, Gemini uses its default static processing. [Google recommends](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding#choose_a_processing_mode) static processing for short clips, whole-video summaries, latency-sensitive queries, or frame-level precision across the clip; agentic processing for long videos or retrieval of specific moments or topics. Choose the processing mode for the workload.

For [Gemini models that support agentic video processing](https://ai.google.dev/gemini-api/docs/video-understanding#agentic-video-understanding), set `media_processing` on a video input:

```py {title="agentic_video.py" test="skip"}
from pydantic_ai import Agent, VideoUrl
from pydantic_ai.models.google import GoogleModel

agent = Agent(GoogleModel('gemini-4-hydrogen'))
result = agent.run_sync(
    [
        'Summarize the important moments with timestamps.',
        VideoUrl(
            url='https://www.youtube.com/watch?v=dQw4w9WgXcQ',
            vendor_metadata={'media_processing': 'AGENTIC'},
        ),
    ]
)
print(result.output)
```

Video source and processing mode are independent. A YouTube [`VideoUrl`][pydantic_ai.messages.VideoUrl] uses Gemini's [direct YouTube input](https://ai.google.dev/gemini-api/docs/generate-content/video-understanding#youtube): Pydantic AI sends the URL without downloading or uploading the video, including when agentic processing is enabled. Google lists YouTube URLs among the [supported agentic video sources](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/video-understanding#agentic-video-processing-support).

On the Gemini Developer API, each processing step appears in the response as a [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] and [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] named `media_processing`. Vertex AI returns signature-only parts for these steps, so they do not appear in the message history.

!!! note "Follow-up video processing"

    Pydantic AI omits media-processing steps from later requests because Gemini currently rejects them. The ordinary final response and its thought signature remain in history. When that history includes the original video, follow-up requests process it again and can incur additional video-processing tokens.

## Image generation

Use [`ImageGenerator`][pydantic_ai.images.ImageGenerator] with a `google:` image model for direct generation and
reference-image editing through the Gemini API, or with a `google-cloud:` model to run the same models on Vertex AI:

```python {title="google_image_generation.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images.google import GoogleImageGenerationSettings

settings = GoogleImageGenerationSettings(
    google_image_config={'aspect_ratio': '1:1', 'image_size': '1K'}
)

gemini_api_generator = ImageGenerator('google:gemini-3.1-flash-lite-image', settings=settings)
vertex_generator = ImageGenerator('google-cloud:gemini-3.1-flash-image', settings=settings)
```

Construct [`GoogleImageGenerationModel`][pydantic_ai.images.google.GoogleImageGenerationModel] with a
[`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] to set the Vertex project and location
explicitly.

The direct adapter accepts inline images and downloadable image URLs on both APIs, and forwards a reference the selected
transport hosts itself as a `fileData` part instead of downloading it, exactly as [`GoogleModel`][pydantic_ai.models.google.GoogleModel]
does: on the Gemini Developer API that is a Files API URI, as an [`UploadedFile`][pydantic_ai.messages.UploadedFile] or an
`ImageUrl`; on Vertex AI, where the Files API is not available, it is a Cloud Storage `gs://bucket/path` URI, as an
`UploadedFile` whose `file_id` starts with `gs://` or an `ImageUrl` whose URL does and that isn't `force_download`.
Vertex reads the object server-side, so a multi-megabyte reference never passes through your process, and neither
transport accepts the other's references: a Files API id on Vertex raises
[`UserError`][pydantic_ai.exceptions.UserError]. Which API a model talks to is read off the
client, not the provider name, so a Vertex-backed client passed to
[`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] is treated as Vertex, and a Gemini Developer API client
passed to [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] keeps Files API support. See the
[image-generation guide](../image-generation.md) for the common API and geometry behavior. The adapter requests an
image-only response because [`ImageGenerator`][pydantic_ai.images.ImageGenerator] returns generated images rather than
Gemini's optional conversational text.

Every generated image carries an unconditional
[SynthID watermark](https://ai.google.dev/responsible/docs/safeguards/synthid). The Gemini 3 image models are thinking
models: thinking is always on and billed, and its tokens are included in the result's `usage`.

## Model settings

You can customize model behavior using [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings]:

```python
from google.genai.types import HarmBlockThreshold, HarmCategory

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

settings = GoogleModelSettings(
    max_tokens=1024,
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('{{ google:medium.name }}')
agent = Agent(model, model_settings=settings)
...
```

`presence_penalty` and `frequency_penalty` aren't sent through the Gemini API, which rejects them for every model, or on Google Cloud for Gemini 3.5 Flash-Lite, Gemini 3.6 Flash and newer, and the Gemini 3 image models, which reject them too.

### Configure thinking

Use the provider-agnostic [`Thinking`][pydantic_ai.capabilities.Thinking] capability to enable thinking:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('{{ google:medium }}', capabilities=[Thinking(effort='medium')])
...
```

For advanced usage, you can pass Google's native thinking config through [`GoogleModelSettings.google_thinking_config`][pydantic_ai.models.google.GoogleModelSettings.google_thinking_config]:

```python
from google.genai.types import ThinkingLevel

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('{{ google:medium.name }}')
model_settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True, 'thinking_level': ThinkingLevel.MEDIUM})
agent = Agent(model, model_settings=model_settings)
...
```

Pydantic AI resolves each model's supported levels from Google's documented thinking table and snaps a
requested effort to the nearest supported level. `gemini-3.1-flash-image` is the exception: Pydantic AI applies its
levels only when the client talks to the Gemini API, and Vertex AI keeps the default levels. Google
[documents](https://ai.google.dev/gemini-api/docs/image-generation) `minimal, high` for `gemini-3.1-flash-image`, but
only the Gemini API enforces them. For a model id the table doesn't cover, declare its
levels with [`GoogleModelProfile.google_thinking_levels`][pydantic_ai.profiles.google.GoogleModelProfile.google_thinking_levels]
(default: every level but `xhigh`); unsupported efforts resolve to the nearest supported level.

`gemini-4-argon`, `gemini-4-carbon`, `gemini-4-hydrogen` and `gemini-4-neon` support the `low`, `medium` and `high`
levels, so `thinking=False` and `'minimal'` resolve to `low`. `gemini-4-hydrogen` and `gemini-4-neon` also support
`xhigh`, their deepest reasoning, which `thinking='xhigh'` and `'max'` select; on models without it, they resolve to
`high`. The
`google-genai` SDK's `ThinkingLevel` enum doesn't list `XHIGH` yet, so the SDK warns
`XHIGH is not a valid ThinkingLevel` and sends the level anyway.

On Google Cloud, `gemini-4-argon` rejects a `thinking_budget` in `google_thinking_config` with a 400 error: use
`thinking_level` instead.

Gemma 4 (`gemma-4-31b-it`, `gemma-4-26b-a4b-it`) supports only the `minimal` and `high` levels on the Gemini API and
rejects any `thinking_budget`, so `thinking=False` and `'low'` resolve to `minimal`, and `'medium'` and above to
`high`. Without a `thinking` setting it thinks by default there.

On Gemini 2.5, `thinking='minimal'` sends a `thinking_budget` of 128 tokens. The exception is `gemini-2.5-flash-lite`
on the Gemini API, which rejects budgets below 512, so it gets 512.
To send another budget for `minimal`, set
[`GoogleModelProfile.google_minimal_thinking_budget`][pydantic_ai.profiles.google.GoogleModelProfile.google_minimal_thinking_budget]
in the model's `profile`.

`thinking=False` turns thinking off where the model allows it. The Pro models can't turn it off, so they think as
little as they allow instead: Gemini 3 Pro models use their lowest thinking level (`low` on `gemini-3.1-pro-preview`),
and Gemini 2.5 Pro uses its minimum `thinking_budget` of 128 tokens. Neither asks for the thoughts to be returned.
On Google Cloud, `gemini-3.1-pro-preview` accepts `thinking_budget=0`, which the Gemini API rejects for it, so
there `thinking=False` sends `thinking_budget=0` and the model doesn't think.

From December 1, 2026, the Gemini API [routes `gemini-2.5-flash` to `gemini-3.5-flash-lite`](https://aistudio.google.com/docs/deprecations),
which rejects `thinking_budget=0`. So on the Gemini API, `thinking=False` on `gemini-2.5-flash` sends the minimum
`thinking_budget` of 128 tokens, like Gemini 2.5 Pro. Before the reroute, `gemini-2.5-flash` thinks a little with that
budget; `gemini-3.5-flash-lite` doesn't. On Google Cloud, `gemini-2.5-flash` keeps `thinking_budget=0`.

See [Thinking](../capabilities/thinking.md) for the unified API and [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking) for Google's native thinking configuration.

### Safety settings

You can customize the safety settings by setting the `google_safety_settings` field.

```python
from google.genai.types import HarmBlockThreshold, HarmCategory

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('{{ google:medium.name }}')
agent = Agent(model, model_settings=model_settings)
...
```

See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for more on safety settings.


### Logprobs

You can return logprobs from the model in your response by setting `google_logprobs` and `google_top_logprobs` in the [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings].

This feature is only supported for non-streaming requests and Google Cloud.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

model_settings = GoogleModelSettings(
    google_logprobs=True, google_top_logprobs=2,
)

model = GoogleModel(
    model_name='{{ google:medium.name }}',
    provider=GoogleCloudProvider(location='europe-west1'),
)
agent = Agent(model, model_settings=model_settings)

result = agent.run_sync('Your prompt here')
# Access logprobs from provider_details
provider_details = result.response.provider_details or {}
logprobs = provider_details.get('logprobs')
avg_logprobs = provider_details.get('avg_logprobs')
```

See the [Google Dev Blog](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/) for more information.

### Model Armor (Google Cloud only)

[Model Armor](https://docs.cloud.google.com/model-armor/overview) is a Google Cloud security service that screens prompts and responses for risks like prompt injection, jailbreaking, and sensitive data leakage.

You can configure it via `google_model_armor_config` in [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings]:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

model_settings = GoogleModelSettings(
    google_model_armor_config={
        'prompt_template_name': 'projects/my-project/locations/europe-west4/templates/prompt-template',
        'response_template_name': 'projects/my-project/locations/europe-west4/templates/response-template',
    }
)

model = GoogleModel(
    model_name='{{ google:medium.name }}',
    provider=GoogleCloudProvider(location='europe-west4'),
)
agent = Agent(model, model_settings=model_settings)
...
```

Templates must be created in advance in the [Google Cloud Console](https://console.cloud.google.com/security/modelarmor) and must reside in the same region as the model endpoint. See the [Model Armor Vertex AI integration docs](https://docs.cloud.google.com/model-armor/model-armor-vertex-integration) for supported locations.

When a prompt or response is blocked, a [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] is raised.

Note that Model Armor screening — both prompt and response templates — only works with non-streaming requests (`agent.run()`). With streaming (`agent.run_stream()`), Google Cloud does not apply Model Armor: the prompt is not screened and the response text is returned unscreened. If you require streaming and need Model Armor protection, pre-screen prompts using the [`google-cloud-modelarmor` SDK](https://pypi.org/project/google-cloud-modelarmor/) before calling the agent.

### Context caching (`google_cached_content`)

When you've created a Gemini [cached content resource](https://ai.google.dev/gemini-api/docs/caching), pass its resource name through [`google_cached_content`][pydantic_ai.models.google.GoogleModelSettings.google_cached_content] to reuse it across requests. (Gemini also caches prompts implicitly, so the unified [`ModelSettings.cache`][pydantic_ai.settings.ModelSettings.cache] setting, which configures request-side caching on other providers, adds nothing to a Google request.)

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(
    google_cached_content='projects/p/locations/global/cachedContents/your-cache-id',
)

agent = Agent(GoogleModel('{{ google:medium.name }}'), model_settings=model_settings)
...
```

!!! warning "Cached fields are owned by the cache resource"
    The cache resource owns `system_instruction`, `tools`, and `tool_config` — Pydantic AI strips them from outgoing requests when `google_cached_content` is set, so agent instructions and registered tools are ignored on cached requests. A `UserWarning` is emitted whenever stripping drops a field, so the mismatch is discoverable.

??? example "Create a cached content resource"
    Pydantic AI doesn't wrap the cache-management API — create the resource with the underlying [google-genai](https://googleapis.github.io/python-genai/) SDK, then pass its name through `google_cached_content`:

    ```python {test="skip"}
    from google.genai.types import Content, CreateCachedContentConfig, Part

    from pydantic_ai.providers.google import GoogleProvider

    provider = GoogleProvider(api_key='your-api-key')

    cache = provider.client.caches.create(
        model='{{ google:medium.name }}',
        config=CreateCachedContentConfig(
            system_instruction='You are a geography expert. Be concise.',
            contents=[Content(role='user', parts=[Part(text='...long context to cache...')])],
            ttl='3600s',
        ),
    )
    print(cache.name)
    #> cachedContents/abc123...
    ```

    Caches have a minimum size (≈2048 tokens for Gemini 2.5 series models, ≈4096 for Gemini 3 series models) and a TTL — see the [Gemini caching docs](https://ai.google.dev/gemini-api/docs/caching) for the current thresholds, pricing, and `list` / `update` / `delete` operations.

## Batch API

`GoogleModel` supports [batch processing](../batch.md) through the Gemini API's
[Batch Mode](https://ai.google.dev/gemini-api/docs/batch-api), which charges 50% of the standard price. A job
targets one model, its requests are uploaded as a file of up to 2 GB, which limits how many it can hold, and it
expires if it hasn't finished within 48 hours. When the process that submitted a job retrieves its results, it
deletes the file, so it doesn't count against the project's file storage. A job collected from another process, for
example from a stored [`PendingModelResponse`](../batch.md#ending-a-run-while-it-waits) history, leaves its file in
place until Google expires it, since only the process that uploaded the file knows it's safe to delete.

Batch Mode isn't available on Google Cloud (Vertex AI), which runs batch prediction jobs from Cloud Storage or
BigQuery instead, or through the [Pydantic AI Gateway](../gateway.md), which routes Google models to Google Cloud:
use `GoogleProvider` with a Gemini API key instead. Configure timeouts and custom headers on the client: per-request `timeout` and `extra_headers`
are unsupported. Retrieving a job needs the model that submitted it, but not the same API key: any key of the project works.

A job that fails or is cancelled before it writes any results doesn't report which requests it held, so retrieving its results raises the job's error as a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. `model_request_batch` reports that error for each of the job's requests, and an agent run waiting on the job raises it, unless it's an error that [resubmitting](../batch.md#failed-requests) may fix, such as a server error, in which case the requests are submitted again. A job that expired, which also writes no results, raises a 504 `ModelHTTPError` saying so, and its requests are submitted again too.

## Long decode

`gemini-4-argon`, `gemini-4-carbon`, `gemini-4-hydrogen` and `gemini-4-neon` cap the output of each request; Google's
[long decode docs](https://ai.google.dev/gemini-api/docs/generate-content/long-decode) give the caps and defaults. When
[`max_tokens`][pydantic_ai.settings.ModelSettings.max_tokens] is above the cap, a request that reaches it returns a
continuation token, and the `google-genai` SDK continues the response from it in another request, until the response
is done or `max_tokens` is spent:

```python {test="skip"}
from pydantic_ai import Agent

agent = Agent('google:gemini-4-argon', model_settings={'max_tokens': 100_000})
result = agent.run_sync('Write the complete specification.')
print(result.output)
```

The SDK merges the requests into one [`ModelResponse`][pydantic_ai.messages.ModelResponse], so to Pydantic AI a long
decode is one model request:

- Its usage is the sum over the requests, which is what Google bills. Each request's input counts the output so far
  again.
- [Usage limits](../agent.md#usage-limits) count it as one request, and a run that isn't streamed checks its token
  limits once every request is done: `max_tokens` is what bounds a long decode.
  [`per_request_input_tokens_limit`][pydantic_ai.usage.UsageLimits.per_request_input_tokens_limit] is checked against
  the input tokens of the first request.
- Timeouts apply to each request. The Google providers create their client with the 20-minute read timeout Google
  recommends.
- If a later request fails, the model request fails. A run that isn't streamed loses the output of the earlier requests,
  already billed. A streamed run keeps what it streamed as a response with
  [`state='interrupted'`][pydantic_ai.messages.ModelResponse.state] that carries the finished requests' usage, which
  [`capture_run_messages()`][pydantic_ai.capture_run_messages] returns.
- With `google-genai` 2.31.0 and later, a streamed run sends a later request that fails before it streams any output
  3 more times from the same continuation token, with no wait. If they all fail, the model request fails as above.
  This applies to errors the HTTP client raises: a connection that can't be opened or that drops, a timeout, or the
  `HTTPStatusError` of a [retrying transport](../retries.md#transport-retries) that gave up. It doesn't apply to an
  error status the SDK reads itself. `retry_options` and
  [`max_retries`](../retries.md#setting-the-sdk-retry-count-per-request) can't turn this off, and they also retry a
  connection that can't be opened, or a timeout before the response arrives, on each of those 4 requests: with
  `max_retries=2`, such an error goes on the wire 12 times.
- The SDK resumes a dropped stream from its last continuation token. If output was streamed after that token, the
  model request fails as above instead, since the resumed request would repeat that output.
- On [Temporal](../durable_execution/temporal.md#activity-configuration), the whole long decode runs in one model
  request activity, so its `start_to_close_timeout` must cover every request. A retried activity starts the response
  over. The merged response is the activity's one result, so a very long output can exceed Temporal's
  [payload limit](../durable_execution/temporal.md#large-payloads).

Without `max_tokens`, Google's default output limit applies, and a response that reaches it ends with
`finish_reason='length'`.

On [Google Cloud](google-cloud.md), `gemini-4-argon` is served on the `global` location and the `us` and `eu`
multi-regions, not in single regions such as the default `us-central1`: pass one of those as the provider's
[`location`](google-cloud.md#customizing-location-or-project).

## Computer use

To let Gemini use a computer, add the Pydantic AI Harness [`ComputerUse`](../harness/computer-use.md) capability. On
the Gemini API, it sends Google's own
[computer use tool](https://ai.google.dev/gemini-api/docs/computer-use) as a
[`ComputerTool`][pydantic_ai.native_tools.ComputerTool] to the Gemini 4 models, Gemini 3.8 Flash, Gemini 3.5 Flash-Lite
and Gemini 3 Flash Preview. Other models, every model on Vertex AI, and any model with `ComputerUse(native=False)` get a
regular function tool named `computer` instead.

Google's tool can't zoom or read the pointer position. It can't be sent beside a function tool named like one of its
functions, such as `click` or `type`, so with such a tool the model gets the `computer` function tool instead.

### Actions that need approval

Gemini's safety checks can flag an action, such as deleting files or accepting terms, as one a person must confirm. The
flagged call waits for [approval](../deferred-tools.md#human-in-the-loop-tool-approval), with Gemini's explanation in
its metadata under `reason`. Resolve it with a
[`HandleDeferredToolCalls`](../deferred-tools.md#resolving-deferred-calls-with-a-handler) handler, or add
[`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] to the agent's `output_type` to end the run and resume
it later. Without either, the run fails at the first flagged call.

```python {test="skip" lint="skip" gateway_toggle="skip"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai_harness import ComputerUse


async def ask_a_person(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    approvals = {}
    for call in requests.approvals:
        reason = requests.metadata[call.tool_call_id]['reason']
        approvals[call.tool_call_id] = input(f'{reason} Approve? [y/N] ') == 'y'
    return requests.build_results(approvals=approvals)


agent = Agent(
    '{{ google:medium }}',
    capabilities=[my_desktop_sandbox, ComputerUse(), HandleDeferredToolCalls(handler=ask_a_person)],
)
```

Google sends no safety decisions on streamed requests, so streamed runs never pause for approval.

## Streaming cancellation

!!! note "Transport cancellation"
    [`cancel()`][pydantic_ai.result.StreamedRunResult.cancel] safely interrupts an active local stream pull, including one running in another task. The `google-genai` SDK exposes no documented per-stream transport handle, so closing the returned iterator does not guarantee immediate HTTP teardown or indicate when remote generation stops and billing ends. See [googleapis/python-genai#2425](https://github.com/googleapis/python-genai/issues/2425).

## Decisions API {#decisions-api}

The Gemini API's [Decisions API](https://aistudio.google.com/docs/decisions-api) runs Gemini Decision as a [decision model](decision.md): it answers typed questions about text or images, each with a probability or a distribution over the options, rather than writing text. [`GoogleDecisionsModel`][pydantic_ai.models.google_decisions.GoogleDecisionsModel] is the Pydantic AI model class for it, so an agent's output type and tools become the questions as described on the [Decision models](decision.md) page. The `'google-decisions:'` prefix picks it:

```python {gateway_toggle="skip"}
from pydantic_ai import Agent

agent = Agent('google-decisions:gemini-decision-1', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

It runs on the Gemini API, reads the same `GOOGLE_API_KEY`, and takes the same [`GoogleProvider`](#api-key-gemini-api) as `GoogleModel`, including a [custom HTTP client](#custom-http-client) and [retry options](#http-retries):

```python {gateway_toggle="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google_decisions import GoogleDecisionsModel
from pydantic_ai.providers.google import GoogleProvider

model = GoogleDecisionsModel('gemini-decision-1', provider=GoogleProvider(api_key='your-api-key'))
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
...
```

All the questions of a request go to the API in one call, and its usage is reported in input tokens: the model writes no output.

The API takes no conversation turns, so the conversation goes as the [state](decision.md#where-the-wording-comes-from) other decision models are sent: one text, with the latest prompt as what to judge and the history beside it.

Image input works through [`BinaryContent`][pydantic_ai.BinaryContent], [`BinaryImage`][pydantic_ai.messages.BinaryImage] or [`ImageUrl`][pydantic_ai.messages.ImageUrl]; `ImageUrl` images are downloaded and sent inline. Each image follows its `<image N>` reference in the state. See [image input](../input.md#image-input). Set `vendor_metadata={'media_resolution': {'level': 'MEDIA_RESOLUTION_LOW'}}` on an image to send it at a lower resolution, which costs fewer input tokens. `GoogleModel` reads the same [per-part media resolution](https://ai.google.dev/gemini-api/docs/media-resolution), so an image keeps its resolution when a step [escalates](decision.md#escalating-to-a-language-model) to a Gemini model; only the `level` is sent.

`timeout`, `max_retries`, `extra_headers` and `extra_body` are forwarded to the request; `extra_body` can't set `questions`, which the model keys itself. The other generic settings, such as `temperature`, are ignored: the API takes none. [`GoogleDecisionsModelSettings`][pydantic_ai.models.google_decisions.GoogleDecisionsModelSettings] takes the two [thresholds](decision.md#confidence-and-thresholds) every decision model has, `decision_boolean_threshold` and `decision_route_threshold`.

`GoogleDecisionsModel` checks the API's request limits before sending a request or downloading image URLs:

- **255 options in one pick-one question.** A pick-one field counts its own options, and the [route question](decision.md#routes-which-thing-to-do) counts every tool plus every output type. A question over the limit raises a [`UserError`][pydantic_ai.exceptions.UserError].
- **255 levels in one rubric.** A field of more whole numbers from 0 becomes a [pick-one](decision.md#what-each-field-type-does) instead.
- **1000 questions in one request.** A `list` or `dict` of options counts one question per option, and with tools or a union of output types, every route's fields can go in the same request as the route question when they fit the question cap and the [size cutoff](decision.md#routes-which-thing-to-do). Otherwise the model picks the route first and fills it in a second request. An initial request whose fields alone exceed the cap raises a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] before images are downloaded, so a [`FallbackModel`](overview.md#fallback-model) can take over.

Every question goes with instructions, which the API requires: a pick-one, a rubric or a yes/no with described answers that has nothing else to ask is sent a generic question, and its options carry the meaning. An option's label can be up to 128 characters; a longer one fails the request with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError].

!!! note "Measure on your own data"
    A threshold tuned on another decision model does not carry over to this one. Measure accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.
