# Google

The `GoogleModel` is a model that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to
access Google's Gemini models via both the Gemini API and Google Cloud (formerly known as Vertex AI).

Two providers wrap those endpoints:

- [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] — the Gemini API (Google AI Studio), surfaced under the `'google:'` prefix.
- [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] — [Google Cloud](google-cloud.md) (formerly known as Vertex AI), surfaced under the `'google-cloud:'` prefix.

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

agent = Agent('google:gemini-3.7-flash')
...
```

Or you can explicitly create the provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-api-key')
model = GoogleModel('gemini-3.7-flash', provider=provider)
agent = Agent(model)
...
```

### Google Cloud (Enterprise)

See [Google Cloud](google-cloud.md) for Gemini and Claude access, authentication, and Model Garden setup.

- <span id="application-default-credentials"></span>[Application Default Credentials](google-cloud.md#application-default-credentials)
- <span id="service-account"></span>[Service Account](google-cloud.md#service-account)
- <span id="api-key"></span>[API Key](google-cloud.md#api-key)
- <span id="customizing-location-or-project"></span>[Customizing Location or Project](google-cloud.md#customizing-location-or-project)

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
model = GoogleModel('gemini-3.7-flash', provider=provider)
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

## Custom HTTP Client

Google providers use `httpx2` by default. You can pass a custom `httpx2.AsyncClient` to control transport settings:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

custom_http_client = AsyncClient(timeout=30)
model = GoogleModel(
    'gemini-3.7-flash',
    provider=GoogleProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

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
    'gemini-3.7-flash',
    provider=GoogleProvider(api_key='your-api-key', retry_options=retry_options),
)
agent = Agent(model)
...
```

This passes the options through to the SDK's [`HttpOptions.retry_options`](https://googleapis.github.io/python-genai/genai.html#genai.types.HttpOptions.retry_options). See the [Vertex AI retry strategy documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/retry-strategy) for guidance on choosing values.

## Document, Image, Audio, and Video Input

`GoogleModel` supports multi-modal input, including documents, images, audio, and video.

YouTube video URLs can be passed directly to Google models:

```py {title="youtube_input.py" test="skip" lint="skip"}
from pydantic_ai import Agent, VideoUrl
from pydantic_ai.models.google import GoogleModel

agent = Agent(GoogleModel('gemini-3.7-flash'))
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

agent = Agent(GoogleModel('gemini-3.7-flash', provider=provider))
result = agent.run_sync(
    [
        'What company is this logo from?',
        DocumentUrl(url=file.uri, media_type=file.mime_type),
    ]
)
print(result.output)
```

See the [input documentation](../input.md) for more details and examples.

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
    temperature=0.2,
    max_tokens=1024,
    top_k=40,
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('gemini-3.7-flash')
agent = Agent(model, model_settings=settings)
...
```

### Configure thinking

Use the provider-agnostic [`Thinking`][pydantic_ai.capabilities.Thinking] capability to enable thinking:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('google:gemini-3.7-flash', capabilities=[Thinking(effort='medium')])
...
```

For advanced usage, you can pass Google's native thinking config through [`GoogleModelSettings.google_thinking_config`][pydantic_ai.models.google.GoogleModelSettings.google_thinking_config]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-3.7-flash')
model_settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True, 'thinking_level': 'MEDIUM'})
agent = Agent(model, model_settings=model_settings)
...
```

Pydantic AI resolves each model's supported levels from Google's documented thinking table and snaps a
requested effort to the nearest supported level. For a model id the table doesn't cover, declare its
levels with [`GoogleModelProfile.google_thinking_levels`][pydantic_ai.profiles.google.GoogleModelProfile.google_thinking_levels]
(default: the full scale); unsupported efforts resolve to the nearest supported level.

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
model = GoogleModel('gemini-3.7-flash')
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
    model_name='gemini-3.7-flash',
    provider=GoogleCloudProvider(location='europe-west1'),
)
agent = Agent(model, model_settings=model_settings)

result = agent.run_sync('Your prompt here')
# Access logprobs from provider_details
logprobs = result.response.provider_details.get('logprobs')
avg_logprobs = result.response.provider_details.get('avg_logprobs')
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
    model_name='gemini-3.7-flash',
    provider=GoogleCloudProvider(location='europe-west4'),
)
agent = Agent(model, model_settings=model_settings)
...
```

Templates must be created in advance in the [Google Cloud Console](https://console.cloud.google.com/security/modelarmor) and must reside in the same region as the model endpoint. See the [Model Armor Vertex AI integration docs](https://docs.cloud.google.com/model-armor/model-armor-vertex-integration) for supported locations.

When a prompt or response is blocked, a [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError] is raised.

Note that Model Armor screening — both prompt and response templates — only works with non-streaming requests (`agent.run()`). With streaming (`agent.run_stream()`), Google Cloud does not apply Model Armor: the prompt is not screened and the response text is returned unscreened. If you require streaming and need Model Armor protection, pre-screen prompts using the [`google-cloud-modelarmor` SDK](https://pypi.org/project/google-cloud-modelarmor/) before calling the agent.

### Context caching (`google_cached_content`)

When you've created a Gemini [cached content resource](https://ai.google.dev/gemini-api/docs/caching), pass its resource name through [`google_cached_content`][pydantic_ai.models.google.GoogleModelSettings.google_cached_content] to reuse it across requests:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(
    google_cached_content='projects/p/locations/global/cachedContents/your-cache-id',
)

agent = Agent(GoogleModel('gemini-3.7-flash'), model_settings=model_settings)
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
        model='gemini-3.7-flash',
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

## Streaming cancellation

!!! note "Transport cancellation"
    [`cancel()`][pydantic_ai.result.StreamedRunResult.cancel] safely interrupts an active local stream pull, including one running in another task. The `google-genai` SDK exposes no documented per-stream transport handle, so closing the returned iterator does not guarantee immediate HTTP teardown or indicate when remote generation stops and billing ends. See [googleapis/python-genai#2425](https://github.com/googleapis/python-genai/issues/2425).
