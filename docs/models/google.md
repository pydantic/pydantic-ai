# Google

The `GoogleModel` is a model that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to
access Google's Gemini models via both the Gemini API and Google Cloud (formerly known as Vertex AI).

Two providers wrap those endpoints:

- [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] — the Gemini API (Google AI Studio), surfaced under the `'google:'` prefix.
- [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] — Google Cloud (formerly known as Vertex AI), surfaced under the `'google-cloud:'` prefix.

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

agent = Agent('google:gemini-3-pro-preview')
...
```

Or you can explicitly create the provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-api-key')
model = GoogleModel('gemini-3-pro-preview', provider=provider)
agent = Agent(model)
...
```

### Google Cloud (Enterprise)

If you are an enterprise user, you can also use `GoogleModel` to access Gemini via Google Cloud (formerly known as Vertex AI).

This interface has a number of advantages over the Gemini API:

1. The Google Cloud API comes with more enterprise readiness guarantees.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with Google Cloud to guarantee capacity.
3. If you're running Pydantic AI inside Google Cloud, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

You can authenticate using [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials), a service account, or an [API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode).

Whichever way you authenticate, you'll need to have the Vertex AI API (now branded as Google Cloud AI) enabled in your Google Cloud account.

#### Application Default Credentials

If you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you can use the `GoogleCloudProvider` by name:

```python {test="ci_only"}
from pydantic_ai import Agent

agent = Agent('google-cloud:gemini-3-pro-preview')
...
```

Or you can explicitly create the provider and model:

```python {test="ci_only"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider()
model = GoogleModel('gemini-3-pro-preview', provider=provider)
agent = Agent(model)
...
```

#### Service Account

To use a service account JSON file, explicitly create the provider and model:

```python {title="google_model_service_account.py" test="skip"}
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform'],
)
provider = GoogleCloudProvider(credentials=credentials, project='your-project-id')
model = GoogleModel('gemini-3-flash-preview', provider=provider)
agent = Agent(model)
...
```

#### API Key

To use Google Cloud with an API key, [create a key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode) and set it as an environment variable:

```bash
export GOOGLE_API_KEY=your-api-key
```

You can then use `GoogleModel` via the `GoogleCloudProvider` by name:

```python {test="ci_only"}
from pydantic_ai import Agent

agent = Agent('google-cloud:gemini-3-pro-preview')
...
```

Or you can explicitly create the provider and model:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(api_key='your-api-key')
model = GoogleModel('gemini-3-pro-preview', provider=provider)
agent = Agent(model)
...
```

#### Customizing Location or Project

You can specify the location and/or project when using Google Cloud:

```python {title="google_model_location.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(location='asia-east1', project='your-google-cloud-project-id')
model = GoogleModel('gemini-3-pro-preview', provider=provider)
agent = Agent(model)
...
```

#### Service tier (`service_tier`, `google_cloud_service_tier`)

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
model = GoogleModel('gemini-3-flash-preview', provider=provider)
agent = Agent(model)

result = agent.run_sync(
    'Hello!',
    model_settings=GoogleModelSettings(google_cloud_service_tier='pt_then_flex'),
)
```

Swap `'pt_then_flex'` for any [`GoogleCloudServiceTier`][pydantic_ai.models.google.GoogleCloudServiceTier] value — e.g. `'pt_then_priority'` for [Priority PayGo](https://cloud.google.com/vertex-ai/generative-ai/docs/priority-paygo) spillover, or `'flex_only'` / `'priority_only'` to bypass PT entirely.

After the request, inspect [`ModelResponse`][pydantic_ai.messages.ModelResponse] `provider_details.get('traffic_type')` (e.g. `ON_DEMAND_FLEX`, `ON_DEMAND_PRIORITY`) to see which tier served it, when the API returns it.

#### Model Garden

You can access models from the [Model Garden](https://cloud.google.com/model-garden?hl=en) that support the `generateContent` API and are available under your Google Cloud project, including but not limited to Gemini, using one of the following `model_name` patterns:

- `{model_id}` for Gemini models
- `{publisher}/{model_id}`
- `publishers/{publisher}/models/{model_id}`
- `projects/{project}/locations/{location}/publishers/{publisher}/models/{model_id}`

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(
    project='your-google-cloud-project-id',
    location='us-central1',  # the region where the model is available
)
model = GoogleModel('meta/llama-3.3-70b-instruct-maas', provider=provider)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the `GoogleProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

custom_http_client = AsyncClient(timeout=30)
model = GoogleModel(
    'gemini-3-pro-preview',
    provider=GoogleProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## HTTP Retries

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
    'gemini-3-pro-preview',
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

agent = Agent(GoogleModel('gemini-3-flash-preview'))
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

agent = Agent(GoogleModel('gemini-3-flash-preview', provider=provider))
result = agent.run_sync(
    [
        'What company is this logo from?',
        DocumentUrl(url=file.uri, media_type=file.mime_type),
    ]
)
print(result.output)
```

See the [input documentation](../input.md) for more details and examples.

## Model settings

You can customize model behavior using [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings]:

```python
from google.genai.types import HarmBlockThreshold, HarmCategory

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

settings = GoogleModelSettings(
    temperature=0.2,
    max_tokens=1024,
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('gemini-3-pro-preview')
agent = Agent(model, model_settings=settings)
...
```

### Configure thinking

Use the provider-agnostic [`Thinking`][pydantic_ai.capabilities.Thinking] capability to enable thinking:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('google:gemini-3.5-flash', capabilities=[Thinking(effort='medium')])
...
```

For advanced usage, you can pass Google's native thinking config through [`GoogleModelSettings.google_thinking_config`][pydantic_ai.models.google.GoogleModelSettings.google_thinking_config]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-3.5-flash')
model_settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True, 'thinking_level': 'MEDIUM'})
agent = Agent(model, model_settings=model_settings)
...
```

See [Thinking](../thinking.md) for the unified API and [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking) for Google's native thinking configuration.

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
model = GoogleModel('gemini-3-flash-preview')
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
    model_name='gemini-2.5-flash',
    provider=GoogleCloudProvider(location='europe-west1'),
)
agent = Agent(model, model_settings=model_settings)

result = agent.run_sync('Your prompt here')
# Access logprobs from provider_details
logprobs = result.response.provider_details.get('logprobs')
avg_logprobs = result.response.provider_details.get('avg_logprobs')
```

See the [Google Dev Blog](https://developers.googleblog.com/unlock-gemini-reasoning-with-logprobs-on-vertex-ai/) for more information.

### Context caching (`google_cached_content`)

When you've created a Gemini [cached content resource](https://ai.google.dev/gemini-api/docs/caching), pass its resource name through [`google_cached_content`][pydantic_ai.models.google.GoogleModelSettings.google_cached_content] to reuse it across requests:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(
    google_cached_content='projects/p/locations/global/cachedContents/your-cache-id',
)

agent = Agent(GoogleModel('gemini-2.5-pro'), model_settings=model_settings)
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
        model='gemini-2.5-flash',
        config=CreateCachedContentConfig(
            system_instruction='You are a geography expert. Be concise.',
            contents=[Content(role='user', parts=[Part(text='...long context to cache...')])],
            ttl='3600s',
        ),
    )
    print(cache.name)
    #> cachedContents/abc123...
    ```

    Caches have a minimum size (≈1024 tokens for `gemini-2.5-flash`, ≈4096 for `gemini-2.5-pro`) and a TTL — see the [Gemini caching docs](https://ai.google.dev/gemini-api/docs/caching) for the current thresholds, pricing, and `list` / `update` / `delete` operations.

## Streaming cancellation

!!! warning "Cancellation limitations"
    The `google-genai` SDK exposes streaming responses only as an async iterator, with no separate handle for closing the underlying HTTP transport. Because of a [Python language rule on async generators](https://peps.python.org/pep-0525/), [`cancel()`][pydantic_ai.result.StreamedRunResult.cancel] cannot interrupt an in-flight chunk read while another coroutine is iterating the stream. Pydantic AI marks the response with `state='interrupted'`, but upstream generation may continue until the surrounding `async with agent.run_stream(...)` block exits.

    For reliable cancellation, either pass `debounce_by=None` to [`stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text], [`stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output], or [`stream_response()`][pydantic_ai.result.StreamedRunResult.stream_response] and call `cancel()` from the same task that's iterating:

    ```python {title="cancel_google.py" test="skip"}
    from pydantic_ai import Agent

    agent = Agent('google:gemini-3-pro-preview')


    def should_stop(chunk: str) -> bool:
        return len(chunk) > 100


    async def main():
        async with agent.run_stream('Write a long essay about Python') as result:
            async for chunk in result.stream_text(debounce_by=None):
                if should_stop(chunk):
                    await result.cancel()
                    break
    ```

    Or, if you need to keep debouncing, wrap the stream with [`contextlib.aclosing`](https://docs.python.org/3/library/contextlib.html#contextlib.aclosing) so the iterator is closed before `cancel()` runs:

    ```python {title="cancel_google_aclosing.py" test="skip"}
    from contextlib import aclosing

    from pydantic_ai import Agent

    agent = Agent('google:gemini-3-pro-preview')


    def should_stop(chunk: str) -> bool:
        return len(chunk) > 100


    async def main():
        async with agent.run_stream('Write a long essay about Python') as result:
            async with aclosing(result.stream_text()) as stream:
                async for chunk in stream:
                    if should_stop(chunk):
                        break
            await result.cancel()
    ```

    Calling `cancel()` from a different task while iteration is in progress is not currently reliable on this provider.
