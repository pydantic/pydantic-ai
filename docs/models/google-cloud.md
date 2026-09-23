# Google Cloud

Use Gemini and other supported Model Garden models through [`GoogleModel`][pydantic_ai.models.google.GoogleModel], or Claude through [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel]. Google Cloud's model APIs are also known as Vertex AI.

| Models | Configuration | Model selection |
| --- | --- | --- |
| Gemini | [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] | `google-cloud:<model-name>` |
| Model Garden models with a `generateContent` API | [Model Garden](#model-garden) | `GoogleModel` with `GoogleCloudProvider` |
| Claude | [Anthropic client for Vertex AI](#claude-on-google-cloud) | `AnthropicModel` with `AnthropicProvider` |

For Gemini through Google AI Studio instead, see [Google's Gemini API](google.md#api-key-gemini-api).

## Install

For Gemini and Model Garden models using `GoogleModel`, install the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```

## Authentication

If you are an enterprise user, you can also use `GoogleModel` to access Gemini via Google Cloud (formerly known as Vertex AI).

This interface has a number of advantages over the Gemini API:

1. The Google Cloud API comes with more enterprise readiness guarantees.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with Google Cloud to guarantee capacity.
3. If you're running Pydantic AI inside Google Cloud, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

You can authenticate using [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials), a service account, or an [API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode).

Whichever way you authenticate, you'll need to have the Vertex AI API (now branded as Google Cloud AI) enabled in your Google Cloud account.

### Application Default Credentials

If you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you can use the `GoogleCloudProvider` by name:

```python {test="ci_only"}
from pydantic_ai import Agent

agent = Agent('google-cloud:gemini-3.7-flash')
...
```

Or you can explicitly create the provider and model:

```python {test="ci_only"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider()
model = GoogleModel('gemini-3.7-flash', provider=provider)
agent = Agent(model)
...
```

### Service Account

To use a service account JSON file, explicitly create the provider and model:

```python {title="google_model_service_account.py" test="skip"}
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

credentials = service_account.Credentials.from_service_account_file('path/to/service-account.json')
provider = GoogleCloudProvider(credentials=credentials, project='your-project-id')
model = GoogleModel('gemini-3.7-flash', provider=provider)
agent = Agent(model)
...
```

!!! note "Credential scopes"
    [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] automatically applies
    `https://www.googleapis.com/auth/cloud-platform` to credentials that require scopes. Existing scopes are preserved.

### API Key

To use Google Cloud with an API key, [create a key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode) and set it as an environment variable:

```bash
export GOOGLE_API_KEY=your-api-key
```

You can then use `GoogleModel` via [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] by name:

```python {test="ci_only"}
from pydantic_ai import Agent

agent = Agent('google-cloud:gemini-3.7-flash')
...
```

Or you can explicitly create the provider and model:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(api_key='your-api-key')
model = GoogleModel('gemini-3.7-flash', provider=provider)
agent = Agent(model)
...
```

!!! note "Authentication precedence"
    Explicit `credentials` select credential-based authentication. Explicit `project` or `location`
    selects [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
    `GOOGLE_APPLICATION_CREDENTIALS` also takes precedence over an API key from the environment.
    `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` configure the ADC path but do not override
    an environment API key by themselves. Without explicit ADC arguments, an explicit `api_key`
    selects Express Mode.

### Customizing Location or Project

You can specify the location and/or project when using Google Cloud:

```python {title="google_model_location.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(location='global', project='your-google-cloud-project-id')
model = GoogleModel('gemini-3.7-flash', provider=provider)
agent = Agent(model)
...
```

In addition to the single-region values listed in
[`GoogleCloudLocation`][pydantic_ai.providers.google.GoogleCloudLocation], `GoogleCloudProvider` accepts the
`'global'` location and the `'us'`/`'eu'` multi-regions. The multi-region values are routed to the
`aiplatform.{us,eu}.rep.googleapis.com` data-residency endpoints — use them when an org policy blocks the
global endpoint for data residency, or when a model is initially available only on `global` and the
multi-regions rather than a single region. Model availability differs between single regions, multi-regions,
and `global`; see the
[Vertex AI locations docs](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#available-regions).

```python {title="google_model_multi_region.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google_cloud import GoogleCloudProvider

provider = GoogleCloudProvider(location='us', project='your-google-cloud-project-id')
model = GoogleModel('gemini-3.7-flash', provider=provider)
agent = Agent(model)
...
```

## Model Garden

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

## Model settings and features

Gemini on Google Cloud uses the same [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings] as the Gemini API. See the [Google model guide](google.md#model-settings) for thinking, safety settings, multimodal inputs, and context caching, including differences between the two services.

Google Cloud also supports [service tiers and provisioned throughput](google.md#service-tier-service_tier-google_cloud_service_tier) and [Model Armor](google.md#model-armor-google-cloud-only).

## Claude on Google Cloud

For Claude, install the `anthropic` optional group and pass an `AsyncAnthropicVertex` client to [`AnthropicProvider`][pydantic_ai.providers.anthropic.AnthropicProvider]. See [Claude on Google Cloud](anthropic.md#google-cloud) for the setup example. This uses Anthropic's Messages API, so configure it with [Anthropic model settings](anthropic.md#model-settings).
