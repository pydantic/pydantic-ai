# Gemini

!!! note
    We've developed a new Google model called `GoogleModel` which uses `google-genai` under the hood.

    Honestly, Google packages are a mess, and that's why we've used plain `httpx` instead of relying on their own client
    to create `GeminiModel`. That said, it's easier to use the `google-genai` package directly, since they keep the package
    up-to-date with the latest API changes. For that reason, we've created a new model called `GoogleModel` which uses
    `google-genai` under the hood.

    Check it out [here](../api/models/google.md).

Pydantic AI supports Google's Gemini models through two different APIs:

- Generative Language API (`generativelanguage.googleapis.com`)
- Vertex AI API (`*-aiplatform.googleapis.com`)

## Gemini via Generative Language API

### Install

To use `GeminiModel` models, you just need to install `pydantic-ai` or `pydantic-ai-slim`, no extra dependencies are required.

### Configuration

`GeminiModel` lets you use Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.

`GeminiModelName` contains a list of available Gemini models that can be used through this interface.

To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/apikey) and select "Create API key".

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GEMINI_API_KEY=your-api-key
```

You can then use `GeminiModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.0-flash')
...
```

!!! note
    The `google-gla` provider prefix represents the [Google **G**enerative **L**anguage **A**PI](https://ai.google.dev/api/all-methods) for `GeminiModel`s.
    `google-vertex` is used with [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

Or initialise the model directly with just the model name and provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-gla')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleGLAProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the `GoogleGLAProvider` with a custom `http_client`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

custom_http_client = AsyncClient(timeout=30)
model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleGLAProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Gemini via VertexAI

If you are an enterprise user, you should use the `google-vertex` provider with `GeminiModel` which uses the `*-aiplatform.googleapis.com` API.

`GeminiModelName` contains a list of available Gemini models that can be used through this interface.

### Install

To use the `google-vertex` provider with `GeminiModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `vertexai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[vertexai]"
```

### Configuration

This interface has a number of advantages over `generativelanguage.googleapis.com` documented above:

1. The VertexAI API comes with more enterprise readiness guarantees.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with VertexAI to guarantee capacity.
3. If you're running Pydantic AI inside GCP, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

The big disadvantage is that for local development you may need to create and configure a "service account", which can be challenging to get right.

Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.

### Application default credentials

Luckily if you're running Pydantic AI inside GCP, or you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you should be able to use `VertexAIModel` without any additional setup.

To use `VertexAIModel`, with [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) configured (e.g. with `gcloud`), you can simply use:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-vertex')
agent = Agent(model)
...
```

Internally this uses [`google.auth.default()`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) from the `google-auth` package to obtain credentials.

!!! note "Won't fail until `agent.run()`"
    Because `google.auth.default()` requires network requests and can be slow, it's not run until you call `agent.run()`.

You may also need to pass the `project_id` argument to `GoogleVertexProvider` if application default credentials don't set a project, if you pass `project_id` and it conflicts with the project set by application default credentials, an error is raised.

### Service account

If instead of application default credentials, you want to authenticate with a service account, you'll need to create a service account, add it to your GCP project (note: this step is necessary even if you created the service account within the project), give that service account the "Vertex AI Service Agent" role, and download the service account JSON file.

Once you have the JSON file, you can use it thus:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleVertexProvider(service_account_file='path/to/service-account.json'),
)
agent = Agent(model)
...
```

Alternatively, if you already have the service account information in memory, you can pass it as a dictionary:

```python
import json

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

service_account_info = json.loads(
    '{"type": "service_account", "project_id": "my-project-id"}'
)
model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleVertexProvider(service_account_info=service_account_info),
)
agent = Agent(model)
...
```

### Customizing region

Whichever way you authenticate, you can specify which region requests will be sent to via the `region` argument.

Using a region close to your application can improve latency and might be important from a regulatory perspective.

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

model = GeminiModel(
    'gemini-2.0-flash', provider=GoogleVertexProvider(region='asia-east1')
)
agent = Agent(model)
...
```

You can also customize the `GoogleVertexProvider` with a custom `http_client`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_vertex import GoogleVertexProvider

custom_http_client = AsyncClient(timeout=30)
model = GeminiModel(
    'gemini-2.0-flash',
    provider=GoogleVertexProvider(region='asia-east1', http_client=custom_http_client),
)
agent = Agent(model)
...
```

### Model settings

You can use the [`GeminiModelSettings`][pydantic_ai.models.gemini.GeminiModelSettings] class to customize the model request.

#### Disable thinking

You can disable thinking by setting the `thinking_budget` to `0` on the `google_thinking_config`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings

model_settings = GeminiModelSettings(gemini_thinking_config={'thinking_budget': 0})
model = GeminiModel('gemini-2.0-flash')
agent = Agent(model, model_settings=model_settings)
...
```

Check out the [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking) for more on thinking.

#### Safety settings

You can customize the safety settings by setting the `google_safety_settings` field.

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings

model_settings = GeminiModelSettings(
    gemini_safety_settings=[
        {
            'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
            'threshold': 'BLOCK_ONLY_HIGH',
        }
    ]
)
model = GeminiModel('gemini-2.0-flash')
agent = Agent(model, model_settings=model_settings)
...
```

Check out the [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for more on safety settings.
