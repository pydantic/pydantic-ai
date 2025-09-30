# Google

The `GoogleModel` is a model that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to
access Google's Gemini models via both the Generative Language API and Vertex AI.

## Install

To use `GoogleModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```

---

## Configuration

`GoogleModel` lets you use Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods) (`generativelanguage.googleapis.com`) or [Vertex AI API](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) (`*-aiplatform.googleapis.com`).

### API Key (Generative Language API)

To use Gemini via the Generative Language API, go to [aistudio.google.com](https://aistudio.google.com/apikey) and create an API key.

Once you have the API key, set it as an environment variable:

```bash
export GOOGLE_API_KEY=your-api-key
```

You can then use `GoogleModel` by explicitly creating a provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-api-key')
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

### Vertex AI (Enterprise/Cloud)

If you are an enterprise user, you can use the `google-vertex` provider with `GoogleModel` to access Gemini via Vertex AI.

This interface has a number of advantages over the Generative Language API:

1. The VertexAI API comes with more enterprise readiness guarantees.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with VertexAI to guarantee capacity.
3. If you're running Pydantic AI inside GCP, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

The big disadvantage is that for local development you may need to create and configure a "service account", which can be challenging to get right.

Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.

To use Vertex AI, you may need to set up [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) or use a service account. You can also specify the region.

#### Application Default Credentials

If you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you can use:

```python {test="ci_only"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True)
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

#### Service Account

To use a service account JSON file:

```python {title="google_model_service_account.py" test="skip"}
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform'],
)
provider = GoogleProvider(credentials=credentials, project='your-project-id')
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

#### Customizing Location

You can specify the location when using Vertex AI:

```python {title="google_model_location.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True, location='asia-east1')
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

#### Customizing Model

You can access models from the [Model Garden](https://cloud.google.com/model-garden?hl=en) that support the generateContent API and are available under your GCP project, including but not limited to Gemini, using one of the following `model_name` patterns: 

- `{model_id}` for Gemini models
- `{publisher}/{model_id}`
- `publishers/{publisher}/models/{model_id}`
- `projects/{project}/locations/{location}/publishers/{publisher}/models/{model_id}`

```python {test="skip"}
from google.oauth2 import service_account

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

credentials = service_account.Credentials.from_service_account_file(
    'path/to/service-account.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform'],
)
provider = GoogleProvider(
    credentials=credentials,
    project='your-gcp-project-id',
    location='us-central1',  # the region where the model is available
)
model = GoogleModel('meta/llama-3.3-70b-instruct-maas', provider=provider)
agent = Agent(model)
...
```

## Provider Argument

You can supply a custom `GoogleProvider` instance using the `provider` argument to configure advanced client options, such as setting a custom `base_url`.

This is useful if you're using a custom-compatible endpoint with the Google Generative Language API.

```python
from google.genai import Client
from google.genai.types import HttpOptions

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

client = Client(
    api_key='gemini-custom-api-key',
    http_options=HttpOptions(base_url='gemini-custom-base-url'),
)
provider = GoogleProvider(client=client)
model = GoogleModel('gemini-1.5-flash', provider=provider)
agent = Agent(model)
...
```

## Model Settings

You can customize model behavior using [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings]:

```python
from google.genai.types import HarmBlockThreshold, HarmCategory

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

settings = GoogleModelSettings(
    temperature=0.2,
    max_tokens=1024,
    google_thinking_config={'thinking_budget': 2048},
    google_safety_settings=[
        {
            'category': HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            'threshold': HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        }
    ]
)
model = GoogleModel('gemini-1.5-flash')
agent = Agent(model, model_settings=settings)
...
```

See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for more on safety settings, and [thinking config](https://ai.google.dev/gemini-api/docs/thinking).

## Document, Image, Audio, and Video Input

`GoogleModel` supports multi-modal input, including documents, images, audio, and video. See the [input documentation](../input.md) for details and examples.

## Model settings

You can use the [`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings] class to customize the model request.

### Disable thinking

You can disable thinking by setting the `thinking_budget` to `0` on the `google_thinking_config`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(google_thinking_config={'thinking_budget': 0})
model = GoogleModel('gemini-2.0-flash')
agent = Agent(model, model_settings=model_settings)
...
```

Check out the [Gemini API docs](https://ai.google.dev/gemini-api/docs/thinking) for more on thinking.

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
model = GoogleModel('gemini-2.0-flash')
agent = Agent(model, model_settings=model_settings)
...
```

See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for more on safety settings.
