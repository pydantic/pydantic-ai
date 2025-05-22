# Google

The `GoogleModel` is a model that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to
access Google's Gemini models via both the Generative Language API and Vertex AI.

## Install

To use `GoogleModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```

---

!!! warning "Explicit instantiation required"
    You **cannot** currently use `Agent('google-gla:gemini-1.5-flash')` or `Agent('google-vertex:gemini-1.5-flash')` directly with `GoogleModel`. The model name inference will select [`GeminiModel`](../models/gemini.md) instead of `GoogleModel`.

    To use `GoogleModel`, you **must** explicitly instantiate a [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] and pass it to
    [`GoogleModel`][pydantic_ai.models.google.GoogleModel], then pass the model to [`Agent`][pydantic_ai.Agent].

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

To use Vertex AI, you may need to set up [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) or use a service account. You can also specify the region.

#### Application Default Credentials

If you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you can use:

```python
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

credentials = service_account.Credentials.from_service_account_file('path/to/service-account.json')
provider = GoogleProvider(credentials=credentials)
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

## Provider Argument

You can provide a custom `Provider` via the `provider` argument, for example to use a pre-initialized `genai.Client` or to customize authentication:

```python
from google import genai

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

client = genai.Client()
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

!!! warning
    When using Gemini models, document content is always sent as binary data, regardless of whether you use `DocumentUrl` or `BinaryContent`.
    This is due to differences in how Vertex AI and Google AI handle document inputs.

    See [this discussion](https://discuss.ai.google.dev/t/i-am-using-google-generative-ai-model-gemini-1-5-pro-for-image-analysis-but-getting-error/34866/4)
    for more details.

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
