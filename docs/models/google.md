# Google

The `GoogleModel` is a model that uses the [`google-genai`](https://pypi.org/project/google-genai/) package under the hood to
access Google's Gemini models via both the Generative Language API and Vertex AI.

## Install

To use `GoogleModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```


## Configuration

`GoogleModel` lets you use Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods) (`generativelanguage.googleapis.com`) or [Vertex AI API](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models) (`*-aiplatform.googleapis.com`).

### API Key (Generative Language API)

To use Gemini via the Generative Language API, go to [aistudio.google.com](https://aistudio.google.com/apikey) and create an API key.

Once you have the API key, set it as an environment variable:

```bash
export GOOGLE_API_KEY=your-api-key
```

You can then use `GoogleModel` by name (where GLA stands for Generative Language API):

```python
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.5-pro')
...
```

Or you can explicitly create the provider:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(api_key='your-api-key')
model = GoogleModel('gemini-2.5-pro', provider=provider)
agent = Agent(model)
...
```

### Vertex AI (Enterprise/Cloud)

If you are an enterprise user, you can also use `GoogleModel` to access Gemini via Vertex AI.

This interface has a number of advantages over the Generative Language API:

1. The VertexAI API comes with more enterprise readiness guarantees.
2. You can [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput) with Vertex AI to guarantee capacity.
3. If you're running Pydantic AI inside GCP, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective, and might improve latency.

You can authenticate using [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials), a service account, or an [API key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode).

Whichever way you authenticate, you'll need to have Vertex AI enabled in your GCP account.

#### Application Default Credentials

If you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you can use `GoogleProvider` in Vertex AI mode by name:

```python {test="ci_only"}
from pydantic_ai import Agent

agent = Agent('google-vertex:gemini-2.5-pro')
...
```

Or you can explicitly create the provider and model:

```python {test="ci_only"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True)
model = GoogleModel('gemini-2.5-pro', provider=provider)
agent = Agent(model)
...
```

#### Service Account

To use a service account JSON file, explicitly create the provider and model:

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
model = GoogleModel('gemini-2.5-flash', provider=provider)
agent = Agent(model)
...
```

#### API Key

To use Vertex AI with an API key, [create a key](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys?usertype=expressmode) and set it as an environment variable:

```bash
export GOOGLE_API_KEY=your-api-key
```

You can then use `GoogleModel` in Vertex AI mode by name:

```python {test="ci_only"}
from pydantic_ai import Agent

agent = Agent('google-vertex:gemini-2.5-pro')
...
```

Or you can explicitly create the provider and model:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True, api_key='your-api-key')
model = GoogleModel('gemini-2.5-pro', provider=provider)
agent = Agent(model)
...
```

#### Customizing Location or Project

You can specify the location and/or project when using Vertex AI:

```python {title="google_model_location.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(vertexai=True, location='asia-east1', project='your-gcp-project-id')
model = GoogleModel('gemini-2.5-pro', provider=provider)
agent = Agent(model)
...
```

#### Model Garden

You can access models from the [Model Garden](https://cloud.google.com/model-garden?hl=en) that support the `generateContent` API and are available under your GCP project, including but not limited to Gemini, using one of the following `model_name` patterns:

- `{model_id}` for Gemini models
- `{publisher}/{model_id}`
- `publishers/{publisher}/models/{model_id}`
- `projects/{project}/locations/{location}/publishers/{publisher}/models/{model_id}`

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

provider = GoogleProvider(
    project='your-gcp-project-id',
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
    'gemini-2.5-pro',
    provider=GoogleProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```


## Document, Image, Audio, and Video Input

`GoogleModel` supports multi-modal input, including documents, images, audio, and video. See the [input documentation](../input.md) for details and examples.

## Enhanced JSON Schema Support

As of November 2025, Google Gemini models (2.5+) provide enhanced support for JSON Schema features when using [`NativeOutput`](../output.md#native-output), enabling more sophisticated structured outputs:

### Supported Features

- **Property Ordering**: The order of properties in your Pydantic model definition is now preserved in the output
- **Title Fields**: The `title` field is supported for providing short property descriptions
- **Union Types (`anyOf` and `oneOf`)**: Full support for conditional structures using Python's `Union` or `|` type syntax
- **Recursive Schemas (`$ref` and `$defs`)**: Full support for self-referential models and reusable schema definitions, enabling tree structures and recursive data
- **Numeric Constraints**: `minimum` and `maximum` constraints are respected (note: `exclusiveMinimum` and `exclusiveMaximum` are not yet supported)
- **Optional Fields (`type: 'null'`)**: Proper handling of optional fields with `None` values
- **Additional Properties**: Dictionary fields with `dict[str, T]` are fully supported
- **Tuple Types (`prefixItems`)**: Support for tuple-like array structures

### Example: Recursive Schema

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.output import NativeOutput

class TreeNode(BaseModel):
    """A tree node that can contain child nodes."""
    value: int
    children: list['TreeNode'] | None = None

model = GoogleModel('gemini-2.5-pro')
agent = Agent(model, output_type=NativeOutput(TreeNode))

result = await agent.run('Create a tree with root value 1 and two children with values 2 and 3')
# result.output will be a TreeNode with proper structure
```

### Example: Union Types

```python
from typing import Union, Literal
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.output import NativeOutput

class Success(BaseModel):
    status: Literal['success']
    data: str

class Error(BaseModel):
    status: Literal['error']
    error_message: str

class Response(BaseModel):
    result: Union[Success, Error]

model = GoogleModel('gemini-2.5-pro')
agent = Agent(model, output_type=NativeOutput(Response))

result = await agent.run('Process this request successfully')
# result.output.result will be either Success or Error
```

See the [structured output documentation](../output.md) for more details on using `NativeOutput` with Pydantic models.

## Model settings

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
model = GoogleModel('gemini-2.5-flash')
agent = Agent(model, model_settings=settings)
...
```

### Disable thinking

You can disable thinking by setting the `thinking_budget` to `0` on the `google_thinking_config`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model_settings = GoogleModelSettings(google_thinking_config={'thinking_budget': 0})
model = GoogleModel('gemini-2.5-flash')
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
model = GoogleModel('gemini-2.5-flash')
agent = Agent(model, model_settings=model_settings)
...
```

See the [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for more on safety settings.
