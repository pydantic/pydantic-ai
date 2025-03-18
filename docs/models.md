PydanticAI is Model-agnostic and has built in support for the following model providers:

* [OpenAI](#openai)
* [Anthropic](#anthropic)
* Gemini via two different APIs: [Generative Language API](#gemini) and [VertexAI API](#gemini-via-vertexai)
* [Ollama](#ollama)
* [Groq](#groq)
* [Mistral](#mistral)
* [Cohere](#cohere)
* [Bedrock](#bedrock)

See [OpenAI-compatible models](#openai-compatible-models) for more examples on how to use models such as [OpenRouter](#openrouter), and [Grok (xAI)](#grok-xai) that support the OpenAI SDK.

You can also [add support for other models](#implementing-custom-models).

PydanticAI also comes with [`TestModel`](api/models/test.md) and [`FunctionModel`](api/models/function.md) for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right packages installed.

## Models, Interfaces, and Providers


PydanticAI uses a few key terms to describe how it interacts with different LLMs:

* **Model**: This refers to the specific LLM model you want to handle your requests (e.g., `gpt-4o`, `claude-3-5-sonnet-latest`,
    `gemini-1.5-flash`). It's the "brain" that processes your prompts and generates responses.  You specify the
    _Model_ as a parameter to the _Interface_.
* **Interface**: This refers to a PydanticAI class used to make requests following a specific LLM API
    (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a
    vendor-SDK-agnostic API, ensuring a single PydanticAI agent is portable to different LLM vendors without
    any other code changes just by swapping out the _Interface_ it uses. Currently, interface classes are named
    roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIModel`, `AnthropicModel`, `GeminiModel`,
    etc. These `Model` classes will soon be renamed to `<VendorSdk>Interface` to reflect this terminology better.
* **Provider**: This refers to _Interface_-specific classes which handle the authentication and connections to an LLM vendor.
    Passing a non-default _Provider_ as a parameter to an _Interface_ is how you can ensure that your agent will make
    requests to a specific endpoint, or make use of a specific approach to authentication (e.g., you can use Vertex-specific
    auth with the `GeminiModel` by way of the `VertexProvider`). In particular, this is how you can make use of an AI gateway,
    or an LLM vendor that offers API compatibility with the vendor SDK used by an existing interface (such as `OpenAIModel`).

In short, you select a *model*, PydanticAI uses the appropriate *interface* class, and the *provider* handles the
connection and authentication to the underlying service.

## OpenAI

### Install

To use OpenAI models, you need to either install [`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `openai` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

### Configuration

To use [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] through their main API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

You can then use [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] by name:

```python {title="openai_model_by_name.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
...
```

Or initialise the model directly with just the model name:

```python {title="openai_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o')
agent = Agent(model)
...
```
By default, the `OpenAIModel` uses the [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider.__init__]
with the `base_url` set to `https://api.openai.com/v1`.

### `provider` argument

You can provide a custom [`Provider`][pydantic_ai.providers.Provider] via the [`provider` argument][pydantic_ai.models.openai.OpenAIModel.__init__]:

```python {title="openai_model_provider.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

### Custom OpenAI Client

`OpenAIProvider` also accepts a custom `AsyncOpenAI` client via the
[`openai_client` parameter][pydantic_ai.providers.openai.OpenAIProvider.__init__], so you can customise the
`organization`, `project`, `base_url` etc. as defined in the [OpenAI API docs](https://platform.openai.com/docs/api-reference).

You could also use the [`AsyncAzureOpenAI`](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints)
client to use the Azure OpenAI API.

```python {title="openai_azure.py"}
from openai import AsyncAzureOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncAzureOpenAI(
    azure_endpoint='...',
    api_version='2024-07-01-preview',
    api_key='your-api-key',
)

model = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=client),
)
agent = Agent(model)
...
```

## Anthropic

### Install

To use [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel] models, you need to either install [`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `anthropic` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[anthropic]'
```

### Configuration

To use [Anthropic](https://anthropic.com) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.

[`AnthropicModelName`][pydantic_ai.models.anthropic.AnthropicModelName] contains a list of available Anthropic models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

You can then use [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel] by name:

```py title="anthropic_model_by_name.py"
from pydantic_ai import Agent

agent = Agent('anthropic:claude-3-5-sonnet-latest')
...
```

Or initialise the model directly with just the model name:

```py title="anthropic_model_init.py"
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-3-5-sonnet-latest')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom [`Provider`][pydantic_ai.providers.Provider] via the [`provider` argument][pydantic_ai.models.anthropic.AnthropicModel.__init__]:

```py title="anthropic_model_provider.py"
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'claude-3-5-sonnet-latest', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

### Custom HTTP Client

You can customize the `AnthropicProvider` with a custom `httpx.AsyncClient`:

```py title="anthropic_model_custom_provider.py"
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
    'claude-3-5-sonnet-latest',
    provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Gemini

### Install

To use [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] models, you just need to install [`pydantic-ai`](install.md) or [`pydantic-ai-slim`](install.md#slim-install), no extra dependencies are required.

### Configuration

[`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] let's you use the Google's Gemini models through their [Generative Language API](https://ai.google.dev/api/all-methods), `generativelanguage.googleapis.com`.

[`GeminiModelName`][pydantic_ai.models.gemini.GeminiModelName] contains a list of available Gemini models that can be used through this interface.

To use `GeminiModel`, go to [aistudio.google.com](https://aistudio.google.com/apikey) and select "Create API key".

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GEMINI_API_KEY=your-api-key
```

You can then use [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] by name:

```python {title="gemini_model_by_name.py"}
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.0-flash')
...
```

!!! note
    The `google-gla` provider prefix represents the [Google **G**enerative **L**anguage **A**PI](https://ai.google.dev/api/all-methods) for `GeminiModel`s.
    `google-vertex` is used with [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models).

Or initialise the model directly with just the model name and provider:

```python {title="gemini_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-gla')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom [`Provider`][pydantic_ai.providers.Provider] via the [`provider` argument][pydantic_ai.models.gemini.GeminiModel.__init__]:

```python {title="gemini_model_provider.py"}
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
```python {title="gemini_model_custom_provider.py"}
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

If you are an enterprise user, you should use the `google-vertex` provider with [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel] which uses the `*-aiplatform.googleapis.com` API.

[`GeminiModelName`][pydantic_ai.models.gemini.GeminiModelName] contains a list of available Gemini models that can be used through this interface.

### Install

To use the `google-vertex` provider with [`GeminiModel`][pydantic_ai.models.gemini.GeminiModel], you need to either install
[`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `vertexai` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[vertexai]'
```

### Configuration

This interface has a number of advantages over `generativelanguage.googleapis.com` documented above:

1. The VertexAI API comes with more enterprise readiness guarantees.
2. You can
   [purchase provisioned throughput](https://cloud.google.com/vertex-ai/generative-ai/docs/provisioned-throughput#purchase-provisioned-throughput)
   with VertexAI to guarantee capacity.
3. If you're running PydanticAI inside GCP, you don't need to set up authentication, it should "just work".
4. You can decide which region to use, which might be important from a regulatory perspective,
   and might improve latency.

The big disadvantage is that for local development you may need to create and configure a "service account", which I've found extremely painful to get right in the past.

Whichever way you authenticate, you'll need to have VertexAI enabled in your GCP account.

### Application default credentials

Luckily if you're running PydanticAI inside GCP, or you have the [`gcloud` CLI](https://cloud.google.com/sdk/gcloud) installed and configured, you should be able to use `VertexAIModel` without any additional setup.

To use `VertexAIModel`, with [application default credentials](https://cloud.google.com/docs/authentication/application-default-credentials) configured (e.g. with `gcloud`), you can simply use:

```python {title="vertexai_application_default_credentials.py"}
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel('gemini-2.0-flash', provider='google-vertex')
agent = Agent(model)
...
```

Internally this uses [`google.auth.default()`](https://google-auth.readthedocs.io/en/master/reference/google.auth.html) from the `google-auth` package to obtain credentials.

!!! note "Won't fail until `agent.run()`"

    Because `google.auth.default()` requires network requests and can be slow, it's not run until you call `agent.run()`.

You may also need to pass the [`project_id` argument to `GoogleVertexProvider`][pydantic_ai.providers.google_vertex.GoogleVertexProvider] if application default credentials don't set a project, if you pass `project_id` and it conflicts with the project set by application default credentials, an error is raised.

### Service account

If instead of application default credentials, you want to authenticate with a service account, you'll need to create a service account, add it to your GCP project (note: AFAIK this step is necessary even if you created the service account within the project), give that service account the "Vertex AI Service Agent" role, and download the service account JSON file.

Once you have the JSON file, you can use it thus:

```python {title="vertexai_service_account.py"}
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

```python {title="vertexai_service_account.py" hl_lines="7-9"}
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

### Customising region

Whichever way you authenticate, you can specify which region requests will be sent to via the [`region` argument][pydantic_ai.providers.google_vertex.GoogleVertexProvider].

Using a region close to your application can improve latency and might be important from a regulatory perspective.

```python {title="vertexai_region.py"}
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
```python {title="vertexai_custom_provider.py"}
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

## Groq

### Install

To use [`GroqModel`][pydantic_ai.models.groq.GroqModel], you need to either install [`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `groq` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[groq]'
```

### Configuration

To use [Groq](https://groq.com/) through their API, go to [console.groq.com/keys](https://console.groq.com/keys) and follow your nose until you find the place to generate an API key.

[`GroqModelName`][pydantic_ai.models.groq.GroqModelName] contains a list of available Groq models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export GROQ_API_KEY='your-api-key'
```

You can then use [`GroqModel`][pydantic_ai.models.groq.GroqModel] by name:

```python {title="groq_model_by_name.py"}
from pydantic_ai import Agent

agent = Agent('groq:llama-3.3-70b-versatile')
...
```

Or initialise the model directly with just the model name:

```python {title="groq_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel

model = GroqModel('llama-3.3-70b-versatile')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom [`Provider`][pydantic_ai.providers.Provider] via the
[`provider` argument][pydantic_ai.models.groq.GroqModel.__init__]:

```python {title="groq_model_provider.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

model = GroqModel(
    'llama-3.3-70b-versatile', provider=GroqProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the [`GroqProvider`][pydantic_ai.providers.groq.GroqProvider] with a
custom `httpx.AsyncHTTPClient`:

```python {title="groq_model_custom_provider.py"}
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.groq import GroqProvider

custom_http_client = AsyncClient(timeout=30)
model = GroqModel(
    'llama-3.3-70b-versatile',
    provider=GroqProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Mistral

### Install

To use [`MistralModel`][pydantic_ai.models.mistral.MistralModel], you need to either install [`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `mistral` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[mistral]'
```

### Configuration

To use [Mistral](https://mistral.ai) through their API, go to [console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/) and follow your nose until you find the place to generate an API key.

[`LatestMistralModelNames`][pydantic_ai.models.mistral.LatestMistralModelNames] contains a list of the most popular Mistral models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export MISTRAL_API_KEY='your-api-key'
```

You can then use [`MistralModel`][pydantic_ai.models.mistral.MistralModel] by name:

```python {title="mistral_model_by_name.py"}
from pydantic_ai import Agent

agent = Agent('mistral:mistral-large-latest')
...
```

Or initialise the model directly with just the model name:

```python {title="mistral_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

model = MistralModel('mistral-small-latest')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom [`Provider`][pydantic_ai.providers.Provider] via the
[`provider` argument][pydantic_ai.models.mistral.MistralModel.__init__]:

```python {title="groq_model_provider.py"}
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

model = MistralModel(
    'mistral-large-latest', provider=MistralProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

You can also customize the provider with a custom `httpx.AsyncHTTPClient`:

```python {title="groq_model_custom_provider.py"}
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel
from pydantic_ai.providers.mistral import MistralProvider

custom_http_client = AsyncClient(timeout=30)
model = MistralModel(
    'mistral-large-latest',
    provider=MistralProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Cohere

### Install

To use [`CohereModel`][pydantic_ai.models.cohere.CohereModel], you need to either install [`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `cohere` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[cohere]'
```

### Configuration

To use [Cohere](https://cohere.com/) through their API, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key.

[`CohereModelName`][pydantic_ai.models.cohere.CohereModelName] contains a list of the most popular Cohere models.

### Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export CO_API_KEY='your-api-key'
```

You can then use [`CohereModel`][pydantic_ai.models.cohere.CohereModel] by name:

```python {title="cohere_model_by_name.py"}
from pydantic_ai import Agent

agent = Agent('cohere:command')
...
```

Or initialise the model directly with just the model name:

```python {title="cohere_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command', api_key='your-api-key')
agent = Agent(model)
...
```

### `api_key` argument

If you don't want to or can't set the environment variable, you can pass it at runtime via the [`api_key` argument][pydantic_ai.models.cohere.CohereModel.__init__]:

```python {title="cohere_model_api_key.py"}
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel

model = CohereModel('command', api_key='your-api-key')
agent = Agent(model)
...
```

## Bedrock

### Install

To use [`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel], you need to either install [`pydantic-ai`](install.md), or install [`pydantic-ai-slim`](install.md#slim-install) with the `bedrock` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[bedrock]'
```

### Configuration

To use [AWS Bedrock](https://aws.amazon.com/bedrock/), you'll need an AWS account with Bedrock enabled and appropriate credentials. You can use either AWS credentials directly or a pre-configured boto3 client.

[`BedrockModelName`][pydantic_ai.models.bedrock.BedrockModelName] contains a list of available Bedrock models, including models from Anthropic, Amazon, Cohere, Meta, and Mistral.

### Environment variables

You can set your AWS credentials as environment variables:

```bash
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_REGION='us-east-1'  # or your preferred region
```

You can then use [`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel] by name:

```python {title="bedrock_model_by_name.py", test="skip"}
from pydantic_ai import Agent

agent = Agent('bedrock:anthropic.claude-3-sonnet-20240229-v1:0')
...
```

Or initialize the model directly with just the model name:

```python {title="bedrock_model_init.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

model = BedrockConverseModel('anthropic.claude-3-sonnet-20240229-v1:0')
agent = Agent(model)
...
```

### `provider` argument

You can provide a custom [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider] via the [`provider` argument][pydantic_ai.models.bedrock.BedrockConverseModel.__init__]. This is useful when you want to specify credentials directly or use a custom boto3 client:

```python {title="bedrock_model_provider.py"}
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using AWS credentials directly
model = BedrockConverseModel(
    'anthropic.claude-3-sonnet-20240229-v1:0',
    provider=BedrockProvider(
        region_name='us-east-1',
        aws_access_key_id='your-access-key',
        aws_secret_access_key='your-secret-key',
    ),
)
agent = Agent(model)
...
```

You can also pass a pre-configured boto3 client:

```python {title="bedrock_model_boto3.py"}
import boto3

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using a pre-configured boto3 client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
model = BedrockConverseModel(
    'anthropic.claude-3-sonnet-20240229-v1:0',
    provider=BedrockProvider(bedrock_client=bedrock_client),
)
agent = Agent(model)
...
```

## OpenAI-compatible Models

Many of the models are compatible with OpenAI API, and thus can be used with [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] in PydanticAI.
Before getting started, check the [OpenAI](#openai) section for installation and configuration instructions.

To use another OpenAI-compatible API, you can make use of the [`base_url`][pydantic_ai.providers.openai.OpenAIProvider.__init__]
and [`api_key`][pydantic_ai.providers.openai.OpenAIProvider.__init__] arguments from `OpenAIProvider`:

```python {title="deepseek_model_init.py" hl_lines="5-6"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com', api_key='your-api-key'
    ),
)
agent = Agent(model)
...
```

You can also use the `provider` argument with a custom provider class like the [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider]:

```python {title="deepseek_model_init_provider_class.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

model = OpenAIModel(
    'deepseek-chat',
    provider=DeepSeekProvider(api_key='your-deepseek-api-key'),
)
agent = Agent(model)
...
```
You can also customize any provider with a custom `http_client`:
```python {title="deepseek_model_init_provider_custom.py"}
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider

custom_http_client = AsyncClient(timeout=30)
model = OpenAIModel(
    'deepseek-chat',
    provider=DeepSeekProvider(
        api_key='your-deepseek-api-key', http_client=custom_http_client
    ),
)
agent = Agent(model)
...
```

### Ollama

To use [Ollama](https://ollama.com/), you must first download the Ollama client, and then download a model using the [Ollama model library](https://ollama.com/library).

You must also ensure the Ollama server is running when trying to make requests to it. For more information, please see the [Ollama documentation](https://github.com/ollama/ollama/tree/main/docs).

#### Example local usage

With `ollama` installed, you can run the server with the model you want to use:

```bash {title="terminal-run-ollama"}
ollama run llama3.2
```

(this will pull the `llama3.2` model if you don't already have it downloaded)

Then run your code, here's a minimal example:

```python {title="ollama_example.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class CityLocation(BaseModel):
    city: str
    country: str


ollama_model = OpenAIModel(
    model_name='llama3.2', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)
agent = Agent(ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

#### Example using a remote server

```python {title="ollama_example_with_remote_server.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

ollama_model = OpenAIModel(
    model_name='qwen2.5-coder:7b',  # (1)!
    provider=OpenAIProvider(base_url='http://192.168.1.74:11434/v1'),  # (2)!
)


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent(model=ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""
```

1. The name of the model running on the remote server
2. The url of the remote server

### Azure AI Foundry

If you want to use [Azure AI Foundry](https://ai.azure.com/) as your provider, you can do so by using the
[`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] class.

```python {title="azure_provider_example.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIModel(
    'gpt-4o',
    provider=AzureProvider(
        azure_endpoint='your-azure-endpoint',
        api_version='your-api-version',
        api_key='your-api-key',
    ),
)
agent = Agent(model)
...
```

### OpenRouter

To use [OpenRouter](https://openrouter.ai), first create an API key at [openrouter.ai/keys](https://openrouter.ai/keys).

Once you have the API key, you can use it with the [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]:

```python {title="openrouter_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key='your-openrouter-api-key',
    ),
)
agent = Agent(model)
...
```

### Grok (xAI)

Go to [xAI API Console](https://console.x.ai/) and create an API key.
Once you have the API key, you can use it with the [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]:

```python {title="grok_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'grok-2-1212',
    provider=OpenAIProvider(base_url='https://api.x.ai/v1', api_key='your-xai-api-key'),
)
agent = Agent(model)
...
```

### Perplexity

Follow the Perplexity [getting started](https://docs.perplexity.ai/guides/getting-started)
guide to create an API key. Then, you can query the Perplexity API with the following:

```py {title="perplexity_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'sonar-pro',
    provider=OpenAIProvider(
        base_url='https://api.perplexity.ai',
        api_key='your-perplexity-api-key',
    ),
)
agent = Agent(model)
...
```

### Fireworks AI

Go to [Fireworks.AI](https://fireworks.ai/) and create an API key in your account settings.
Once you have the API key, you can use it with the [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]:

```python {title="fireworks_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'accounts/fireworks/models/qwq-32b',  # model library available at https://fireworks.ai/models
    provider=OpenAIProvider(
        base_url='https://api.fireworks.ai/inference/v1',
        api_key='your-fireworks-api-key',
    ),
)
agent = Agent(model)
...
```

### Together AI

Go to [Together.ai](https://www.together.ai/) and create an API key in your account settings.
Once you have the API key, you can use it with the [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider]:

```python {title="together_model_init.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',  # model library available at https://www.together.ai/models
    provider=OpenAIProvider(
        base_url='https://api.together.xyz/v1',
        api_key='your-together-api-key',
    ),
)
agent = Agent(model)
...
```

## Implementing Custom Models

To implement support for models not already supported, you will need to subclass the [`Model`][pydantic_ai.models.Model] abstract base class.

For streaming, you'll also need to implement the following abstract base class:

* [`StreamedResponse`][pydantic_ai.models.StreamedResponse]

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to PydanticAI, see the [contributing guidelines](contributing.md#new-model-rules).

## Fallback

You can use [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to attempt multiple models
in sequence until one returns a successful result. Under the hood, PydanticAI automatically switches
from one model to the next if the current model returns a 4xx or 5xx status code.

In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key),
and then falls back to the Anthropic model.

```python {title="fallback_model.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel

openai_model = OpenAIModel('gpt-4o', api_key='not-valid')
anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
print(response.data)
#> Paris

print(response.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='What is the capital of France?',
                timestamp=datetime.datetime(...),
                part_kind='user-prompt',
            )
        ],
        kind='request',
    ),
    ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='claude-3-5-sonnet-latest',
        timestamp=datetime.datetime(...),
        kind='response',
    ),
]
"""
```

The `ModelResponse` message above indicates in the `model_name` field that the result was returned by the Anthropic model, which is the second model specified in the `FallbackModel`.

!!! note
    Each model's options should be configured individually. For example, `base_url`, `api_key`, and custom clients should be set on each model itself, not on the `FallbackModel`.

In this next example, we demonstrate the exception-handling capabilities of `FallbackModel`.
If all models fail, a [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] is raised, which
contains all the exceptions encountered during the `run` execution.

=== "Python >=3.11"

    ```python {title="fallback_model_failure.py" py="3.11"}
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.openai import OpenAIModel

    openai_model = OpenAIModel('gpt-4o', api_key='not-valid')
    anthropic_model = AnthropicModel('claude-3-5-sonnet-latest', api_key='not-valid')
    fallback_model = FallbackModel(openai_model, anthropic_model)

    agent = Agent(fallback_model)
    try:
        response = agent.run_sync('What is the capital of France?')
    except* ModelHTTPError as exc_group:
        for exc in exc_group.exceptions:
            print(exc)
    ```

=== "Python <3.11"

    Since [`except*`](https://docs.python.org/3/reference/compound_stmts.html#except-star) is only supported
    in Python 3.11+, we use the [`exceptiongroup`](https://github.com/agronholm/exceptiongroup) backport
    package for earlier Python versions:

    ```python {title="fallback_model_failure.py" noqa="F821" test="skip"}
    from exceptiongroup import catch

    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelHTTPError
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.openai import OpenAIModel


    def model_status_error_handler(exc_group: BaseExceptionGroup) -> None:
        for exc in exc_group.exceptions:
            print(exc)


    openai_model = OpenAIModel('gpt-4o', api_key='not-valid')
    anthropic_model = AnthropicModel('claude-3-5-sonnet-latest', api_key='not-valid')
    fallback_model = FallbackModel(openai_model, anthropic_model)

    agent = Agent(fallback_model)
    with catch({ModelHTTPError: model_status_error_handler}):
        response = agent.run_sync('What is the capital of France?')
    ```

By default, the `FallbackModel` only moves on to the next model if the current model raises a
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. You can customize this behavior by
passing a custom `fallback_on` argument to the `FallbackModel` constructor.
