# OpenAI

## Install

To use OpenAI models or OpenAI-compatible APIs, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

To use `OpenAIModel` with the OpenAI API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

You can then use `OpenAIModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel('gpt-4o')
agent = Agent(model)
...
```

By default, the `OpenAIModel` uses the `OpenAIProvider` with the `base_url` set to `https://api.openai.com/v1`.

## Configure the provider

If you want to pass parameters in code to the provider, you can programmatically instantiate the
[OpenAIProvider][pydantic_ai.providers.openai.OpenAIProvider] and pass it to the model:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

## Custom OpenAI Client

`OpenAIProvider` also accepts a custom `AsyncOpenAI` client via the `openai_client` parameter, so you can customise the `organization`, `project`, `base_url` etc. as defined in the [OpenAI API docs](https://platform.openai.com/docs/api-reference).

```python {title="custom_openai_client.py"}
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncOpenAI(max_retries=3)
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
agent = Agent(model)
...
```

You could also use the [`AsyncAzureOpenAI`](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints) client
to use the Azure OpenAI API. Note that the `AsyncAzureOpenAI` is a subclass of `AsyncOpenAI`.

```python
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

## OpenAI Responses API

Pydantic AI also supports OpenAI's [Responses API](https://platform.openai.com/docs/api-reference/responses) through the `OpenAIResponsesModel` class.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model)
...
```

The Responses API has built-in tools that you can use instead of building your own:

- [Web search](https://platform.openai.com/docs/guides/tools-web-search): allow models to search the web for the latest information before generating a response.
- [File search](https://platform.openai.com/docs/guides/tools-file-search): allow models to search your files for relevant information before generating a response.
- [Computer use](https://platform.openai.com/docs/guides/tools-computer-use): allow models to use a computer to perform tasks on your behalf.

You can use the `OpenAIResponsesModelSettings` class to make use of those built-in tools:

```python
from openai.types.responses import WebSearchToolParam  # (1)!

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model_settings = OpenAIResponsesModelSettings(
    openai_builtin_tools=[WebSearchToolParam(type='web_search_preview')],
)
model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model=model, model_settings=model_settings)

result = agent.run_sync('What is the weather in Tokyo?')
print(result.output)
"""
As of 7:48 AM on Wednesday, April 2, 2025, in Tokyo, Japan, the weather is cloudy with a temperature of 53°F (12°C).
"""
```

1. The file search tool and computer use tool can also be imported from `openai.types.responses`.

You can learn more about the differences between the Responses API and Chat Completions API in the [OpenAI API docs](https://platform.openai.com/docs/guides/responses-vs-chat-completions).

## OpenAI-compatible Models

Many providers and models are compatible with the OpenAI API, and can be used with `OpenAIModel` in Pydantic AI.
Before getting started, check the [installation and configuration](#install) instructions above.

To use another OpenAI-compatible API, you can make use of the `base_url` and `api_key` arguments from `OpenAIProvider`:

```python
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

Various providers also have their own provider classes so that you don't need to specify the base URL yourself and you can use the standard `<PROVIDER>_API_KEY` environment variable to set the API key.
When a provider has its own provider class, you can use the `Agent("<provider>:<model>")` shorthand, e.g. `Agent("deepseek:deepseek-chat")` or `Agent("openrouter:google/gemini-2.5-pro-preview")`, instead of building the `OpenAIModel` explicitly. Similarly, you can pass the provider name as a string to the `provider` argument on `OpenAIModel` instead of building instantiating the provider class explicitly.

#### Model Profile

Sometimes, the provider or model you're using will have slightly different requirements than OpenAI's API or models, like having different restrictions on JSON schemas for tool definitions, or not supporting tool definitions to be marked as strict.

When using an alternative provider class provided by Pydantic AI, an appropriate model profile is typically selected automatically based on the model name.
If the model you're using is not working correctly out of the box, you can tweak various aspects of how model requests are constructed by providing your own [`ModelProfile`][pydantic_ai.profiles.ModelProfile] (for behaviors shared among all model classes) or [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (for behaviors specific to `OpenAIModel`):

```py
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com', api_key='your-api-key'
    ),
    profile=OpenAIModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,  # Supported by any model class on a plain ModelProfile
        openai_supports_strict_tool_definition=False  # Supported by OpenAIModel only, requires OpenAIModelProfile
    )
)
agent = Agent(model)
```

### DeepSeek

To use the [DeepSeek](https://deepseek.com) provider, first create an API key by following the [Quick Start guide](https://api-docs.deepseek.com/).
Once you have the API key, you can use it with the [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider]:

```python
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

```python
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

```bash
ollama run llama3.2
```

(this will pull the `llama3.2` model if you don't already have it downloaded)

Then run your code, here's a minimal example:

```python
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
agent = Agent(ollama_model, output_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

#### Example using a remote server

```python
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


agent = Agent(model=ollama_model, output_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65)
```

1. The name of the model running on the remote server
2. The url of the remote server

### Azure AI Foundry

If you want to use [Azure AI Foundry](https://ai.azure.com/) as your provider, you can do so by using the
[`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] class.

```python
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

Once you have the API key, you can use it with the [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

model = OpenAIModel(
    'anthropic/claude-3.5-sonnet',
    provider=OpenRouterProvider(api_key='your-openrouter-api-key'),
)
agent = Agent(model)
...
```

### Vercel AI Gateway

To use [Vercel's AI Gateway](https://vercel.com/docs/ai-gateway), first follow the [documentation](https://vercel.com/docs/ai-gateway) instructions on obtaining an API key or OIDC token.

You can set your credentials using one of these environment variables:

```bash
export VERCEL_AI_GATEWAY_API_KEY='your-ai-gateway-api-key'
# OR
export VERCEL_OIDC_TOKEN='your-oidc-token'
```

Once you have set the environment variable, you can use it with the [`VercelProvider`][pydantic_ai.providers.vercel.VercelProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.vercel import VercelProvider

# Uses environment variable automatically
model = OpenAIModel(
    'anthropic/claude-4-sonnet',
    provider=VercelProvider(),
)
agent = Agent(model)

# Or pass the API key directly
model = OpenAIModel(
    'anthropic/claude-4-sonnet',
    provider=VercelProvider(api_key='your-vercel-ai-gateway-api-key'),
)
agent = Agent(model)
...
```

### Grok (xAI)

Go to [xAI API Console](https://console.x.ai/) and create an API key.
Once you have the API key, you can use it with the [`GrokProvider`][pydantic_ai.providers.grok.GrokProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.grok import GrokProvider

model = OpenAIModel(
    'grok-2-1212',
    provider=GrokProvider(api_key='your-xai-api-key'),
)
agent = Agent(model)
...
```

### MoonshotAI

Create an API key in the [Moonshot Console](https://platform.moonshot.ai/console).
With that key you can instantiate the [`MoonshotAIProvider`][pydantic_ai.providers.moonshotai.MoonshotAIProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.moonshotai import MoonshotAIProvider

model = OpenAIModel(
    'kimi-k2-0711-preview',
    provider=MoonshotAIProvider(api_key='your-moonshot-api-key'),
)
agent = Agent(model)
...
```

### GitHub Models

To use [GitHub Models](https://docs.github.com/en/github-models), you'll need a GitHub personal access token with the `models: read` permission.

Once you have the token, you can use it with the [`GitHubProvider`][pydantic_ai.providers.github.GitHubProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.github import GitHubProvider

model = OpenAIModel(
    'xai/grok-3-mini',  # GitHub Models uses prefixed model names
    provider=GitHubProvider(api_key='your-github-token'),
)
agent = Agent(model)
...
```

You can also set the `GITHUB_API_KEY` environment variable:

```bash
export GITHUB_API_KEY='your-github-token'
```

GitHub Models supports various model families with different prefixes. You can see the full list on the [GitHub Marketplace](https://github.com/marketplace?type=models) or the public [catalog endpoint](https://models.github.ai/catalog/models).

### Perplexity

Follow the Perplexity [getting started](https://docs.perplexity.ai/guides/getting-started)
guide to create an API key. Then, you can query the Perplexity API with the following:

```python
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
Once you have the API key, you can use it with the [`FireworksProvider`][pydantic_ai.providers.fireworks.FireworksProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.fireworks import FireworksProvider

model = OpenAIModel(
    'accounts/fireworks/models/qwq-32b',  # model library available at https://fireworks.ai/models
    provider=FireworksProvider(api_key='your-fireworks-api-key'),
)
agent = Agent(model)
...
```

### Together AI

Go to [Together.ai](https://www.together.ai/) and create an API key in your account settings.
Once you have the API key, you can use it with the [`TogetherProvider`][pydantic_ai.providers.together.TogetherProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.together import TogetherProvider

model = OpenAIModel(
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',  # model library available at https://www.together.ai/models
    provider=TogetherProvider(api_key='your-together-api-key'),
)
agent = Agent(model)
...
```

### Heroku AI

To use [Heroku AI](https://www.heroku.com/ai), you can use the [`HerokuProvider`][pydantic_ai.providers.heroku.HerokuProvider]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.heroku import HerokuProvider

model = OpenAIModel(
    'claude-3-7-sonnet',
    provider=HerokuProvider(api_key='your-heroku-inference-key'),
)
agent = Agent(model)
...
```

You can set the `HEROKU_INFERENCE_KEY` and `HEROKU_INFERENCE_URL` environment variables to set the API key and base URL, respectively:

```bash
export HEROKU_INFERENCE_KEY='your-heroku-inference-key'
export HEROKU_INFERENCE_URL='https://us.inference.heroku.com'
```
