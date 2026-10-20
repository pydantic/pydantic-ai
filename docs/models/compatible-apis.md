---
description: "Use OpenAI-compatible inference platforms and gateways such as Fireworks AI, Together AI, Vercel AI Gateway, LiteLLM and vLLM with Pydantic AI, or connect to any custom endpoint."
---

# Other compatible APIs

Many inference platforms, gateways, and self-hosted servers implement the OpenAI Chat Completions API. Pydantic AI has a provider for each service on this page, so using one works like using OpenAI: set the service's API key environment variable and pass `<provider>:<model>` to [`Agent`][pydantic_ai.Agent]:

```python
from pydantic_ai import Agent

agent = Agent('fireworks:accounts/fireworks/models/qwq-32b')
...
```

The provider sets the endpoint and authentication, and selects a [model profile](#model-profile) that accounts for the service's model names and API behavior. Services with their own guide, such as [DeepSeek](deepseek.md), [OpenRouter](openrouter.md), and [Ollama](ollama.md), are listed in the [provider directory](overview.md#provider-directory).

To connect to an endpoint Pydantic AI has no provider for, see [Other endpoints](#other-endpoints).

## Install

These integrations use the OpenAI SDK:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Alibaba Cloud Model Studio (DashScope)

To use Qwen models via [Alibaba Cloud Model Studio (DashScope)](https://www.alibabacloud.com/en/product/modelstudio), you can set the `ALIBABA_API_KEY` (or `DASHSCOPE_API_KEY`) environment variable and use [`AlibabaProvider`][pydantic_ai.providers.alibaba.AlibabaProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('alibaba:qwen-max')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.alibaba import AlibabaProvider

model = OpenAIChatModel(
    'qwen-max',
    provider=AlibabaProvider(api_key='your-api-key'),
)
agent = Agent(model)
...
```

The `AlibabaProvider` uses the international DashScope compatible endpoint `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` by default. You can override this by passing a custom `base_url`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.alibaba import AlibabaProvider

model = OpenAIChatModel(
    'qwen-max',
    provider=AlibabaProvider(
        api_key='your-api-key',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',  # China region
    ),
)
agent = Agent(model)
...
```

!!! note "Document input is not supported"
    The DashScope compatible-mode Chat Completions API does not accept document content parts, so passing a [`DocumentUrl`][pydantic_ai.messages.DocumentUrl] or document [`BinaryContent`][pydantic_ai.messages.BinaryContent] to an [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] backed by [`AlibabaProvider`][pydantic_ai.providers.alibaba.AlibabaProvider] raises a `UserError`.

## Fireworks AI

Go to [Fireworks.AI](https://fireworks.ai/) and create an API key in your account settings.

You can set the `FIREWORKS_API_KEY` environment variable and use [`FireworksProvider`][pydantic_ai.providers.fireworks.FireworksProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('fireworks:accounts/fireworks/models/qwq-32b')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.fireworks import FireworksProvider

model = OpenAIChatModel(
    'accounts/fireworks/models/qwq-32b',  # model library available at https://fireworks.ai/models
    provider=FireworksProvider(api_key='your-fireworks-api-key'),
)
agent = Agent(model)
...
```

## Heroku AI

To use [Heroku AI](https://www.heroku.com/ai), first create an API key.

You can set the `HEROKU_INFERENCE_KEY` and (optionally) `HEROKU_INFERENCE_URL` environment variables and use [`HerokuProvider`][pydantic_ai.providers.heroku.HerokuProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('heroku:claude-sonnet-4-5')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.heroku import HerokuProvider

model = OpenAIChatModel(
    'claude-sonnet-4-5',
    provider=HerokuProvider(api_key='your-heroku-inference-key'),
)
agent = Agent(model)
...
```

## LiteLLM

[`LiteLLMProvider`][pydantic_ai.providers.litellm.LiteLLMProvider] connects to a [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy), which translates requests for the upstream providers it is configured with. Pass the proxy URL (for example `http://localhost:<port>`) as `api_base` and your LiteLLM key (or a placeholder) as `api_key`. Requests are sent as Chat Completions to `api_base` as-is, so you can also point it at an OpenAI-compatible upstream directly, such as `https://api.openai.com/v1` with your OpenAI key, but not at an API that only LiteLLM can translate. Use the `custom/` model name prefix for custom LLMs.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

model = OpenAIChatModel(
    'openai/gpt-5.2',
    provider=LiteLLMProvider(
        api_base='<api-base-url>',
        api_key='<api-key>'
    )
)
agent = Agent(model)

result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
...
```

!!! note
    If your model rejects requests with more than one leading system message (for example, you
    see `System message must be at the beginning.`), set
    `openai_chat_supports_multiple_system_messages=False` on its profile. See
    [Models that accept only one leading system message](#models-that-accept-only-one-leading-system-message)
    for details.

## Nebius AI Studio

Go to [Nebius AI Studio](https://studio.nebius.com/) and create an API key.

You can set the `NEBIUS_API_KEY` environment variable and use [`NebiusProvider`][pydantic_ai.providers.nebius.NebiusProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('nebius:Qwen/Qwen3-32B-fast')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.nebius import NebiusProvider

model = OpenAIChatModel(
    'Qwen/Qwen3-32B-fast',
    provider=NebiusProvider(api_key='your-nebius-api-key'),
)
agent = Agent(model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

## OVHcloud AI Endpoints

To use OVHcloud AI Endpoints, you need to create a new API key. To do so, go to the [OVHcloud manager](https://ovh.com/manager), then in Public Cloud > AI Endpoints > API keys. Click on `Create a new API key` and copy your new key.

You can explore the [catalog](https://endpoints.ai.cloud.ovh.net/catalog) to find which models are available.

You can set the `OVHCLOUD_API_KEY` environment variable and use [`OVHcloudProvider`][pydantic_ai.providers.ovhcloud.OVHcloudProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('ovhcloud:gpt-oss-120b')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

If you need to configure the provider, you can use the [`OVHcloudProvider`][pydantic_ai.providers.ovhcloud.OVHcloudProvider] class:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ovhcloud import OVHcloudProvider

model = OpenAIChatModel(
    'gpt-oss-120b',
    provider=OVHcloudProvider(api_key='your-api-key'),
)
agent = Agent(model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

## SambaNova

To use [SambaNova Cloud](https://cloud.sambanova.ai/), you need to obtain an API key from the [SambaNova Cloud dashboard](https://cloud.sambanova.ai/dashboard).

SambaNova provides access to multiple model families including Meta Llama, DeepSeek, Qwen, and Mistral models with fast inference speeds.

You can set the `SAMBANOVA_API_KEY` environment variable and use [`SambaNovaProvider`][pydantic_ai.providers.sambanova.SambaNovaProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('sambanova:Meta-Llama-3.1-8B-Instruct')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.sambanova import SambaNovaProvider

model = OpenAIChatModel(
    'Meta-Llama-3.1-8B-Instruct',
    provider=SambaNovaProvider(api_key='your-api-key'),
)
agent = Agent(model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

For a complete list of available models, see the [SambaNova supported models documentation](https://docs.sambanova.ai/docs/en/models/sambacloud-models).

## Together AI

Go to [Together.ai](https://www.together.ai/) and create an API key in your account settings.

You can set the `TOGETHER_API_KEY` environment variable and use [`TogetherProvider`][pydantic_ai.providers.together.TogetherProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('together:meta-llama/Llama-3.3-70B-Instruct-Turbo-Free')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.together import TogetherProvider

model = OpenAIChatModel(
    'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',  # model library available at https://www.together.ai/models
    provider=TogetherProvider(api_key='your-together-api-key'),
)
agent = Agent(model)
...
```

`deepseek-ai/DeepSeek-V4-*` models reject a forced tool choice while thinking is on, and thinking is their default. Pydantic AI therefore never forces tool choice for those models on Together: explicit `tool_choice='required'` or a tool list raises a [`UserError`][pydantic_ai.exceptions.UserError], and resolved output-tool forcing is sent as `tool_choice='auto'`; unlike with [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider], the restriction is unconditional because whether Together honors DeepSeek's thinking toggle is unverified.

## Vercel AI Gateway

To use [Vercel's AI Gateway](https://vercel.com/docs/ai-gateway), first follow the [documentation](https://vercel.com/docs/ai-gateway) instructions on obtaining an API key or OIDC token.

You can set the `VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN` environment variable and use [`VercelProvider`][pydantic_ai.providers.vercel.VercelProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('vercel:anthropic/claude-sonnet-4-5')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vercel import VercelProvider

model = OpenAIChatModel(
    'anthropic/claude-sonnet-4-5',
    provider=VercelProvider(api_key='your-vercel-ai-gateway-api-key'),
)
agent = Agent(model)
...
```

## vLLM

[vLLM](https://docs.vllm.ai/) is a high-throughput inference server with an OpenAI-compatible API. Connect with [`VLLMProvider`][pydantic_ai.providers.vllm.VLLMProvider], setting `base_url` directly or through `VLLM_BASE_URL`. For authenticated servers, set `api_key` or `VLLM_API_KEY`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vllm import VLLMProvider

model = OpenAIChatModel(
    'Qwen/Qwen3.8-27B',
    provider=VLLMProvider(base_url='http://localhost:8000/v1'),
)
agent = Agent(model)

result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

With those environment variables set, you can instead reference the provider by name:

```python
from pydantic_ai import Agent

agent = Agent('vllm:Qwen/Qwen3.8-27B')

result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

!!! note "Tool calling requires server configuration"
    For agents that let the model decide whether to call a tool, start vLLM with `--enable-auto-tool-choice` and select the model-specific parser with `--tool-call-parser`. See the [vLLM tool calling guide](https://docs.vllm.ai/en/stable/features/tool_calling/) for supported models and parser values.

!!! note "Multiple system messages are merged by default"
    Some vLLM chat templates reject multiple leading system messages, so `VLLMProvider` merges them by default. To opt out, pass an [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] with `openai_chat_supports_multiple_system_messages=True`. See [Models that accept only one leading system message](#models-that-accept-only-one-leading-system-message).

## Other endpoints

For an OpenAI-compatible service or server that Pydantic AI has no provider for, use [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] with its `base_url` and `api_key`, or set the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables. Use [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] for a Chat Completions endpoint, or [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] if the service implements the Responses API:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>', api_key='your-api-key'
    ),
)
agent = Agent(model)
...
```

Compatibility with one API does not imply support for the other or for every OpenAI feature. `OpenAIProvider` also assumes OpenAI's models: it selects a profile from the model name as if it were an OpenAI model, whatever the `base_url`. For other models, features such as `thinking` or structured output may be ignored or rejected until you configure the [model profile](#model-profile).

## Model profile

A model profile tells the model class how to shape requests for a particular model and API: which JSON schema restrictions its tool definitions have, whether tools can be marked as strict, how reasoning is configured, and so on. The providers above select one automatically based on the model name, so you only need this section if a model behind a custom endpoint doesn't work correctly out of the box.

Pass your own [`ModelProfile`][pydantic_ai.profiles.ModelProfile] (for behaviors shared among all model classes) or [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (for behaviors specific to the OpenAI model classes) to tweak how requests are constructed:

```py
from pydantic_ai import Agent, InlineDefsJsonSchemaTransformer
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'model_name',
    provider=OpenAIProvider(
        base_url='https://<openai-compatible-api-endpoint>.com', api_key='your-api-key'
    ),
    profile=OpenAIModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,  # Supported by any model class via the base ModelProfile
        openai_supports_strict_tool_definition=False,  # Supported by OpenAIChatModel and OpenAIResponsesModel
        openai_chat_supports_multiple_system_messages=False,  # Supported by OpenAIChatModel only — for strict providers (e.g. some vLLM/LiteLLM setups) that require exactly one initial system message
        openai_chat_supports_max_completion_tokens=False,  # Supported by OpenAIChatModel only — for providers (e.g. OpenRouter) that only accept the older `max_tokens` field instead of `max_completion_tokens`
    )
)
agent = Agent(model)
```

A model supporting reasoning does not mean the endpoint accepts OpenAI's `reasoning_effort` values, so set the flags according to what the endpoint accepts, not only what the model can do.

### Gateways that serve several model families {#custom-openai-compatible-provider}

If you run your own gateway that has no provider class and routes to models from several developers, you can subclass [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] and override [`model_profile()`][pydantic_ai.providers.Provider.model_profile] to pick a profile from each model name, instead of passing `profile=` on every model. The helpers in [`pydantic_ai.profiles`](../api/profiles.md) return the profile for each model family:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import (
    OpenAIJsonSchemaTransformer,
    OpenAIModelProfile,
    openai_model_profile,
)
from pydantic_ai.providers.openai import OpenAIProvider


class GatewayProvider(OpenAIProvider):
    @property
    def name(self) -> str:
        return 'my-gateway'

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile:
        provider_to_profile = {
            'openai': openai_model_profile,
            'groq': groq_model_profile,
            'moonshotai': moonshotai_model_profile,
        }
        provider_name, _, model_name = model_name.partition('/')
        profile = None
        if profile_func := provider_to_profile.get(provider_name):
            profile = profile_func(model_name)
        return merge_profile(
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
            profile,
        )


provider = GatewayProvider(
    base_url='https://gateway.example/v1',
    api_key='your-gateway-api-key',
)
model = OpenAIChatModel('openai/gpt-5.6-sol', provider=provider)
agent = Agent(model)
```

The model name is only normalized for the profile lookup; the name sent to the gateway stays unchanged. Pydantic AI merges the returned profile with [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE], and you can pass gateway-specific overrides as a final argument to [`merge_profile()`][pydantic_ai.profiles.merge_profile]. The model class doesn't change with the profile: `OpenAIChatModel` still sends a Chat Completions request, whichever model family the gateway routes it to.

### Detect incomplete streamed responses

Some OpenAI-compatible APIs can close a Chat Completions stream cleanly without a terminal
`finish_reason`, making a partial response look complete: like a non-streamed response without a finish
reason, it gets a [`finish_reason`][pydantic_ai.messages.ModelResponse.finish_reason] of `'stop'`. If your provider guarantees that complete
streams include a finish reason, set
[`openai_chat_streaming_requires_finish_reason=True`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_streaming_requires_finish_reason]
in the model profile. Pydantic AI will then raise [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]
when the stream reaches EOF without one. The option defaults to `False` because some compatible APIs
do not guarantee the field.

### Models that accept only one leading system message

Some models are served with a chat template (applied server-side, for example by [vLLM](https://docs.vllm.ai/), [LiteLLM](#litellm), or TGI) that accepts only a single system message at the start of the conversation and rejects additional ones. Sending more than one fails with a `400` error such as `System message must be at the beginning.` or `Conversation roles must alternate ...`, seen with some newer Qwen, Mistral, Gemma, and Command-R models. It's easy to hit without intending to, since more than one leading system message can be produced in several ways.

Set `openai_chat_supports_multiple_system_messages=False` on the model's [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (as shown in [Model profile](#model-profile)) to merge the leading run of system messages into one, joined with two newlines, before the request is sent. The merge is lossless, so it's safe to enable whenever a backend rejects multiple system messages.
