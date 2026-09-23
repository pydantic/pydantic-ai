# Other compatible APIs

Many hosted providers, gateways, and local inference servers implement OpenAI-compatible APIs. Use the [provider directory](overview.md#provider-directory) to find a named integration, or configure a custom endpoint below.

Chat Completions endpoints use [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]; Responses endpoints use [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel]. Compatibility with one API does not imply support for the other or for every OpenAI feature.

## Install

Install Pydantic AI with the OpenAI SDK used by this integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Use the provider class for the service you are calling when one is available, such as [`OpenRouterProvider`][pydantic_ai.providers.openrouter.OpenRouterProvider], [`LiteLLMProvider`][pydantic_ai.providers.litellm.LiteLLMProvider] for a [LiteLLM proxy](#litellm), or [`VLLMProvider`][pydantic_ai.providers.vllm.VLLMProvider] for a [local or remote vLLM server](#vllm).
These providers configure authentication and select [model profiles](#model-profile) that account for the service's model names and API behavior.
You can also use the `Agent("<provider>:<model>")` shorthand, e.g. `Agent("openrouter:openai/gpt-5.6-sol")`, or pass the provider name to `OpenAIChatModel(provider=...)`.

If the service has no dedicated provider, you can use [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] with a custom `base_url` and `api_key`, or the `OPENAI_BASE_URL` and `OPENAI_API_KEY` environment variables:

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

!!! note "A custom URL does not change model profile selection"

    `OpenAIProvider` still selects a profile using OpenAI model names, even with a custom `base_url`.
    It does not infer the service from the URL or resolve gateway IDs such as `groq/qwen/qwen3-32b` to another provider's profile.
    An incorrect profile can cause settings such as `thinking` to be ignored or apply the wrong restrictions to sampling and tool schemas.
    For a service without a dedicated provider, configure the [model profile](#model-profile) or define a [custom provider](#custom-openai-compatible-provider) to match both the model and the gateway's API behavior.

## Model Profile

Sometimes, the provider or model you're using will have slightly different requirements than OpenAI's API or models, like having different restrictions on JSON schemas for tool definitions, or not supporting tool definitions to be marked as strict.

When using an alternative provider class provided by Pydantic AI, an appropriate model profile is typically selected automatically based on the model name.
For a custom endpoint, profile selection and request translation must agree: a model supporting reasoning does not mean its API accepts OpenAI's `reasoning_effort` values.
If the model you're using is not working correctly out of the box, you can tweak various aspects of how model requests are constructed by providing your own [`ModelProfile`][pydantic_ai.profiles.ModelProfile] (for behaviors shared among all model classes) or [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (for behaviors specific to `OpenAIChatModel`):

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

### Custom providers for gateways {#custom-openai-compatible-provider}

If your gateway routes requests to multiple providers, subclass [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] and override [`model_profile()`][pydantic_ai.providers.Provider.model_profile] to resolve its model IDs.
This selects a profile for every model using that provider, so you do not need to pass `profile=` on each model.
Only normalize the name for profile lookup; the model ID sent to the gateway stays unchanged.

Like the built-in providers, use helpers from [`pydantic_ai.profiles`](../api/profiles.md) to select the underlying model's profile, then apply any gateway-specific overrides.
For example, this gateway uses `openrouter/<provider>/<model>` for OpenRouter routes and `<provider>/<model>` for its other routes:

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
        provider_name, _, model_name = model_name.removeprefix('openrouter/').partition('/')
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
model = OpenAIChatModel('openrouter/openai/gpt-5.6-sol', provider=provider)
agent = Agent(model)
```

Extend the mapping and normalization rules for the models and aliases your gateway serves.
The OpenAI JSON schema transformer is a fallback; a model-family helper can supply its own transformer.
The returned profile replaces `OpenAIProvider`'s profile selection; Pydantic AI merges it with [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE] automatically.
Add gateway-specific overrides as a final argument to [`merge_profile()`][pydantic_ai.profiles.merge_profile], after the model-family profile.

Profile selection does not switch the model class: `OpenAIChatModel` still constructs an OpenAI Chat Completions request.
Set capability flags according to what your gateway accepts; the model-family helpers cannot account for its request translation or API restrictions.

### Detect incomplete streamed responses

Some OpenAI-compatible APIs can close a Chat Completions stream cleanly without a terminal
`finish_reason`, making a partial response look complete. If your provider guarantees that complete
streams include a finish reason, set
[`openai_chat_streaming_requires_finish_reason=True`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_streaming_requires_finish_reason]
in the model profile. Pydantic AI will then raise [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]
when the stream reaches EOF without one. The option defaults to `False` because some compatible APIs
do not guarantee the field.

### Models that accept only one leading system message

Some models are served with a chat template (applied server-side, for example by [vLLM](https://docs.vllm.ai/), [LiteLLM](#litellm), or TGI) that accepts only a single system message at the start of the conversation and rejects additional ones. Sending more than one fails with a `400` error such as `System message must be at the beginning.` or `Conversation roles must alternate ...`, seen with some newer Qwen, Mistral, Gemma, and Command-R models. It's easy to hit without intending to, since more than one leading system message can be produced in several ways.

Set `openai_chat_supports_multiple_system_messages=False` on the model's [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (as shown above) to merge the leading run of system messages into one, joined with two newlines, before the request is sent. The merge is lossless, so it's safe to enable whenever a backend rejects multiple system messages.

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

## Vercel AI Gateway

To use [Vercel's AI Gateway](https://vercel.com/docs/ai-gateway), first follow the [documentation](https://vercel.com/docs/ai-gateway) instructions on obtaining an API key or OIDC token.

You can set the `VERCEL_AI_GATEWAY_API_KEY` and `VERCEL_OIDC_TOKEN` environment variables and use [`VercelProvider`][pydantic_ai.providers.vercel.VercelProvider] by name:

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

## GitHub Models

!!! warning "GitHub Models has been retired"
    GitHub Models was [retired on July 30, 2026](https://docs.github.com/en/github-models) — the playground, model catalog, and inference API are no longer available. [`GitHubProvider`][pydantic_ai.providers.github.GitHubProvider] is therefore deprecated and will be removed in v3.

    For model access going forward, GitHub recommends [Azure AI Foundry](https://ai.azure.com/) or [GitHub Copilot](https://docs.github.com/en/copilot), which Pydantic AI supports through [`GitHubCopilotModel`](github-copilot.md).

## Perplexity

Follow the Perplexity [getting started](https://docs.perplexity.ai/guides/getting-started)
guide to create an API key, then initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'sonar-pro',
    provider=OpenAIProvider(
        base_url='https://api.perplexity.ai',
        api_key='your-perplexity-api-key',
    ),
)
agent = Agent(model)
...
```

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

To use [LiteLLM](https://www.litellm.ai/), set the configs as outlined in the [doc](https://docs.litellm.ai/docs/set_keys). In `LiteLLMProvider`, you can pass `api_base` and `api_key`. The value of these configs will depend on your setup. For example, if you are using OpenAI models, then you need to pass `https://api.openai.com/v1` as the `api_base` and your OpenAI API key as the `api_key`. If you are using a LiteLLM proxy server running on your local machine, then you need to pass `http://localhost:<port>` as the `api_base` and your LiteLLM API key (or a placeholder) as the `api_key`.

To use custom LLMs, use `custom/` prefix in the model name.

Once you have the configs, use the [`LiteLLMProvider`][pydantic_ai.providers.litellm.LiteLLMProvider] as follows:

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

You can customize the base URL if needed:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.sambanova import SambaNovaProvider

model = OpenAIChatModel(
    'DeepSeek-R1-0528',
    provider=SambaNovaProvider(
        api_key='your-api-key',
        base_url='https://custom.endpoint.com/v1',
    ),
)
agent = Agent(model)
...
```

## Atlas Cloud

[Atlas Cloud](https://www.atlascloud.ai/) is an OpenAI-compatible API gateway that provides access to 300+ models from a single endpoint, including DeepSeek, Qwen, Claude, GPT, and Gemini.

Atlas Cloud doesn't have a dedicated provider class, so you can use it with [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] by setting the `base_url` and `api_key`.
For its non-OpenAI model IDs, configure the [model profile](#model-profile) or a [custom provider](#custom-openai-compatible-provider) for the model and gateway behavior:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIChatModel(
    'deepseek-ai/deepseek-v4-pro',
    provider=OpenAIProvider(
        base_url='https://api.atlascloud.ai/v1',
        api_key='your-atlas-cloud-api-key',
    ),
)
agent = Agent(model)
...
```

## Rapid-MLX (Apple Silicon)

[Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) is an OpenAI-compatible inference server for Apple Silicon, built on Apple's MLX framework.

```bash
pip install rapid-mlx
rapid-mlx serve mlx-community/Qwen3.5-4B-MLX-4bit
```

The server listens on `http://localhost:8000/v1` and implements the OpenAI chat completions API, so you can point [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] at it:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

rapid_mlx_model = OpenAIChatModel(
    model_name='default',
    provider=OpenAIProvider(
        base_url='http://localhost:8000/v1',
        api_key='not-needed',
    ),
)
agent = Agent(rapid_mlx_model)
```
