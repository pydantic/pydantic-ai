---
description: "Use OpenAI-compatible inference platforms and gateways such as Fireworks AI, Together AI, Vercel AI Gateway, LiteLLM and vLLM with Pydantic AI, or connect to any custom endpoint."
---

# Other compatible APIs

Many inference platforms, gateways, and self-hosted servers implement the OpenAI Chat Completions API. Pydantic AI has a provider for most services on this page, so using one works like using OpenAI: set the service's API key environment variable and pass `<provider>:<model>` to [`Agent`][pydantic_ai.Agent]:

```python
from pydantic_ai import Agent

agent = Agent('fireworks:accounts/fireworks/models/qwq-32b')
...
```

The provider sets the endpoint and authentication, and selects a [model profile](#model-profile) that accounts for the service's model names and API behavior. Services with their own guide, such as [DeepSeek](deepseek.md), [OpenRouter](openrouter.md), and [Ollama](ollama.md), are listed in the [provider directory](overview.md#provider-directory).

To connect to an endpoint Pydantic AI has no provider for, such as [OCI Generative AI](#oci-generative-ai), see [Other endpoints](#other-endpoints).

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

Use [`NativeOutput`][pydantic_ai.output.NativeOutput] for Fireworks' schema-constrained structured output.
Fireworks enforces the schema through its OpenAI-compatible `response_format` parameter.
As the [Fireworks structured output documentation](https://docs.fireworks.ai/structured-responses/structured-response-formatting) describes, `pattern` enforcement is best-effort: unsupported regex constructs, such as lookahead and lookbehind, are ignored, leaving the field as an unconstrained string. Pydantic AI still validates the output, so a value that doesn't match the pattern triggers an output retry.

## Heroku AI

To use [Heroku AI](https://www.heroku.com/ai), first create an API key.

You can set the `HEROKU_INFERENCE_KEY` and (optionally) `HEROKU_INFERENCE_URL` environment variables and use [`HerokuProvider`][pydantic_ai.providers.heroku.HerokuProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('heroku:claude-4-5-sonnet')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.heroku import HerokuProvider

model = OpenAIChatModel(
    'claude-4-5-sonnet',
    provider=HerokuProvider(api_key='your-heroku-inference-key'),
)
agent = Agent(model)
...
```

## LiteLLM

!!! note "No `litellm` package dependency"
    `LiteLLMProvider` does not install or import the `litellm` Python package. It talks to a LiteLLM proxy over HTTP using the OpenAI client.

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

## Otari

[Otari](https://otari.ai) is Mozilla.ai's open-source LLM gateway, which adds API key management, budgets, usage tracking and fallback routing in front of the providers it routes to. Use the hosted gateway at [otari.ai](https://otari.ai), or [run your own](https://github.com/mozilla-ai/otari).

Create an API key on the **API keys** page of the otari.ai dashboard, set the `OTARI_API_KEY` environment variable and use [`OtariProvider`][pydantic_ai.providers.otari.OtariProvider] by name. Name the model as `<provider>:<model>`, or use the name of one of your routing policies or aliases:

```python
from pydantic_ai import Agent

agent = Agent('otari:anthropic:claude-sonnet-5-5')
...
```

Requests go to otari.ai's US region by default. Pass `region='eu'` for an account in the EU region, or point `base_url` (or the `OTARI_BASE_URL` environment variable) at your self-hosted gateway's `/api/v1` URL:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.otari import OtariProvider

eu_agent = Agent(
    OpenAIChatModel(
        'anthropic:claude-sonnet-5-5',
        provider=OtariProvider(api_key='your-otari-api-key', region='eu'),
    )
)
self_hosted_agent = Agent(
    OpenAIChatModel(
        'openai:gpt-5.2',
        provider=OtariProvider(api_key='your-otari-api-key', base_url='http://localhost:8000/api/v1'),
    )
)
...
```

The provider part of the model name selects the [model profile](#model-profile), so `anthropic:claude-sonnet-5-5` gets Claude's. After a host of many vendors' models, the vendor that follows decides: `groq:openai/gpt-oss-120b` gets gpt-oss's.

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

## Stav

[Stav](https://stav.ai) is a European AI inference gateway: one API key for models from many labs, addressed as `creator/model` (for example `deepseek-ai/deepseek-v4-flash-0731` or `mistral-ai/mistral-large-4`), with the option to keep compute in the EU or EEA. Create an API key on the [API keys page](https://stav.ai/app/connect/api-keys) of your Stav workspace.

You can set the `STAV_API_KEY` environment variable and use [`StavProvider`][pydantic_ai.providers.stav.StavProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('stav:deepseek-ai/deepseek-v4-flash-0731')
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

By default Stav may run a request in any region. To keep compute inside a jurisdiction, pass `region`: `'eu'`, `'eea'`, a country code such as `'no'`, or a list such as `['no', 'se']` that allows any of them. Stav refuses a request it can't serve there instead of sending it elsewhere, with a 403 `endpoint_not_admissible` error that Pydantic AI raises as [`ModelAuthenticationError`][pydantic_ai.exceptions.ModelAuthenticationError]; the key is fine, so widen the region or pick another model:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.stav import StavProvider

model = OpenAIChatModel(
    'mistral-ai/mistral-large-4',
    provider=StavProvider(api_key='your-stav-api-key', region='eu'),
)
agent = Agent(model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

Set `STAV_BASE_URL` or pass `base_url` to point the provider at another Stav endpoint.

Stav also serves the Responses API, so you can use [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] with the same provider. OpenAI and xAI models are passed through to their own Responses API; Stav translates other models onto Chat Completions and lists any request parameters it had to drop in the `X-Stav-Dropped-Params` response header. Unlike OpenAI, Stav doesn't store responses unless you set [`openai_store`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_store], which only passed-through models accept.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.stav import StavProvider

model = OpenAIResponsesModel(
    'openai/gpt-5-mini',
    provider=StavProvider(api_key='your-stav-api-key', region='eea'),
)
agent = Agent(model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
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

`deepseek-ai/DeepSeek-V4-*` models, including point-release ids such as `deepseek-ai/DeepSeek-V4.1-Flash`, reject a forced tool choice while thinking is on, and thinking is their default. Pydantic AI therefore never forces tool choice for those models on Together: explicit `tool_choice='required'` or a tool list raises a [`UserError`][pydantic_ai.exceptions.UserError], and resolved output-tool forcing is sent as `tool_choice='auto'`; unlike with [`DeepSeekProvider`][pydantic_ai.providers.deepseek.DeepSeekProvider], the restriction is unconditional because whether Together honors DeepSeek's thinking toggle is unverified.

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

For an OpenAI-compatible service or server that Pydantic AI has no provider for, use [`OpenAICompatibleProvider`][pydantic_ai.providers.openai_compatible.OpenAICompatibleProvider] with a name for the service, its `base_url` and, if it needs one, its `api_key`. Use [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] for a Chat Completions endpoint, or [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] if the service implements the Responses API:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

model = OpenAIChatModel(
    'model_name',
    provider=OpenAICompatibleProvider(
        'my-service',
        base_url='https://<openai-compatible-api-endpoint>',
        api_key='your-api-key',
    ),
)
agent = Agent(model)
...
```

The name is recorded as the [`provider_name`][pydantic_ai.messages.ModelResponse.provider_name] of every response and in OpenTelemetry attributes, and it decides which data from earlier responses is sent back to the service: response and item IDs only go back to a provider with the same name, and encrypted reasoning only to one with the same [canonical system](../message-history.md#sending-history-to-another-provider), which defaults to the name. So choose a name that stays the same for the service.

Compatibility with one API does not imply support for the other or for every OpenAI feature. The provider picks a profile from the model name: a `vendor/model` name, as many gateways and Hugging Face repo ids use, gets the profile of the model family that vendor or organization makes, whatever its case, so `Qwen/Qwen3.8-27B` and `qwen/qwen3.8-27b` get the same profile and `openai/gpt-oss-120b` gets the gpt-oss one. Any other name gets the profile OpenAI-compatible APIs share, without the features only OpenAI's own API supports. If features such as `thinking` or structured output are ignored or rejected, configure the [model profile](#model-profile). A family profile can also leave out a feature the shared profile has: MiniMax documents no `response_format` on its [OpenAI-compatible API](https://platform.minimax.io/docs/api-reference/text-openai-api), so [`NativeOutput`](../output.md#native-output) raises a `UserError` for the MiniMax models the family profile describes, under `minimax/` and `MiniMaxAI/` names, here and on [LiteLLM](#litellm). Use [tool output](../output.md#tool-output) or [prompted output](../output.md#prompted-output) instead.

[`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] also takes a `base_url`, but it's for OpenAI itself, or a transparent proxy to it: it assumes OpenAI's models and features, and sends OpenAI's encrypted reasoning and compaction items to the endpoint.

### Obscurify.ai

[Obscurify.ai](https://obscurify.ai/?utm_source=pydanticai) is a privacy-focused service with an OpenAI-compatible Chat Completions API serving 200+ models named `vendor/model`, such as `deepseek/deepseek-v4-flash` and `z-ai/glm-5.3`. Accounts are a random account number with no email, and prompts and responses are not stored on the server. Get an API key from your [account page](https://obscurify.ai/account) and use `https://obscurify.ai/v1` as the `base_url`. The model names are listed in the [API docs](https://obscurify.ai/api-docs) and at `https://obscurify.ai/v1/models`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

model = OpenAIChatModel(
    'deepseek/deepseek-v4-flash',
    provider=OpenAICompatibleProvider(
        'obscurify',
        base_url='https://obscurify.ai/v1',
        api_key='your-obscurify-api-key',
    ),
)
agent = Agent(model)
...
```

### Oracle Cloud Infrastructure (OCI) Generative AI {#oci-generative-ai}

[OCI Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm) serves an [OpenAI-compatible API](https://docs.oracle.com/en-us/iaas/Content/generative-ai/openai-compatible-api.htm) at `https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1`. Create a [Generative AI API key](https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm) in the region of the model you want to use, and pass it as `api_key`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    'oci',
    base_url='https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1',
    api_key='your-oci-genai-api-key',
    profile=lambda model_name: OpenAICompatibleProvider.model_profile(model_name.replace('.', '/', 1)),
)
model = OpenAIChatModel('openai.gpt-oss-120b', provider=provider)
agent = Agent(model)
...
```

OCI [names models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm) `<vendor>.<model>`, such as `openai.gpt-oss-120b` or `meta.llama-3.3-70b-instruct`, so without the `profile` function every OCI model gets the profile OpenAI-compatible APIs share. The function gives each model the family profile of its `vendor/model` spelling, as [Gateways that serve several model families](#custom-openai-compatible-provider) explains.

The [API key documentation](https://docs.oracle.com/en-us/iaas/Content/generative-ai/api-keys.htm) lists the model families and regions an API key can use. Meta Llama models support only the Chat Completions API, so use [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] for them; xAI Grok and OpenAI gpt-oss models also support the Responses API, which [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] uses.

!!! note "Allow API keys in an IAM policy"
    An API key works only once an IAM policy lets API keys use Generative AI. To authorize every key in a compartment, create the policy before the key. To limit the policy to one key, create the key first and use its OCID. See [Adding Key Permissions](https://docs.oracle.com/en-us/iaas/Content/generative-ai/add-api-permission.htm) for the policy statements.

### WaveSpeed AI {#wavespeed-ai}

[WaveSpeed AI](https://wavespeed.ai/llm) serves models from many vendors through an [OpenAI-compatible Chat Completions API](https://wavespeed.ai/docs/llm-service-quick-start) at `https://llm.wavespeed.ai/v1`. Pass your WaveSpeed API key as `api_key`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

provider = OpenAICompatibleProvider(
    'wavespeed',
    base_url='https://llm.wavespeed.ai/v1',
    api_key='your-wavespeed-api-key',
)
model = OpenAIChatModel('qwen/qwen3.8-flash', provider=provider)
agent = Agent(model)
...
```

WaveSpeed names models `<vendor>/<model>`, so each model gets the profile of its family, as [Other endpoints](#other-endpoints) explains. `GET https://llm.wavespeed.ai/v1/models` [lists the current model IDs](https://wavespeed.ai/docs/supported-llm-models) without an API key.

## Non-standard finish reasons

When an OpenAI-compatible API returns a `finish_reason` outside the five that OpenAI defines, `OpenAIChatModel` keeps the response, gives it a [`finish_reason`][pydantic_ai.messages.ModelResponse.finish_reason] of `None`, and keeps the raw value in [`provider_details['finish_reason']`][pydantic_ai.messages.ModelResponse.provider_details].

## Gemini usage

Chat Completions streams normally report usage once, on the last chunk, and [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel] adds up whatever the chunks report. Gemini's OpenAI-compatible endpoint instead repeats its running totals on every chunk, so [`OpenAIProvider`][pydantic_ai.providers.openai.OpenAIProvider] sets [`openai_chat_streaming_repeats_usage_totals`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_streaming_repeats_usage_totals] for `gemini*` model names, and each chunk's usage replaces the last.

Chat Completions also counts reasoning tokens in `completion_tokens`. Gemini's endpoint leaves its thinking tokens out of `completion_tokens` and counts them only in `total_tokens`: with thinking turned off, `total_tokens` is exactly `prompt_tokens + completion_tokens`. `OpenAIProvider` therefore sets [`openai_chat_completion_tokens_omit_reasoning`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_completion_tokens_omit_reasoning] for the same names. When a response's or chunk's `usage` has no `completion_tokens_details`, the amount by which `total_tokens` exceeds `prompt_tokens + completion_tokens` then counts as output and reasoning tokens.

Both flags also apply through a custom `base_url`. A server that serves these names turns a flag off in the model's [profile](overview.md#models-and-providers): `openai_chat_streaming_repeats_usage_totals=False` when it reports per-chunk deltas, and `openai_chat_completion_tokens_omit_reasoning=False` when its `total_tokens` counts anything besides input, output and thinking tokens.

## Response data Pydantic AI doesn't map

OpenAI-compatible APIs often return more than the Chat Completions API defines: vendor fields on the response, choice or message, or a message's `content` as a list of parts rather than a string. Mistral models on the Albert API, for example, return `[{"type": "text", "text": "..."}, {"type": "reference", "reference_ids": [1]}]` when an answer carries references. `OpenAIChatModel` joins the text parts into the response's text and ignores the rest.

To keep that data, set [`openai_include_unmapped_response_data`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_include_unmapped_response_data]. It's stored, exactly as received, in [`provider_details['unmapped']`][pydantic_ai.messages.ModelResponse.provider_details], keyed by where it came from:

```python {test="skip"}
{
    'response': {...},  # vendor fields on the completion
    'choice': {...},  # vendor fields on the choice
    'message': {...},  # vendor fields on the message
    'content_parts': [{'type': 'reference', 'reference_ids': [1]}],
}
```

A streamed response collects it per chunk, as `{'chunks': [...]}` with `delta` in place of `message`, once the stream ends. Fields Pydantic AI already maps, like `reasoning_content`, aren't repeated. The data is for reading only: it's kept with the message history but never sent back to a provider.

## Model profile

A model profile tells the model class how to shape requests for a particular model and API: which JSON schema restrictions its tool definitions have, whether tools can be marked as strict, how reasoning is configured, and so on. The providers above select one automatically based on the model name, so you only need this section if a model behind a custom endpoint doesn't work correctly out of the box.

Pass your own [`ModelProfile`][pydantic_ai.profiles.ModelProfile] (for behaviors shared among all model classes) or [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (for behaviors specific to the OpenAI model classes) to tweak how requests are constructed:

```py
from pydantic_ai import Agent, InlineDefsJsonSchemaTransformer
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

model = OpenAIChatModel(
    'model_name',
    provider=OpenAICompatibleProvider(
        'my-service',
        base_url='https://<openai-compatible-api-endpoint>.com',
        api_key='your-api-key',
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

A gateway that routes to models from several developers usually names them `vendor/model`, which `OpenAICompatibleProvider` already picks a profile for. If your gateway names them otherwise, such as `openai.gpt-5.6-sol` in this example, or needs gateway-specific overrides, pass `profile=` to the provider instead of to every model: a [`ModelProfile`][pydantic_ai.profiles.ModelProfile] is merged over the profile picked for each model, and a function from the model name to a profile replaces it. The helpers in [`pydantic_ai.profiles`](../api/profiles.md) return the profile for each model family:

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
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider


def gateway_model_profile(model_name: str) -> ModelProfile:
    provider_to_profile = {
        'openai': openai_model_profile,
        'groq': groq_model_profile,
        'moonshotai': moonshotai_model_profile,
    }
    provider_name, _, model_name = model_name.partition('.')
    profile = None
    if profile_func := provider_to_profile.get(provider_name):
        profile = profile_func(model_name)
    return merge_profile(
        OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer),
        profile,
    )


provider = OpenAICompatibleProvider(
    'my-gateway',
    base_url='https://gateway.example/v1',
    api_key='your-gateway-api-key',
    profile=gateway_model_profile,
)
model = OpenAIChatModel('openai.gpt-5.6-sol', provider=provider)
agent = Agent(model)
```

The model name is only normalized for the profile lookup; the name sent to the gateway stays unchanged. Pydantic AI merges the returned profile with [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE], and you can pass gateway-specific overrides as a final argument to [`merge_profile()`][pydantic_ai.profiles.merge_profile]. The model class doesn't change with the profile: `OpenAIChatModel` still sends a Chat Completions request, whichever model family the gateway routes it to.

If the gateway passes requests through unchanged to a provider whose encrypted reasoning it returns, such as OpenAI, pass that provider's canonical system as `canonical_system`, and the systems whose data the gateway accepts as `compatible_systems`. See [Sending history to another provider](../message-history.md#sending-history-to-another-provider).

### Detect incomplete streamed responses

Some OpenAI-compatible APIs can close a Chat Completions stream cleanly without a terminal
`finish_reason`, making a partial response look complete: like a non-streamed response without a finish
reason, it gets a [`finish_reason`][pydantic_ai.messages.ModelResponse.finish_reason] of `'stop'`. If your provider guarantees that complete
streams include a finish reason, set
[`openai_chat_streaming_requires_finish_reason=True`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_streaming_requires_finish_reason]
in the model profile. Pydantic AI will then raise [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]
when the stream reaches EOF without one. The option defaults to `False` because some compatible APIs
do not guarantee the field.

### Tool call IDs that repeat across responses

Pydantic AI pairs each tool call with its result by the call's ID across the whole conversation. Some OpenAI-compatible APIs make tool call IDs unique only within one response, numbering the calls of every response from `call_0` again, so two different calls in one conversation share an ID. If yours does, set [`tool_call_ids_are_response_scoped=True`][pydantic_ai.profiles.ModelProfile.tool_call_ids_are_response_scoped] in the model profile. Each ID the API sends is then stored with a suffix derived from its response, such as `call_0_k3ZcR9aQ`, and sent back to the same endpoint as `call_0`. Other providers and endpoints, such as a [fallback model](overview.md#fallback-model), get the stored ID. The flag works with both `OpenAIChatModel` and `OpenAIResponsesModel`, and the [Bedrock Mantle](bedrock.md#bedrock-mantle) provider sets it for the GPT-5 models it serves on the Responses API.

### Models that accept only one leading system message

Some models are served with a chat template (applied server-side, for example by [vLLM](https://docs.vllm.ai/), [LiteLLM](#litellm), or TGI) that accepts only a single system message at the start of the conversation and rejects additional ones. Sending more than one fails with a `400` error such as `System message must be at the beginning.` or `Conversation roles must alternate ...`, seen with some newer Qwen, Mistral, Gemma, and Command-R models. It's easy to hit without intending to, since more than one leading system message can be produced in several ways.

Set `openai_chat_supports_multiple_system_messages=False` on the model's [`OpenAIModelProfile`][pydantic_ai.profiles.openai.OpenAIModelProfile] (as shown in [Model profile](#model-profile)) to merge the leading run of system messages into one, joined with two newlines, before the request is sent. The merge is lossless, so it's safe to enable whenever a backend rejects multiple system messages.
