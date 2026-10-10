---
description: "Find supported model developers, cloud platforms, inference services, gateways and local servers, with setup guides, model selection and fallback configuration."
---

# Models and Providers

Pydantic AI supports model developers, cloud platforms, inference services, gateways, and local model servers. Find your service below and follow its setup guide for installation, authentication, and supported features.

## Provider directory

Pass a name in the form `<provider>:<model>` to [`Agent`][pydantic_ai.Agent] to select a provider by its prefix. Some routes require an explicit model or client instead; each setup guide shows the available options.

| Provider and setup | Service | Model selection |
| --- | --- | --- |
| [Pydantic AI Gateway](../gateway.md) | Gateway | `gateway/<provider>:` |
| [OpenAI](openai.md) | Model developer | `openai:`, `openai-chat:`, `openai-responses:` |
| [Anthropic](anthropic.md) | Model developer | `anthropic:` |
| [Google / Gemini API](google.md) | Model developer | `google:` |
| [AWS Bedrock](bedrock.md) | Cloud platform | `bedrock:`, `bedrock-mantle:`; Anthropic client |
| [Google Cloud / Vertex AI](google-cloud.md) | Cloud platform | `google-cloud:`; Anthropic client |
| [Microsoft Azure / Foundry](azure.md) | Cloud platform | `azure:`, `azure-responses:`; Anthropic client |
| [Alibaba Cloud / Qwen (DashScope)](compatible-apis.md#alibaba-cloud-model-studio-dashscope) | Cloud platform; model developer | `alibaba:` |
| [Cerebras](cerebras.md) | Inference platform | `cerebras:` |
| [Cohere](cohere.md) | Model developer | `cohere:` |
| [Crusoe](crusoe.md) | Inference platform | `crusoe:` |
| [DeepSeek](deepseek.md) | Model developer | `deepseek:`; explicit Responses model |
| [Fireworks AI](compatible-apis.md#fireworks-ai) | Inference platform | `fireworks:` |
| [GitHub Copilot](github-copilot.md) | Subscription access | `github-copilot:` |
| [Groq](groq.md) | Inference platform | `groq:` |
| [Heroku AI](compatible-apis.md#heroku-ai) | Cloud platform | `heroku:` |
| [Hugging Face](huggingface.md) | Inference platform | `huggingface:` |
| [LiteLLM](compatible-apis.md#litellm) | Self-hosted gateway | `litellm:` |
| [Mistral](mistral.md) | Model developer | `mistral:` |
| [Moonshot AI / Kimi](moonshotai.md) | Model developer | `moonshotai:` |
| [Nebius AI Studio](compatible-apis.md#nebius-ai-studio) | Inference platform | `nebius:` |
| [Ollama](ollama.md) | Local inference; cloud inference | `ollama:` |
| [OpenAI Codex](openai-codex.md) | Subscription access | `openai-codex:` |
| [OpenAI Decisions API](openai.md#decisions-api) | [Decision model](decision.md) | `openai-decisions:` |
| [OpenRouter](openrouter.md) | Gateway | `openrouter:` |
| [OVHcloud AI Endpoints](compatible-apis.md#ovhcloud-ai-endpoints) | Cloud platform | `ovhcloud:` |
| [SambaNova](compatible-apis.md#sambanova) | Inference platform | `sambanova:` |
| [Snowflake Cortex](snowflake.md) | Cloud platform | `snowflake:` |
| [System One API](system-one.md) | [Decision models](decision.md) such as CLM, Laya, and Ollama's | `system-one:` |
| [Together AI](compatible-apis.md#together-ai) | Inference platform | `together:` |
| [TypeSafe (Jev)](typesafe.md) | [Decision model](decision.md) | `typesafe:` |
| [Vercel AI Gateway](compatible-apis.md#vercel-ai-gateway) | Gateway | `vercel:` |
| [vLLM](compatible-apis.md#vllm) | Self-hosted inference | `vllm:` |
| [xAI](xai.md) | Model developer | `xai:` |
| [Z.AI](zai.md) | Model developer | `zai:` |

!!! tip "One key for every model"
    The easiest way to try models from several providers is the [Pydantic AI Gateway](../gateway.md): one API key for models from OpenAI, Anthropic, Google Cloud, Groq, and AWS Bedrock, with spending limits and cost monitoring in [Pydantic Logfire](../logfire.md). Set `PYDANTIC_AI_GATEWAY_API_KEY` and add the `gateway/` prefix to the model string, for example `Agent('gateway/anthropic:claude-fable-5-1')`. The [Gateway quick start](../gateway.md#quick-start) shows how to create a key.

The service descriptions help you find a deployment option; a company may offer more than one kind of service. Feature support depends on the model and API you select, even when two services use the same API format.

For an OpenAI-compatible endpoint not listed here, see [Other endpoints](compatible-apis.md#other-endpoints); for any other API, implement a [custom model](#custom-models).

For testing and development, use [`TestModel`](../api/models/test.md) or [`FunctionModel`](../api/models/function.md).

### OpenAI-compatible Providers {#openai-compatible-providers}

Many entries in the directory use OpenAI-compatible APIs. Their setup guides cover which model class to use; [Other compatible APIs](compatible-apis.md) explains custom endpoints and model profiles.

## Models, providers, and profiles {#models-and-providers}

Pydantic AI uses a few key terms to describe how it interacts with different LLMs:

- **Model**: This refers to the Pydantic AI class used to make requests following a specific LLM API
  (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a
  vendor-SDK-agnostic API, ensuring a single Pydantic AI agent is portable to different LLM vendors without
  any other code changes just by swapping out the Model it uses. Model classes are named
  roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIChatModel`, `AnthropicModel`, `GoogleModel`,
  etc. When using a Model class, you specify the actual LLM model name (e.g., `gpt-5`,
  `claude-sonnet-4-5`, `gemini-3-flash-preview`) as a parameter.
- **Provider**: This refers to provider-specific classes which handle the authentication and connections
  to an LLM vendor. Passing a non-default _Provider_ as a parameter to a Model is how you can ensure
  that your agent will make requests to a specific endpoint, or make use of a specific approach to
  authentication (e.g., you can use Azure auth with the `OpenAIChatModel` by way of the `AzureProvider`).
  In particular, this is how you can make use of an AI gateway, or an LLM vendor that offers API compatibility
  with the vendor SDK used by an existing Model (such as `OpenAIChatModel`).
- **Profile**: This refers to a description of how requests to a specific model or family of models need to be
  constructed to get the best results, independent of the model and provider classes used.
  For example, different models have different restrictions on the JSON schemas that can be used for tools,
  and the same schema transformer needs to be used for Gemini models whether you're using `GoogleModel`
  with model name `gemini-3-pro-preview`, or `OpenAIChatModel` with `OpenRouterProvider` and model name `google/gemini-3-pro-preview`.

When you instantiate an [`Agent`][pydantic_ai.Agent] with just a name formatted as `<provider>:<model>`, e.g. `openai:gpt-5.2` or `openrouter:google/gemini-3-pro-preview`,
Pydantic AI will automatically select the appropriate model class, provider, and profile.
If you want to use a different provider or profile, you can instantiate a model class directly and pass in `provider` and/or `profile` arguments.

### Inspecting a model's profile {#inspecting-a-models-profile}

A model's [`ModelProfile`][pydantic_ai.profiles.ModelProfile] also describes what the model can do. It is a `TypedDict` whose keys are all optional, so you read capability flags with `.get()` on `model.profile` — for example [`supports_text_output`][pydantic_ai.profiles.ModelProfile.supports_text_output], [`supports_tools`][pydantic_ai.profiles.ModelProfile.supports_tools], [`supports_json_schema_output`][pydantic_ai.profiles.ModelProfile.supports_json_schema_output], and [`supported_native_tools`][pydantic_ai.profiles.ModelProfile.supported_native_tools]. This is useful when you want to branch on a capability rather than discover a limitation at request time — for example checking whether a model supports text generation, tool calling, native JSON-schema output, or a specific native tool before relying on it:

```python
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebSearchTool

model = TestModel()
profile = model.profile

print(profile.get('supports_tools'))
#> True
print(profile.get('supports_text_output'))
#> True
print(profile.get('supports_json_schema_output'))
#> False
print(WebSearchTool in profile.get('supported_native_tools', frozenset()))
#> True
```

`model.profile` is usually the fully *resolved* profile: keys from [`DEFAULT_PROFILE`][pydantic_ai.profiles.DEFAULT_PROFILE] are merged with the provider's defaults, so `profile.get('supports_tools')` returns a value. If you supply `profile=` as a callable (or otherwise have a partial profile dict), pass the default as the fallback, `profile.get('supports_tools', DEFAULT_PROFILE['supports_tools'])` (after importing `DEFAULT_PROFILE`), to tolerate missing keys.
Individual model adapters expose their resolved profile the same way, so the same check works whether the model was selected automatically from a `<provider>:<model>` name or instantiated directly. A [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] is different: it has no single profile because its candidate models may have different capabilities. Inspect the profile of each model in `fallback_model.models` instead. Don't confuse profiles with [Capabilities](../capabilities/overview.md), which are reusable bundles of tools, hooks, and settings you add to an agent — the profile describes what the underlying model itself supports.

The profile also carries the model's [`context_window`][pydantic_ai.profiles.ModelProfile.context_window]: the maximum number of tokens it can handle in a single request, or `None` when unknown. Every model, including a `FallbackModel` and wrappers like [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel], exposes it as [`model.context_window`][pydantic_ai.models.AbstractModel.context_window]; a fallback model reports the smallest window among its candidates, so history that fits it fits whichever candidate answers. Inside a run, [`ctx.context_window_used`][pydantic_ai.tools.RunContext.context_window_used] reports the fraction in use, or `None` when it cannot be calculated. See [Compact when the context window fills](../message-history.md#compact-when-the-context-window-fills) for an example that handles this case.

## HTTP Client Lifecycle

When a [`Provider`][pydantic_ai.providers.Provider] creates its own HTTP client (i.e. you don't pass a custom `http_client`), it owns that client's lifecycle. Using the [`Agent`][pydantic_ai.Agent] as an async context manager keeps the HTTP client open across every run inside the block, so the runs reuse its connections, and closes it cleanly on exit:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')

async def main():
    async with agent:
        result = await agent.run('What is the capital of France?')
        print(result.output)
        #> The capital of France is Paris.
```

You can also use a [`Model`][pydantic_ai.models.Model] or [`Provider`][pydantic_ai.providers.Provider] directly as an async context manager for the same effect.

An agent you don't enter this way enters its model for the duration of each run instead. The provider then closes its HTTP client when the run ends and creates a new one for the next run, so no connections are reused between runs. A model name passed to a run, as in `agent.run(..., model='openai:gpt-5.2')`, goes further: it creates a new provider, and with it a new HTTP client, for every run. In a long-lived service such as a web server, enter the agent once when the service starts and run it inside that block, and to switch models per run, pass `Model` instances you created and entered (`async with model:`) once, rather than model names.

Pydantic AI only ever closes an HTTP client it created itself. A client you pass in is yours to close, whether it's an `http_client` or a provider SDK client, such as `openai_client`, `anthropic_client` (including `AsyncAnthropicVertex`), Google's `client`, or `xai_client`.

### Configuring the HTTP client

The HTTP clients Pydantic AI creates have a 600-second timeout with a 5-second connect timeout, and a connection pool of up to 1000 connections, of which up to 100 are kept alive while idle. These are the defaults of the OpenAI and Anthropic SDKs' own clients. To change them, create a client with [`create_async_httpx2_client()`][pydantic_ai.models.create_async_httpx2_client], which keeps the remaining defaults and Pydantic AI's `User-Agent`, and pass it to the provider as `http_client`:

```python {title="configure_http_client.py"}
import httpx2

from pydantic_ai import Agent
from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


async def main():
    async with create_async_httpx2_client(
        timeout=httpx2.Timeout(120, connect=5, pool=10),
        limits=httpx2.Limits(max_connections=200, max_keepalive_connections=50),
    ) as http_client:
        model = OpenAIChatModel('gpt-5.2', provider=OpenAIProvider(http_client=http_client))
        agent = Agent(model)
        result = await agent.run('What is the capital of France?')
        print(result.output)
        #> The capital of France is Paris.
```

Because you created the client, you close it, here by leaving the `async with` block. Passing the same client to several providers makes them share its connection pool.

The Groq, Cohere and GitHub providers take a legacy `httpx.AsyncClient` instead, which you build yourself with the same arguments, for example `httpx.AsyncClient(timeout=httpx.Timeout(120, connect=5), limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))`.

## Custom Models

!!! note
    If a model API is compatible with the OpenAI API, you do not need a custom model class and can provide your own [custom provider](compatible-apis.md) instead.

To implement support for a model API that's not already supported, you will need to subclass the [`Model`][pydantic_ai.models.Model] abstract base class.
For streaming, you'll also need to implement the [`StreamedResponse`][pydantic_ai.models.StreamedResponse] abstract base class.

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIChatModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to Pydantic AI, see the [contributing guidelines](../contributing.md#new-model-rules).

## HTTP Request Concurrency

You can limit the number of concurrent HTTP requests to a model using the
[`ConcurrencyLimitedModel`][pydantic_ai.ConcurrencyLimitedModel] wrapper.
This is useful for respecting rate limits or managing resource usage when running many agents in parallel.

```python {title="model_concurrency.py"}
import asyncio

from pydantic_ai import Agent, ConcurrencyLimitedModel

# Wrap a model with concurrency limiting
model = ConcurrencyLimitedModel('openai:gpt-4o', limiter=5)

# Multiple agents can share this rate-limited model
agent = Agent(model)


async def main():
    # These will be rate-limited to 5 concurrent HTTP requests
    results = await asyncio.gather(
        *[agent.run(f'Question {i}') for i in range(20)]
    )
    print(len(results))
    #> 20
```

The `limiter` parameter accepts:

- An integer for simple limiting (e.g., `limiter=5`)
- A [`ConcurrencyLimit`][pydantic_ai.ConcurrencyLimit] for advanced configuration with backpressure control
- A [`ConcurrencyLimiter`][pydantic_ai.ConcurrencyLimiter] for sharing limits across multiple models

### Shared Concurrency Limits

To share a concurrency limit across multiple models (e.g., different models from the same provider),
you can create a [`ConcurrencyLimiter`][pydantic_ai.ConcurrencyLimiter] and pass it to
multiple `ConcurrencyLimitedModel` instances:

```python {title="shared_concurrency.py"}
import asyncio

from pydantic_ai import Agent, ConcurrencyLimitedModel, ConcurrencyLimiter

# Create a shared limiter with a descriptive name
shared_limiter = ConcurrencyLimiter(max_running=10, name='openai-pool')

# Both models share the same concurrency limit
model1 = ConcurrencyLimitedModel('openai:gpt-4o', limiter=shared_limiter)
model2 = ConcurrencyLimitedModel('openai:gpt-4o-mini', limiter=shared_limiter)

agent1 = Agent(model1)
agent2 = Agent(model2)


async def main():
    # Total concurrent requests across both agents limited to 10
    results = await asyncio.gather(
        *[agent1.run(f'Question {i}') for i in range(10)],
        *[agent2.run(f'Question {i}') for i in range(10)],
    )
    print(len(results))
    #> 20
```

An agent and the `ConcurrencyLimitedModel` used for its own model request must use separate
`ConcurrencyLimiter` instances. If you set `Agent(max_concurrency=...)` as well as
`ConcurrencyLimitedModel(limiter=...)`, using the same instance raises
[`UserError`][pydantic_ai.exceptions.UserError].
Nested `ConcurrencyLimitedModel` wrappers also need different limiter instances.

Re-entering a `ConcurrencyLimiter` through an agent on the same task raises `RuntimeError`; use a separate limiter
for the nested run. When an agent delegates to another agent through a tool, each run or model
request acquires its own slot. A shared pool must have enough capacity for the parent and nested
operation to run at the same time.

When instrumentation is enabled, requests waiting for a concurrency slot appear as spans with
attributes showing the queue depth and configured limits. The `name` parameter on
`ConcurrencyLimiter` helps identify shared limiters in traces.

<!-- TODO(Marcelo): We need to create a section in the docs about reliability. -->

## Handling Model API Errors

When a request to a model provider fails, Pydantic AI raises a
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] rather than the provider SDK's own exception, which
remains available as `__cause__`. When the provider says what went wrong, the error is an instance of one of
these categories, whichever provider it came from:

| Exception | Raised when |
|---|---|
| [`ModelRateLimitError`][pydantic_ai.exceptions.ModelRateLimitError] | A rate limit was reached, e.g. an HTTP 429. Waiting and retrying may help. |
| [`ModelQuotaExceededError`][pydantic_ai.exceptions.ModelQuotaExceededError] | The provider said the account's quota, credits, or billing is exhausted, e.g. an HTTP 402 or OpenAI's `insufficient_quota`. Waiting won't help. |
| [`ModelUnavailableError`][pydantic_ai.exceptions.ModelUnavailableError] | The provider or model is temporarily unavailable, e.g. an HTTP 503 or a gRPC `UNAVAILABLE`. |
| [`ModelOverloadedError`][pydantic_ai.exceptions.ModelOverloadedError] | The provider said it's unavailable because it's overloaded, e.g. an HTTP 529 or Anthropic's `overloaded_error`. A subclass of `ModelUnavailableError`. |
| [`ModelServerError`][pydantic_ai.exceptions.ModelServerError] | The provider failed with another server error, e.g. an HTTP 500, 502, or 504. |
| [`ModelConnectionError`][pydantic_ai.exceptions.ModelConnectionError] | The request could not reach the provider, or its response could not be read. Its [`phase`][pydantic_ai.exceptions.ModelConnectionError.phase] says whether the request may have reached the provider. |
| [`ModelTimeoutError`][pydantic_ai.exceptions.ModelTimeoutError] | An attempt timed out at the transport layer. A subclass of `ModelConnectionError`. |
| [`ModelContextWindowExceededError`][pydantic_ai.exceptions.ModelContextWindowExceededError] | The input exceeds the model's context window. |

A provider's own signal decides between categories that share a status code: a 429 is a rate limit unless the
provider says the quota is exhausted, and a 503 is only unavailable unless the provider says it's overloaded.

Other errors are raised as a plain `ModelAPIError` or [`ModelHTTPError`](#http-errors), as they always were.
Categories are assigned from the provider's error code or type where it sends one, and from the error message
only where the provider offers nothing else. Every `ModelAPIError` also exposes the provider's
[`provider_error_code`][pydantic_ai.exceptions.ModelAPIError.provider_error_code],
[`provider_error_type`][pydantic_ai.exceptions.ModelAPIError.provider_error_type], and error
[`body`][pydantic_ai.exceptions.ModelAPIError.body], when available.

An error is reported the same way whether the provider sent it before a stream opened or inside an already open
stream. For example, both an HTTP 503 from Bedrock and a `serviceUnavailableException` in a Bedrock stream raise a
`ModelHTTPError` with status 503 that is also a `ModelUnavailableError`. The second has
[`in_stream`][pydantic_ai.exceptions.ModelAPIError.in_stream] set, so you can still tell them apart.

This lets you decide what to do about a failure without knowing which provider it came from:

```python {title="handle_model_errors.py" test="skip" lint="skip"}
from pydantic_ai import Agent, ModelContextWindowExceededError, ModelRateLimitError

agent = Agent('openai:gpt-5.2')

try:
    result = agent.run_sync('What is the capital of France?', message_history=history)
except ModelRateLimitError as exc:
    raise MyRateLimitException('AI service is rate-limited. Try again shortly.', retry_after=exc.retry_after)
except ModelContextWindowExceededError:
    result = agent.run_sync('What is the capital of France?', message_history=summarize(history))
```

[`retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] is the number of seconds the provider asked
you to wait, as a `float`, or `None` if it didn't say.

See [Compaction](../capabilities/compaction.md) for keeping a conversation under the context window in the
first place, which is preferable to waiting for `ModelContextWindowExceededError`: model performance usually degrades well
before the hard limit.

### HTTP errors

When a provider returns a 4xx or 5xx response, the error is a
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError], and also an instance of the error category above
that it belongs to, if any. The exception exposes the
[`status_code`][pydantic_ai.exceptions.ModelHTTPError.status_code], the response
[`body`][pydantic_ai.exceptions.ModelAPIError.body], and the provider's
**response headers** via the [`headers`][pydantic_ai.exceptions.ModelHTTPError.headers]
attribute (a `dict[str, str]` with lowercase keys, or `None` for providers that don't
surface headers, such as gRPC-based providers). Its
[`retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] is parsed from the `retry-after-ms` or
`Retry-After` header, handling both the integer delta-seconds and HTTP-date formats, and
[`provider_retry_hint`][pydantic_ai.exceptions.ModelHTTPError.provider_retry_hint] from the `x-should-retry` header
that OpenAI and Anthropic send to say whether retrying may help.

When [OpenAI](openai.md), [Anthropic](anthropic.md), the [Google Gemini API](google.md),
[Amazon Bedrock](bedrock.md), or [Groq](groq.md) reports that a requested model identifier is
unavailable, Pydantic AI adds a close known match to the error message when one exists. The
suggestion is also available as
[`suggested_model_id`][pydantic_ai.exceptions.ModelHTTPError.suggested_model_id]. This is
best-effort guidance after the provider rejects a request, not local validation: unknown model
identifiers remain valid so custom deployments and newly released models continue to work.

When a provider error has a known fix outside the request, such as the [data retention](bedrock.md#data-retention)
mode Amazon Bedrock requires for some models, Pydantic AI appends guidance to the error message. It is also
available as [`hint`][pydantic_ai.exceptions.ModelHTTPError.hint].

!!! note
    Some `ModelHTTPError`s don't come with an HTTP status of their own, and get the status the provider uses for
    the same error instead, so they're handled the same way:

    - xAI's gRPC status codes are mapped to their HTTP equivalents, like `RESOURCE_EXHAUSTED` to 429. These have
      `headers=None`.
    - OpenRouter errors parsed from a 200-OK response report the error's own `code`.
    - An error sent inside an already open stream gets the status of the same error before the stream opens, like
      529 for Anthropic's `overloaded_error`. These have
      [`in_stream`][pydantic_ai.exceptions.ModelAPIError.in_stream] set, and their `headers`, if any, are the
      stream's.
      A request can be streamed without you asking for it (e.g. when the agent has an event stream handler), which
      is why the error doesn't otherwise depend on it.

A [custom model](#custom-models) can raise the same errors with
[`ModelHTTPError.for_category`][pydantic_ai.exceptions.ModelHTTPError.for_category], which builds a `ModelHTTPError`
that is also an instance of the given category, so code that handles either catches it:

```python {title="custom_model_errors.py"}
from pydantic_ai import ModelHTTPError, ModelRateLimitError


def map_error(status_code: int, body: object) -> ModelHTTPError:
    if status_code == 429:
        return ModelHTTPError.for_category(ModelRateLimitError, status_code=429, model_name='my-model', body=body)
    return ModelHTTPError(status_code, 'my-model', body)


error = map_error(429, {'error': 'Too many requests'})
print(isinstance(error, ModelHTTPError), isinstance(error, ModelRateLimitError))
#> True True
```

## Fallback Model

You can use [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to attempt multiple models
in sequence until one succeeds. Pydantic AI can switch to the next model when the current model
raises an exception (like a 4xx/5xx API error) **or** when the response content indicates a semantic
failure (like a truncated response or a failed native tool call).

By default, fallback triggers on [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] (4xx/5xx API errors, and
connection failures or timeouts before the response starts), so you don't need to configure anything for the most
common use case. That includes every [error category](#handling-model-api-errors): rate limits, exhausted quota,
unavailable or overloaded providers, server errors, connection failures and timeouts, and
[context window overflows](#context-window-overflow). To fall back only on some of them, pass those categories as
`fallback_on`, e.g. `fallback_on=(ModelRateLimitError, ModelUnavailableError, ModelServerError)`.

This behavior is controlled by the `fallback_on` parameter (see
[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]), which accepts exception types,
exception handlers, and response handlers — all of which can be sync or async.

!!! note
    The provider SDKs on which Models are based (like OpenAI, Anthropic, etc.) often have built-in retry logic that can delay the `FallbackModel` from activating.

    When using `FallbackModel`, it's recommended to disable provider SDK retries to ensure immediate fallback, for example by setting `max_retries=0` on a [custom OpenAI client](openai.md#custom-openai-client) or a [custom Anthropic client](anthropic.md#custom-http-client). See [The layers](../retries.md#the-layers) in the retries guide for the full retry map.

In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key),
and then falls back to the Anthropic model.

```python {title="fallback_model.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

openai_model = OpenAIChatModel('gpt-5.2')
anthropic_model = AnthropicModel('claude-sonnet-4-5')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
response = agent.run_sync('What is the capital of France?')
print(response.output)
#> The capital of France is Paris.

print(response.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='What is the capital of France?',
                timestamp=datetime.datetime(...),
            )
        ],
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
    ),
    ModelResponse(
        parts=[TextPart(content='The capital of France is Paris.')],
        usage=RequestUsage(cost=Decimal('0.000273'), input_tokens=56, output_tokens=7),
        model_name='claude-sonnet-4-5',
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
        failed_attempts=[
            ModelRequestAttempt(
                model_name='gpt-5.2',
                provider_name='openai',
                outcome='error',
                error="ModelHTTPError: status_code: 401, model_name: gpt-5.2, body: {'error': 'Invalid API Key'}",
                timestamp=datetime.datetime(...),
                duration=datetime.timedelta(...),
            )
        ],
    ),
]
"""
```

The `ModelResponse` message above indicates in the `model_name` field that the output was returned by the Anthropic model, which is the second model specified in the `FallbackModel`.

!!! note
    Each model's options should be configured individually. For example, `base_url`, `api_key`, and custom clients should be set on each model itself, not on the `FallbackModel`.

### Per-Model Settings

You can configure different [`ModelSettings`][pydantic_ai.settings.ModelSettings] for each model in a fallback chain by passing the `settings` parameter when creating each model. This is particularly useful when different providers have different optimal configurations:

```python {title="fallback_model_per_settings.py"}
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

# Configure each model with provider-specific optimal settings
openai_model = OpenAIChatModel(
    'gpt-5.2',
    settings=ModelSettings(temperature=0.7, max_tokens=1000)  # Higher creativity for OpenAI
)
anthropic_model = AnthropicModel(
    'claude-sonnet-4-5',
    settings=ModelSettings(temperature=0.2, max_tokens=1000)  # Lower temperature for consistency
)

fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)

result = agent.run_sync('Write a creative story about space exploration')
print(result.output)
"""
In the year 2157, Captain Maya Chen piloted her spacecraft through the vast expanse of the Andromeda Galaxy. As she discovered a planet with crystalline mountains that sang in harmony with the cosmic winds, she realized that space exploration was not just about finding new worlds, but about finding new ways to understand the universe and our place within it.
"""
```

In this example, if the OpenAI model fails, the agent will automatically fall back to the Anthropic model with its own configured settings. The `FallbackModel` itself doesn't have settings - it uses the individual settings of whichever model successfully handles the request.

### Exception Handling

The next example demonstrates the exception-handling capabilities of `FallbackModel`.
If all models fail, a [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup] is raised, which
contains all the exceptions encountered during the `run` execution.

```python {title="fallback_model_failure.py" py="3.11"}
from pydantic_ai import Agent, ModelAPIError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

openai_model = OpenAIChatModel('gpt-5.2')
anthropic_model = AnthropicModel('claude-sonnet-4-5')
fallback_model = FallbackModel(openai_model, anthropic_model)

agent = Agent(fallback_model)
try:
    response = agent.run_sync('What is the capital of France?')
except* ModelAPIError as exc_group:
    for exc in exc_group.exceptions:
        print(exc)
```

By default, the `FallbackModel` only moves on to the next model if the current model raises a
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], which includes
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. You can customize this behavior by
passing a custom `fallback_on` argument to the `FallbackModel` constructor.

### Context Window Overflow

`FallbackModel` also falls back when a model raises
[`ModelContextWindowExceededError`][pydantic_ai.exceptions.ModelContextWindowExceededError], since a later model in the chain may
have a larger context window. If the whole chain fails, the overflow is in the
[`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup], where `except* ModelContextWindowExceededError`
reaches it. If none of your models has a larger window, you can avoid sending the oversized request to each of
them by excluding the overflow from `fallback_on`, so it propagates as is:

```python {title="fallback_without_overflow.py" test="skip" lint="skip"}
from pydantic_ai import Agent, ModelContextWindowExceededError, ModelAPIError
from pydantic_ai.models.fallback import FallbackModel


def fallback_on(exc: Exception) -> bool:
    return isinstance(exc, ModelAPIError) and not isinstance(exc, ModelContextWindowExceededError)


agent = Agent(FallbackModel('openai:gpt-5.2', 'anthropic:claude-sonnet-4-5', fallback_on=fallback_on))

try:
    result = agent.run_sync('What is the capital of France?', message_history=history)
except ModelContextWindowExceededError:
    result = agent.run_sync('What is the capital of France?', message_history=summarize(history))
```

!!! note
    Validation errors (from [structured output](../output.md#structured-output) or [tool parameters](../tools.md)) do **not** trigger fallback. These errors use the [retry mechanism](../agent.md#reflection-and-self-correction) instead, which re-prompts the same model to try again. This is intentional: validation errors stem from the non-deterministic nature of LLMs and may succeed on retry, whereas API errors (4xx/5xx) generally indicate issues that won't resolve by retrying the same request.

### Failed attempts and usage

The response that answers lists every attempt the `FallbackModel` moved on from in [`failed_attempts`][pydantic_ai.messages.ModelResponse.failed_attempts], in order. Each [`ModelRequestAttempt`][pydantic_ai.messages.ModelRequestAttempt] records the model and provider it tried, whether it raised an error (with the error as `'ExceptionType: message'`) or returned a response a [response handler](#response-based-fallback) rejected, when it started and how long it took, and the usage the provider reported for it. An attempt that raised has no usage, as the provider reported none. When every model fails, the same attempts are on the [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup]'s `attempts`.

A rejected response was still generated and billed, so its tokens and cost are added to the run's [`RunUsage`][pydantic_ai.usage.RunUsage] and count towards the token and cost limits in [`UsageLimits`][pydantic_ai.usage.UsageLimits]. They stay on the attempt rather than being folded into the answering response's `usage`, and the rejected response itself is not added to the message history. A fallback attempt isn't a response the agent acted on, so it doesn't count towards [`request_limit`][pydantic_ai.usage.UsageLimits.request_limit].

The rejected responses' usage counts when every model fails, too. If it exceeds a limit, [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] is raised with the `FallbackExceptionGroup` as its `__cause__`.

!!! note
    The rejected responses' usage is not recorded when a later model raises an error that `fallback_on` doesn't cover, or, under [Temporal](../durable_execution/temporal.md), when every model fails: the exception group reaches the workflow without its attempts.

```python {title="fallback_model_attempts.py"}
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage


def truncated(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('The capital of France is')],
        usage=RequestUsage(input_tokens=60, output_tokens=100),
        finish_reason='length',
    )


def complete(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[TextPart('The capital of France is Paris.')],
        usage=RequestUsage(input_tokens=60, output_tokens=8),
    )


def was_truncated(response: ModelResponse) -> bool:
    return response.finish_reason == 'length'


fallback_model = FallbackModel(
    FunctionModel(truncated, model_name='primary'),
    FunctionModel(complete, model_name='backup'),
    fallback_on=was_truncated,
)
agent = Agent(fallback_model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
print(result.usage)
#> RunUsage(input_tokens=120, output_tokens=108, requests=1)
print(result.response.usage)
#> RequestUsage(input_tokens=60, output_tokens=8)
for attempt in result.response.failed_attempts or []:
    print(attempt.model_name, attempt.outcome, attempt.usage)
    #> primary rejected RequestUsage(input_tokens=60, output_tokens=100)
```

When the agent is [instrumented](../logfire.md), the model request (`chat`) span describes the model that answered and ends OK. Each attempt the `FallbackModel` moved on from gets its own child span, `model request attempt <model name>`, which ends with an ERROR status, the way a failed tool call gets its own span inside a successful agent run. An error is recorded on that span as an exception, and a response rejected by a [response handler](#response-based-fallback) carries its usage, cost and finish reason instead, as a `chat` span would. Each attempt span names the model it tried in its `gen_ai.provider.name` and `gen_ai.request.model` attributes, and gives its zero-based position among the request's attempts in `pydantic_ai.model_request.attempt`. With [`include_content=False`](../logfire.md#excluding-prompts-and-completions), the exception keeps only its type. The attempt span is created once the attempt has failed, so spans the tried model opens itself, such as a [decision model](decision.md)'s `decide` span, sit beside it under the `chat` span rather than inside it.

### Response-Based Fallback

In addition to exception-based fallback, you can also trigger fallback based on the **content** of a model's response. This is useful when a model returns a successful HTTP response (no exception), but the response content indicates a semantic failure — for example, an unexpected finish reason or a native tool reporting failure.

!!! note "Non-streaming only"
    Response-based fallback currently only works with non-streaming requests (`agent.run()` and `agent.run_sync()`).
    For streaming requests (`agent.run_stream()`), only exception-based fallback is supported.

The `fallback_on` parameter accepts:

- A tuple of exception types: `(ModelAPIError, ModelHTTPError)`
- An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
- A response handler (sync or async): `def check(r: ModelResponse) -> bool`
- A list mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

Handler type is auto-detected by inspecting type hints on the first parameter. If the first parameter is hinted as [`ModelResponse`][pydantic_ai.messages.ModelResponse], it's a response handler. Otherwise (including untyped handlers and lambdas), it's an exception handler.

As the hints are resolved at runtime, every annotated type in the handler signature must be imported at runtime rather than only under `if TYPE_CHECKING:`. If any annotation can't be resolved, a [`UserError`][pydantic_ai.exceptions.UserError] is raised instead of the handler being silently treated as an exception handler.

#### Finish Reason Example

A simple use case is checking the model's finish reason — for example, falling back if the response was truncated due to length limits:

```python {title="fallback_on_finish_reason.py"}
from pydantic_ai import Agent
from pydantic_ai.messages import FinishReason, ModelResponse
from pydantic_ai.models.fallback import FallbackModel


def bad_finish_reason(response: ModelResponse) -> bool:
    """Fallback if the model stopped due to length limit, content filter, or error."""
    reason: FinishReason | None = response.finish_reason
    # Trigger fallback for problematic finish reasons
    return reason in ('length', 'content_filter', 'error')


fallback_model = FallbackModel(
    'openai:gpt-5.2',
    'anthropic:claude-sonnet-4-5',
    fallback_on=bad_finish_reason,
)

agent = Agent(fallback_model)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

!!! warning "Solo response handlers replace default exception fallback"
    When you pass a single response handler as `fallback_on` (as above), it **replaces** the default `(ModelAPIError,)` exception fallback entirely. This means API errors (4xx/5xx) will propagate as exceptions instead of triggering fallback to the next model.

    To keep exception-based fallback alongside a response handler, pass them together as a list — see the [mixed example below](#combining-handlers).

!!! note
    The [agent loop](../agent.md) only acts on a finish reason when the response has no actionable
    output. A `'length'` finish reason on an empty or thinking-only response raises
    [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] (typically the model hit
    the token limit mid-thinking), and an empty or thinking-only response with a `'content_filter'` finish
    reason raises [`ContentFilterError`][pydantic_ai.exceptions.ContentFilterError]. Other empty or thinking-only
    responses are re-prompted, up to the output retry limit. Non-empty responses are handled normally
    regardless of finish reason — a tool call truncated mid-arguments is re-prompted like any other
    invalid-arguments failure, and only once its retry budget is exhausted does it surface as
    [`IncompleteToolCall`][pydantic_ai.exceptions.IncompleteToolCall] (a subclass of
    `UnexpectedModelBehavior`).

    All of these are raised from the agent loop, after `model.request()` has already returned
    successfully, so no exception-based `fallback_on` can catch them — not the default
    `fallback_on=(ModelAPIError,)` (which wouldn't match anyway, as `ContentFilterError` inherits from
    `UnexpectedModelBehavior`, not [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]), and not an
    explicit `fallback_on=(ContentFilterError,)` either. To fall back on a bad finish reason instead,
    use a response handler (see the [example above](#finish-reason-example)) that inspects
    [`finish_reason`][pydantic_ai.messages.ModelResponse.finish_reason]: it rejects the response inside
    `FallbackModel` before the agent loop sees it, so the next model is tried instead of an exception
    being raised. Note that it rejects *every* response with that finish reason, including non-empty
    ones the agent loop would have accepted. To instead raise on `content_filter` responses that still
    carry partial or refusal text, add the
    [`RaiseContentFilterError`][pydantic_ai.capabilities.RaiseContentFilterError] capability.

#### Native Tool Failure Example

A more complex use case is when using native tools like web search or URL fetching. For example, Google's [`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] may return a successful response with a status indicating the URL fetch failed:

```python {title="fallback_on_native_tool.py"}
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel


def web_fetch_failed(response: ModelResponse) -> bool:
    """Check if a web_fetch native tool failed to retrieve content."""
    for call, result in response.native_tool_calls:
        if call.tool_name != 'web_fetch':
            continue
        if not isinstance(result.content, list):
            continue
        for item in result.content:
            if isinstance(item, dict):
                status = item.get('url_retrieval_status', '')
                if status and status != 'URL_RETRIEVAL_STATUS_SUCCESS':
                    return True
    return False


google_model = GoogleModel('gemini-2.5-flash')
anthropic_model = AnthropicModel('claude-sonnet-4-5')

# Auto-detected as response handler via type hint
fallback_model = FallbackModel(
    google_model,
    anthropic_model,
    fallback_on=web_fetch_failed,
)

agent = Agent(fallback_model)

# If Google's web_fetch fails, automatically falls back to Anthropic
result = agent.run_sync('Summarize https://ai.pydantic.dev')
print(result.output)
"""
Pydantic AI is a Python agent framework for building production-grade LLM applications.
"""
```

Response handlers receive the [`ModelResponse`][pydantic_ai.messages.ModelResponse] returned by the model and should return `True` to trigger fallback to the next model, or `False` to accept the response.

#### Combining Handlers

You can combine exception types, exception handlers, and response handlers in a single list:

```python {title="fallback_on_mixed.py" requires="fallback_on_native_tool.py"}
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.fallback import FallbackModel

from fallback_on_native_tool import anthropic_model, google_model, web_fetch_failed

fallback_model = FallbackModel(
    google_model,
    anthropic_model,
    fallback_on=[
        ModelAPIError,  # Exception type
        lambda exc: 'rate limit' in str(exc).lower(),  # Exception handler (untyped lambda)
        web_fetch_failed,  # Response handler (auto-detected via type hint)
    ],
)
```

### Exception Handling in Middleware and Decorators

When using `FallbackModel`, it's important to understand that [`FallbackExceptionGroup`][pydantic_ai.exceptions.FallbackExceptionGroup]
inherits from Python's [`ExceptionGroup`](https://docs.python.org/3/library/exceptions.html#ExceptionGroup). This means
that existing exception handling code that catches specific exceptions (like `ModelAPIError`) won't automatically catch
the individual exceptions wrapped inside the group.

For example, if you have middleware or a decorator that catches `ModelAPIError`:

```python {title="middleware_without_fallback.py"}
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from pydantic_ai import ModelAPIError

T = TypeVar('T')


# This handler will NOT catch ModelAPIError when using FallbackModel!
def handle_api_errors(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except ModelAPIError as e:  # Won't catch FallbackExceptionGroup
            print(f'API error: {e}')
            raise

    return wrapper
```

This decorator will miss `ModelAPIError` exceptions when using `FallbackModel`, because they're wrapped in a
`FallbackExceptionGroup` containing one exception per failed model, in the order the models were tried.

To handle both cases, you can use Python 3.11+ `except*` syntax, which catches matching exceptions from
exception groups as well as bare exceptions. Note that `except*` always delivers the caught exceptions as an
`ExceptionGroup` (even if the original was a bare exception), so re-raising will propagate an `ExceptionGroup`
rather than the original exception type:

```python {title="middleware_with_fallback.py" py="3.11"}
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from pydantic_ai import ModelAPIError

T = TypeVar('T')


def handle_api_errors(func: Callable[..., T]) -> Callable[..., T]:
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except* ModelAPIError as exc_group:
            for exc in exc_group.exceptions:
                print(f'API error: {exc}')
            raise

    return wrapper
```

You can also catch `FallbackExceptionGroup` directly if you want to handle it specifically:

```python {title="catch_fallback_exception_group.py" test="skip"}
from pydantic_ai import Agent, FallbackExceptionGroup
from pydantic_ai.models.fallback import FallbackModel

agent = Agent(FallbackModel('openai:gpt-5-mini', 'anthropic:claude-sonnet-4-6'))

try:
    response = agent.run_sync('What is the capital of France?')
except FallbackExceptionGroup as exc_group:
    print(f'All {len(exc_group.exceptions)} models failed:')
    for exc in exc_group.exceptions:
        print(f'  - {exc}')
```
