# Model Providers

Pydantic AI is model-agnostic and has built-in support for multiple model providers:

* [OpenAI](openai.md)
* [Anthropic](anthropic.md)
* [Gemini](google.md) (via two different APIs: Generative Language API and VertexAI API)
* [Bedrock](bedrock.md)
* [Cerebras](cerebras.md)
* [Cohere](cohere.md)
* [Groq](groq.md)
* [Hugging Face](huggingface.md)
* [Mistral](mistral.md)
* [OpenRouter](openrouter.md)
* [Outlines](outlines.md)

## OpenAI-compatible Providers

In addition, many providers are compatible with the OpenAI API, and can be used with `OpenAIChatModel` in Pydantic AI:

- [Alibaba Cloud Model Studio (DashScope)](openai.md#alibaba-cloud-model-studio-dashscope)
- [Azure AI Foundry](openai.md#azure-ai-foundry)
- [DeepSeek](openai.md#deepseek)
- [Fireworks AI](openai.md#fireworks-ai)
- [GitHub Models](openai.md#github-models)
- [Grok (xAI)](openai.md#grok-xai)
- [Heroku](openai.md#heroku-ai)
- [LiteLLM](openai.md#litellm)
- [Nebius AI Studio](openai.md#nebius-ai-studio)
- [Ollama](openai.md#ollama)
- [OVHcloud AI Endpoints](openai.md#ovhcloud-ai-endpoints)
- [Perplexity](openai.md#perplexity)
- [Together AI](openai.md#together-ai)
- [Vercel AI Gateway](openai.md#vercel-ai-gateway)

Pydantic AI also comes with [`TestModel`](../api/models/test.md) and [`FunctionModel`](../api/models/function.md)
for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right
packages installed. If you try to use the model without having done so, you'll be told what to install.

## Models and Providers

Pydantic AI uses a few key terms to describe how it interacts with different LLMs:

- **Model**: This refers to the Pydantic AI class used to make requests following a specific LLM API
  (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a
  vendor-SDK-agnostic API, ensuring a single Pydantic AI agent is portable to different LLM vendors without
  any other code changes just by swapping out the Model it uses. Model classes are named
  roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIChatModel`, `AnthropicModel`, `GoogleModel`,
  etc. When using a Model class, you specify the actual LLM model name (e.g., `gpt-5`,
  `claude-sonnet-4-5`, `gemini-2.5-flash`) as a parameter.
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
  with model name `gemini-2.5-pro-preview`, or `OpenAIChatModel` with `OpenRouterProvider` and model name `google/gemini-2.5-pro-preview`.

When you instantiate an [`Agent`][pydantic_ai.Agent] with just a name formatted as `<provider>:<model>`, e.g. `openai:gpt-5` or `openrouter:google/gemini-2.5-pro-preview`,
Pydantic AI will automatically select the appropriate model class, provider, and profile.
If you want to use a different provider or profile, you can instantiate a model class directly and pass in `provider` and/or `profile` arguments.

## Custom Models

!!! note
    If a model API is compatible with the OpenAI API, you do not need a custom model class and can provide your own [custom provider](openai.md#openai-compatible-models) instead.

To implement support for a model API that's not already supported, you will need to subclass the [`Model`][pydantic_ai.models.Model] abstract base class.
For streaming, you'll also need to implement the [`StreamedResponse`][pydantic_ai.models.StreamedResponse] abstract base class.

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIChatModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to Pydantic AI, see the [contributing guidelines](../contributing.md#new-model-rules).


<!-- TODO(Marcelo): We need to create a section in the docs about reliability. -->

## Fallback Model

You can use [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to attempt multiple models
in sequence until one successfully returns a result. Under the hood, Pydantic AI automatically switches
from one model to the next if the current model returns a 4xx or 5xx status code.

!!! note
    The provider SDKs on which Models are based (like OpenAI, Anthropic, etc.) often have built-in retry logic that can delay the `FallbackModel` from activating.

    When using `FallbackModel`, it's recommended to disable provider SDK retries to ensure immediate fallback, for example by setting `max_retries=0` on a [custom OpenAI client](openai.md#custom-openai-client).

In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key),
and then falls back to the Anthropic model.

<!-- TODO(Marcelo): Do not skip this test. For some reason it becomes a flaky test if we don't skip it. -->

```python {title="fallback_model.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

openai_model = OpenAIChatModel('gpt-5')
anthropic_model = AnthropicModel('claude-sonnet-4-5')
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
        model_name='claude-sonnet-4-5',
        timestamp=datetime.datetime(...),
        kind='response',
        provider_response_id=None,
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
    'gpt-5',
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

=== "Python >=3.11"

    ```python {title="fallback_model_failure.py" py="3.11"}
    from pydantic_ai import Agent, ModelAPIError
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.openai import OpenAIChatModel

    openai_model = OpenAIChatModel('gpt-5')
    anthropic_model = AnthropicModel('claude-sonnet-4-5')
    fallback_model = FallbackModel(openai_model, anthropic_model)

    agent = Agent(fallback_model)
    try:
        response = agent.run_sync('What is the capital of France?')
    except* ModelAPIError as exc_group:
        for exc in exc_group.exceptions:
            print(exc)
    ```

=== "Python <3.11"

    Since [`except*`](https://docs.python.org/3/reference/compound_stmts.html#except-star) is only supported
    in Python 3.11+, we use the [`exceptiongroup`](https://github.com/agronholm/exceptiongroup) backport
    package for earlier Python versions:

    ```python {title="fallback_model_failure.py" noqa="F821" test="skip"}
    from exceptiongroup import catch

    from pydantic_ai import Agent, ModelAPIError
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.openai import OpenAIChatModel


    def model_status_error_handler(exc_group: BaseExceptionGroup) -> None:
        for exc in exc_group.exceptions:
            print(exc)


    openai_model = OpenAIChatModel('gpt-5')
    anthropic_model = AnthropicModel('claude-sonnet-4-5')
    fallback_model = FallbackModel(openai_model, anthropic_model)

    agent = Agent(fallback_model)
    with catch({ModelAPIError: model_status_error_handler}):
        response = agent.run_sync('What is the capital of France?')
    ```

By default, the `FallbackModel` only moves on to the next model if the current model raises a
[`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], which includes
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. You can customize this behavior by
passing a custom `fallback_on` argument to the `FallbackModel` constructor.

!!! note
    Validation errors (from [structured output](../output.md#structured-output) or [tool parameters](../tools.md)) do **not** trigger fallback. These errors use the [retry mechanism](../agents.md#reflection-and-self-correction) instead, which re-prompts the same model to try again. This is intentional: validation errors stem from the non-deterministic nature of LLMs and may succeed on retry, whereas API errors (4xx/5xx) generally indicate issues that won't resolve by retrying the same request.

!!! note "Streaming limitation"
    For streaming requests, exception-based fallback only catches errors during stream **initialization** (e.g., connection errors, authentication failures). If an exception occurs mid-stream after events have started flowing, it will propagate to the caller without triggering fallback. To handle mid-stream failures, use [`fallback_on_part`](#part-based-fallback-streaming) which buffers the stream and can cleanly switch to a fallback model.

### Response-Based Fallback

In addition to exception-based fallback, you can also trigger fallback based on the **content** of a model's response using the `fallback_on_response` parameter. This is useful when a model returns a successful HTTP response (no exception), but the response content indicates a semantic failure.

A common use case is when using built-in tools like web search or URL fetching. For example, Google's `WebFetchTool` may return a successful response with a status indicating the URL fetch failed:

```python {title="fallback_on_response.py" test="skip" lint="skip"}
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import BuiltinToolCallPart, BuiltinToolReturnPart, ModelMessage, ModelResponse
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel


def web_fetch_failed(response: ModelResponse, messages: list[ModelMessage]) -> bool:
    """Check if a web_fetch built-in tool failed to retrieve content."""
    call: BuiltinToolCallPart
    result: BuiltinToolReturnPart
    for call, result in response.builtin_tool_calls:
        if call.tool_name != 'web_fetch':
            continue
        if not isinstance(result.content, dict):
            continue
        content: dict[str, Any] = result.content
        status = content.get('url_retrieval_status', '')
        if status and status != 'URL_RETRIEVAL_STATUS_SUCCESS':
            return True
    return False


google_model = GoogleModel('gemini-2.5-flash')
anthropic_model = AnthropicModel('claude-sonnet-4-5')

fallback_model = FallbackModel(
    google_model,
    anthropic_model,
    fallback_on_response=web_fetch_failed,
)

agent = Agent(fallback_model)

# If Google's web_fetch fails, automatically falls back to Anthropic
result = agent.run_sync('Summarize https://example.com')
print(result.output)
```

The `fallback_on_response` callback receives two arguments:

- `response`: The [`ModelResponse`][pydantic_ai.messages.ModelResponse] returned by the model
- `messages`: The list of [`ModelMessage`][pydantic_ai.messages.ModelMessage] that were sent to the model

The callback should return `True` to trigger fallback to the next model, or `False` to accept the response.

!!! note
    When using `fallback_on_response` with streaming (`run_stream`), the entire response is buffered before being returned. This means the caller won't receive partial results until the full response is ready and the fallback condition has been evaluated. This is necessary because the response content must be fully available to evaluate the fallback condition.

### Part-Based Fallback (Streaming)

For streaming requests, you can use `fallback_on_part` to check each response part as it arrives from the model. This enables **early abort** when failure conditions are detectable before the full response completes—saving tokens and reducing latency by starting the fallback sooner.

This is particularly useful when built-in tool results (like `web_fetch`) arrive early in the stream:

```python {title="fallback_on_part.py" test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.messages import BuiltinToolReturnPart, ModelMessage, ModelResponsePart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.google import GoogleModel


def web_fetch_failed_part(part: ModelResponsePart, messages: list[ModelMessage]) -> bool:
    """Check if a web_fetch built-in tool part indicates failure."""
    if not isinstance(part, BuiltinToolReturnPart) or part.tool_name != 'web_fetch':
        return False
    if not isinstance(part.content, dict):
        return False
    status = part.content.get('url_retrieval_status', '')
    return bool(status) and status != 'URL_RETRIEVAL_STATUS_SUCCESS'


google_model = GoogleModel('gemini-2.5-flash')
anthropic_model = AnthropicModel('claude-sonnet-4-5')

fallback_model = FallbackModel(
    google_model,
    anthropic_model,
    fallback_on_part=web_fetch_failed_part,
)

agent = Agent(fallback_model)

# With streaming, fallback can occur as soon as the failed tool result arrives
async with agent.run_stream('Summarize https://example.com') as result:
    output = await result.get_output()
print(output)
```

The `fallback_on_part` callback receives:

- `part`: A [`ModelResponsePart`][pydantic_ai.messages.ModelResponsePart] that has completed streaming
- `messages`: The list of [`ModelMessage`][pydantic_ai.messages.ModelMessage] that were sent to the model

You can use both `fallback_on_part` and `fallback_on_response` together. Parts are checked during streaming, and if the stream completes without part rejection, the full response is checked with `fallback_on_response`.

!!! warning "Buffering trade-off"
    Like `fallback_on_response`, using `fallback_on_part` buffers the response before returning it to the caller. This means the caller won't receive events progressively—they'll receive the complete response after all parts have been validated.

    The benefit of `fallback_on_part` over `fallback_on_response` is **not** live streaming to the caller, but rather:

    - **Token savings**: Stop consuming a response as soon as a failure is detected, rather than waiting for it to complete
    - **Faster fallback**: Start the next model immediately instead of waiting for a doomed response to finish
    - **Cost reduction**: Pay only for the tokens consumed before the failure was detected

    If you need progressive streaming to the caller and only want fallback for connection/initialization errors, you can omit `fallback_on_part` and `fallback_on_response`. However, be aware that exception-based fallback (`fallback_on`) only catches errors during stream **initialization**—if an exception occurs mid-stream after events have started flowing, it will propagate to the caller without triggering fallback.

!!! note
    `fallback_on_part` only applies to streaming requests (`run_stream`). For non-streaming requests, use `fallback_on_response` instead.
