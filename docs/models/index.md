# Model Providers

PydanticAI is model-agnostic and has built-in support for multiple model providers:

* [OpenAI](openai.md)
* [DeepSeek](openai.md#openai-compatible-models)
* [Anthropic](anthropic.md)
* [Gemini](gemini.md) (via two different APIs: Generative Language API and VertexAI API)
* [Ollama](openai.md#ollama)
* [Groq](groq.md)
* [Mistral](mistral.md)
* [Cohere](cohere.md)
* [Bedrock](bedrock.md)

## OpenAI-compatible Providers

Many models are compatible with the OpenAI API, and can be used with `OpenAIModel` in PydanticAI:

* [OpenRouter](openai.md#openrouter)
* [Grok (xAI)](openai.md#grok-xai)
* [Perplexity](openai.md#perplexity)
* [Fireworks AI](openai.md#fireworks-ai)
* [Together AI](openai.md#together-ai)
* [Azure AI Foundry](openai.md#azure-ai-foundry)

PydanticAI also comes with [`TestModel`](../api/models/test.md) and [`FunctionModel`](../api/models/function.md)
for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right
packages installed.

## Models and Providers

PydanticAI uses a few key terms to describe how it interacts with different LLMs:

* **Model**: This refers to the PydanticAI class used to make requests following a specific LLM API
    (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a
    vendor-SDK-agnostic API, ensuring a single PydanticAI agent is portable to different LLM vendors without
    any other code changes just by swapping out the Model it uses. Model classes are named
    roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIModel`, `AnthropicModel`, `GeminiModel`,
    etc. When using a Model class, you specify the actual LLM model name (e.g., `gpt-4o`,
    `claude-3-5-sonnet-latest`, `gemini-1.5-flash`) as a parameter.
* **Provider**: This refers to Model-specific classes which handle the authentication and connections
    to an LLM vendor. Passing a non-default _Provider_ as a parameter to a Model is how you can ensure
    that your agent will make requests to a specific endpoint, or make use of a specific approach to
    authentication (e.g., you can use Vertex-specific auth with the `GeminiModel` by way of the `VertexProvider`).
    In particular, this is how you can make use of an AI gateway, or an LLM vendor that offers API compatibility
    with the vendor SDK used by an existing Model (such as `OpenAIModel`).

In short, you select a specific model name (like `gpt-4o`), PydanticAI uses the appropriate Model class (like `OpenAIModel`), and the provider handles the connection and authentication to the underlying service.

## Custom Models

To implement support for models not already supported, you will need to subclass the [`Model`][pydantic_ai.models.Model] abstract base class.

For streaming, you'll also need to implement the following abstract base class:

* [`StreamedResponse`][pydantic_ai.models.StreamedResponse]

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to PydanticAI, see the [contributing guidelines](../contributing.md#new-model-rules).

<!-- TODO(Marcelo): We need to create a section in the docs about reliability. -->
## Fallback Model

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

openai_model = OpenAIModel('gpt-4o')
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

    openai_model = OpenAIModel('gpt-4o')
    anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
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


    openai_model = OpenAIModel('gpt-4o')
    anthropic_model = AnthropicModel('claude-3-5-sonnet-latest')
    fallback_model = FallbackModel(openai_model, anthropic_model)

    agent = Agent(fallback_model)
    with catch({ModelHTTPError: model_status_error_handler}):
        response = agent.run_sync('What is the capital of France?')
    ```

By default, the `FallbackModel` only moves on to the next model if the current model raises a
[`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError]. You can customize this behavior by
passing a custom `fallback_on` argument to the `FallbackModel` constructor.
