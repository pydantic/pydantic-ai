# Model Providers

Pydantic AI is model-agnostic and has built-in support for multiple model providers:

* [OpenAI](openai.md)
* [Anthropic](anthropic.md)
* [Gemini](gemini.md) (via two different APIs: Generative Language API and VertexAI API)
* [Groq](groq.md)
* [Mistral](mistral.md)
* [Cohere](cohere.md)
* [Bedrock](bedrock.md)
* [Hugging Face](huggingface.md)

## OpenAI-compatible Providers

In addition, many providers are compatible with the OpenAI API, and can be used with `OpenAIModel` in Pydantic AI:

- [DeepSeek](openai.md#deepseek)
- [Grok (xAI)](openai.md#grok-xai)
- [Ollama](openai.md#ollama)
- [OpenRouter](openai.md#openrouter)
- [Vercel AI Gateway](openai.md#vercel-ai-gateway)
- [Perplexity](openai.md#perplexity)
- [Fireworks AI](openai.md#fireworks-ai)
- [Together AI](openai.md#together-ai)
- [Azure AI Foundry](openai.md#azure-ai-foundry)
- [Heroku](openai.md#heroku-ai)
- [GitHub Models](openai.md#github-models)

Pydantic AI also comes with [`TestModel`](../api/models/test.md) and [`FunctionModel`](../api/models/function.md)
for testing and development.

To use each model provider, you need to configure your local environment and make sure you have the right
packages installed.

## Models and Providers

Pydantic AI uses a few key terms to describe how it interacts with different LLMs:

- **Model**: This refers to the Pydantic AI class used to make requests following a specific LLM API
  (generally by wrapping a vendor-provided SDK, like the `openai` python SDK). These classes implement a
  vendor-SDK-agnostic API, ensuring a single Pydantic AI agent is portable to different LLM vendors without
  any other code changes just by swapping out the Model it uses. Model classes are named
  roughly in the format `<VendorSdk>Model`, for example, we have `OpenAIModel`, `AnthropicModel`, `GeminiModel`,
  etc. When using a Model class, you specify the actual LLM model name (e.g., `gpt-4o`,
  `claude-3-5-sonnet-latest`, `gemini-1.5-flash`) as a parameter.
- **Provider**: This refers to provider-specific classes which handle the authentication and connections
  to an LLM vendor. Passing a non-default _Provider_ as a parameter to a Model is how you can ensure
  that your agent will make requests to a specific endpoint, or make use of a specific approach to
  authentication (e.g., you can use Vertex-specific auth with the `GeminiModel` by way of the `VertexProvider`).
  In particular, this is how you can make use of an AI gateway, or an LLM vendor that offers API compatibility
  with the vendor SDK used by an existing Model (such as `OpenAIModel`).
- **Profile**: This refers to a description of how requests to a specific model or family of models need to be
  constructed to get the best results, independent of the model and provider classes used.
  For example, different models have different restrictions on the JSON schemas that can be used for tools,
  and the same schema transformer needs to be used for Gemini models whether you're using `GoogleModel`
  with model name `gemini-2.5-pro-preview`, or `OpenAIModel` with `OpenRouterProvider` and model name `google/gemini-2.5-pro-preview`.

When you instantiate an [`Agent`][pydantic_ai.Agent] with just a name formatted as `<provider>:<model>`, e.g. `openai:gpt-4o` or `openrouter:google/gemini-2.5-pro-preview`,
Pydantic AI will automatically select the appropriate model class, provider, and profile.
If you want to use a different provider or profile, you can instantiate a model class directly and pass in `provider` and/or `profile` arguments.

## Custom Models

To implement support for a model API that's not already supported, you will need to subclass the [`Model`][pydantic_ai.models.Model] abstract base class.
For streaming, you'll also need to implement the [`StreamedResponse`][pydantic_ai.models.StreamedResponse] abstract base class.

The best place to start is to review the source code for existing implementations, e.g. [`OpenAIModel`](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/models/openai.py).

For details on when we'll accept contributions adding new models to Pydantic AI, see the [contributing guidelines](../contributing.md#new-model-rules).

If a model API is compatible with the OpenAI API, you do not need a custom model class and can provide your own [custom provider](openai.md#openai-compatible-models) instead.

<!-- TODO(Marcelo): We need to create a section in the docs about reliability. -->

## Fallback Model

You can use [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] to attempt multiple models
in sequence until one successfully returns a result. Under the hood, Pydantic AI automatically switches
from one model to the next if the current model returns a 4xx or 5xx status code.

In the following example, the agent first makes a request to the OpenAI model (which fails due to an invalid API key),
and then falls back to the Anthropic model.

<!-- TODO(Marcelo): Do not skip this test. For some reason it becomes a flaky test if we don't skip it. -->

```python {title="fallback_model.py" test="skip"}
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
        vendor_id=None,
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
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings

# Configure each model with provider-specific optimal settings
openai_model = OpenAIModel(
    'gpt-4o',
    settings=ModelSettings(temperature=0.7, max_tokens=1000)  # Higher creativity for OpenAI
)
anthropic_model = AnthropicModel(
    'claude-3-5-sonnet-latest',
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
