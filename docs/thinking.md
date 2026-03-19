# Thinking

Thinking (or reasoning) is the process by which a model works through a problem step-by-step before
providing its final answer.

This capability is typically disabled by default and depends on the specific model being used.
See the sections below for how to enable thinking for each provider.

## OpenAI

When using the [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], text output inside `<think>` tags are converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).

Some [OpenAI-compatible model providers](models/openai.md#openai-compatible-models) might also support native thinking parts that are not delimited by tags. Instead, they are sent and received as separate, custom fields in the API. Typically, if you are calling the model via the `<provider>:<model>` shorthand, Pydantic AI handles it for you. Nonetheless, you can still configure the fields with [`openai_chat_thinking_field`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_thinking_field].

If your provider recommends to send back these custom fields not changed, for caching or interleaved thinking benefits, you can also achieve this with [`openai_chat_send_back_thinking_parts`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_send_back_thinking_parts].

### OpenAI Responses

The [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] can generate native thinking parts.
To enable this functionality, you need to set the
[`OpenAIResponsesModelSettings.openai_reasoning_effort`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_effort] and [`OpenAIResponsesModelSettings.openai_reasoning_summary`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_summary] [model settings](agent.md#model-run-settings).

By default, the unique IDs of reasoning, text, and function call parts from the message history are sent to the model, which can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
if the message history you're sending does not match exactly what was received from the Responses API in a previous response, for example if you're using a [history processor](message-history.md#processing-message-history).
To disable this, you can disable the [`OpenAIResponsesModelSettings.openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids] [model setting](agent.md#model-run-settings).

```python {title="openai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
    openai_reasoning_summary='detailed',
)
agent = Agent(model, model_settings=settings)
...
```

!!! note "Raw reasoning without summaries"
    Some OpenAI-compatible APIs (such as LM Studio, vLLM, or OpenRouter with gpt-oss models) may return raw reasoning content without reasoning summaries. In this case, [`ThinkingPart.content`][pydantic_ai.messages.ThinkingPart.content] will be empty, but the raw reasoning is available in `provider_details['raw_content']`. Following [OpenAI's guidance](https://cookbook.openai.com/examples/responses_api/reasoning_items) that raw reasoning should not be shown directly to users, we store it in `provider_details` rather than in the main `content` field.

## Anthropic

To enable thinking, use the [`AnthropicModelSettings.anthropic_thinking`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_thinking] [model setting](agent.md#model-run-settings).

!!! note
    Extended thinking (`type: 'enabled'` with `budget_tokens`) is deprecated on `claude-opus-4-6`+. For those models, use [adaptive thinking](#adaptive-thinking--effort) instead.

```python {title="anthropic_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-sonnet-4-5')
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
)
agent = Agent(model, model_settings=settings)
...
```

### Interleaved Thinking

To enable [interleaved thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#interleaved-thinking), you need to include the beta header in your model settings:

```python {title="anthropic_interleaved_thinking.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-sonnet-4-5')
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 10000},
    extra_headers={'anthropic-beta': 'interleaved-thinking-2025-05-14'},
)
agent = Agent(model, model_settings=settings)
...
```

### Adaptive Thinking & Effort

Starting with `claude-opus-4-6`, Anthropic supports [adaptive thinking](https://docs.anthropic.com/en/docs/build-with-claude/adaptive-thinking), where the model dynamically decides when and how much to think based on the complexity of each request. This replaces extended thinking (`type: 'enabled'` with `budget_tokens`) which is deprecated on Opus 4.6. Adaptive thinking also automatically enables interleaved thinking.

```python {title="anthropic_adaptive_thinking.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-opus-4-6')
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'adaptive'},
    anthropic_effort='high',
)
agent = Agent(model, model_settings=settings)
...
```

The [`anthropic_effort`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_effort] setting controls how much effort the model puts into its response (independent of thinking). See the [Anthropic effort docs](https://docs.anthropic.com/en/docs/build-with-claude/effort) for details.

!!! note
    Older models (`claude-sonnet-4-5`, `claude-opus-4-5`, etc.) do not support adaptive thinking and require `{'type': 'enabled', 'budget_tokens': N}` as shown [above](#anthropic).

## Google

To enable thinking, use the [`GoogleModelSettings.google_thinking_config`][pydantic_ai.models.google.GoogleModelSettings.google_thinking_config] [model setting](agent.md#model-run-settings).

```python {title="google_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-3-pro-preview')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
agent = Agent(model, model_settings=settings)
...
```

## xAI

xAI reasoning models (Grok) support native thinking. To preserve the thinking content for multi-turn conversations, enable [`XaiModelSettings.xai_include_encrypted_content`][pydantic_ai.models.xai.XaiModelSettings.xai_include_encrypted_content].

```python {title="xai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel, XaiModelSettings

model = XaiModel('grok-4-fast-reasoning')
settings = XaiModelSettings(xai_include_encrypted_content=True)
agent = Agent(model, model_settings=settings)
...
```

## Bedrock

Although Bedrock Converse doesn't provide a unified API to enable thinking, you can still use [`BedrockModelSettings.bedrock_additional_model_requests_fields`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_additional_model_requests_fields] [model setting](agent.md#model-run-settings) to pass provider-specific configuration:

=== "Claude"

    ```python {title="bedrock_claude_thinking_part.py"}
    from pydantic_ai import Agent
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
    model_settings = BedrockModelSettings(
        bedrock_additional_model_requests_fields={
            'thinking': {'type': 'enabled', 'budget_tokens': 1024}
        }
    )
    agent = Agent(model=model, model_settings=model_settings)

    ```
=== "OpenAI"


    ```python {title="bedrock_openai_thinking_part.py"}
    from pydantic_ai import Agent
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

    model = BedrockConverseModel('openai.gpt-oss-120b-1:0')
    model_settings = BedrockModelSettings(
        bedrock_additional_model_requests_fields={'reasoning_effort': 'low'}
    )
    agent = Agent(model=model, model_settings=model_settings)

    ```
=== "Qwen"


    ```python {title="bedrock_qwen_thinking_part.py"}
    from pydantic_ai import Agent
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

    model = BedrockConverseModel('qwen.qwen3-32b-v1:0')
    model_settings = BedrockModelSettings(
        bedrock_additional_model_requests_fields={'reasoning_config': 'high'}
    )
    agent = Agent(model=model, model_settings=model_settings)

    ```

=== "Deepseek"
    Reasoning is [always enabled](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html) for Deepseek model

    ```python {title="bedrock_deepseek_thinking_part.py"}
    from pydantic_ai import Agent
    from pydantic_ai.models.bedrock import BedrockConverseModel

    model = BedrockConverseModel('us.deepseek.r1-v1:0')
    agent = Agent(model=model)

    ```

## Groq

Groq supports different formats to receive thinking parts:

- `"raw"`: The thinking part is included in the text content inside `<think>` tags, which are automatically converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
- `"hidden"`: The thinking part is not included in the text content.
- `"parsed"`: The thinking part has its own structured part in the response which is converted into a [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] object.

To enable thinking, use the [`GroqModelSettings.groq_reasoning_format`][pydantic_ai.models.groq.GroqModelSettings.groq_reasoning_format] [model setting](agent.md#model-run-settings):

```python {title="groq_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('qwen-qwq-32b')
settings = GroqModelSettings(groq_reasoning_format='parsed')
agent = Agent(model, model_settings=settings)
...
```

## OpenRouter

To enable thinking, use the [`OpenRouterModelSettings.openrouter_reasoning`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_reasoning] [model setting](agent.md#model-run-settings).

```python {title="openrouter_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

model = OpenRouterModel('openai/gpt-5.2')
settings = OpenRouterModelSettings(openrouter_reasoning={'effort': 'high'})
agent = Agent(model, model_settings=settings)
...
```

## Mistral

Thinking is supported by the `magistral` family of models. It does not need to be specifically enabled.

## Cohere

Thinking is supported by the `command-a-reasoning-08-2025` model. It does not need to be specifically enabled.

## Hugging Face

Text output inside `<think>` tags is automatically converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).

## Outlines

Some local models run through Outlines include in their text output a thinking part delimited by tags. In that case, it will be handled by Pydantic AI that will separate the thinking part from the final answer without the need to specifically enable it. The thinking tags used by default are `"<think>"` and `"</think>"`. If your model uses different tags, you can specify them in the [model profile](models/openai.md#model-profile) using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field.

Outlines currently does not support thinking along with structured output. If you provide an `output_type`, the model text output will not contain a thinking part with the associated tags, and you may experience degraded performance.
