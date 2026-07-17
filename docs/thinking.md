# Thinking

Thinking (or reasoning) is the process by which a model works through a problem step-by-step before
providing its final answer.

The simplest way to enable thinking across supported providers is the [`Thinking`][pydantic_ai.capabilities.Thinking] capability.
Provider-specific settings are available for advanced usage when you need direct access to a provider's native thinking controls.

## Unified thinking settings

Use the [`Thinking` capability](capabilities.md#thinking) to enable thinking:

```python {title="thinking_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('anthropic:claude-opus-4-7', capabilities=[Thinking(effort='high')])
```

You can also set the underlying `thinking` field in [`ModelSettings`][pydantic_ai.settings.ModelSettings] directly:

```python {title="unified_thinking.py"}
from pydantic_ai import Agent

agent = Agent('anthropic:claude-opus-4-7', model_settings={'thinking': 'high'})
```

The [`Thinking.effort`][pydantic_ai.capabilities.Thinking.effort] value accepts:

- `True` — enable thinking with the provider's default effort level
- `False` — disable thinking (silently ignored on always-on models)
- `'minimal'` / `'low'` / `'medium'` / `'high'` / `'xhigh'` — enable thinking at a specific effort level (unsupported levels map to the closest available value)

These are the same values accepted by the underlying `thinking` model setting.
When omitted, the model uses its default behavior. Provider-specific settings (documented in the sections below) take precedence when both are set.

### Provider translation

The `Thinking` capability maps each effort value to the selected provider's native format:

| Provider | `Thinking()` / `Thinking(effort=True)` | `Thinking(effort='high')` | Notes |
|---|---|---|---|
| Anthropic (Opus 4.6+) | `anthropic_thinking={'type': 'adaptive'}` | `{type: 'adaptive'}` + `effort='high'` | Claude Opus 4.7, 4.8, and Sonnet 5 also support `effort='xhigh'` |
| Anthropic (older) | `anthropic_thinking={'type': 'enabled', 'budget_tokens': 10000}` | `budget_tokens=16384` | Budget-based; `'low'` → 2048 tokens |
| OpenAI | `reasoning_effort='medium'` | `reasoning_effort='high'` | |
| Google (Gemini 3+) | `include_thoughts=True` | `thinking_level='HIGH'` | |
| Google (Gemini 2.5) | `include_thoughts=True` | `thinking_budget=24576` | |
| Groq | `reasoning_format='parsed'` (gpt-oss also `reasoning_effort='medium'`) | `reasoning_format='parsed'` (gpt-oss also `reasoning_effort='high'`) | gpt-oss: unified effort → `reasoning_effort` (`low`/`medium`/`high`, via `extra_body`; always-on, so `thinking=False` is silently ignored); qwen3: `thinking=False` → `reasoning_effort='none'` (true disable, via `extra_body`); other reasoning models → `'hidden'` (suppresses output only) |
| Mistral | `reasoning_effort='high'` | `reasoning_effort='high'` | Only on adjustable-reasoning models (e.g. `mistral-small-latest`, `mistral-medium-3-5`); `magistral` reasons always-on and gets no `reasoning_effort`. Mistral exposes only `'high'`/`'none'`, so every enabled level (incl. `'minimal'`) → `'high'` and only `thinking=False` → `'none'` |
| OpenRouter | `reasoning={'effort': 'medium', 'enabled': True}` | `reasoning={'effort': 'high', 'enabled': True}` | `thinking=False` → `effort='none'`; always-on routes silently ignore; via `extra_body` |
| Cerebras | `reasoning_effort` omitted (reasons by default) | `reasoning_effort` omitted | `thinking=False` → `reasoning_effort='none'`; gpt-oss reasons always-on, so `thinking=False` is silently ignored |
| xAI | `reasoning_effort` omitted on Grok 4.3 (uses its default) | `reasoning_effort='high'` | Grok 4.3 supports `'none'`, `'low'`, `'medium'`, and `'high'`, and `thinking=True` omits the parameter so the model applies its own default; Grok 3 Mini only supports `'low'` and `'high'` (so `thinking=True` → `'high'`) and silently ignores `thinking=False`; Grok 4.5 supports `'low'`, `'medium'`, and `'high'` but not `'none'`, so it reasons always-on (`thinking=True` → `'medium'`) and silently ignores `thinking=False` |
| Bedrock (Claude 4.6+) | `thinking.type='adaptive'` | `{type: 'adaptive'}` + `output_config.effort='high'` | Effort lives in the sibling `output_config` field per AWS docs; `xhigh` maps to `max` |
| Bedrock (Claude older) | `thinking.type='enabled'` | `budget_tokens=16384` | Budget-based |
| Bedrock (OpenAI) | `reasoning_effort='medium'` | `reasoning_effort='high'` | Converse rejects `'none'`; `thinking=False` silently ignored |
| Bedrock (Qwen) | `reasoning_config='high'` | `reasoning_config='high'` | Only `'low'` and `'high'`; `thinking=False` silently ignored |

## OpenAI

When using the [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], text output inside `<think>` tags are converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).

Some [OpenAI-compatible model providers](models/openai.md#openai-compatible-models) might also support native thinking parts that are not delimited by tags. Instead, they are sent and received as separate, custom fields in the API. Typically, if you are calling the model via the `<provider>:<model>` shorthand, Pydantic AI handles it for you. Nonetheless, you can still configure the fields with [`openai_chat_thinking_field`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_thinking_field].

If your provider recommends to send back these custom fields not changed, for caching or interleaved thinking benefits, you can also achieve this with [`openai_chat_send_back_thinking_parts`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_chat_send_back_thinking_parts].

### OpenAI Responses

The [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] can generate native thinking parts.
To enable this functionality, you need to set the
[`OpenAIResponsesModelSettings.openai_reasoning_effort`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_effort] and [`OpenAIResponsesModelSettings.openai_reasoning_summary`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_summary] [model settings](agent.md#model-run-settings).
Models that support it can additionally use a `pro` [reasoning mode](models/openai.md#reasoning-mode), which is independent of the effort and never set by the unified `thinking` setting.

By default, the unique IDs of reasoning, text, and function call parts from the message history are sent to the model, which can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
if the message history you're sending does not match exactly what was received from the Responses API in a previous response, for example if you're using a [history processor](message-history.md#processing-message-history).
To disable this, you can disable the [`OpenAIResponsesModelSettings.openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids] [model setting](agent.md#model-run-settings).

```python {title="openai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')
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
    Extended thinking (`type: 'enabled'` with `budget_tokens`) is deprecated on `claude-opus-4-6` and removed on `claude-opus-4-7`, `claude-opus-4-8`, and `claude-sonnet-5`. For those models, use [adaptive thinking](#adaptive-thinking--effort) instead.

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

Starting with `claude-opus-4-6`, Anthropic supports [adaptive thinking](https://docs.anthropic.com/en/docs/build-with-claude/adaptive-thinking), where the model dynamically decides when and how much to think based on the complexity of each request. This replaces extended thinking (`type: 'enabled'` with `budget_tokens`) which is deprecated on Opus 4.6 and removed on Opus 4.7, 4.8, and Sonnet 5. Claude Opus 4.7, 4.8, and Sonnet 5 also add the `xhigh` effort level. Adaptive thinking also automatically enables interleaved thinking.

```python {title="anthropic_adaptive_thinking.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-opus-4-8')
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

Thinking tokens count against Anthropic's loop-wide [task budgets](models/anthropic.md#task-budgets-beta), so adaptive thinking naturally scales down as the budget depletes.

## Google

For advanced usage, use the [`GoogleModelSettings.google_thinking_config`][pydantic_ai.models.google.GoogleModelSettings.google_thinking_config] [model setting](agent.md#model-run-settings).

```python {title="google_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-3.5-flash')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True, 'thinking_level': 'MEDIUM'})
agent = Agent(model, model_settings=settings)
...
```

See the [Google model docs](models/google.md#configure-thinking) for more details.

## xAI

xAI reasoning models (Grok) support native thinking. To preserve the thinking content for multi-turn conversations, enable [`XaiModelSettings.xai_include_encrypted_content`][pydantic_ai.models.xai.XaiModelSettings.xai_include_encrypted_content].

```python {title="xai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel, XaiModelSettings

model = XaiModel('grok-4.3')
settings = XaiModelSettings(xai_include_encrypted_content=True)
agent = Agent(model, model_settings=settings)
...
```

## Bedrock

For Claude Sonnet 4.6+ and Opus 4.6+, Pydantic AI's unified `thinking` setting translates to AWS's required [adaptive thinking](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-adaptive-thinking.html) shape automatically — set [`ModelSettings.thinking`][pydantic_ai.settings.ModelSettings.thinking] and you're done.

For older Claude models or to pin a specific `budget_tokens`, you can still use [`BedrockModelSettings.bedrock_additional_model_requests_fields`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_additional_model_requests_fields] [model setting](agent.md#model-run-settings) to pass provider-specific configuration directly:

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

The unified [`ModelSettings.thinking`][pydantic_ai.settings.ModelSettings.thinking] setting works across providers: it selects `reasoning_format='parsed'` so thinking parts are returned, and for the gpt-oss family its effort level also drives Groq's `reasoning_effort` (`minimal`/`low` → `'low'`, `medium` → `'medium'`, `high`/`xhigh` → `'high'`, `True` → `'medium'`).

Two composable [model settings](agent.md#model-run-settings) give finer control: [`GroqModelSettings.groq_reasoning_format`][pydantic_ai.models.groq.GroqModelSettings.groq_reasoning_format] selects how thinking parts are returned (the formats above), and [`GroqModelSettings.groq_reasoning_effort`][pydantic_ai.models.groq.GroqModelSettings.groq_reasoning_effort] (sent to Groq as `reasoning_effort`) controls how much the model reasons, taking precedence over the unified `thinking` mapping:

```python {title="groq_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('openai/gpt-oss-120b')
settings = GroqModelSettings(groq_reasoning_format='parsed', groq_reasoning_effort='medium')
agent = Agent(model, model_settings=settings)
...
```

!!! note
    Most Groq reasoning models do not support truly disabling thinking. When `thinking=False` is set via the unified setting, the behavior is family-specific: the qwen3 family truly disables reasoning via `reasoning_effort='none'` (and when combined with an explicit `groq_reasoning_effort` on qwen3, the disable wins and `groq_reasoning_effort` is ignored, with a warning); the gpt-oss family reasons always-on and cannot be disabled, so `thinking=False` is silently ignored; other reasoning models send `reasoning_format='hidden'`, which suppresses reasoning output but the model may still reason internally.

!!! note
    The accepted `reasoning_effort` values are family-specific (see the [Groq docs](https://console.groq.com/docs/reasoning#reasoning-effort)): the gpt-oss family accepts `'low'`, `'medium'`, and `'high'`, so unified `thinking` effort levels map onto those; the qwen3 family accepts only `'none'` and `'default'`, so unified enable-levels there control `reasoning_format` but send no `reasoning_effort` (there is no gradation to map). An explicit `groq_reasoning_effort` always takes precedence over the unified mapping.

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

!!! note "Wire format details"
    Truthy [`thinking`][pydantic_ai.settings.ModelSettings.thinking] values send both `effort` and `enabled: True` on the wire. The explicit `enabled: True` is a no-op for reasoning-by-default models but load-bearing for reasoning-optional routes (parts of the `google/gemma-*` family, for example) that otherwise leave reasoning disabled despite `effort` being set.

    [`thinking=False`][pydantic_ai.settings.ModelSettings.thinking] sends `reasoning={'effort': 'none'}` — the [documented OpenRouter disable signal](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) — on routes whose upstream can honor disable (e.g. `anthropic/claude-sonnet-4.5`, `z-ai/glm-4.6`). On routes whose upstream is always-on (e.g. `openai/o3`, `openai/gpt-5`, `mistralai/magistral-medium-*`, `deepseek/deepseek-r1`, `x-ai/grok-3-mini`), `thinking=False` is silently ignored at the model-profile gate, matching the same model's direct-route behavior. Set [`OpenRouterModelSettings.openrouter_reasoning`][pydantic_ai.models.openrouter.OpenRouterModelSettings.openrouter_reasoning] directly when you want explicit per-route control.

## Z.AI

To enable thinking, use the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] [model setting](agent.md#model-run-settings). To preserve thinking content across multi-turn conversations, also set [`ZaiModelSettings.zai_clear_thinking`][pydantic_ai.models.zai.ZaiModelSettings.zai_clear_thinking] to `False`.

```python {title="zai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel, ZaiModelSettings

model = ZaiModel('glm-5')
settings = ZaiModelSettings(thinking=True, zai_clear_thinking=False)
agent = Agent(model, model_settings=settings)
...
```

## Mistral

The `magistral` family always reasons and does not need to be specifically enabled; `thinking=False` is silently ignored. Mistral has [deprecated](https://docs.mistral.ai/resources/deprecated/native-reasoning) the `magistral` family in favor of the adjustable-reasoning models below.

Models with adjustable reasoning (the Mistral Small 4 and Medium 3.5 families: `mistral-small-latest`, `mistral-small-2603`, `mistral-medium-latest`, `mistral-medium-3-5`, `mistral-medium-2604`) are controlled via the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting, which maps to Mistral's `reasoning_effort`. Mistral exposes only `'high'` (full thinking) and `'none'` (thinking suppressed), so every enabled level maps to `'high'` and only `thinking=False` maps to `'none'`. Older `mistral-small-*` / `mistral-medium-*` snapshots do not support reasoning, so `thinking` is silently ignored for them.

## Cohere

Thinking is supported by the `command-a-reasoning-08-2025` model. It does not need to be specifically enabled.

## Hugging Face

Text output inside `<think>` tags is automatically converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).
