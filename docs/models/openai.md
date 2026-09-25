---
description: "Use OpenAI GPT models with Pydantic AI via the Responses or Chat Completions API, or any OpenAI-compatible API such as DeepSeek, Azure, vLLM or LiteLLM."
---

# OpenAI

## Install

To use OpenAI models or OpenAI-compatible APIs, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

To use OpenAI models with the OpenAI API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key.

!!! tip
    You can also use your [ChatGPT/Codex subscription](openai-codex.md) instead of an API key: the `openai-codex:` prefix authenticates against the Codex backend with the same OAuth flow as the official Codex CLI.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

The bare `'openai:'` prefix resolves to [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel], which uses the modern [Responses API](https://platform.openai.com/docs/api-reference/responses).

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-6-sol')
...
```

To pin to the legacy [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) instead, use the `'openai-chat:'` prefix, which resolves to [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel].
For `gpt-6-sol` and `gpt-6-luna`, Chat Completions supports function calling only when `openai_reasoning_effort='none'`. Use the Responses API when you need reasoning and tools together.

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

model = OpenAIResponsesModel('gpt-6-sol')
agent = Agent(model)
...
```

By default, the model uses the `OpenAIProvider` with the `base_url` set to `https://api.openai.com/v1`.

## Configure the provider

If you want to pass parameters in code to the provider, you can programmatically instantiate the
[OpenAIProvider][pydantic_ai.providers.openai.OpenAIProvider] and pass it to the model:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

## Custom OpenAI Client

`OpenAIProvider` also accepts a custom `AsyncOpenAI` client via the `openai_client` parameter, so you can customise the `organization`, `project`, `base_url` etc. as defined in the [OpenAI API docs](https://platform.openai.com/docs/api-reference).

The client retries failed requests on its own, independently of the agent's retry budgets. It defaults to `max_retries=2`, so one model request can reach the network up to three times. It honors the `x-should-retry` response header; without that header, it retries status 408, 409, 429 or 5xx, plus timeouts and connection errors, but not other 4xx responses such as 400 or 401. Set `max_retries=0` to keep the retry policy in your transport alone. See [Retry multiplication](../retries.md#retry-multiplication) for how the layers stack.

```python {title="custom_openai_client.py"}
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncOpenAI(max_retries=3)
model = OpenAIResponsesModel('gpt-5.2', provider=OpenAIProvider(openai_client=client))
agent = Agent(model)
...
```

You could also use the [`AsyncAzureOpenAI`](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/switching-endpoints) client
to use the Azure OpenAI API. Note that the `AsyncAzureOpenAI` is a subclass of `AsyncOpenAI`.

```python
from openai import AsyncAzureOpenAI

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

client = AsyncAzureOpenAI(
    azure_endpoint='...',
    api_version='2024-07-01-preview',
    api_key='your-api-key',
)

model = OpenAIChatModel(
    'gpt-5.2',
    provider=OpenAIProvider(openai_client=client),
)
agent = Agent(model)
...
```

## Image generation

Use [`ImageGenerator`][pydantic_ai.images.ImageGenerator] with an `openai:` image model for direct generation and
reference-image editing. This uses OpenAI's Images API rather than a conversational Responses model:

```python {title="openai_image_generation.py"}
from pydantic_ai import ImageGenerator
from pydantic_ai.images.openai import OpenAIImageGenerationSettings

generator = ImageGenerator(
    'openai:gpt-image-2',
    settings=OpenAIImageGenerationSettings(
        dimensions=(1280, 720),
        openai_quality='low',
        openai_output_format='jpeg',
    ),
)
```

OpenAI accepts [`BinaryImage`][pydantic_ai.messages.BinaryImage] and
[`ImageUrl`][pydantic_ai.messages.ImageUrl] reference inputs. Its image-edit endpoint requires file content and does not
accept an [`UploadedFile`][pydantic_ai.messages.UploadedFile] provider file ID. Transparent-background support varies
by model and requires PNG or WebP output. Provider-specific settings are forwarded to OpenAI so newly supported values
are not blocked by stale client-side checks. See the [image-generation guide](../image-generation.md) for generation,
editing, geometry, and normalized settings.

GPT Image models require a verified organization on a paid usage tier: on an unverified organization every request fails
with a rate-limit error before anything generates. Complex prompts can take up to two minutes to process.

## Model settings

You can customize model behavior using [`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
settings = OpenAIResponsesModelSettings(
    temperature=0.2,
    service_tier='flex',
)
agent = Agent(model, model_settings=settings)
...
```

### Service tier

OpenAI supports controlling the [service tier](https://platform.openai.com/docs/api-reference/responses/create#responses-create-service_tier) to trade off latency and cost.
You can use the unified [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier] field or the provider-specific [`openai_service_tier`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_service_tier] field. Both accept `'auto'`, `'default'`, `'flex'`, and `'priority'`, passed through unchanged. `openai_service_tier` takes precedence over the unified field when both are set.

OpenAI may serve a request on a different tier than the one requested, for example when a `'priority'` request is downgraded. The tier that actually served the request is stored in `ModelResponse.provider_details['service_tier']`.

### Prompt caching

GPT-5.6 and GPT-6 models support OpenAI's [implicit and explicit prompt cache breakpoints](https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-breakpoints) with both the Responses and Chat Completions APIs. OpenAI creates an implicit breakpoint by default. To control the cacheable prefix precisely, insert [`CachePoint`][pydantic_ai.messages.CachePoint] after the user content block that should end the prefix:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.openai import OpenAIResponsesModelSettings

settings = OpenAIResponsesModelSettings(
    openai_prompt_cache_key='product-docs-v1',
    openai_prompt_cache_options={'mode': 'explicit', 'ttl': '30m'},
)
agent = Agent('openai:gpt-5.6-sol', model_settings=settings)

result = agent.run_sync([
    'Long-lived reference material...',
    CachePoint(),
    'Answer using the reference material.',
])
```

Caching requires a prefix of at least 1024 tokens; shorter prefixes are not cached even when explicitly marked. With `mode='implicit'` (the default), OpenAI may write one implicit and up to three explicit breakpoints. With `mode='explicit'`, it may write up to four explicit breakpoints and no implicit breakpoint. The TTL is request-wide: OpenAI currently accepts only `'30m'`, configured through [`openai_prompt_cache_options`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_prompt_cache_options], and ignores the generic per-marker [`CachePoint.ttl`][pydantic_ai.messages.CachePoint.ttl] value. For GPT-5.6 and later models, set a stable [`openai_prompt_cache_key`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_prompt_cache_key] to use OpenAI's more reliable matching for both implicit and explicit caching. Requests without a key may still receive automatic cache hits, but do not use the improved matching. Use different keys to partition unrelated workloads.

When OpenAI reports prompt cache writes, Pydantic AI exposes them as [`result.usage.cache_write_tokens`][pydantic_ai.usage.RunUsage.cache_write_tokens]. Cache reads are available as [`result.usage.cache_read_tokens`][pydantic_ai.usage.RunUsage.cache_read_tokens]. For GPT-5.6 and later model families, OpenAI bills cache writes at 1.25 times the uncached input token rate.

### Moderation

Both the Responses and Chat Completions APIs can run [moderation](https://platform.openai.com/docs/guides/moderation) on the input and output of a request. Moderation is off by default; enable it with [`openai_moderation`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_moderation]:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
settings = OpenAIResponsesModelSettings(
    openai_moderation={'model': 'omni-moderation-latest'}
)
agent = Agent(model, model_settings=settings)

result = agent.run_sync('Your prompt here')
moderation = (result.response.provider_details or {}).get('moderation')
```

When the response includes moderation results, they are stored under the `'moderation'` key of [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details], with `input` and `output` entries each carrying the flagged status, per-category flags, and category scores.

With [`OpenAIChatModel`](#chat-completions-api), use [`OpenAIChatModelSettings`][pydantic_ai.models.openai.OpenAIChatModelSettings] instead. The results are surfaced the same way on both the non-streaming and streaming paths, except that the Chat Completions API nests each entry one level deeper, under a `results` list.

## Responses API features

The features below are specific to the Responses API and only available on [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] (the default). For background on how the Responses API differs from Chat Completions, see the [OpenAI API docs](https://platform.openai.com/docs/guides/migrate-to-responses).

### Reasoning mode

The GPT-5.6 and GPT-6 families can use OpenAI's [`standard` and `pro` reasoning modes](https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode). `standard` is the default; `pro` performs more model work to improve reliability on difficult tasks, at the cost of higher latency and token usage. The mode is independent of the reasoning effort: any combination of mode and effort is valid, and the unified [`thinking`](../capabilities/thinking.md) setting only ever influences the effort, so `pro` is used only when you set it explicitly.

Configure the mode with [`openai_reasoning_mode`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_mode]; there is no separate `pro` model to select:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')
settings = OpenAIResponsesModelSettings(openai_reasoning_mode='pro')
agent = Agent(model, model_settings=settings)
...
```

The setting is ignored on models that don't support reasoning mode, per [`OpenAIModelProfile.openai_responses_supports_reasoning_mode`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_responses_supports_reasoning_mode].

### Reasoning context

Reasoning models can use OpenAI's [reasoning context](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-across-calls) to control which prior-turn reasoning items are available to the model when sampling. `auto` defers to the model's own default (OpenAI treats it exactly like not sending the field), `current_turn` makes only the active turn's reasoning available, and `all_turns` renders compatible reasoning items from earlier turns into the next sample. `all_turns` requires access to earlier response items via [`previous_response_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_previous_response_id], a conversation, or replayed history; on a first request it behaves like `current_turn`.

Pydantic AI sends `all_turns` by default on models that support it, so that earlier-turn reasoning stays available without opting in — consistent with how prior thinking is sent back to other models. This renders earlier reasoning into each follow-up sample, which costs additional input tokens; set `auto` explicitly to defer to OpenAI's own per-model default, or `current_turn` to keep earlier turns out of the sample.

Configure the context with [`openai_reasoning_context`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_context]:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')
settings = OpenAIResponsesModelSettings(openai_reasoning_context='all_turns')
agent = Agent(model, model_settings=settings)
...
```

`auto` and `current_turn` are sent to any model that supports reasoning. `all_turns` is sent only to models whose profile sets [`OpenAIModelProfile.openai_responses_supports_reasoning_context`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_responses_supports_reasoning_context] (currently the GPT-5.4, GPT-5.5, GPT-5.6, and GPT-6 families); on other models it is ignored.

### Native tools

The Responses API has native tools that you can use instead of building your own:

- [Web search](https://platform.openai.com/docs/guides/tools-web-search): allow models to search the web for the latest information before generating a response.
- [Code interpreter](https://platform.openai.com/docs/guides/tools-code-interpreter): allow models to write and run Python code in a sandboxed environment before generating a response.
- [Image generation](https://platform.openai.com/docs/guides/tools-image-generation): allow models to generate images based on a text prompt.
- [File search](https://platform.openai.com/docs/guides/tools-file-search): allow models to search your files for relevant information before generating a response.
- [Computer use](https://platform.openai.com/docs/guides/tools-computer-use): allow models to use a computer to perform tasks on your behalf.

Web search, Code interpreter, Image generation, and File search are natively supported through the [Native tools](../native-tools.md) feature.

Computer use can be enabled by passing an [`openai.types.responses.ComputerToolParam`](https://github.com/openai/openai-python/blob/main/src/openai/types/responses/computer_tool_param.py) in the `openai_native_tools` setting on [`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings]. It doesn't currently generate [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] or [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart] parts in the message history, or streamed events; please submit an issue if you need native support for this native tool.

```python {title="computer_use_tool.py" test="skip"}
from openai.types.responses import ComputerToolParam

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model_settings = OpenAIResponsesModelSettings(
    openai_native_tools=[
        ComputerToolParam(
            type='computer',
        )
    ],
)
model = OpenAIResponsesModel('gpt-5.2')
agent = Agent(model=model, model_settings=model_settings)

result = agent.run_sync('Open a new browser tab')
print(result.output)
```

#### Referencing earlier responses

The Responses API supports referencing earlier model responses in a new request using a `previous_response_id` parameter, to ensure the full [conversation state](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses#passing-context-from-the-previous-response) including [reasoning items](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context) is kept in context without having to resend it. This is available through the [`openai_previous_response_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_previous_response_id] field in
[`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings].

When the field is set to `'auto'`, Pydantic AI automatically selects the most recent `provider_response_id` from the message history and omits messages that came before it, letting the OpenAI API reconstruct them from server-side state. The same chaining is applied inside a run across tool-call continuations and retries, so OpenAI never sees duplicate copies of the same messages.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
agent = Agent(model=model)

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

model_settings = OpenAIResponsesModelSettings(openai_previous_response_id='auto')
result2 = agent.run_sync(
    'Explain?',
    message_history=result1.new_messages(),
    model_settings=model_settings
)
print(result2.output)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.
```

As an alternative to passing `message_history`, you can pass a concrete `provider_response_id` from an earlier run as the seed. Pydantic AI uses the seed for the first request in the new run, then automatically chains to the response returned for that request on any subsequent in-run calls — so the chain still extends correctly if the run includes tool-call continuations or retries.

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
agent = Agent(model=model)

result = agent.run_sync('The secret is 1234')
response_id = result.response.provider_response_id
assert response_id is not None
model_settings = OpenAIResponsesModelSettings(openai_previous_response_id=response_id)
result = agent.run_sync('What is the secret code?', model_settings=model_settings)
print(result.output)
#> 1234
```

!!! note
    Referencing a stored response requires the response to have actually been stored. OpenAI stores responses by default; if you've disabled storage via [`openai_store=False`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_store] or your organization has Zero Data Retention enabled, chaining is unavailable and the full message history must be sent on every request.

#### Using durable conversations

OpenAI's [Conversations API](https://platform.openai.com/docs/guides/conversation-state?api-mode=responses#using-the-conversations-api) works with the Responses API to persist conversation state in a durable conversation object. If you already have an OpenAI conversation ID, pass it with [`openai_conversation_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_conversation_id]:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
agent = Agent(model=model)

model_settings = OpenAIResponsesModelSettings(openai_conversation_id='conv_...')
result = agent.run_sync('What did we discuss last time?', model_settings=model_settings)
print(result.output)
```

When a response belongs to a conversation, Pydantic AI stores the returned ID in `ModelResponse.provider_details['conversation_id']`. Setting `openai_conversation_id='auto'` uses the most recent same-provider conversation ID from the message history and sends only the new input items after that response.

When message-level [`conversation_id`][pydantic_ai.messages.ModelResponse.conversation_id] values are available, `auto` only reuses an OpenAI conversation from the current Pydantic AI conversation; pass a concrete OpenAI conversation ID to reuse one explicitly:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
agent = Agent(model=model)

model_settings = OpenAIResponsesModelSettings(openai_conversation_id='conv_...')
result = agent.run_sync('What did we discuss last time?', model_settings=model_settings)

follow_up_settings = OpenAIResponsesModelSettings(openai_conversation_id='auto')
result2 = agent.run_sync(
    'Summarize the next step.',
    message_history=result.new_messages(),
    model_settings=follow_up_settings,
)
print(result2.output)
```

Pydantic AI does not create OpenAI conversations for you. Use the OpenAI client to create the conversation, then pass its ID to `openai_conversation_id`. The `conversation` and `previous_response_id` parameters are mutually exclusive in the OpenAI API, so `openai_conversation_id` cannot be combined with [`openai_previous_response_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_previous_response_id].

#### Message Compaction

The Responses API supports [compacting message history](https://developers.openai.com/api/docs/guides/compaction) to reduce token usage in long conversations. Compaction produces an encrypted summary that replaces older messages while preserving context.

The easiest way to enable compaction is with the [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] capability:

```python {title="openai_compaction.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAICompaction

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[OpenAICompaction()],
)
```

By default, `OpenAICompaction` runs in **stateful mode**: it configures OpenAI's server-side auto-compaction via the `context_management` field on the regular `/responses` request, and OpenAI triggers compaction whenever the input token count crosses a threshold it manages for you. This mode is compatible with [`openai_previous_response_id='auto'`](#referencing-earlier-responses) and [`openai_conversation_id`](#using-durable-conversations).

After compaction, subsequent requests send only the compacted window, from the latest compaction item onward. The Responses API processes and bills replayed items that precede a compaction item, so omitting them keeps the compacted context from growing again.

To override the threshold, pass [`token_threshold`][pydantic_ai.models.openai.OpenAICompaction]:

```python {title="openai_compaction_token_threshold.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAICompaction

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[OpenAICompaction(token_threshold=100_000)],
)
```

As an alternative, `OpenAICompaction` supports a **stateless mode** (`stateless=True`) that calls the stateless `/responses/compact` endpoint via a `before_model_request` hook. Use this in [ZDR](https://openai.com/enterprise-privacy/) environments where OpenAI must not retain conversation data, when using [`openai_store=False`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_store], or when you need explicit out-of-band control over when compaction runs. Stateless mode requires you to specify either a [`message_count_threshold`][pydantic_ai.models.openai.OpenAICompaction] or a custom `trigger` callable:

```python {title="openai_compaction_stateless.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAICompaction

agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[OpenAICompaction(message_count_threshold=20)],
)
```

The mode is inferred from which parameters you pass: supplying `message_count_threshold` or `trigger` implies stateless mode, otherwise stateful mode is used. You can also pass `stateless=True` or `stateless=False` explicitly. Mixing parameters from different modes raises [`UserError`][pydantic_ai.exceptions.UserError].

!!! tip
    Stateful compaction pairs especially well with [`openai_previous_response_id='auto'`](#referencing-earlier-responses) or [`openai_conversation_id`](#using-durable-conversations). Both rely on OpenAI's server-side conversation state, so OpenAI can use a previously compacted context as the starting point for the next turn without you having to resend it.

For lower-level use cases, you can call [`compact_messages`][pydantic_ai.models.openai.OpenAIResponsesModel.compact_messages] directly on the model.

### Text phases

Models that support it label each assistant message with a `phase`: `commentary` for the preamble the model writes while it works, and `final_answer` for the answer itself. Pydantic AI surfaces it as `'phase'` in [`TextPart.provider_details`][pydantic_ai.messages.TextPart.provider_details], and on models known to accept the field it also sends it back on the next request so the model keeps the distinction across turns.

When [streaming](../agent.md#streaming-all-events), the phase is set on the [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] that opens each text part (including its first content chunk), so you can route commentary and the final answer differently as they're generated. Prefer [`run_stream_events`][pydantic_ai.agent.AbstractAgent.run_stream_events] for this: [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] treats the first text part as the final output, which is often `commentary` on models that emit a preamble.

```python {title="openai_phase.py"}
from pydantic_ai import Agent, PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta

agent = Agent('openai:gpt-5.5')


async def main():
    final_answer_indexes: set[int] = set()
    async with agent.run_stream_events('What is the capital of France?') as events:
        async for event in events:
            if isinstance(event, PartStartEvent):
                # Indexes are scoped to a single model response and start over on the
                # next one, so a new part at an index supersedes what was there before.
                final_answer_indexes.discard(event.index)
                if isinstance(event.part, TextPart):
                    phase = (event.part.provider_details or {}).get('phase')
                    if phase == 'final_answer':
                        final_answer_indexes.add(event.index)
                        print(event.part.content)
            elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                if event.index in final_answer_indexes:
                    print(event.delta.content_delta)
```

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

A `'phase'` key appears in `provider_details` whenever the model labels its output, but it is only sent back on models that [`OpenAIModelProfile.openai_supports_phase`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_supports_phase] marks as accepting it. On every other model the label is surfaced to you and dropped from follow-up requests.

### Background mode

For long-running requests, such as large reasoning or tool-heavy jobs that may exceed the practical duration of a synchronous request, OpenAI's Responses API offers a [background mode](https://platform.openai.com/docs/guides/background) that runs the request server-side and lets you retrieve the result once it's ready. Enable it with [`openai_background`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_background]:

```python {title="openai_background.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.2')
settings = OpenAIResponsesModelSettings(openai_background=True)
agent = Agent(model, model_settings=settings)
...
```

When the response comes back still pending (`'queued'` or `'in_progress'`), Pydantic AI continues it to completion transparently, so you don't need to do anything. This works for both [`agent.run`][pydantic_ai.agent.AbstractAgent.run] and [`agent.run_stream`][pydantic_ai.agent.AbstractAgent.run_stream], and the result is stitched into a single [`ModelResponse`][pydantic_ai.messages.ModelResponse] — when streaming, live token activity is surfaced as it's generated and arrives as one continuous stream.

Because the request is queued server-side, the time to the first token is higher than for a synchronous request. While a background response is still pending, Pydantic AI polls for completion at a fixed interval.

!!! note
    If a run is suspended mid-request (its final [`ModelResponse.state`][pydantic_ai.messages.ModelResponse.state] is `'suspended'`) and persisted in message history, passing that history back resumes the same background response rather than starting a new one. Resuming after the provider's retention window raises [`SuspendedResponseExpired`][pydantic_ai.exceptions.SuspendedResponseExpired]. Abandoning or cancelling the run cancels the server-side background job.

## Chat Completions API

If you need the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat) instead of the default [Responses API](https://platform.openai.com/docs/api-reference/responses), pin to it with the `'openai-chat:'` prefix or [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel]:

```python
from pydantic_ai import Agent

agent = Agent('openai-chat:gpt-5.2')
...
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

model = OpenAIChatModel('gpt-5.2')
agent = Agent(model)
...
```

Five [`ModelSettings`][pydantic_ai.settings.ModelSettings] fields reach OpenAI only through this API — `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias` and `stop_sequences`. The Responses API accepts none of them, so they are dropped on the default `openai:` path.

For other services using the same API format, see the [provider directory](overview.md#provider-directory) and [compatible API configuration](compatible-apis.md).

## OpenAI-compatible Models

Find services in the [provider directory](overview.md#provider-directory), or use [Other compatible APIs](compatible-apis.md) to configure a custom endpoint. The provider guides and configuration topics previously on this page are now available here:

- <span id="model-profile"></span>[Model Profile](compatible-apis.md#model-profile)
- <span id="custom-openai-compatible-provider"></span>[Custom providers for gateways](compatible-apis.md#custom-openai-compatible-provider)
- <span id="detect-incomplete-streamed-responses"></span>[Detect incomplete streamed responses](compatible-apis.md#detect-incomplete-streamed-responses)
- <span id="models-that-accept-only-one-leading-system-message"></span>[Models that accept only one leading system message](compatible-apis.md#models-that-accept-only-one-leading-system-message)
- <span id="deepseek"></span>[DeepSeek](deepseek.md)
- <span id="alibaba-cloud-model-studio-dashscope"></span>[Alibaba Cloud Model Studio (DashScope)](compatible-apis.md#alibaba-cloud-model-studio-dashscope)
- <span id="ollama"></span>[Ollama](ollama.md)
- <span id="azure-ai-foundry"></span>[Azure AI Foundry](azure.md)
- <span id="connecting-to-an-existing-api-version-based-deployment"></span>[Connecting to an existing `api-version`-based deployment](azure.md#connecting-to-an-existing-api-version-based-deployment)
- <span id="using-azure-with-the-responses-api"></span>[Using Azure with the Responses API](azure.md#using-azure-with-the-responses-api)
- <span id="vercel-ai-gateway"></span>[Vercel AI Gateway](compatible-apis.md#vercel-ai-gateway)
- <span id="moonshotai"></span>[MoonshotAI](moonshotai.md)
- <span id="github-models"></span>[GitHub Models](compatible-apis.md#github-models)
- <span id="perplexity"></span>[Perplexity](compatible-apis.md#perplexity)
- <span id="fireworks-ai"></span>[Fireworks AI](compatible-apis.md#fireworks-ai)
- <span id="together-ai"></span>[Together AI](compatible-apis.md#together-ai)
- <span id="heroku-ai"></span>[Heroku AI](compatible-apis.md#heroku-ai)
- <span id="litellm"></span>[LiteLLM](compatible-apis.md#litellm)
- <span id="vllm"></span>[vLLM](compatible-apis.md#vllm)
- <span id="nebius-ai-studio"></span>[Nebius AI Studio](compatible-apis.md#nebius-ai-studio)
- <span id="ovhcloud-ai-endpoints"></span>[OVHcloud AI Endpoints](compatible-apis.md#ovhcloud-ai-endpoints)
- <span id="sambanova"></span>[SambaNova](compatible-apis.md#sambanova)
- <span id="atlas-cloud"></span>[Atlas Cloud](compatible-apis.md#atlas-cloud)
- <span id="rapid-mlx-apple-silicon"></span>[Rapid-MLX (Apple Silicon)](compatible-apis.md#rapid-mlx-apple-silicon)
