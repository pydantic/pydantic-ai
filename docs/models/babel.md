# Babel

The models in `pydantic_ai.models.babel` are drop-in variants of the [OpenAI](openai.md), [Anthropic](anthropic.md), [Google](google.md) and [Bedrock](bedrock.md) models whose wire mapping is done by [babel](https://github.com/pydantic/babel).

Babel authors each provider's request, response and streaming translation once, as a small pure transform that is compiled to Python, Rust and TypeScript and verified byte-for-byte against a shared corpus of real provider traffic. A babel model keeps everything else from its parent: the provider and its authentication, the HTTP client, model settings, model profiles, native tools and the agent loop. Only the translation between the message history and the provider's wire format changes.

Use a babel model when you want the same wire mapping in Pydantic AI that a babel-backed gateway or TypeScript or Rust service uses, so a request looks the same on the wire whichever of them sends it.

## Install

To use the babel models, install `pydantic-ai-slim` with the `babel` optional group alongside the provider's own group:

```bash
pip/uv-add "pydantic-ai-slim[babel,openai]"
```

Each model lives in its own module and imports only its provider's SDK, so installing one provider's group is enough for that model.

## Usage

Construct a babel model exactly like the model it replaces:

```python
from pydantic_ai import Agent
from pydantic_ai.models.babel.openai import BabelOpenAIChatModel

model = BabelOpenAIChatModel('gpt-5.2')
agent = Agent(model)
...
```

The same applies to [`BabelAnthropicModel`][pydantic_ai.models.babel.anthropic.BabelAnthropicModel], [`BabelGoogleModel`][pydantic_ai.models.babel.google.BabelGoogleModel] and [`BabelBedrockConverseModel`][pydantic_ai.models.babel.bedrock.BabelBedrockConverseModel], including the `provider` and `settings` arguments their parents take:

```python
from pydantic_ai import Agent
from pydantic_ai.models.babel.anthropic import BabelAnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = BabelAnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='your-api-key'))
agent = Agent(model)
...
```

## What stays the same

- Provider-specific model settings, such as `anthropic_cache_instructions`, `bedrock_cache_messages` or `openai_continuous_usage_stats`, are applied as they are by the parent model.
- Model profile facts about the wire are handed to babel as capability data and applied the way the parent applies them: the OpenAI system-prompt role and single-system-message merge, and on Bedrock the tool-result block, the `status` field, a leading assistant turn and whether thinking parts are replayed.
- Prompt-cache breakpoints (`CachePoint`) and the Anthropic 4-breakpoint limit behave as in [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel].
- A `ThinkingPart` replays its signature only to the provider that produced it, so a history that moves between providers never sends one provider's signature to another.
- A failed tool return takes the provider's error channel (Anthropic `is_error`, Bedrock `status`, Gemini `{"error": ...}`), or is wrapped as `{"error": ...}` for Chat Completions, as the parent models send it. Files a `CodeExecutionTool` uploads reach the Anthropic container as they do from the parent.
- `provider_details`, `finish_reason` and the response `state` are read from the raw response the way the parent model reads them: the raw finish reason, and per provider the refusal, safety ratings, blocked-prompt feedback, logprobs, moderation, service tier, guardrail trace, container id and input transformations. Inline files a model returns become `FilePart`s.
- Model profile flags the parents apply to a stream, such as `ignore_streamed_leading_whitespace` and `openai_chat_streaming_requires_finish_reason`, apply the same way.
- Media URLs a provider cannot fetch itself (audio and documents for OpenAI, everything for Google, everything but `s3://` objects for Bedrock, and any `FileUrl` with `force_download` set) are downloaded and inlined, with the same SSRF protection as the parent models. Media a provider cannot take at all raises `NotImplementedError`, as it does from the parent.
- A `SystemPromptPart` placed after the opening ones in a hand-built history is delivered in place as `<system>`-tagged user text, the way Pydantic AI delivers it to any model whose wire has no inline system role, since babel carries one request-level system prompt.

## Limitations

- Server-side (native) tool calls and results are replayed only to the provider that produced them; the babel models do not yet map grounding sources or file outputs.
- Deferred tools, tool search and capability-loading parts are not supported and raise a `UserError`.
- Gemini's per-file `media_resolution` and `video_metadata` from a file's `vendor_metadata` are not sent; only OpenAI's image `detail` is carried.
- Thinking parts are not replayed to Chat Completions, whatever `openai_chat_send_back_thinking_parts` says, as babel's `openai-chat` codec has no reasoning field yet.
- A `CachePoint` in a user prompt is not carried to Bedrock (the `bedrock_cache_instructions` and `bedrock_cache_messages` settings are), as babel's Converse codec has no cache breakpoint yet.

## Building your own

The boundary the babel models use is public: [`messages_to_ir`][pydantic_ai.models.babel.messages_to_ir] renders a message history as babel's IR, [`reconcile_ir`][pydantic_ai.models.babel.reconcile_ir] fits it to a target's capability facts, [`ir_to_model_response`][pydantic_ai.models.babel.ir_to_model_response] reads a decoded IR response back, and [`fold_stream_emits`][pydantic_ai.models.babel.fold_stream_emits] routes streaming emits into a response. A babel model for another provider babel supports is a small subclass that overrides the parent model's mapping methods with those calls.
