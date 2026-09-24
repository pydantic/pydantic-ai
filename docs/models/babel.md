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

- Provider-specific model settings, such as `anthropic_cache_instructions` or `openai_continuous_usage_stats`, are applied as they are by the parent model.
- Prompt-cache breakpoints (`CachePoint`) and the Anthropic 4-breakpoint limit behave as in [`AnthropicModel`][pydantic_ai.models.anthropic.AnthropicModel].
- A `ThinkingPart` replays its signature only to the provider that produced it, so a history that moves between providers never sends one provider's signature to another.
- Media URLs a provider cannot fetch itself (audio and documents for OpenAI, everything for Google and Bedrock, and any `FileUrl` with `force_download` set) are downloaded and inlined, with the same SSRF protection as the parent models.

## Limitations

- Server-side (native) tool calls and results are replayed only to the provider that produced them; the babel models do not yet map grounding sources or file outputs.
- Deferred tools, tool search and capability-loading parts are not supported and raise a `UserError`.
- Response metadata beyond the model name, response id and finish reason is not carried into `provider_details`, except for OpenAI's raw finish reason and timestamp.

## Building your own

The boundary the babel models use is public: [`messages_to_ir`][pydantic_ai.models.babel.messages_to_ir] renders a message history as babel's IR, [`ir_to_model_response`][pydantic_ai.models.babel.ir_to_model_response] reads a decoded IR response back, and [`fold_stream_emits`][pydantic_ai.models.babel.fold_stream_emits] routes streaming emits into a response. A babel model for another provider babel supports is a small subclass that overrides the parent model's mapping methods with those three calls.
