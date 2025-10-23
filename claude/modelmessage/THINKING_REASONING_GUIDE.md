# Pydantic AI: Thinking/Reasoning Features Guide

## Overview

Thinking (also called reasoning) is the process by which a model works through a problem step-by-step before providing its final answer. This capability is typically **disabled by default** and depends on the specific LLM provider being used.

## Message Structure

Thinking parts are represented as `ThinkingPart` objects in the message system. The message hierarchy is:

```
ModelMessage (Union of ModelRequest | ModelResponse)
├── ModelResponse
│   └── ModelResponsePart (Union of text/tool/thinking)
│       ├── TextPart
│       ├── ToolCallPart
│       └── ThinkingPart  <-- Thinking response
└── ModelRequest
    └── ModelRequestPart
```

### ThinkingPart Class

```python
@dataclass
class ThinkingPart:
    """A thinking response from a model."""
    content: str
    """The thinking content of the response."""
    
    id: str | None = None
    """The identifier of the thinking part."""
    
    signature: str | None = None
    """The signature of the thinking.
    
    Supported by:
    * Anthropic (corresponds to the `signature` field)
    * Bedrock (corresponds to the `signature` field)
    * Google (corresponds to the `thought_signature` field)
    * OpenAI (corresponds to the `encrypted_content` field)
    """
```

### ThinkingPartDelta Class

For streaming responses:

```python
@dataclass
class ThinkingPartDelta:
    """A partial update (delta) for a `ThinkingPart` to append new thinking content."""
    
    content_delta: str | None = None
    """The incremental thinking content to add to the existing `ThinkingPart` content."""
    
    signature_delta: str | None = None
    """Optional signature delta. Note this is never treated as a delta — it can replace None."""
    
    provider_name: str | None = None
    """Optional provider name for the thinking part. Signatures are only sent back to the same provider."""
    
    part_delta_kind: Literal['thinking'] = 'thinking'
    """Part delta type identifier, used as a discriminator."""
```

## Provider-Specific Implementation

### OpenAI

#### OpenAIChatModel (Chat Completions)

The `OpenAIChatModel` automatically parses text output inside `<think>` tags and converts them to `ThinkingPart` objects.

**Customization:**
- Use the `thinking_tags` field on the model profile to customize the tags (default: `<think>...</think>`)
- See [models/openai.md](#model-profile) for more details

#### OpenAIResponsesModel (Responses API)

The `OpenAIResponsesModel` can generate **native thinking parts** with dedicated support for reasoning.

**Enable with:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',          # or 'medium', 'high'
    openai_reasoning_summary='detailed',    # or 'concise'
)
agent = Agent(model, model_settings=settings)
```

**Settings:**

- `openai_reasoning_effort`: Controls the reasoning budget ('low', 'medium', 'high')
- `openai_reasoning_summary`: Controls summary detail level ('detailed' or 'concise')
- `openai_send_reasoning_ids`: Whether to send unique IDs of reasoning/text/function call parts from message history (enabled by default)
  - **Warning:** May cause errors if your message history doesn't exactly match what was received from the API
  - Disable if using [history processors](message-history.md#processing-message-history)

**Important:** For Responses API, the message history IDs matter. If you're manipulating message history (e.g., with a history processor), you may need to disable `openai_send_reasoning_ids` to avoid errors like:
```
"Item 'rs_123' of type 'reasoning' was provided without its required following item."
```

### Anthropic

**Enable with:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-sonnet-4-0')
settings = AnthropicModelSettings(
    anthropic_thinking={
        'type': 'enabled',
        'budget_tokens': 1024  # Maximum tokens for thinking
    }
)
agent = Agent(model, model_settings=settings)
```

**Key features:**
- Thinking is explicitly enabled with a token budget
- Returns `ThinkingPart` objects with optional `signature` field
- See [Anthropic's extended thinking docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)

### Google (Gemini)

**Enable with:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-2.5-pro')
settings = GoogleModelSettings(
    google_thinking_config={'include_thoughts': True}
)
agent = Agent(model, model_settings=settings)
```

**Key features:**
- Thinking configuration supports the `include_thoughts` flag
- Returns `ThinkingPart` objects with optional `thought_signature` field
- See [Google's thinking documentation](https://ai.google.dev/gemini-api/docs/thinking)

### Groq

Groq supports different formats for receiving thinking parts:

**Formats:**
- `"raw"`: Thinking part included in text content inside `<think>` tags (auto-converted to `ThinkingPart`)
- `"hidden"`: Thinking part not included in text content
- `"parsed"`: Thinking part as its own structured part (converted to `ThinkingPart`)

**Enable with:**

```python
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('qwen-qwq-32b')
settings = GroqModelSettings(groq_reasoning_format='parsed')
agent = Agent(model, model_settings=settings)
```

### Mistral

Thinking is supported by the `magistral` family of models and **does not need to be explicitly enabled**. It works automatically.

### Cohere

Thinking is supported by the `command-a-reasoning-08-2025` model and **does not need to be explicitly enabled**. It works automatically.

### Hugging Face

Text output inside `<think>` tags is automatically converted to `ThinkingPart` objects.

**Customization:**
- Use the `thinking_tags` field on the model profile to customize the tags (default: `<think>...</think>`)

### Bedrock

(Documentation indicates support but specific configuration details are not yet documented)

## Using Model-Specific Settings

Model-specific settings follow the same precedence hierarchy as general `ModelSettings`:

1. **Model-level defaults** - Set when creating a model instance
2. **Agent-level defaults** - Set during Agent initialization  
3. **Run-time overrides** - Passed to `run()`/`run_sync()`/`run_stream()` functions (highest priority)

### Example: Run-time Override

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent('anthropic:claude-sonnet-4-0')

result = agent.run_sync(
    'Analyze this complex problem',
    model_settings=AnthropicModelSettings(
        anthropic_thinking={
            'type': 'enabled',
            'budget_tokens': 2048
        }
    )
)
```

## Streaming Thinking

When using streaming with `run_stream()` or `run_stream_events()`, thinking content is streamed as `ThinkingPartDelta` events:

```python
import asyncio
from pydantic_ai import Agent, PartDeltaEvent, ThinkingPartDelta

agent = Agent('anthropic:claude-sonnet-4-0')

async def handle_event(event):
    if isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, ThinkingPartDelta):
            print(f'Thinking delta: {event.delta.content_delta}')

async def main():
    async with agent.run_stream(
        'Complex problem',
        event_stream_handler=handle_event
    ) as response:
        # Process streamed response
        pass

asyncio.run(main())
```

### Stream Events Example

From the agents documentation:

```python
async def handle_event(event: AgentStreamEvent):
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
```

## API Reference

### Model Settings Classes

- **`ModelSettings`** - Base settings with common parameters (temperature, max_tokens, timeout)
- **`OpenAIResponsesModelSettings`** - OpenAI Responses model specific settings
  - `openai_reasoning_effort`: Literal['low', 'medium', 'high']
  - `openai_reasoning_summary`: Literal['detailed', 'concise']
  - `openai_send_reasoning_ids`: bool
- **`AnthropicModelSettings`** - Anthropic model specific settings
  - `anthropic_thinking`: BetaThinkingConfigParam
- **`GoogleModelSettings`** - Google Gemini model specific settings
  - `google_thinking_config`: ThinkingConfigDict
- **`GroqModelSettings`** - Groq model specific settings
  - `groq_reasoning_format`: Literal['hidden', 'raw', 'parsed']

### Response Parts

- **`ThinkingPart`** - Represents a thinking response with content and optional signature
- **`ThinkingPartDelta`** - Streaming delta for thinking content
- **`TextPart`** - Regular text response
- **`ToolCallPart`** - Tool invocation from the model

## Key Design Patterns

### Pattern 1: Basic Thinking (Anthropic)

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent('anthropic:claude-sonnet-4-0')

result = agent.run_sync(
    'Solve this math problem: 17 * 23 = ?',
    model_settings=AnthropicModelSettings(
        anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}
    )
)

# Access thinking from the result
for message in result.all_messages():
    for part in message.parts:
        if isinstance(part, ThinkingPart):
            print(f"Model was thinking: {part.content}")
        elif isinstance(part, TextPart):
            print(f"Model answered: {part.content}")
```

### Pattern 2: Streaming Thinking Events

```python
from pydantic_ai import Agent, AgentStreamEvent, PartDeltaEvent, ThinkingPartDelta

agent = Agent('anthropic:claude-sonnet-4-0')

async def event_handler(ctx, event_stream):
    async for event in event_stream:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            print(f"Thinking: {event.delta.content_delta}")

async def main():
    async with agent.run_stream(
        'Complex query',
        event_stream_handler=event_handler,
        model_settings=AnthropicModelSettings(
            anthropic_thinking={'type': 'enabled', 'budget_tokens': 2048}
        )
    ) as response:
        async for text in response.stream_text():
            print(f"Response: {text}")
```

### Pattern 3: OpenAI Responses with Reasoning

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5')
agent = Agent(
    model,
    model_settings=OpenAIResponsesModelSettings(
        openai_reasoning_effort='high',
        openai_reasoning_summary='detailed',
    )
)

result = agent.run_sync('Analyze this complex scenario...')
```

## Important Considerations

1. **Model Support**: Not all models support thinking/reasoning. Check provider documentation for model compatibility.

2. **Token Budget**: For Anthropic, you must specify a token budget for thinking. This controls how much the model "thinks".

3. **Message History**: When using OpenAI Responses API with reasoning:
   - IDs of reasoning/text/function call parts are sent by default
   - If you manipulate message history, you may need to disable `openai_send_reasoning_ids`
   - History processors need special consideration

4. **Cost Implications**: Reasoning/thinking typically increases API costs and latency due to the additional compute.

5. **Streaming**: Thinking content can be streamed as `ThinkingPartDelta` events, allowing real-time observation of the model's reasoning process.

6. **Signature Field**: Anthropic, Bedrock, Google, and OpenAI provide signature/encrypted content fields to cryptographically verify thinking integrity across API calls.

## See Also

- [Agents Documentation](agents.md) - General agent usage and model settings
- [Messages Documentation](api/messages.md) - Complete message structure reference
- [Models Overview](models/overview.md) - Overview of all supported models
- [OpenAI Thinking Documentation](models/openai.md)
- [Anthropic Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Google Thinking Documentation](https://ai.google.dev/gemini-api/docs/thinking)
- [Groq Reasoning Documentation](https://console.groq.com/docs/reasoning)
