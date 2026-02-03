# Messages Reference

Source: `pydantic_ai_slim/pydantic_ai/messages.py`

## Message Structure

Conversations are built from `ModelRequest` and `ModelResponse` messages.

```
ModelRequest  →  contains request parts (system prompt, user prompt, tool returns)
ModelResponse →  contains response parts (text, tool calls, thinking)
```

### ModelRequest Parts

| Part | Description |
|------|-------------|
| `SystemPromptPart` | System/instruction prompt |
| `UserPromptPart` | User message (text or multimedia) |
| `ToolReturnPart` | Return value from a tool call |
| `RetryPromptPart` | Retry message sent after validation failure |

### ModelResponse Parts

| Part | Description |
|------|-------------|
| `TextPart` | Plain text response |
| `ToolCallPart` | Tool call with name, args, and ID |
| `ThinkingPart` | Model reasoning/thinking content |

## Accessing Messages

```python {title="messages_access.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')
result = agent.run_sync('What is the capital of France?')

# All messages in the conversation
all_msgs = result.all_messages()
print(len(all_msgs))
#> 2

# Only messages from this run
new_msgs = result.new_messages()
print(len(new_msgs))
#> 2
```

## Message History — Multi-Turn Conversations

Pass previous messages to continue a conversation:

```python {title="conversation_example.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.output)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),
)
print(result2.output)
#> Albert Einstein's most famous equation is (E = mc^2).
```

## Multimedia Content Types

### Images

```python
from pydantic_ai import ImageUrl, BinaryImage

# From URL
image = ImageUrl(url='https://example.com/image.png')

# From binary data
image = BinaryImage(data=b'...', media_type='image/png')
```

### Audio

```python
from pydantic_ai import AudioUrl

audio = AudioUrl(url='https://example.com/audio.mp3')
```

### Video

```python
from pydantic_ai import VideoUrl

video = VideoUrl(url='https://example.com/video.mp4')
```

### Documents

```python
from pydantic_ai import DocumentUrl

doc = DocumentUrl(url='https://example.com/doc.pdf')
```

### Using Multimedia in Prompts

```python
from pydantic_ai import Agent, ImageUrl

agent = Agent('openai:gpt-5')
result = await agent.run([
    'What is in this image?',
    ImageUrl(url='https://example.com/photo.jpg'),
])
```

## ModelMessagesTypeAdapter

For serializing/deserializing message history:

```python
from pydantic_ai import ModelMessagesTypeAdapter

# Serialize to JSON
json_bytes = ModelMessagesTypeAdapter.dump_json(messages)

# Deserialize from JSON
messages = ModelMessagesTypeAdapter.validate_json(json_bytes)
```

## UserContent Type

Content that can appear in a user prompt:

```python
UserContent = str | ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent | FileUrl | FilePart
```

## Processing Message History

The `history_processors` parameter on `Agent` intercepts and modifies message history before each model request. This is essential for:
- Managing token usage in long conversations
- Filtering sensitive information
- Summarizing old messages to preserve context

### Basic Usage

```python {title="simple_history_processor.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove all ModelResponse messages, keeping only ModelRequest messages."""
    return [msg for msg in messages if isinstance(msg, ModelRequest)]

# Create agent with history processor
agent = Agent('openai:gpt-5', history_processors=[filter_responses])

# Example: Create some conversation history
message_history = [
    ModelRequest(parts=[UserPromptPart(content='What is 2+2?')]),
    ModelResponse(parts=[TextPart(content='2+2 equals 4')]),  # This will be filtered out
]

# When you run the agent, the history processor will filter out ModelResponse messages
# result = agent.run_sync('What about 3+3?', message_history=message_history)
```

### Keep Only Recent Messages

```python {title="keep_recent_messages.py"}
from pydantic_ai import Agent, ModelMessage


async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only the last 5 messages to manage token usage."""
    return messages[-5:] if len(messages) > 5 else messages

agent = Agent('openai:gpt-5', history_processors=[keep_recent_messages])

# Example: Even with a long conversation history, only the last 5 messages are sent to the model
long_conversation_history: list[ModelMessage] = []  # Your long conversation history here
# result = agent.run_sync('What did we discuss?', message_history=long_conversation_history)
```

**Warning:** When slicing, ensure tool calls and returns remain paired or the LLM may error.

### Context-Aware Processor with RunContext

```python {title="context_aware_processor.py"}
from pydantic_ai import Agent, ModelMessage, RunContext


def context_aware_processor(
    ctx: RunContext[None],
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    # Access current usage
    current_tokens = ctx.usage.total_tokens

    # Filter messages based on context
    if current_tokens > 1000:
        return messages[-3:]  # Keep only recent messages when token usage is high
    return messages

agent = Agent('openai:gpt-5', history_processors=[context_aware_processor])
```

### Summarize Old Messages

Use a secondary agent to summarize older messages:

```python {title="summarize_old_messages.py"}
from pydantic_ai import Agent, ModelMessage

# Use a cheaper model to summarize old messages.
summarize_agent = Agent(
    'openai:gpt-5-mini',
    instructions="""
Summarize this conversation, omitting small talk and unrelated topics.
Focus on the technical discussion and next steps.
""",
)


async def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Summarize the oldest 10 messages
    if len(messages) > 10:
        oldest_messages = messages[:10]
        summary = await summarize_agent.run(message_history=oldest_messages)
        # Return the last message and the summary
        return summary.new_messages() + messages[-1:]

    return messages


agent = Agent('openai:gpt-5', history_processors=[summarize_old_messages])
```

### Testing History Processors with FunctionModel

```python {title="test_history_processor.py"}
import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.fixture
def received_messages() -> list[ModelMessage]:
    return []


@pytest.fixture
def function_model(received_messages: list[ModelMessage]) -> FunctionModel:
    def capture_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Capture the messages that the provider actually receives
        received_messages.clear()
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Provider response')])

    return FunctionModel(capture_model_function)


def test_history_processor(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, history_processors=[filter_responses])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
    ]

    agent.run_sync('Question 2', message_history=message_history)
    assert received_messages == [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelRequest(parts=[UserPromptPart(content='Question 2')]),
    ]
```

### Multiple Processors

Processors are applied in sequence:

```python {title="multiple_history_processors.py"}
from pydantic_ai import Agent, ModelMessage, ModelRequest


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    return [msg for msg in messages if isinstance(msg, ModelRequest)]


def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-5:]


agent = Agent('openai:gpt-5', history_processors=[filter_responses, summarize_old_messages])
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `ModelMessage` | `pydantic_ai.ModelMessage` | Union of request/response |
| `ModelRequest` | `pydantic_ai.ModelRequest` | Request message |
| `ModelResponse` | `pydantic_ai.ModelResponse` | Response message |
| `TextPart` | `pydantic_ai.TextPart` | Text response part |
| `ToolCallPart` | `pydantic_ai.ToolCallPart` | Tool call part |
| `ToolReturnPart` | `pydantic_ai.ToolReturnPart` | Tool return part |
| `ThinkingPart` | `pydantic_ai.ThinkingPart` | Thinking/reasoning part |
| `SystemPromptPart` | `pydantic_ai.SystemPromptPart` | System prompt part |
| `UserPromptPart` | `pydantic_ai.UserPromptPart` | User prompt part |
| `RetryPromptPart` | `pydantic_ai.RetryPromptPart` | Retry prompt part |
| `ImageUrl` | `pydantic_ai.ImageUrl` | Image from URL |
| `BinaryImage` | `pydantic_ai.BinaryImage` | Image from bytes |
| `AudioUrl` | `pydantic_ai.AudioUrl` | Audio from URL |
| `VideoUrl` | `pydantic_ai.VideoUrl` | Video from URL |
| `DocumentUrl` | `pydantic_ai.DocumentUrl` | Document from URL |
| `ModelMessagesTypeAdapter` | `pydantic_ai.ModelMessagesTypeAdapter` | JSON serialization |
