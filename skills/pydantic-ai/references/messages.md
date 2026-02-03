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

agent = Agent('openai:gpt-4o')
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
