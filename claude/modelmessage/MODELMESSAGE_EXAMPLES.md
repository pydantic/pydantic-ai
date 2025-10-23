# Pydantic AI ModelMessage - Code Examples & Usage Patterns

## Table of Contents
1. Basic Message Creation
2. Serialization/Deserialization
3. Working with Tool Calls
4. Multi-Modal Content
5. Streaming Events
6. Agent Integration Patterns
7. Advanced Scenarios

---

## 1. Basic Message Creation

### Creating a Simple Request
```python
from pydantic_ai import ModelRequest, SystemPromptPart, UserPromptPart

# Create request with system and user prompts
request = ModelRequest(
    parts=[
        SystemPromptPart(content="You are a helpful assistant."),
        UserPromptPart(content="What is Python?"),
    ]
)

# Or use the factory method
request = ModelRequest.user_text_prompt(
    user_prompt="What is Python?",
    instructions="Be concise"
)
```

### Creating a Response
```python
from pydantic_ai import ModelResponse, TextPart, RequestUsage, FinishReason

response = ModelResponse(
    parts=[
        TextPart(content="Python is a high-level programming language."),
    ],
    usage=RequestUsage(input_tokens=10, output_tokens=15),
    model_name="gpt-4o",
    provider_name="openai",
    finish_reason="stop"
)
```

---

## 2. Serialization/Deserialization

### Save Messages to JSON
```python
from pydantic_ai import ModelMessagesTypeAdapter, ModelMessage
import json

# Create messages
messages: list[ModelMessage] = [request, response]

# Serialize to JSON string
json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
json_str = json_bytes.decode('utf-8')

# Or save to file
with open('messages.json', 'w') as f:
    f.write(json_str)
```

### Load Messages from JSON
```python
import json

# Load from file
with open('messages.json', 'r') as f:
    json_str = f.read()

# Deserialize
messages = ModelMessagesTypeAdapter.validate_python(json.loads(json_str))

# Access parts
for msg in messages:
    if msg.kind == 'request':
        print(f"Request with {len(msg.parts)} parts")
    else:
        print(f"Response with {len(msg.parts)} parts")
```

### Convert to Python Dicts
```python
# Convert to JSON-compatible dicts
dict_data = ModelMessagesTypeAdapter.dump_python(
    messages, 
    mode='json'
)

# Pretty print
import json
print(json.dumps(dict_data, indent=2))
```

---

## 3. Working with Tool Calls

### Creating a Tool Call Response
```python
from pydantic_ai import ToolCallPart, ModelResponse

response = ModelResponse(
    parts=[
        ToolCallPart(
            tool_name="get_weather",
            args='{"city": "New York"}',  # JSON string format
            tool_call_id="call_123"
        ),
    ]
)

# Extract tool call
tool_calls = response.tool_calls
for tool_call in tool_calls:
    print(f"Tool: {tool_call.tool_name}")
    print(f"Args dict: {tool_call.args_as_dict()}")
    print(f"Args JSON: {tool_call.args_as_json_str()}")
```

### Creating a Tool Return
```python
from pydantic_ai import ToolReturnPart

# Tool return with dict content
tool_return = ToolReturnPart(
    tool_name="get_weather",
    content={"temperature": 75, "conditions": "sunny"},
    tool_call_id="call_123"
)

# Tool return with string content
tool_return = ToolReturnPart(
    tool_name="search",
    content="Found 5 results",
    tool_call_id="call_124"
)
```

### Building Complete Request After Tool Execution
```python
from pydantic_ai import ModelRequest, ToolReturnPart, UserPromptPart

# First response had a tool call
response = ModelResponse(
    parts=[
        TextPart(content="Let me check the weather"),
        ToolCallPart(tool_name="get_weather", args='{"city": "NYC"}', tool_call_id="call_1")
    ]
)

# Execute tool and create next request
tool_result = get_weather(city="NYC")

next_request = ModelRequest(
    parts=[
        UserPromptPart(content="What is the weather in NYC?"),
        ToolReturnPart(
            tool_name="get_weather",
            content=tool_result,
            tool_call_id="call_1"
        ),
    ]
)
```

### Handling Validation Errors
```python
from pydantic_ai import RetryPromptPart
from pydantic_core import ErrorDetails

# Tool validation failed
validation_errors: list[ErrorDetails] = [...]

retry_request = ModelRequest(
    parts=[
        RetryPromptPart(
            content=validation_errors,
            tool_name="get_weather",
            tool_call_id="call_123"
        ),
    ]
)

# Or with custom message
retry_request = ModelRequest(
    parts=[
        RetryPromptPart(
            content="The coordinates must be within valid ranges (-90 to 90 lat, -180 to 180 lon)",
            tool_name="search_location",
            tool_call_id="call_124"
        ),
    ]
)
```

---

## 4. Multi-Modal Content

### Working with Images
```python
from pydantic_ai import UserPromptPart, ImageUrl, BinaryContent

# Image from URL
image_url = ImageUrl(
    url="https://example.com/image.jpg",
    identifier="img_1",
    vendor_metadata={"detail": "high"}  # For OpenAI
)

request = ModelRequest(
    parts=[
        UserPromptPart(
            content=["Describe this image:", image_url]
        ),
    ]
)

# Binary image from bytes
binary_image = BinaryContent(
    data=image_bytes,
    media_type="image/jpeg",
    identifier="img_2"
)

request = ModelRequest(
    parts=[
        UserPromptPart(
            content=["Analyze this image:", binary_image]
        ),
    ]
)
```

### Working with Documents
```python
from pydantic_ai import DocumentUrl

pdf_url = DocumentUrl(
    url="https://example.com/document.pdf",
    identifier="doc_1",
    force_download=True  # Download and send as bytes
)

request = ModelRequest(
    parts=[
        UserPromptPart(
            content=[
                "Summarize this PDF:",
                pdf_url
            ]
        ),
    ]
)

# Get format info
print(pdf_url.format)  # 'pdf'
print(pdf_url.media_type)  # 'application/pdf'
```

### Working with Audio
```python
from pydantic_ai import AudioUrl

audio_url = AudioUrl(
    url="https://example.com/audio.mp3",
    identifier="audio_1"
)

request = ModelRequest(
    parts=[
        UserPromptPart(
            content=[
                "Transcribe this audio:",
                audio_url
            ]
        ),
    ]
)
```

### Working with Video
```python
from pydantic_ai import VideoUrl

# Regular video
video_url = VideoUrl(
    url="https://example.com/video.mp4",
    identifier="video_1"
)

# YouTube video (auto-detected)
youtube_url = VideoUrl(
    url="https://youtu.be/dQw4w9WgXcQ",
    identifier="youtube_1"
)

# With Google metadata
gemini_video = VideoUrl(
    url="https://example.com/video.mp4",
    identifier="video_2",
    vendor_metadata={
        "video_metadata": {
            "start_offset": {"seconds": 0},
            "end_offset": {"seconds": 30}
        }
    }
)

request = ModelRequest(
    parts=[
        UserPromptPart(
            content=["Analyze this video:", youtube_url]
        ),
    ]
)
```

### Data URI Conversion
```python
from pydantic_ai import BinaryContent

# Create from data URI
data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
binary = BinaryContent.from_data_uri(data_uri)

# Convert to data URI
data_uri_str = binary.data_uri
```

---

## 5. Streaming Events

### Processing Streaming Events
```python
from pydantic_ai import (
    PartStartEvent, PartDeltaEvent, FinalResultEvent,
    TextPartDelta, ToolCallPartDelta
)

# Simulate streaming response processing
events = [
    PartStartEvent(index=0, part=TextPart(content="")),
    PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="Hello")),
    PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=" ")),
    PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world")),
    FinalResultEvent(tool_name=None, tool_call_id=None),
]

# Reconstruct parts from events
parts = {}
for event in events:
    if isinstance(event, PartStartEvent):
        parts[event.index] = event.part
    elif isinstance(event, PartDeltaEvent):
        parts[event.index] = event.delta.apply(parts[event.index])
    elif isinstance(event, FinalResultEvent):
        print(f"Final result: {event.tool_name or 'text'}")

# Collect final parts in order
final_parts = [parts[i] for i in sorted(parts.keys())]
```

### Streaming Tool Calls
```python
# Tool call built incrementally
events = [
    PartStartEvent(
        index=1,
        part=ToolCallPart(tool_name="", args=None, tool_call_id="call_1")
    ),
    PartDeltaEvent(
        index=1,
        delta=ToolCallPartDelta(tool_name_delta="get_")
    ),
    PartDeltaEvent(
        index=1,
        delta=ToolCallPartDelta(tool_name_delta="weather")
    ),
    PartDeltaEvent(
        index=1,
        delta=ToolCallPartDelta(
            args_delta='{"city": "NYC"}'  # JSON string delta
        )
    ),
]

# Process stream
tool_part = None
for event in events:
    if isinstance(event, PartStartEvent):
        tool_part = event.part
    elif isinstance(event, PartDeltaEvent):
        tool_part = event.delta.apply(tool_part)

print(f"Final tool call: {tool_part.tool_name}({tool_part.args_as_dict()})")
```

---

## 6. Agent Integration Patterns

### Complete Agent Loop with Messages
```python
from pydantic_ai import Agent

agent = Agent(model="openai:gpt-4o")

# Agent maintains message_history internally
result = agent.run_sync(
    user_prompt="What's the weather in NYC?",
    tools=[get_weather_tool],
)

# After run, messages are stored in agent context
# You can't directly access history, but it's used internally
```

### Custom Message History Processing
```python
from pydantic_ai import Agent, ModelMessage

# Define a history processor
def my_history_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Filter out sensitive information
    filtered = []
    for msg in messages:
        if msg.kind == 'response':
            # Filter tool details from responses
            new_parts = []
            for part in msg.parts:
                if not isinstance(part, ToolCallPart):
                    new_parts.append(part)
            msg = replace(msg, parts=new_parts)
        filtered.append(msg)
    return filtered

# Would be used via agent configuration
# agent = Agent(..., message_history_processor=my_history_processor)
```

---

## 7. Advanced Scenarios

### Extended Thinking with Signatures
```python
from pydantic_ai import ThinkingPart

# Claude thinking response
thinking = ThinkingPart(
    content="Let me think through this problem step by step...",
    signature="sig_123abc",  # Encrypted signature from Anthropic
    provider_name="anthropic"
)

response = ModelResponse(
    parts=[
        thinking,
        TextPart(content="Based on my analysis...")
    ]
)
```

### Built-in Tools (Provider-Specific)
```python
from pydantic_ai import BuiltinToolCallPart, BuiltinToolReturnPart

# Google's image generation tool
builtin_call = BuiltinToolCallPart(
    tool_name="image_generation",
    args='{"prompt": "sunset over mountains"}',
    provider_name="google"
)

response = ModelResponse(
    parts=[
        TextPart(content="I'll generate an image for you."),
        builtin_call,
    ]
)

# Tool result
builtin_return = BuiltinToolReturnPart(
    tool_name="image_generation",
    content={"image_url": "https://example.com/generated.jpg"},
    tool_call_id=builtin_call.tool_call_id,
    provider_name="google"
)
```

### File Response from Model
```python
from pydantic_ai import FilePart, BinaryImage

# Model generated an image
file_part = FilePart(
    content=BinaryImage(
        data=generated_image_bytes,
        media_type="image/png",
        identifier="gen_img_1"
    ),
    provider_name="openai"  # Only send back to OpenAI
)

response = ModelResponse(
    parts=[
        TextPart(content="Here's the generated image:"),
        file_part,
    ]
)

# Extract images
images = response.images  # List[BinaryImage]
```

### ToolReturn for Rich Responses
```python
from pydantic_ai import ToolReturn, ToolReturnPart

# Tool can return both structured data AND custom content
def fetch_data(query: str) -> ToolReturn:
    results = search_database(query)
    
    # Return structured data + formatted content
    return ToolReturn(
        return_value=results,  # Actual tool return
        content=[  # Content to send to model
            f"Found {len(results)} results:\n",
            ImageUrl(url=results[0].preview_image),
            f"\nTop result: {results[0].title}\n{results[0].description}"
        ]
    )

# Agent will use return_value for tool output
# but send content to model for further processing
```

### Backward Compatibility with Old Message Format
```python
# Old DB format with 'vendor_details' and 'vendor_id'
old_message = {
    "kind": "response",
    "parts": [...],
    "vendor_details": {"finish_reason": "STOP"},
    "vendor_id": "chatcmpl-ABC123"
}

# Will deserialize correctly with alias validation
message = ModelMessagesTypeAdapter.validate_python(old_message)

# Access new field names
print(message.provider_details)  # Works!
print(message.provider_response_id)  # Works!
```

### Metadata for Application Use
```python
from pydantic_ai import ToolReturnPart

# Store app-level metadata that doesn't go to LLM
tool_return = ToolReturnPart(
    tool_name="database_query",
    content=query_results,
    metadata={
        "query_execution_time_ms": 125,
        "cache_hit": True,
        "query_complexity": "high"
    }
)

# Application can access this
if tool_return.metadata.get("cache_hit"):
    print("Used cached result")
```

---

## Complete End-to-End Example

```python
from pydantic_ai import (
    ModelRequest, ModelResponse, SystemPromptPart, UserPromptPart,
    TextPart, ToolCallPart, ToolReturnPart, ModelMessagesTypeAdapter
)

# 1. Create initial request
request1 = ModelRequest(
    parts=[
        SystemPromptPart(content="You are a weather assistant."),
        UserPromptPart(content="What's the weather in Paris?"),
    ]
)

# 2. Simulate model response with tool call
response1 = ModelResponse(
    parts=[
        TextPart(content="Let me check the weather in Paris."),
        ToolCallPart(
            tool_name="get_weather",
            args='{"city": "Paris"}',
            tool_call_id="call_1"
        )
    ],
    model_name="gpt-4o"
)

# 3. Execute tool and create follow-up request
tool_result = {"temperature": 22, "conditions": "sunny"}

request2 = ModelRequest(
    parts=[
        ToolReturnPart(
            tool_name="get_weather",
            content=tool_result,
            tool_call_id="call_1"
        )
    ]
)

# 4. Final response
response2 = ModelResponse(
    parts=[
        TextPart(content="The weather in Paris is currently 22°C and sunny.")
    ],
    model_name="gpt-4o"
)

# 5. Store complete conversation
messages = [request1, response1, request2, response2]

# 6. Serialize and save
json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
with open("conversation.json", "wb") as f:
    f.write(json_bytes)

# 7. Load and continue conversation
with open("conversation.json", "rb") as f:
    loaded_messages = ModelMessagesTypeAdapter.validate_json(f.read())

print(f"Loaded {len(loaded_messages)} messages")
for msg in loaded_messages:
    if msg.kind == "response":
        for part in msg.parts:
            if isinstance(part, TextPart):
                print(f"Assistant: {part.content}")
```

---

## Key Takeaways

1. **Discriminators**: Always check `part_kind` or `kind` field to determine message type
2. **Serialization**: Use `ModelMessagesTypeAdapter` for JSON serialization
3. **Tool Arguments**: Can be either JSON strings or dicts - use `args_as_dict()`/`args_as_json_str()`
4. **Multi-Modal**: Include multiple `UserContent` items in `UserPromptPart`
5. **Streaming**: Use `apply()` methods on deltas to reconstruct parts incrementally
6. **Backward Compatibility**: Old field names still work via alias validation
7. **Metadata**: Non-LLM metadata stored in `metadata` field won't be sent to model
8. **Provider-Specific**: Check `provider_name` for provider-only features like thinking signatures

