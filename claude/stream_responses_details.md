# stream_text() vs stream_responses(): Comprehensive Migration Guide

## Executive Summary

`stream_text()` and `stream_responses()` are two different methods for consuming streaming results from `agent.run_stream()`. The key difference is:

- **`stream_text()`**: Text-only streaming. Works ONLY when the agent outputs plain text (no tools, no structured output).
- **`stream_responses()`**: Universal streaming. Works with ALL response types including text, tool calls, and structured output.

**Critical Issue**: If your agent uses tools and you call `stream_text()`, you'll get a `UserError` because `stream_text()` requires `TextOutputSchema`, which is incompatible with tool usage.

## Key Differences

### Return Types

| Method | Returns | Content |
|--------|---------|---------|
| `stream_text()` | `AsyncIterator[str]` | Just the text content, extracted from the response |
| `stream_responses()` | `AsyncIterator[tuple[ModelResponse, bool]]` | Complete `ModelResponse` object + `is_last` boolean |

### What They Yield

#### `stream_text(delta=False)` (default)
```python
async for text in result.stream_text():
    # text: str - accumulated text so far
    print(text)
    # Output progression:
    # "The first"
    # "The first known"
    # "The first known use of..."
```

#### `stream_text(delta=True)`
```python
async for text_chunk in result.stream_text(delta=True):
    # text_chunk: str - just the new text since last iteration
    print(text_chunk)
    # Output progression:
    # "The first"
    # " known"
    # " use of..."
```

#### `stream_responses()`
```python
async for message, is_last in result.stream_responses():
    # message: ModelResponse - complete response object
    # is_last: bool - True on the final iteration

    # Access various parts:
    text = message.text  # str | None - the text content
    tool_calls = message.tool_calls  # list[ToolCallPart] - any tool calls
    parts = message.parts  # Sequence[ModelResponsePart] - all parts
    usage = message.usage  # RequestUsage - token usage
```

### What's in a ModelResponse?

A `ModelResponse` (from `pydantic_ai/messages.py:1097-1332`) contains:

```python
@dataclass
class ModelResponse:
    parts: Sequence[ModelResponsePart]  # All parts of the response
    usage: RequestUsage                 # Token usage stats
    model_name: str | None             # Model that generated response
    timestamp: datetime                # When response was created
    kind: Literal['response']          # Message type identifier
    provider_name: str | None          # Provider (openai, anthropic, etc.)
    finish_reason: FinishReason | None # Why the model stopped

    # Convenient properties:
    @property
    def text(self) -> str | None       # Extract text from TextParts

    @property
    def tool_calls(self) -> list[ToolCallPart]  # Extract tool calls

    @property
    def thinking(self) -> str | None   # Extract thinking parts (for models that support it)

    @property
    def files(self) -> list[BinaryContent]  # Extract files

    @property
    def images(self) -> list[BinaryImage]  # Extract images
```

**ModelResponsePart** can be:
- `TextPart` - plain text content
- `ToolCallPart` - a tool the model wants to call
- `BuiltinToolCallPart` - built-in tool call (e.g., web search)
- `BuiltinToolReturnPart` - result from built-in tool
- `ThinkingPart` - model's internal reasoning (Claude, o1, etc.)
- `FilePart` - generated files/images

## When to Use Each Method

### Use `stream_text()` when:
- ✅ Your agent outputs **plain text only** (no `output_type` or `output_type=str`)
- ✅ Your agent has **no tools** registered
- ✅ You only care about the text content, nothing else
- ✅ You want the simplest possible API

### Use `stream_responses()` when:
- ✅ Your agent uses **any tools** (including built-in tools)
- ✅ Your agent has **structured output** (`output_type` is a Pydantic model, TypedDict, etc.)
- ✅ You need access to **metadata** (usage, tool calls, thinking, etc.)
- ✅ You need **fine-grained control** over validation
- ✅ You want to handle **multiple response types** (text + tool calls + files)

## Migration Guide: stream_text() → stream_responses()

### Pattern 1: Simple Text Streaming

**Before (stream_text):**
```python
async with agent.run_stream(prompt) as result:
    async for text in result.stream_text():
        print(text)  # Accumulated text
```

**After (stream_responses):**
```python
async with agent.run_stream(prompt) as result:
    async for message, is_last in result.stream_responses():
        text = message.text
        if text:  # text can be None if there are only tool calls
            print(text)
```

### Pattern 2: Delta Text Streaming

**Before (stream_text with delta):**
```python
async with agent.run_stream(prompt) as result:
    full_text = ""
    async for chunk in result.stream_text(delta=True):
        full_text += chunk
        print(chunk, end="", flush=True)
```

**After (stream_responses):**
```python
async with agent.run_stream(prompt) as result:
    last_text = ""
    async for message, is_last in result.stream_responses():
        current_text = message.text or ""
        if current_text != last_text:
            delta = current_text[len(last_text):]
            print(delta, end="", flush=True)
            last_text = current_text
```

### Pattern 3: Real-Time UI Updates

**Before (stream_text for markdown rendering):**
```python
from rich.live import Live
from rich.markdown import Markdown

async with agent.run_stream(prompt) as result:
    with Live('', console=console) as live:
        async for text in result.stream_text():
            live.update(Markdown(text))
```

**After (stream_responses):**
```python
from rich.live import Live
from rich.markdown import Markdown

async with agent.run_stream(prompt) as result:
    with Live('', console=console) as live:
        async for message, is_last in result.stream_responses():
            if message.text:
                live.update(Markdown(message.text))
```

### Pattern 4: Handling Tool Calls

**This is the key scenario where stream_text() fails!**

```python
# Agent with tools - stream_text() will raise UserError!
agent = Agent('openai:gpt-4o')

@agent.tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

# ❌ This will FAIL if the model uses the tool:
async with agent.run_stream('What is the weather in Paris?') as result:
    async for text in result.stream_text():  # UserError!
        print(text)

# ✅ This works correctly:
async with agent.run_stream('What is the weather in Paris?') as result:
    async for message, is_last in result.stream_responses():
        # Check for text content
        if message.text:
            print(f"Text: {message.text}")

        # Check for tool calls
        for tool_call in message.tool_calls:
            print(f"Tool: {tool_call.tool_name}({tool_call.args})")
```

### Pattern 5: Accessing Metadata

**stream_text() only gives you text:**
```python
async with agent.run_stream(prompt) as result:
    async for text in result.stream_text():
        # You have: text (str)
        # You DON'T have: usage, tool calls, thinking, timestamps, etc.
        pass
```

**stream_responses() gives you everything:**
```python
async with agent.run_stream(prompt) as result:
    async for message, is_last in result.stream_responses():
        print(f"Model: {message.model_name}")
        print(f"Usage: {message.usage}")
        print(f"Text: {message.text}")
        print(f"Tool calls: {message.tool_calls}")
        print(f"Thinking: {message.thinking}")
        print(f"Is last: {is_last}")
```

### Pattern 6: Streaming Structured Output

**stream_text() cannot handle this at all:**

```python
class UserProfile(TypedDict):
    name: str
    email: str

agent = Agent('openai:gpt-4o', output_type=UserProfile)

# ❌ This will FAIL - structured output is incompatible with stream_text()
async with agent.run_stream('Extract: John Doe, john@example.com') as result:
    async for text in result.stream_text():  # UserError!
        pass
```

**stream_responses() works with custom validation:**

```python
from pydantic import ValidationError

async with agent.run_stream('Extract: John Doe, john@example.com') as result:
    async for message, is_last in result.stream_responses(debounce_by=0.01):
        try:
            profile = await result.validate_response_output(
                message,
                allow_partial=not is_last
            )
            print(profile)  # Progressively more complete
        except ValidationError:
            continue  # Wait for more data
```

## Implementation Details

### How stream_text() Works (pydantic_ai/result.py:85-108)

1. **Validation Check**: Immediately checks if `output_schema` is `TextOutputSchema`
   - If not → raises `UserError('stream_text() can only be used with text responses')`
   - This is why it fails with tools/structured output!

2. **Text Extraction**: Uses `_stream_response_text()` which:
   - Filters streaming events for `TextPart` and `TextPartDelta` events only
   - Ignores tool calls, thinking, files, etc.
   - Accumulates text content

3. **Validation**: If `delta=False`, runs output validators on accumulated text

4. **Returns**: Pure `str` values

### How stream_responses() Works (pydantic_ai/result.py:446-472)

1. **No Validation**: Accepts ANY response type

2. **Full Message Streaming**:
   - Yields the current `ModelResponse` state
   - Groups events by temporal debouncing (default 0.1s)
   - Includes ALL parts: text, tool calls, thinking, files

3. **Tuple Return**:
   - First element: `ModelResponse` - complete current state
   - Second element: `bool` - whether this is the final message

4. **No Automatic Validation**: You decide when/how to validate

## Common Errors and Solutions

### Error: "stream_text() can only be used with text responses"

**Cause**: You're using `stream_text()` but your agent has:
- Tools registered (`@agent.tool` or `tools=...`)
- Structured output type (`output_type=SomeModel`)
- Built-in tools enabled

**Solution**: Use `stream_responses()` instead.

### Error: Message has no text attribute

**Cause**: Using `stream_responses()` but not checking if `message.text` exists

**Solution**:
```python
async for message, is_last in result.stream_responses():
    if message.text:  # ✅ Always check
        print(message.text)
```

### Error: Cannot access tool calls with stream_text()

**Cause**: `stream_text()` only returns text strings

**Solution**: Use `stream_responses()` to access `message.tool_calls`

## Performance Considerations

### Debouncing

Both methods support `debounce_by` parameter:

```python
# Default: group events within 0.1 seconds
result.stream_text(debounce_by=0.1)
result.stream_responses(debounce_by=0.1)

# No debouncing: every event triggers a yield
result.stream_text(debounce_by=None)
result.stream_responses(debounce_by=None)

# Custom: group events within 0.5 seconds
result.stream_text(debounce_by=0.5)
result.stream_responses(debounce_by=0.5)
```

**Why debounce?**
- Reduces overhead of validation/processing
- Especially important for structured output where validation is expensive
- Trade-off: higher debounce = less frequent updates but better performance

### Memory Usage

- `stream_text()`: Lower memory (only stores text strings)
- `stream_responses()`: Higher memory (stores complete ModelResponse objects)

For most use cases, this difference is negligible.

## Best Practices

### 1. Prefer stream_responses() for New Code

Unless you have a specific reason to use `stream_text()`, prefer `stream_responses()`:
- More flexible
- Works with all agent configurations
- Access to metadata
- No surprises when adding tools later

### 2. Always Check is_last

```python
async for message, is_last in result.stream_responses():
    # Use is_last for special handling
    if is_last:
        # Final validation, cleanup, etc.
        pass
```

### 3. Handle None Gracefully

```python
async for message, is_last in result.stream_responses():
    text = message.text or ""  # Handle None
    thinking = message.thinking or ""  # Handle None
```

### 4. Use Appropriate Debouncing

- UI updates: `debounce_by=0.1` (default)
- Structured output: `debounce_by=0.01` (lower for more frequent validation attempts)
- No validation needed: `debounce_by=None` (every event)

### 5. Validate Progressively with Structured Output

```python
from pydantic import ValidationError

async for message, is_last in result.stream_responses(debounce_by=0.01):
    try:
        output = await result.validate_response_output(
            message,
            allow_partial=not is_last
        )
        # Use progressively complete output
        update_ui(output)
    except ValidationError:
        # Not enough data yet, continue
        continue
```

## Summary Table

| Feature | stream_text() | stream_responses() |
|---------|--------------|-------------------|
| Works with plain text | ✅ Yes | ✅ Yes |
| Works with tools | ❌ No | ✅ Yes |
| Works with structured output | ❌ No | ✅ Yes |
| Returns type | `AsyncIterator[str]` | `AsyncIterator[tuple[ModelResponse, bool]]` |
| Access to text | ✅ Direct | ✅ Via `message.text` |
| Access to tool calls | ❌ No | ✅ Via `message.tool_calls` |
| Access to metadata | ❌ No | ✅ Yes (usage, timestamps, etc.) |
| Access to thinking | ❌ No | ✅ Via `message.thinking` |
| Delta mode | ✅ `delta=True` | ✅ Manual calculation |
| Automatic validation | ✅ Yes (if delta=False) | ❌ Manual via `validate_response_output()` |
| Debouncing | ✅ Yes | ✅ Yes |
| Complexity | Simple | More complex |
| Use case | Text-only agents | Universal |

## Conclusion

**Migration recommendation**: If your agent uses or might use tools in the future, migrate from `stream_text()` to `stream_responses()`. While `stream_responses()` is slightly more verbose, it works universally and provides access to the complete response structure.

The pattern is straightforward:
1. Replace `result.stream_text()` with `result.stream_responses()`
2. Unpack the tuple: `async for message, is_last in ...`
3. Access text via `message.text` (check for None)
4. Optionally use other parts: `message.tool_calls`, `message.thinking`, etc.
