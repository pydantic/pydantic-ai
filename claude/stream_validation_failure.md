# What Happens When Streaming Output Validation Fails

## The Problem

When streaming structured output with `run_stream()`, validation failure creates a serious UX issue:
1. You stream the `content` field to the user token-by-token
2. The LLM then generates an invalid field (e.g., `foo` instead of `bar`)
3. Final validation fails after streaming completes
4. Now what? You've already shown content to the user that's about to be discarded

## How Pydantic AI Handles This

Looking at the code in `/pydantic_ai_slim/pydantic_ai/result.py:61-70`:

```python
async def stream_output(self, *, debounce_by: float | None = 0.1):
    async for response in self.stream_responses(debounce_by=debounce_by):
        if self._final_result_event is not None:
            try:
                yield await self._validate_response(response, allow_partial=True)
            except ValidationError:
                pass  # Silently ignore partial validation errors
    if self._final_result_event is not None:
        yield await self._validate_response(self._raw_stream_response.get())
        # ^^^ This final validation has NO try/except - will raise on error
```

**Key observations:**
1. **During streaming**: Validation errors are silently ignored (`except ValidationError: pass`)
2. **After streaming completes**: Final validation raises if it fails (no try/except)
3. **No automatic retry during streaming**: The retry happens at a higher level

## What Actually Happens on Validation Failure

When final validation fails after streaming:

1. **The stream ends with an exception** - The final `_validate_response` raises `ValidationError`
2. **No retry happens in `run_stream()`** - It just fails
3. **You must handle this yourself** - Catch the error and decide what to do

## The Real Problem: You've Already Shown Content

**This is a fundamental limitation of streaming structured outputs.** Once you've shown tokens to the user, you can't take them back. If validation fails:

- The content you streamed is invalid/incomplete
- The entire response gets discarded
- No automatic retry happens at the streaming level
- The user saw content that's now gone

## Practical Solutions

### 1. Buffer Until Certain (Conservative)
Don't stream `content` until you know the structure is valid:
```python
async with agent.run_stream(prompt) as result:
    try:
        # Get the complete validated output first
        output = await result.get_output()
        # Only show to user after validation succeeds
        await send_to_user(output['content'])
    except ValidationError:
        # Handle the error - maybe retry the whole thing
        await send_error_to_user("Generation failed, retrying...")
```

### 2. Stream But Warn About Potential Failure (Risky)
```python
async with agent.run_stream(prompt) as result:
    streamed_content = ""
    try:
        async for output in result.stream():
            content = output.get('content', '')
            if len(content) > len(streamed_content):
                delta = content[len(streamed_content):]
                await send_to_user(delta)
                streamed_content = content

        # Final validation happens here
        final = await result.get_output()

    except ValidationError as e:
        # Oops - we already showed content that's invalid
        await send_to_user("\n\n[ERROR: Response was invalid, please ignore above]")
        # You'll need to retry the entire request
```

### 3. Use Plain Text Output for Streaming Parts (Hybrid)
Stream text content separately from structured metadata:
```python
# First pass: Get streaming text
agent_text = Agent('gpt-4', output_type=str)
async with agent_text.run_stream("Generate content") as result:
    async for text in result.stream_text(delta=True):
        await send_to_user(text)
    content = await result.get_output()

# Second pass: Get structured metadata
agent_meta = Agent('gpt-4', output_type=Metadata)
meta = await agent_meta.run(f"Given this content, provide metadata: {content}")
```

### 4. Use Two-Phase Generation (Recommended)
Have the LLM generate structure first, then content:
```python
class OutputStructure(TypedDict):
    # Metadata fields first - fail fast if these are wrong
    next_agent: str
    next_agent_reason: str
    # Content field last - only stream after structure validates
    content: str

# Order matters! LLM generates fields in order
```

## Why There's No Retry During Streaming

The fundamental issue is that retries happen at the graph execution level, not the streaming level. When using `run_stream()`:

1. The streaming happens within a single model response
2. Validation failure only detected after streaming completes
3. By then, you've already yielded content to the user
4. A retry would mean starting over with a new request
5. But you can't "unshow" what the user already saw

## Key Takeaway

**Streaming structured outputs is inherently risky.** If any field fails validation, the entire response is invalid, but you've already shown partial content to users.

Best practices:
1. Put fields that might fail validation first (before `content`)
2. Use TypedDict with `NotRequired` for optional fields
3. Consider buffering the stream until you're confident it's valid
4. Have a UX strategy for handling failed streams (error messages, retry UI, etc.)
5. Consider separating streaming content from structured metadata

The core issue is that Pydantic AI doesn't (and can't) retry during streaming - by the time validation fails, the tokens have already been streamed to your user.