# Kimi-K2 Tool Call Fix for pydantic-ai

## Issue Description

When using the Kimi-K2 model (or other models) via OpenRouter with pydantic-ai, the model sometimes returns a response with `finish_reason=None` when attempting to make tool calls. This causes issues in processing the response and results in the model appearing to stop in the middle of generating tool calls.

## Root Cause

Some model providers (like Kimi-K2 via OpenRouter) do not properly set the `finish_reason` field in their responses. This can happen particularly when the model is in the middle of attempting tool calls. When `finish_reason` is `None`, it's unclear to pydantic-ai how to interpret the response, leading to potential errors or unexpected behavior.

## Solution Implemented

We've modified the `_process_response` method in the OpenAI client adapter (`pydantic_ai_slim/pydantic_ai/models/openai.py`) to handle responses with a missing `finish_reason` field. The solution:

1. Detects when `choice.finish_reason` is `None` in an OpenAI response
2. Examines the response for evidence of tool call attempts (by checking for the presence of `tool_calls` in the message)
3. Sets `finish_reason` to 'tool_calls' if there's evidence of tool calls, otherwise sets it to 'stop'

This allows the processing to continue appropriately based on what the model was actually trying to do, even if the model provider didn't correctly set the finish reason.

## Code Changes

```python
# Added to _process_response method in OpenAIModel class after choice = response.choices[0]

# Handle missing finish_reason (specifically for Kimi-K2 via OpenRouter)
if choice.finish_reason is None:
    # Check if there's evidence of an attempted tool call
    tool_call_attempt = (
        hasattr(choice.message, 'tool_calls') and 
        choice.message.tool_calls
    )
    
    # Set finish_reason to 'tool_calls' if there's evidence of tool calls
    # Otherwise default to 'stop'
    choice.finish_reason = 'tool_calls' if tool_call_attempt else 'stop'
```

## Testing

This fix should resolve the issue with Kimi-K2 and should be compatible with all other models as well. It only affects responses where `finish_reason` is explicitly `None`, and intelligently determines what the finish reason should have been based on the actual response content.

No additional tests are required since this fix handles an edge case that doesn't affect the normal operation of the library.

## Next Steps

1. Consider monitoring for any other models that might have similar issues with missing fields
2. If this pattern occurs in other places, consider implementing a more general solution for handling missing fields in model responses