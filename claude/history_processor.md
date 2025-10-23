# Using history_processor for Assistant Response Prefilling

## Overview
`history_processor` allows you to modify message history before it's sent to the model. This can be used to implement assistant response prefilling (starting the assistant's response with specific text that the model continues from).

## How It Works
1. History processors run at `_agent_graph.py:356-358` right before the model request
2. They receive the full message history and return a modified version
3. The modified history is what gets sent to the model API

## Implementation

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart

def inject_prefill(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Inject a prefilled assistant response after the last user message."""
    if messages and messages[-1].kind == 'request':
        # Add a partial assistant response that the model will continue from
        prefill = ModelResponse(parts=[TextPart("Based on the data, ")])
        return messages + [prefill]
    return messages

# Create agent with the history processor
agent = Agent(
    'openai:gpt-4o',  # Works with OpenAI-compatible APIs
    history_processors=[inject_prefill]
)

# The model will continue from "Based on the data, " when generating response
result = await agent.run("What's the trend?")
```

## Key Points
- Works with `OpenAIModel` and OpenAI-compatible APIs (including DeepInfra)
- No validation in `OpenAIModel` blocks conversations ending with assistant messages
- The prefilled text becomes part of the conversation history sent to the model
- Model behavior depends on API support - test with cURL first to verify your API supports prefilling

## With run_stream
Also works with streaming:
```python
async with agent.run_stream("Your prompt") as result:
    async for delta in result.stream():
        # First tokens will continue from your prefill
        print(delta)
```

## Conditional Prefilling
You can make prefilling conditional based on context:
```python
def smart_prefill(messages: list[ModelMessage]) -> list[ModelMessage]:
    if messages and messages[-1].kind == 'request':
        # Check the user's prompt content
        last_request = messages[-1]
        if "analyze" in str(last_request.parts[0].content).lower():
            prefill = ModelResponse(parts=[TextPart("After analyzing the data, ")])
            return messages + [prefill]
    return messages
```