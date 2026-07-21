# Process History

[`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] is a [capability](overview.md) that wraps a [history processor](../message-history.md#processing-message-history): a function that receives the message history before each model request and returns the (possibly modified) list of messages to send. Use it to trim old turns, redact sensitive content, or summarize long conversations:

```python {title="process_history.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage


def keep_recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-5:]  # (1)!


agent = Agent('openai:gpt-5.2', capabilities=[ProcessHistory(keep_recent)])
```

1. Keep only the five most recent messages. In practice you'll want to keep the first request too, so the system prompt survives — see [Processing Message History](../message-history.md#processing-message-history) for complete patterns.

The processor may be sync or async, and may optionally take a [`RunContext`][pydantic_ai.tools.RunContext] as its first argument to access dependencies and run state. Multiple `ProcessHistory` capabilities apply in registration order. Note that the processed messages *replace* the run's message history, so make a copy first if you need to keep the original.

`ProcessHistory` is a thin wrapper around the [`before_model_request`](../hooks.md) lifecycle hook — hook that event directly for richer control, like short-circuiting the model call. See [Processing Message History](../message-history.md#processing-message-history) for the full guide, including summarization examples and interactions with [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages].
