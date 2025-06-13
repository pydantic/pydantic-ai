# Messages and chat history

PydanticAI provides access to messages exchanged during an agent run. These messages can be used both to continue a coherent conversation, and to understand how an agent performed.

### Accessing Messages from Results

After running an agent, you can access the messages exchanged during that run from the `result` object.

Both [`RunResult`][pydantic_ai.agent.AgentRunResult]
(returned by [`Agent.run`][pydantic_ai.Agent.run], [`Agent.run_sync`][pydantic_ai.Agent.run_sync])
and [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] (returned by [`Agent.run_stream`][pydantic_ai.Agent.run_stream]) have the following methods:

* [`all_messages()`][pydantic_ai.agent.AgentRunResult.all_messages]: returns all messages, including messages from prior runs. There's also a variant that returns JSON bytes, [`all_messages_json()`][pydantic_ai.agent.AgentRunResult.all_messages_json].
* [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages]: returns only the messages from the current run. There's also a variant that returns JSON bytes, [`new_messages_json()`][pydantic_ai.agent.AgentRunResult.new_messages_json].

!!! info "StreamedRunResult and complete messages"
    On [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult], the messages returned from these methods will only include the final result message once the stream has finished.

    E.g. you've awaited one of the following coroutines:

    * [`StreamedRunResult.stream()`][pydantic_ai.result.StreamedRunResult.stream]
    * [`StreamedRunResult.stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text]
    * [`StreamedRunResult.stream_structured()`][pydantic_ai.result.StreamedRunResult.stream_structured]
    * [`StreamedRunResult.get_output()`][pydantic_ai.result.StreamedRunResult.get_output]

    **Note:** The final result message will NOT be added to result messages if you use [`.stream_text(delta=True)`][pydantic_ai.result.StreamedRunResult.stream_text] since in this case the result content is never built as one string.

Example of accessing methods on a [`RunResult`][pydantic_ai.agent.AgentRunResult] :

```python {title="run_result_messages.py" hl_lines="10"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result = agent.run_sync('Tell me a joke.')
print(result.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

# all messages from the run
print(result.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=Usage(requests=1, request_tokens=60, response_tokens=12, total_tokens=72),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
]
"""
```
_(This example is complete, it can be run "as is")_

Example of accessing methods on a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] :

```python {title="streamed_run_result_messages.py" hl_lines="9 40"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')


async def main():
    async with agent.run_stream('Tell me a joke.') as result:
        # incomplete messages before the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        timestamp=datetime.datetime(...),
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    ),
                ]
            )
        ]
        """

        async for text in result.stream_text():
            print(text)
            #> Did you hear
            #> Did you hear about the toothpaste
            #> Did you hear about the toothpaste scandal? They called
            #> Did you hear about the toothpaste scandal? They called it Colgate.

        # complete messages once the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Be a helpful assistant.',
                        timestamp=datetime.datetime(...),
                    ),
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    ),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Did you hear about the toothpaste scandal? They called it Colgate.'
                    )
                ],
                usage=Usage(request_tokens=50, response_tokens=12, total_tokens=62),
                model_name='gpt-4o',
                timestamp=datetime.datetime(...),
            ),
        ]
        """
```
_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

### Using Messages as Input for Further Agent Runs

The primary use of message histories in PydanticAI is to maintain context across multiple agent runs.

To use existing messages in a run, pass them to the `message_history` parameter of
[`Agent.run`][pydantic_ai.Agent.run], [`Agent.run_sync`][pydantic_ai.Agent.run_sync] or
[`Agent.run_stream`][pydantic_ai.Agent.run_stream].

If `message_history` is set and not empty, a new system prompt is not generated — we assume the existing message history includes a system prompt.

```python {title="Reusing messages in a conversation" hl_lines="9 13"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.output)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=Usage(requests=1, request_tokens=60, response_tokens=12, total_tokens=72),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.'
            )
        ],
        usage=Usage(requests=1, request_tokens=61, response_tokens=26, total_tokens=87),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
]
"""
```
_(This example is complete, it can be run "as is")_

## Storing and loading messages (to JSON)

While maintaining conversation state in memory is enough for many applications, often times you may want to store the messages history of an agent run on disk or in a database. This might be for evals, for sharing data between Python and JavaScript/TypeScript, or any number of other use cases.

The intended way to do this is using a `TypeAdapter`.

We export [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter] that can be used for this, or you can create your own.

Here's an example showing how:

```python {title="serialize messages to json"}
from pydantic_core import to_jsonable_python

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter  # (1)!

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
history_step_1 = result1.all_messages()
as_python_objects = to_jsonable_python(history_step_1)  # (2)!
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_python(as_python_objects)

result2 = agent.run_sync(  # (3)!
    'Tell me a different joke.', message_history=same_history_as_step_1
)
```

1. Alternatively, you can create a `TypeAdapter` from scratch:
   ```python {lint="skip" format="skip"}
   from pydantic import TypeAdapter
   from pydantic_ai.messages import ModelMessage
   ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])
   ```
2. Alternatively you can serialize to/from JSON directly:
   ```python {test="skip" lint="skip" format="skip"}
   from pydantic_core import to_json
   ...
   as_json_objects = to_json(history_step_1)
   same_history_as_step_1 = ModelMessagesTypeAdapter.validate_json(as_json_objects)
   ```
3. You can now continue the conversation with history `same_history_as_step_1` despite creating a new agent run.

_(This example is complete, it can be run "as is")_

## Other ways of using messages

Since messages are defined by simple dataclasses, you can manually create and manipulate, e.g. for testing.

The message format is independent of the model used, so you can use messages in different agents, or the same agent with different models.

In the example below, we reuse the message from the first agent run, which uses the `openai:gpt-4o` model, in a second agent run using the `google-gla:gemini-1.5-pro` model.

```python {title="Reusing messages with a different model" hl_lines="17"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', system_prompt='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync(
    'Explain?',
    model='google-gla:gemini-1.5-pro',
    message_history=result1.new_messages(),
)
print(result2.output)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            SystemPromptPart(
                content='Be a helpful assistant.',
                timestamp=datetime.datetime(...),
            ),
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=Usage(requests=1, request_tokens=60, response_tokens=12, total_tokens=72),
        model_name='gpt-4o',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.'
            )
        ],
        usage=Usage(requests=1, request_tokens=61, response_tokens=26, total_tokens=87),
        model_name='gemini-1.5-pro',
        timestamp=datetime.datetime(...),
    ),
]
"""
```

## Processing Message History

Sometimes you may want to modify the message history before it's sent to the model. This could be for privacy
reasons (filtering out sensitive information), to save costs on tokens, to give less context to the LLM, or
custom processing logic.

PydanticAI provides a `history_processors` parameter on `Agent` that allows you to intercept and modify
the message history before each model request.

### Usage

The `history_processors` is a list of callables that take a list of
[`ModelMessage`][pydantic_ai.messages.ModelMessage] and return a modified list of the same type.
Each processor is applied in sequence, and processors can be either synchronous or asynchronous:

```python {title="simple_history_processor.py"}
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove all ModelResponse messages, keeping only ModelRequest messages."""
    return [msg for msg in messages if isinstance(msg, ModelRequest)]

agent = Agent('openai:gpt-4o', history_processors=[filter_responses])

# Create some conversation history
message_history = [
    ModelRequest(parts=[UserPromptPart(content='What is 2+2?')]),
    ModelResponse(parts=[TextPart(content='2+2 equals 4')]),
]

# The history processor will filter out the ModelResponse before sending to the model
result = agent.run_sync('What about 3+3?', message_history=message_history)
```

#### Keep Only Recent Messages

You can use the `history_processor` to only keep the recent messages:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage


def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only the last 5 messages to manage token usage."""
    return messages[-5:] if len(messages) > 5 else messages

agent = Agent('openai:gpt-4o', history_processors=[keep_recent_messages])

# Even with a long conversation history, only the last 5 messages are sent to the model
result = agent.run_sync('What did we discuss?', message_history=long_conversation_history)
```

#### Summarize Old Messages

Use an LLM to summarize older messages to preserve context while reducing tokens. Note that since `history_processor` is called synchronously, this approach works best when you pre-compute summaries:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)


class MessageSummarizer:
    def __init__(self):
        self.summary_agent = Agent('openai:gpt-4o-mini')  # Use a cheaper model for summarization
        self._summary_cache = {}

    def get_summary_key(self, messages: list[ModelMessage]) -> str:
        """Create a simple key based on message count and last message content."""
        if not messages:
            return "empty"
        last_content = ""
        if isinstance(messages[-1], ModelRequest) and messages[-1].parts:
            part = messages[-1].parts[-1]
            if isinstance(part, (UserPromptPart, TextPart)):
                last_content = part.content[:50]  # First 50 chars
        return f"{len(messages)}_{hash(last_content)}"

    def summarize_messages(self, messages: list[ModelMessage]) -> str:
        """Synchronously get a summary, using cache when possible."""
        cache_key = self.get_summary_key(messages)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        # Extract text content from messages for summarization
        message_texts = []
        for i, msg in enumerate(messages):
            if isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        message_texts.append(f"User: {part.content}")
                    elif isinstance(part, SystemPromptPart):
                        message_texts.append(f"System: {part.content}")
            elif isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        message_texts.append(f"Assistant: {part.content}")

        if not message_texts:
            return "No messages to summarize"

        conversation_text = "\n".join(message_texts)
        summary_prompt = f"""Provide a concise summary of this conversation:

{conversation_text}

Summary:"""

        # This would need to be run beforehand or you'd need to implement async handling
        summary = self.summary_agent.run_sync(summary_prompt).output
        self._summary_cache[cache_key] = summary
        return summary

# Create a summarizer instance
summarizer = MessageSummarizer()

def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep recent messages and summarize older ones."""
    if len(messages) <= 5:
        return messages  # Keep all messages if we have 5 or fewer

    # Keep the most recent 3 messages as-is
    recent_messages = messages[-3:]
    old_messages = messages[:-3]

    # Get summary of old messages
    summary_text = summarizer.summarize_messages(old_messages)

    # Create a summary message
    summary_message = ModelRequest(parts=[
        SystemPromptPart(content=f"Previous conversation summary: {summary_text}")
    ])

    # Return summary + recent messages
    return [summary_message] + recent_messages

agent = Agent('openai:gpt-4o', history_processors=[summarize_old_messages])
```

!!! tip "Simpler Alternative: Pre-process Message History"
    For many use cases, it's simpler to summarize messages before passing them to the agent rather than using `history_processor`:

    ```python
    # Pre-process approach (simpler)
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelRequest, SystemPromptPart

    summary_agent = Agent('openai:gpt-4o-mini')
    agent = Agent('openai:gpt-4o')

    # Assume message_history is your existing conversation history
    # and format_messages_for_summary is your helper function

    # Summarize old messages beforehand
    if len(message_history) > 10:
        old_messages_text = format_messages_for_summary(message_history[:-5])
        summary = summary_agent.run_sync(f'Summarize this conversation: {old_messages_text}')

        summary_message = ModelRequest(parts=[
            SystemPromptPart(content=f'Previous conversation: {summary.output}')
        ])
        processed_history = [summary_message] + message_history[-5:]
    else:
        processed_history = message_history

    # Use the pre-processed history
    result = agent.run_sync('Continue the conversation', message_history=processed_history)
    ```

    Use `history_processors` when you need automatic processing for every request, or when the processing logic is simple and fast.

### Testing History Processors

You can test what messages are actually sent to the model provider using `FunctionModel`:

```python {title="Testing history processor behavior"}
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def test_history_processor():
    received_messages = []

    def capture_messages(messages: list[ModelMessage], info: AgentInfo):
        received_messages.clear()
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Test response')])

    def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(
        FunctionModel(capture_messages),
        history_processors=[filter_responses]
    )

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
    ]

    agent.run_sync('Question 2', message_history=message_history)

    # received_messages now contains only the filtered messages
    assert len(received_messages) == 2  # Only the two ModelRequest messages
    assert all(isinstance(msg, ModelRequest) for msg in received_messages)
```

### Multiple Processors and Async Support

You can provide multiple processors that will be applied in sequence:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart


def add_context(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Add context prefix to user prompts
    processed = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            new_parts = []
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    new_parts.append(UserPromptPart(content=f'[CONTEXT] {part.content}'))
                else:
                    new_parts.append(part)
            processed.append(ModelRequest(parts=new_parts))
        else:
            processed.append(msg)
    return processed

def keep_recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Keep only recent messages
    return messages[-5:]

# Processors are applied in order: first add_context, then keep_recent
agent = Agent('openai:gpt-4o', history_processors=[add_context, keep_recent])
```

You can also use async processors for operations that require async calls:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart


async def async_summarizer(messages: list[ModelMessage]) -> list[ModelMessage]:
    if len(messages) > 10:
        # Example: Use an async summarization service
        # summary = await your_async_summarization_service(messages[:-5])
        summary = 'Previous conversation summary'  # Placeholder
        summary_msg = ModelRequest(parts=[SystemPromptPart(content=summary)])
        return [summary_msg] + messages[-5:]
    return messages

agent = Agent('openai:gpt-4o', history_processors=[async_summarizer])
```

### Important Notes

- **Sync and Async Support**: History processors can be either synchronous or asynchronous functions. Async processors are awaited automatically.
- **Message Immutability**: The original message history passed to the agent is not modified. The processor receives a copy and should return a new list.
- **Performance**: History processors are called on every model request, so keep them efficient for large message histories.
- **Error Handling**: If the history processor raises an exception, the agent run will fail. Ensure proper error handling in your processor.
- **Type Safety**: The processor must return a list of `ModelMessage` objects. Invalid return types will cause runtime errors.

## Examples

For a more complete example of using messages in conversations, see the [chat app](examples/chat-app.md) example.
