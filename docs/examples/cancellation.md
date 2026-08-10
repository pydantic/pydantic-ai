Cancel a streaming agent response from an interactive terminal, then continue the conversation with its preserved history.

Demonstrates:

- [cancelling a run](../agent.md#cancelling-a-run)
- [streaming text responses](../output.md#streaming-text)
- [message history](../message-history.md)
- Building an interactive terminal UI with [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/)

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.cancel_and_resume
```

Press ++esc++ while a response is streaming to cancel that turn. The partial history is kept, so
the next prompt continues the same conversation. Press ++ctrl+c++ or ++ctrl+d++ to exit.

## How Cancellation and Resume Work

The reusable `stream_turn` function accepts a cancellation token and returns the history captured
before cancellation. Resume with that history and a fresh token:

```python
import asyncio
from collections.abc import AsyncIterator

from pydantic_ai_examples.cancel_and_resume import stream_turn

from pydantic_ai import Agent, CancellationToken, ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel

first_delta = asyncio.Event()


async def stream_response(
    _messages: list[ModelMessage], _info: AgentInfo
) -> AsyncIterator[str]:
    for chunk in ('You ', 'can ', 'resume ', 'this.'):
        yield chunk
        await asyncio.sleep(0.01)


async def cancel_on_first_delta(token: CancellationToken) -> None:
    await first_delta.wait()
    token.cancel()  # in the interactive example, the Esc key handler calls this


async def main() -> None:
    agent = Agent(FunctionModel(stream_function=stream_response))

    # First turn: cancel the run as soon as the response starts streaming.
    token = CancellationToken()
    canceller = asyncio.create_task(cancel_on_first_delta(token))
    history, cancelled = await stream_turn(
        agent, 'Tell me a story', [], token, lambda _text: first_delta.set()
    )
    await canceller
    print(f'cancelled={cancelled}, kept {len(history)} messages')
    #> cancelled=True, kept 2 messages

    # The next turn resumes from that preserved history with a fresh token.
    history, cancelled = await stream_turn(
        agent, 'Continue', history, CancellationToken(), lambda _text: None
    )
    print(f'cancelled={cancelled}, now {len(history)} messages')
    #> cancelled=False, now 4 messages


asyncio.run(main())
```

[`RunCancelled.all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages] preserves
completed work and partial streamed responses. Any interrupted tool calls are
[repaired automatically](../message-history.md#making-histories-provider-valid) when the history is
used by the next run.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/cancel_and_resume.py"}```
