---
title: Stream a response as it is generated
description: Send text to a terminal, HTTP response, or UI without waiting for the complete model output.
---

# Stream a response as it is generated

Use `run_stream()` when the consumer needs the final text incrementally rather than every internal agent event.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent

agent = Agent('openai:gpt-5.6-sol')


async def main() -> None:
    chunks: list[str] = []
    async with agent.run_stream('Write a one-sentence deployment update.') as result:
        async for text in result.stream_text(delta=True):
            chunks.append(text)
            # In a server, send `text` to a websocket or streaming HTTP response here.

    print(''.join(chunks))
    #> Version 2.4 is deployed successfully in all regions.


if __name__ == '__main__':
    asyncio.run(main())
```

`delta=True` yields only newly arrived text, which can be forwarded directly. Use `run_stream_events()` instead when the client also needs tool-call, thinking, usage, or final-result events.

## Related

See [Streaming text](../output.md#streaming-text) for validation behavior and [streaming all events](../agent.md#streaming-all-events) for tool-call and usage events.
