---
title: Keep conversation history within a fixed window
description: Bound the context sent to the model while retaining the application's complete stored transcript.
---

# Keep conversation history within a fixed window

Use a history processor to limit what reaches the model. Continue storing the complete transcript separately so retention, audit, and deletion policy do not depend on the model context window.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent, ModelMessage
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessagesTypeAdapter


def keep_recent_text_turns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep two complete text-only turns plus the new request."""
    return messages[-5:]


agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[ProcessHistory(keep_recent_text_turns)],
    instructions='Answer using the recent incident updates in context.',
)


async def main() -> None:
    transcript: list[ModelMessage] = []
    for update in (
        'Incident update: checkout latency is high.',
        'Incident update: the team paused deployments.',
        'Incident update: rollback started.',
    ):
        result = await agent.run(update, message_history=transcript)
        transcript.extend(result.new_messages())

    final = await agent.run('What is the latest mitigation?', message_history=transcript)
    transcript.extend(final.new_messages())
    print(final.output)
    #> The latest mitigation is a rollback.
    stored = ModelMessagesTypeAdapter.dump_json(transcript).decode()
    print('checkout latency is high' in stored)
    #> True


if __name__ == '__main__':
    asyncio.run(main())
```

A history processor replaces the history inside that run. Extending a separately stored transcript with `result.new_messages()` prevents trimming from deleting the application's record. Simple slicing is appropriate only for text-only turns. Tool calls and returns must remain paired; for tool-using agents, trim at complete turn boundaries or use Harness compaction. Summarization preserves more context but adds cost and can lose details, so start with deterministic trimming when recent turns are sufficient.

## Related

See [Process History](../capabilities/process-history.md) for safe trimming rules and summarization options.
