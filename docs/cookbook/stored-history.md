---
title: Continue a conversation with stored history
description: Persist one run's messages and use them to continue the same conversation later.
---

# Continue a conversation with stored history

Store `result.all_messages()` and pass it back as `message_history` when conversations need to survive beyond one process or request.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions='Be concise and remember details from earlier turns.',
)


async def main() -> None:
    first = await agent.run('My deployment region is eu-west-1.')

    # Store these bytes against the conversation ID in your database.
    Path('conversation.json').write_bytes(ModelMessagesTypeAdapter.dump_json(first.all_messages()))

    # This can happen in a later request or after a process restart.
    stored_messages = ModelMessagesTypeAdapter.validate_json(Path('conversation.json').read_bytes())
    second = await agent.run('Which deployment region did I choose?', message_history=stored_messages)
    print(second.output)
    #> You chose eu-west-1.


if __name__ == '__main__':
    asyncio.run(main())
```

Pydantic AI does not hide conversation state inside the agent. The application owns the messages, so it can persist, inspect, branch, or delete them. When loading history supplied by an untrusted client, use the documented message-history security controls rather than treating client messages as trusted server state.

## Related

See [Messages and chat history](../message-history.md) for database storage patterns, history processors, and the boundary for untrusted history.
