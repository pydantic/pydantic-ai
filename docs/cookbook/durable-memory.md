---
title: Remember user preferences across restarts
description: Give one agent durable, tenant-isolated memory without placing memory contents in application prompts.
---

# Remember user preferences across restarts

Use Harness `Memory` with an application-controlled namespace when an agent should retain useful facts beyond one conversation.

```bash
pip/uv-add "pydantic-ai-harness[memory]"
```

```python {test="skip"}
import asyncio
from dataclasses import dataclass

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import SqliteMemoryStore

from pydantic_ai import Agent


@dataclass
class AppContext:
    user_id: str


def build_agent() -> Agent[AppContext, str]:
    return Agent(
        'anthropic:claude-sonnet-4-6',
        deps_type=AppContext,
        instructions='Use memory for stable user preferences that will help in future conversations.',
        capabilities=[
            Memory(
                SqliteMemoryStore(database='assistant-memory.db'),
                namespace=lambda ctx: ctx.deps.user_id,
            )
        ],
    )


async def main() -> None:
    deps = AppContext(user_id='user-123')

    first_process = build_agent()
    await first_process.run('Remember that I prefer concise deployment reports.', deps=deps)

    # A newly constructed agent reads the same durable, namespaced memory.
    restarted_process = build_agent()
    result = await restarted_process.run('How should you format my deployment reports?', deps=deps)
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

The namespace comes from authenticated application dependencies and is not exposed as a model tool argument. A second user ID addresses different records. Memory content is model-written, untrusted data: do not use it as authorization state or allow it to override application instructions.
