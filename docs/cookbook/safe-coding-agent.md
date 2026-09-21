---
title: Build a coding agent scoped to one workspace
description: Let an agent inspect, edit, run, and verify a project without granting unrestricted filesystem access.
---

# Build a coding agent scoped to one workspace

Harness `Coder` composes filesystem, search, editing, shell, planning, and output-management capabilities around one workspace boundary.

```bash
pip/uv-add "pydantic-ai-harness[coder]"
```

```python {test="skip"}
import asyncio
from pathlib import Path

from pydantic_ai_harness import Coder

from pydantic_ai import Agent

workspace = Path.cwd().resolve()
agent = Agent(
    'openai:gpt-5.6-sol',
    instructions=(
        'Make the smallest correct change. Read relevant code first, run focused tests, '
        'and report the files changed and verification performed.'
    ),
    capabilities=[Coder(workspace)],
)


async def main() -> None:
    result = await agent.run(
        'Fix the failing tests for the invoice parser. Do not change its public API.'
    )
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

The filesystem capability is rooted at `workspace`; relative paths cannot escape it. The shell still executes real commands with the current process permissions, so run untrusted coding tasks inside a container or sandbox and require approval for deployment, credential, or destructive operations.
