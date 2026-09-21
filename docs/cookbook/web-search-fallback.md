---
title: Search the web with a local fallback
description: Use provider-native web tools when available and a local implementation on other models.
---

# Search the web with a local fallback

Capabilities can select a provider-native tool when supported and retain a local fallback when the agent changes models.

```bash
pip/uv-add "pydantic-ai[duckduckgo]"
```

```python {test="skip"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch, WebSearch

agent = Agent(
    'openai:gpt-5.6-sol',
    instructions='Prefer primary sources and include a direct URL for every factual claim.',
    capabilities=[
        WebSearch(local=True),
        WebFetch(local=True),
    ],
)


async def main() -> None:
    result = await agent.run('What changed in the latest stable Python release?')
    print(result.output)


if __name__ == '__main__':
    asyncio.run(main())
```

Native tools are used when the model profile supports them; otherwise the local tools remain available. Source requirements still need output validation or evaluation when citations are part of the application contract.
