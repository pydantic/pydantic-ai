---
title: Fall back when a model provider fails
description: Retry a run on a second provider when the primary model returns a provider error.
---

# Fall back when a model provider fails

Use `FallbackModel` when an application should continue through a provider outage or retryable model failure.

```bash
pip/uv-add "pydantic-ai-slim[openai,anthropic]"
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key
```

```python {dunder_name="not_main"}
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel(
    'openai:gpt-5.6-sol',
    'anthropic:claude-sonnet-4-6',
)
agent = Agent(model)


async def main() -> None:
    result = await agent.run('Give the incident commander one concise status update.')
    print(result.output)
    #> Checkout is degraded; rollback is in progress.
    # Record result.response.model_name to monitor which provider served the response.


if __name__ == '__main__':
    asyncio.run(main())
```

The models are tried in order. Fallback applies to provider and model failures, not invalid application output; use output validation and bounded retries for that case. Configure credentials for every listed provider and monitor which model served each response so a persistent primary failure does not go unnoticed.

## Related

See [Fallback models](../models/overview.md#fallback-model) for retry rules, exception matching, and streaming behavior.
