---
title: Review a change with parallel specialists
description: Run independent typed reviews concurrently, then combine their findings in deterministic application code.
---

# Review a change with parallel specialists

Use ordinary `asyncio` concurrency when tasks are independent. Give each specialist a narrow output type and combine the results in code instead of asking another model to preserve every finding.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import Agent


class Finding(BaseModel):
    area: Literal['security', 'reliability']
    risk: str
    mitigation: str


security = Agent(
    'openai:gpt-5.6-sol',
    output_type=list[Finding],
    instructions='Return only concrete security risks and mitigations.',
)
reliability = Agent(
    'openai:gpt-5.6-sol',
    output_type=list[Finding],
    instructions='Return only concrete reliability risks and mitigations.',
)


async def main() -> None:
    change = 'Move session storage from Redis to PostgreSQL without a staged rollout.'
    security_result, reliability_result = await asyncio.gather(
        security.run(f'Security review: {change}'),
        reliability.run(f'Reliability review: {change}'),
    )
    findings = security_result.output + reliability_result.output
    for finding in findings:
        print(f'{finding.area}: {finding.risk}')
        #> security: Session rows may cross tenant boundaries
        #> reliability: A direct cutover can lose active sessions


if __name__ == '__main__':
    asyncio.run(main())
```

Set a timeout and usage limit on each run in production. `asyncio.gather()` fails the operation when either specialist raises; use `return_exceptions=True` only when the product can explicitly report a partial review rather than silently treating it as complete.

## Related

See [Multi-agent applications](../multi-agent-applications.md) for delegation and hand-off patterns where agents need to interact.
