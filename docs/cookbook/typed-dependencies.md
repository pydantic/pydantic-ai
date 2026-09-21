---
title: Give tools typed application context
description: Pass request-scoped services and data to agent tools through typed dependencies.
---

# Give tools typed application context

Use `deps_type` and `RunContext` to give tools access to application state without globals or closures.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class SupportContext:
    customer_id: str
    orders: dict[str, str]


agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=SupportContext,
    instructions='Help the customer using their order information.',
)


@agent.tool
def order_status(ctx: RunContext[SupportContext], order_id: str) -> str:
    """Look up the current status of one order."""
    return ctx.deps.orders.get(order_id, 'Order not found')


async def main() -> None:
    deps = SupportContext(
        customer_id='customer-123',
        orders={'A100': 'Shipped; expected Friday'},
    )
    result = await agent.run('Where is order A100?', deps=deps)
    print(result.output)
    #> Order A100 has shipped and is expected Friday.


if __name__ == '__main__':
    asyncio.run(main())
```

The dependency type is part of the agent's type signature, so tools and call sites are checked together. In an application, the same container can hold database connections, API clients, authenticated user details, or other request-scoped services.

## Related

See [Dependencies](../dependencies.md) for dynamic instructions, testing overrides, and dependency typing across an application.
