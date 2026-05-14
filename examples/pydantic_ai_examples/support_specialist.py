"""Customer-support agent demonstrating deferred capability loading.

The agent advertises two specialist capabilities — `orders` and `returns` —
by id and description only. The model reads the catalog, calls
`load_capability(id)` to unlock the matching specialist, and only then sees
that specialist's tools and instructions. The specialist not loaded on a
given run never enters the context window.

Run with:

    uv run -m pydantic_ai_examples.support_specialist
"""

from dataclasses import dataclass, field

import logfire

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.anthropic import AnthropicModelSettings

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


@dataclass
class Order:
    id: str
    item: str
    status: str


@dataclass
class Store:
    orders: dict[str, Order] = field(
        default_factory=lambda: {
            'A-1042': Order('A-1042', 'Pydantic mug', 'delivered'),
            'A-1099': Order('A-1099', 'Pydantic hoodie', 'in_transit'),
        }
    )


orders_tools = FunctionToolset[Store]()


@orders_tools.tool(defer_loading=True)
def order_status(ctx: RunContext[Store], order_id: str) -> str:
    """Look up the current status of an order."""
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'No order found with id {order_id}.'
    return f'Order {order.id} ({order.item}) is currently {order.status}.'


orders_capability = Capability[Store](
    id='orders',
    description='Look up order status by id.',
    instructions='Quote the item name in your reply.',
    toolset=orders_tools,
    defer_loading=True,
)


returns_tools = FunctionToolset[Store]()


@returns_tools.tool(defer_loading=True)
def start_return(ctx: RunContext[Store], order_id: str, reason: str) -> str:
    """Open a return request. Only delivered orders can be returned."""
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'No order found with id {order_id}.'
    if order.status != 'delivered':
        return f'Order {order.id} is {order.status}; only delivered orders can be returned.'
    return f'Return opened for {order.id}. Reason: {reason!r}.'


returns_capability = Capability[Store](
    id='returns',
    description='Open return requests and answer return-policy questions.',
    instructions='Returns are accepted within 30 days of delivery for unused items.',
    toolset=returns_tools,
    defer_loading=True,
)


support_agent = Agent(
    model='anthropic:claude-sonnet-4-6',
    model_settings=AnthropicModelSettings(anthropic_cache_messages=True),
    deps_type=Store,
    instructions='You are a customer-support agent for an e-commerce store.',
    capabilities=[orders_capability, returns_capability],
)


async def main() -> None:
    store = Store()
    for prompt in [
        'Where is order A-1042?',
        "I'd like to return A-1042 — it arrived damaged.",
    ]:
        print(f'\n> {prompt}')
        result = await support_agent.run(prompt, deps=store)
        print(result.output)

        print(result.usage())


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
