"""Customer-support agent demonstrating deferred capability loading.

The agent advertises four specialist capabilities — `orders`, `returns`,
`account`, and `products` — by id and description, but keeps each
specialist's instructions and tools out of the model's initial request.
The model reads the catalog, picks the relevant specialist by calling
`load_capability(id)`, and only then sees that specialist's tools and
instructions. Most user questions only need one specialist, so the other
three never enter the context window.

Run with:

    uv run -m pydantic_ai_examples.support_specialist
"""

from __future__ import annotations

from dataclasses import dataclass, field

import logfire

from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.capabilities import Capability

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Order:
    id: str
    customer_email: str
    item: str
    status: str


@dataclass
class Customer:
    email: str
    shipping_address: str


@dataclass
class Product:
    sku: str
    name: str
    stock: int


@dataclass
class Store:
    """In-memory stand-in for the order/customer/catalog database."""

    orders: dict[str, Order] = field(
        default_factory=lambda: {
            'A-1042': Order('A-1042', 'jane@example.com', 'Pydantic mug', 'delivered'),
            'A-1099': Order(
                'A-1099', 'jane@example.com', 'Pydantic hoodie', 'in_transit'
            ),
        }
    )
    customers: dict[str, Customer] = field(
        default_factory=lambda: {
            'jane@example.com': Customer('jane@example.com', '1 Logfire Lane, London'),
        }
    )
    products: dict[str, Product] = field(
        default_factory=lambda: {
            'PYD-MUG': Product('PYD-MUG', 'Pydantic mug', 42),
            'PYD-HOOD': Product('PYD-HOOD', 'Pydantic hoodie', 0),
            'PYD-CAP': Product('PYD-CAP', 'Pydantic cap', 17),
        }
    )


# --- Orders specialist -----------------------------------------------------

orders_tools = FunctionToolset[Store]()


@orders_tools.tool
def order_status(ctx: RunContext[Store], order_id: str) -> str:
    """Look up the current status of an order by its id."""
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'No order found with id {order_id}.'
    return f'Order {order.id} ({order.item}) is currently {order.status}.'


@orders_tools.tool
def orders_for_customer(ctx: RunContext[Store], email: str) -> str:
    """List all orders placed by a customer."""
    matches = [o for o in ctx.deps.orders.values() if o.customer_email == email]
    if not matches:
        return f'No orders found for {email}.'
    return '\n'.join(f'- {o.id}: {o.item} ({o.status})' for o in matches)


orders_capability = Capability[Store](
    id='orders',
    description="Look up order status and a customer's order history.",
    instructions='Call order_status with the full order id (e.g. "A-1042"). Quote the item name in your reply.',
    toolset=orders_tools,
    defer_loading=True,
)


# --- Returns specialist ----------------------------------------------------

returns_tools = FunctionToolset[Store]()


@returns_tools.tool
def start_return(ctx: RunContext[Store], order_id: str, reason: str) -> str:
    """Open a return request for an order. Only delivered orders can be returned."""
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'No order found with id {order_id}.'
    if order.status != 'delivered':
        return f'Order {order.id} is {order.status}; only delivered orders can be returned.'
    return f'Return opened for {order.id}. Reason recorded: {reason!r}.'


returns_capability = Capability[Store](
    id='returns',
    description='Open return requests and answer questions about the return policy.',
    instructions=(
        'Returns are accepted within 30 days of delivery. Items must be unused. '
        'Always confirm the order is delivered before calling start_return.'
    ),
    toolset=returns_tools,
    defer_loading=True,
)


# --- Account specialist ----------------------------------------------------

account_tools = FunctionToolset[Store]()


@account_tools.tool
def update_shipping_address(
    ctx: RunContext[Store], email: str, new_address: str
) -> str:
    """Update a customer's saved shipping address."""
    customer = ctx.deps.customers.get(email)
    if customer is None:
        return f'No customer found with email {email}.'
    customer.shipping_address = new_address
    return f'Shipping address for {email} updated to: {new_address}.'


account_capability = Capability[Store](
    id='account',
    description='Update saved account details such as the shipping address.',
    instructions='Echo back the new value after updating so the customer can confirm it.',
    toolset=account_tools,
    defer_loading=True,
)


# --- Products specialist ---------------------------------------------------

products_tools = FunctionToolset[Store]()


@products_tools.tool
def search_catalog(ctx: RunContext[Store], query: str) -> str:
    """Search the product catalog by name (case-insensitive substring match)."""
    matches = [p for p in ctx.deps.products.values() if query.lower() in p.name.lower()]
    if not matches:
        return f'No products matched {query!r}.'
    return '\n'.join(f'- {p.sku}: {p.name} (stock: {p.stock})' for p in matches)


@products_tools.tool
def check_stock(ctx: RunContext[Store], sku: str) -> str:
    """Check how many units of a SKU are in stock."""
    product = ctx.deps.products.get(sku)
    if product is None:
        return f'No product found with SKU {sku}.'
    return f'{product.name} ({sku}): {product.stock} in stock.'


products_capability = Capability[Store](
    id='products',
    description='Search the product catalog and check stock levels.',
    instructions='Mention the SKU alongside the product name so the customer can reference it.',
    toolset=products_tools,
    defer_loading=True,
)


# --- Agent -----------------------------------------------------------------

support_agent = Agent(
    'openai:gpt-5.2',
    deps_type=Store,
    instructions='You are a customer-support agent for an e-commerce store.',
    capabilities=[
        orders_capability,
        returns_capability,
        account_capability,
        products_capability,
    ],
)


async def main() -> None:
    store = Store()
    prompts = [
        'Where is order A-1042?',
        "I'd like to return order A-1042 — it arrived damaged.",
        'Please change my shipping address to 9 Validator Way, Bristol. '
        'My email is jane@example.com.',
        'Do you have any Pydantic hoodies in stock?',
    ]
    for prompt in prompts:
        print(f'\n> {prompt}')
        result = await support_agent.run(prompt, deps=store)
        print(result.output)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
