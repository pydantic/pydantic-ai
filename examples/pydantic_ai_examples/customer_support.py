"""Example of a customer support agent for a pizza restaurant with memory, tools, and streaming.

Run with:

    uv run -m pydantic_ai_examples.customer_support
"""

from __future__ import annotations as _annotations

import asyncio
import sqlite3
from collections.abc import AsyncGenerator, Callable
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, LiteralString, TypeVar

import logfire
from typing_extensions import ParamSpec

from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class Order:
    item: str
    quantity: int
    size: str
    order_id: int
    status: str = 'confirmed'


P = ParamSpec('P')
R = TypeVar('R')

THIS_DIR = Path(__file__).parent


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite.

    The SQLite standard library package is synchronous, so we
    use a thread pool executor to run queries asynchronously.
    """

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.customer_support_messages.sqlite'
    ) -> AsyncGenerator[Database]:
        with logfire.span('connect to DB'):
            loop = asyncio.get_running_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            con = await loop.run_in_executor(executor, cls._connect, file)
            slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        con = logfire.instrument_sqlite3(con)
        cur = con.cursor()
        cur.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self._execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            messages,
            commit=True,
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[ModelMessage]:
        c = await self._asyncify(
            self._execute, 'SELECT message_list FROM messages ORDER BY id'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[ModelMessage] = []
        for row in rows:
            messages.extend(ModelMessagesTypeAdapter.validate_json(row[0]))
        return messages

    def _execute(
        self, sql: LiteralString, *args: Any, commit: bool = False
    ) -> sqlite3.Cursor:
        cur = self.con.cursor()
        cur.execute(sql, args)
        if commit:
            self.con.commit()
        return cur

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return await self._loop.run_in_executor(
            self._executor,
            partial(func, **kwargs),
            *args,
        )


@dataclass
class SupportDependencies:
    db: Database
    order_id_counter: int = 1
    orders: dict[int, Order] = field(default_factory=dict)


agent = Agent(
    'openai:gpt-5.2',
    deps_type=SupportDependencies,
    instructions=(
        'You are a customer support agent for "Pydantic Pizza", a pizza restaurant. '
        'Welcome customers and help them place orders. '
        'Use the `get_menu` tool to show the menu, '
        'use `place_order` to place an order, '
        'and use `get_order_status` to check on an existing order. '
        'Be friendly and helpful.'
    ),
)


@agent.tool
async def get_menu(ctx: RunContext[SupportDependencies]) -> str:
    """Get the pizza restaurant menu."""
    return (
        'Menu:\n'
        '- Margherita (small $8, medium $10, large $12)\n'
        '- Pepperoni (small $9, medium $11, large $13)\n'
        '- Vegetarian (small $9, medium $11, large $13)\n'
        '- Hawaiian (small $10, medium $12, large $14)\n'
        '- Meaty (small $11, medium $13, large $15)'
    )


@agent.tool
async def place_order(
    ctx: RunContext[SupportDependencies],
    item: str,
    quantity: int,
    size: str,
) -> str:
    """Place an order for a pizza.

    Args:
        ctx: The run context.
        item: The name of the pizza to order (e.g. Margherita, Pepperoni).
        quantity: The number of pizzas to order.
        size: The size of the pizza (small, medium, or large).
    """
    order_id = ctx.deps.order_id_counter
    ctx.deps.order_id_counter += 1
    order = Order(item=item, quantity=quantity, size=size, order_id=order_id)
    ctx.deps.orders[order_id] = order
    return f'Order placed! Your order ID is {order_id}. You ordered {quantity} x {size} {item} pizza(s).'


@agent.tool
async def get_order_status(ctx: RunContext[SupportDependencies], order_id: int) -> str:
    """Get the status of an order by its order ID.

    Args:
        ctx: The run context.
        order_id: The unique identifier of the order.
    """
    order = ctx.deps.orders.get(order_id)
    if order is None:
        return f'Order {order_id} not found.'
    return f'Order {order_id}: {order.quantity} x {order.size} {order.item} — Status: {order.status}'


async def main():
    async with Database.connect() as db:
        deps = SupportDependencies(db=db)
        print('Welcome to Pydantic Pizza! Type "quit" to exit.\n')
        while True:
            prompt = input('You: ')
            if prompt.strip().lower() == 'quit':
                break
            messages = await db.get_messages()
            async with agent.run_stream(
                prompt, deps=deps, message_history=messages
            ) as result:
                print('Agent: ', end='', flush=True)
                async for text in result.stream_output():
                    print(text, end='', flush=True)
                print()
                await db.add_messages(result.new_messages_json())


if __name__ == '__main__':
    asyncio.run(main())
