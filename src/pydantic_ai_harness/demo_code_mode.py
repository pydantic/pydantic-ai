"""End-to-end CodeMode demo against Anthropic Claude Sonnet 4.6.

Run with:

    uv pip install anthropic logfire
    ANTHROPIC_API_KEY=... LOGFIRE_TOKEN=... uv run python demo_code_mode.py

Both `anthropic` and `logfire` are optional. Without `LOGFIRE_TOKEN` Logfire
prints structured spans to stdout instead of streaming to the UI.
"""

from __future__ import annotations

import asyncio
import os
import sys

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart

from pydantic_harness import CodeMode

try:
    import logfire
except ImportError:  # pragma: no cover - logfire is optional
    logfire = None  # type: ignore[assignment]


def _configure_logfire() -> None:
    """Configure Logfire if available, otherwise do nothing."""
    if logfire is None:
        return
    logfire.configure(
        service_name='pydantic-harness-demo',
        send_to_logfire='if-token-present',
        console=logfire.ConsoleOptions(verbose=True),
    )
    logfire.instrument_pydantic_ai()
    try:
        logfire.instrument_anthropic()  # pyright: ignore[reportUnknownMemberType]
    except Exception:  # pragma: no cover - anthropic instrumentation is optional
        pass


PRODUCTS: dict[str, dict[str, float | int]] = {
    'apple': {'price': 1.20, 'stock': 50},
    'banana': {'price': 0.50, 'stock': 200},
    'cherry': {'price': 3.00, 'stock': 12},
    'date': {'price': 5.50, 'stock': 8},
    'elderberry': {'price': 7.25, 'stock': 4},
}


def get_price(item: str) -> float:
    """Look up the unit price of an item in dollars."""
    if item not in PRODUCTS:
        raise ValueError(f'Unknown item: {item!r}. Available: {sorted(PRODUCTS)}')
    return float(PRODUCTS[item]['price'])


def get_stock(item: str) -> int:
    """Return the current available stock for an item."""
    if item not in PRODUCTS:
        raise ValueError(f'Unknown item: {item!r}. Available: {sorted(PRODUCTS)}')
    return int(PRODUCTS[item]['stock'])


def apply_discount(amount: float, percent: int) -> float:
    """Apply a percentage discount to a dollar amount and return the new total."""
    return round(amount * (1 - percent / 100), 2)


async def main() -> int:
    """Run the CodeMode demo against Claude Sonnet 4.6."""
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('ANTHROPIC_API_KEY not set', file=sys.stderr)
        return 1

    _configure_logfire()

    agent: Agent[None, str] = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[CodeMode[None]()],
        instructions=(
            'You are a shopping assistant. When the user asks a question that requires '
            'multiple tool calls or arithmetic, use the `run_code` tool to write a single '
            'Python snippet that calls the available functions (remember to `await` them) '
            'and prints the final answer. Then summarise the result for the user.'
        ),
    )

    agent.tool_plain(get_price)
    agent.tool_plain(get_stock)
    agent.tool_plain(apply_discount)

    question = (
        "I'd like to buy 4 apples, 3 bananas, 2 cherries, and 1 date. "
        'For any item where I am ordering more than 25% of available stock, '
        'apply a 15% discount on that item only. '
        'What is my total in dollars?'
    )

    if logfire is not None:
        with logfire.span('code_mode_demo', question=question):
            result = await agent.run(question)
    else:
        result = await agent.run(question)

    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == 'run_code':
                args = part.args if isinstance(part.args, dict) else None
                if args and 'code' in args:
                    print(args['code'])
                    print()

    print(result.output)

    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
