# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai[logfire]",
# ]
# ///
"""Run Code Mode vs Traditional Mode evals with Logfire instrumentation.

This script runs the same complex task in both modes multiple times,
with Logfire tracing enabled for later analysis.

Usage:
    source .env && uv run python demos/code_mode/run_evals.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import TypedDict

import logfire

from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.messages import ModelResponse
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset


# =============================================================================
# Mock Data - Same as eval test
# =============================================================================


class TeamMember(TypedDict):
    id: str
    name: str
    role: str
    department: str


class Order(TypedDict):
    order_id: str
    user_id: str
    product_id: str
    quantity: int
    unit_price: float
    status: str


class Product(TypedDict):
    product_id: str
    name: str
    category: str
    base_price: float


class Discount(TypedDict):
    product_id: str
    discount_percent: float
    min_quantity: int


class EvalRunResult(TypedDict):
    mode: str
    run: int
    request_count: int
    input_tokens: int
    output_tokens: int
    correct: bool


_TEAM_MEMBERS: dict[str, list[TeamMember]] = {
    'engineering': [
        {'id': 'u1', 'name': 'Alice', 'role': 'lead', 'department': 'backend'},
        {'id': 'u2', 'name': 'Bob', 'role': 'senior', 'department': 'frontend'},
        {'id': 'u3', 'name': 'Carol', 'role': 'junior', 'department': 'backend'},
        {'id': 'u4', 'name': 'Dan', 'role': 'senior', 'department': 'devops'},
    ],
    'sales': [
        {'id': 'u5', 'name': 'Eve', 'role': 'manager', 'department': 'enterprise'},
        {'id': 'u6', 'name': 'Frank', 'role': 'rep', 'department': 'smb'},
    ],
}

_USER_ORDERS: dict[str, list[Order]] = {
    'u1': [
        {
            'order_id': 'o1',
            'user_id': 'u1',
            'product_id': 'p1',
            'quantity': 2,
            'unit_price': 100.0,
            'status': 'completed',
        },
        {
            'order_id': 'o2',
            'user_id': 'u1',
            'product_id': 'p2',
            'quantity': 1,
            'unit_price': 50.0,
            'status': 'completed',
        },
    ],
    'u2': [
        {
            'order_id': 'o3',
            'user_id': 'u2',
            'product_id': 'p3',
            'quantity': 5,
            'unit_price': 200.0,
            'status': 'completed',
        },
        {
            'order_id': 'o4',
            'user_id': 'u2',
            'product_id': 'p1',
            'quantity': 1,
            'unit_price': 100.0,
            'status': 'pending',
        },
    ],
    'u3': [
        {
            'order_id': 'o5',
            'user_id': 'u3',
            'product_id': 'p2',
            'quantity': 3,
            'unit_price': 50.0,
            'status': 'completed',
        },
    ],
    'u4': [
        {
            'order_id': 'o6',
            'user_id': 'u4',
            'product_id': 'p4',
            'quantity': 10,
            'unit_price': 25.0,
            'status': 'completed',
        },
        {
            'order_id': 'o7',
            'user_id': 'u4',
            'product_id': 'p3',
            'quantity': 2,
            'unit_price': 200.0,
            'status': 'completed',
        },
    ],
    'u5': [
        {
            'order_id': 'o8',
            'user_id': 'u5',
            'product_id': 'p1',
            'quantity': 20,
            'unit_price': 100.0,
            'status': 'completed',
        },
    ],
    'u6': [],
}

_PRODUCTS: dict[str, Product] = {
    'p1': {'product_id': 'p1', 'name': 'Laptop', 'category': 'electronics', 'base_price': 100.0},
    'p2': {'product_id': 'p2', 'name': 'Mouse', 'category': 'electronics', 'base_price': 50.0},
    'p3': {'product_id': 'p3', 'name': 'Monitor', 'category': 'electronics', 'base_price': 200.0},
    'p4': {'product_id': 'p4', 'name': 'Cable', 'category': 'accessories', 'base_price': 25.0},
}

_DISCOUNTS: dict[str, Discount] = {
    'p1': {'product_id': 'p1', 'discount_percent': 10.0, 'min_quantity': 2},
    'p3': {'product_id': 'p3', 'discount_percent': 15.0, 'min_quantity': 3},
    'p4': {'product_id': 'p4', 'discount_percent': 20.0, 'min_quantity': 5},
}


# =============================================================================
# Tool Functions
# =============================================================================


def get_team_members(team_name: str) -> list[TeamMember]:
    """Get all members of a team."""
    return _TEAM_MEMBERS.get(team_name, [])


def get_user_orders(user_id: str) -> list[Order]:
    """Get all orders for a specific user."""
    return _USER_ORDERS.get(user_id, [])


def get_product(product_id: str) -> Product | None:
    """Get product details by ID."""
    return _PRODUCTS.get(product_id)


def get_discount(product_id: str) -> Discount | None:
    """Get discount information for a product if available."""
    return _DISCOUNTS.get(product_id)


# =============================================================================
# Eval Runner
# =============================================================================


PROMPT = """\
Analyze the engineering team's spending with discount savings.

Steps:
1. Get all team members from the 'engineering' team
2. For each team member, fetch their completed orders
3. For each order, get the product details and check if there's a discount
4. Calculate the discount amount for orders that qualify (quantity >= min_quantity)
5. Find the team member who saved the most money from discounts

Return:
- The name of the person who saved the most from discounts
- Their total discount savings amount
- The product(s) that gave them the biggest discount"""


def _create_toolset() -> FunctionToolset[None]:
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_team_members, takes_ctx=False)
    toolset.add_function(get_user_orders, takes_ctx=False)
    toolset.add_function(get_product, takes_ctx=False)
    toolset.add_function(get_discount, takes_ctx=False)
    return toolset


async def run_traditional(run_number: int) -> EvalRunResult:
    """Run task in traditional mode."""
    toolset = _create_toolset()
    agent: Agent[None, str] = Agent('gateway/anthropic:claude-sonnet-4-5')

    with logfire.span('eval_run', mode='traditional', run_number=run_number):
        result = await agent.run(PROMPT, toolsets=[toolset])

    return _extract_eval_metrics(result, 'traditional', run_number)


async def run_code_mode(run_number: int) -> EvalRunResult:
    """Run task in code mode."""
    toolset = _create_toolset()
    code_toolset = CodeModeToolset(wrapped=toolset)
    agent: Agent[None, str] = Agent('gateway/anthropic:claude-sonnet-4-5')

    with logfire.span('eval_run', mode='code_mode', run_number=run_number):
        async with code_toolset:
            result = await agent.run(PROMPT, toolsets=[code_toolset])

    return _extract_eval_metrics(result, 'code_mode', run_number)


def _extract_eval_metrics(result: AgentRunResult[str], mode: str, run_number: int) -> EvalRunResult:
    """Extract metrics from agent result."""
    request_count = 0
    total_input = 0
    total_output = 0
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            request_count += 1
            total_input += msg.usage.input_tokens
            total_output += msg.usage.output_tokens

    correct = 'bob' in result.output.lower() and '150' in result.output

    return EvalRunResult(
        mode=mode,
        run=run_number,
        request_count=request_count,
        input_tokens=total_input,
        output_tokens=total_output,
        correct=correct,
    )


async def main():
    num_runs = 10

    print('=' * 70)
    print('Code Mode vs Traditional Mode Evaluation')
    print(f'Running {num_runs} iterations of each mode')
    print('=' * 70)
    print()

    all_results: list[EvalRunResult] = []

    # Run traditional mode
    print('TRADITIONAL MODE')
    print('-' * 40)
    for i in range(1, num_runs + 1):
        print(f'  Run {i}/{num_runs}...', end=' ', flush=True)
        result = await run_traditional(i)
        all_results.append(result)
        status = 'correct' if result['correct'] else 'WRONG'
        print(
            f'requests={result["request_count"]}, tokens={result["input_tokens"] + result["output_tokens"]}, {status}'
        )

    print()

    # Run code mode
    print('CODE MODE')
    print('-' * 40)
    for i in range(1, num_runs + 1):
        print(f'  Run {i}/{num_runs}...', end=' ', flush=True)
        result = await run_code_mode(i)
        all_results.append(result)
        status = 'correct' if result['correct'] else 'WRONG'
        print(
            f'requests={result["request_count"]}, tokens={result["input_tokens"] + result["output_tokens"]}, {status}'
        )

    print()

    # Summary
    trad_results = [r for r in all_results if r['mode'] == 'traditional']
    code_results = [r for r in all_results if r['mode'] == 'code_mode']

    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)
    print(f'{"Metric":<25} {"Traditional":>15} {"Code Mode":>15} {"Savings":>15}')
    print('-' * 70)

    trad_req = sum(r['request_count'] for r in trad_results) / len(trad_results) if trad_results else 0.0
    code_req = sum(r['request_count'] for r in code_results) / len(code_results) if code_results else 0.0
    req_savings = f'{((trad_req - code_req) / trad_req * 100):.1f}%' if trad_req > 0 else 'N/A'
    print(f'{"Avg Requests":<25} {trad_req:>15.1f} {code_req:>15.1f} {req_savings:>15}')

    trad_in = sum(r['input_tokens'] for r in trad_results) / len(trad_results) if trad_results else 0.0
    trad_out = sum(r['output_tokens'] for r in trad_results) / len(trad_results) if trad_results else 0.0
    code_in = sum(r['input_tokens'] for r in code_results) / len(code_results) if code_results else 0.0
    code_out = sum(r['output_tokens'] for r in code_results) / len(code_results) if code_results else 0.0
    trad_tokens = trad_in + trad_out
    code_tokens = code_in + code_out
    tok_savings = f'{((trad_tokens - code_tokens) / trad_tokens * 100):.1f}%' if trad_tokens > 0 else 'N/A'
    print(f'{"Avg Total Tokens":<25} {trad_tokens:>15.1f} {code_tokens:>15.1f} {tok_savings:>15}')

    trad_correct = sum(1 for r in trad_results if r['correct'])
    code_correct = sum(1 for r in code_results if r['correct'])
    print(f'{"Correct Answers":<25} {trad_correct:>15}/{num_runs} {code_correct:>15}/{num_runs}')

    print('-' * 70)
    print()
    print('View traces at https://logfire.pydantic.dev')
    print()


if __name__ == '__main__':
    # Configure Logfire
    logfire.configure(service_name='code-mode-eval')
    logfire.instrument_pydantic_ai()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nInterrupted')
        sys.exit(1)
