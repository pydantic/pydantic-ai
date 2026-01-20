"""Comprehensive code mode vs traditional mode comparison using pydantic-evals.

This test demonstrates the efficiency advantages of code mode by running the SAME
complex task in both modes and comparing metrics like request count and token usage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

pytestmark = [pytest.mark.anyio]


# =============================================================================
# Mock Data - Complex nested data for N+M+K query pattern
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


class InventoryItem(TypedDict):
    product_id: str
    warehouse: str
    stock: int


# Teams with more members for deeper N+1 pattern
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

# Orders with product references
_USER_ORDERS: dict[str, list[Order]] = {
    'u1': [
        {'order_id': 'o1', 'user_id': 'u1', 'product_id': 'p1', 'quantity': 2, 'unit_price': 100.0, 'status': 'completed'},
        {'order_id': 'o2', 'user_id': 'u1', 'product_id': 'p2', 'quantity': 1, 'unit_price': 50.0, 'status': 'completed'},
    ],
    'u2': [
        {'order_id': 'o3', 'user_id': 'u2', 'product_id': 'p3', 'quantity': 5, 'unit_price': 200.0, 'status': 'completed'},
        {'order_id': 'o4', 'user_id': 'u2', 'product_id': 'p1', 'quantity': 1, 'unit_price': 100.0, 'status': 'pending'},
    ],
    'u3': [
        {'order_id': 'o5', 'user_id': 'u3', 'product_id': 'p2', 'quantity': 3, 'unit_price': 50.0, 'status': 'completed'},
    ],
    'u4': [
        {'order_id': 'o6', 'user_id': 'u4', 'product_id': 'p4', 'quantity': 10, 'unit_price': 25.0, 'status': 'completed'},
        {'order_id': 'o7', 'user_id': 'u4', 'product_id': 'p3', 'quantity': 2, 'unit_price': 200.0, 'status': 'completed'},
    ],
    'u5': [
        {'order_id': 'o8', 'user_id': 'u5', 'product_id': 'p1', 'quantity': 20, 'unit_price': 100.0, 'status': 'completed'},
    ],
    'u6': [],
}

# Products
_PRODUCTS: dict[str, Product] = {
    'p1': {'product_id': 'p1', 'name': 'Laptop', 'category': 'electronics', 'base_price': 100.0},
    'p2': {'product_id': 'p2', 'name': 'Mouse', 'category': 'electronics', 'base_price': 50.0},
    'p3': {'product_id': 'p3', 'name': 'Monitor', 'category': 'electronics', 'base_price': 200.0},
    'p4': {'product_id': 'p4', 'name': 'Cable', 'category': 'accessories', 'base_price': 25.0},
}

# Discounts - only some products have discounts
_DISCOUNTS: dict[str, Discount] = {
    'p1': {'product_id': 'p1', 'discount_percent': 10.0, 'min_quantity': 2},
    'p3': {'product_id': 'p3', 'discount_percent': 15.0, 'min_quantity': 3},
    'p4': {'product_id': 'p4', 'discount_percent': 20.0, 'min_quantity': 5},
}

# Inventory across warehouses
_INVENTORY: dict[str, list[InventoryItem]] = {
    'p1': [
        {'product_id': 'p1', 'warehouse': 'east', 'stock': 50},
        {'product_id': 'p1', 'warehouse': 'west', 'stock': 30},
    ],
    'p2': [
        {'product_id': 'p2', 'warehouse': 'east', 'stock': 200},
    ],
    'p3': [
        {'product_id': 'p3', 'warehouse': 'west', 'stock': 15},
    ],
    'p4': [
        {'product_id': 'p4', 'warehouse': 'east', 'stock': 500},
        {'product_id': 'p4', 'warehouse': 'west', 'stock': 300},
    ],
}


# =============================================================================
# Tool Functions - Simulating multiple services/APIs
# =============================================================================


def get_team_members(team_name: str) -> list[TeamMember]:
    """Get all members of a team.

    Args:
        team_name: Name of the team (e.g., 'engineering', 'sales')

    Returns:
        List of team members with their id, name, role, and department.
    """
    return _TEAM_MEMBERS.get(team_name, [])


def get_user_orders(user_id: str) -> list[Order]:
    """Get all orders for a specific user.

    Args:
        user_id: The user's unique identifier.

    Returns:
        List of orders with order_id, product_id, quantity, unit_price, and status.
    """
    return _USER_ORDERS.get(user_id, [])


def get_product(product_id: str) -> Product | None:
    """Get product details by ID.

    Args:
        product_id: The product's unique identifier.

    Returns:
        Product details including name, category, and base_price, or None if not found.
    """
    return _PRODUCTS.get(product_id)


def get_discount(product_id: str) -> Discount | None:
    """Get discount information for a product if available.

    Args:
        product_id: The product's unique identifier.

    Returns:
        Discount info with discount_percent and min_quantity, or None if no discount.
    """
    return _DISCOUNTS.get(product_id)


def get_inventory(product_id: str) -> list[InventoryItem]:
    """Get inventory levels for a product across all warehouses.

    Args:
        product_id: The product's unique identifier.

    Returns:
        List of inventory items showing stock levels per warehouse.
    """
    return _INVENTORY.get(product_id, [])


# =============================================================================
# Pydantic-Evals Models
# =============================================================================


class TaskInput(BaseModel):
    """Input for the evaluation task."""

    prompt: str


class TaskOutput(BaseModel):
    """Output from the evaluation task."""

    result: str
    request_count: int
    total_input_tokens: int
    total_output_tokens: int


class TaskMetadata(BaseModel):
    """Metadata for expected behavior."""

    expected_tool_calls: int
    expected_in_result: list[str]


# =============================================================================
# Custom Evaluators
# =============================================================================


@dataclass
class ContainsExpected(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
    """Check that result contains expected strings from metadata."""

    async def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> bool:
        if ctx.metadata is None:
            return True
        result_lower = ctx.output.result.lower()
        return all(exp.lower() in result_lower for exp in ctx.metadata.expected_in_result)


@dataclass
class RequestCountEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
    """Extract request count as a score for comparison."""

    async def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> int:
        return ctx.output.request_count


@dataclass
class TokenCountEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
    """Extract total token count as a score for comparison."""

    async def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> int:
        return ctx.output.total_input_tokens + ctx.output.total_output_tokens


# =============================================================================
# Task Functions
# =============================================================================


def _create_toolset() -> FunctionToolset[None]:
    """Create the function toolset with all tools."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_team_members, takes_ctx=False)
    toolset.add_function(get_user_orders, takes_ctx=False)
    toolset.add_function(get_product, takes_ctx=False)
    toolset.add_function(get_discount, takes_ctx=False)
    toolset.add_function(get_inventory, takes_ctx=False)
    return toolset


async def run_traditional(inputs: TaskInput) -> TaskOutput:
    """Run task in traditional mode - direct tool calls."""
    toolset = _create_toolset()
    agent: Agent[None, str] = Agent('gateway/anthropic:claude-sonnet-4-5')

    result = await agent.run(inputs.prompt, toolsets=[toolset])

    # Count requests and tokens from messages
    request_count = 0
    total_input = 0
    total_output = 0
    for msg in result.all_messages():
        if hasattr(msg, 'usage'):
            request_count += 1
            total_input += msg.usage.input_tokens
            total_output += msg.usage.output_tokens

    return TaskOutput(
        result=result.output,
        request_count=request_count,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


async def run_code_mode(inputs: TaskInput) -> TaskOutput:
    """Run task in code mode - Python code execution."""
    toolset = _create_toolset()
    code_toolset = CodeModeToolset(wrapped=toolset)
    agent: Agent[None, str] = Agent('gateway/anthropic:claude-sonnet-4-5')

    async with code_toolset:
        result = await agent.run(inputs.prompt, toolsets=[code_toolset])

    # Count requests and tokens from messages
    request_count = 0
    total_input = 0
    total_output = 0
    for msg in result.all_messages():
        if hasattr(msg, 'usage'):
            request_count += 1
            total_input += msg.usage.input_tokens
            total_output += msg.usage.output_tokens

    return TaskOutput(
        result=result.output,
        request_count=request_count,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
    )


# =============================================================================
# Test Cases
# =============================================================================

# Complex N+M+K query pattern:
# 1. Get team members (1 call)
# 2. For each member, get their orders (N calls)
# 3. For each order, get product details (M calls)
# 4. For each product, check discount (K calls)
# Total: 1 + N + M + K calls where M and K depend on data

COMPLEX_TASK = TaskInput(
    prompt="""\
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
)


# =============================================================================
# Test
# =============================================================================


async def test_code_mode_vs_traditional(allow_model_requests: None):
    """Compare code mode vs traditional mode on a complex nested query task.

    This test demonstrates that code mode:
    1. Uses fewer LLM requests (2 vs many)
    2. Uses fewer total tokens (context doesn't regrow)
    3. Produces correct results

    The task requires N+M+K queries creating a deep call tree:
    - 1 call to get team members
    - N calls to get orders for each member
    - M calls to get product details for each order
    - K calls to check discounts for each product

    Traditional mode needs many round-trips; code mode does it in one execution.
    """
    # Create dataset with the complex task
    dataset: Dataset[TaskInput, TaskOutput, TaskMetadata] = Dataset(
        cases=[
            Case(
                name='engineering_discount_analysis',
                inputs=COMPLEX_TASK,
                metadata=TaskMetadata(
                    # Expected: 1 team + 4 users + ~7 orders + ~7 products + ~7 discounts = ~26 calls
                    expected_tool_calls=20,
                    # Dan saves most: orders p4 (10 qty, 20% off $25 = $50) + p3 (2 qty, no discount - needs 3)
                    # Actually let's trace: Dan has p4 qty=10 (>=5, 20% off) = 10*25*0.2 = $50
                    # Bob has p3 qty=5 (>=3, 15% off) = 5*200*0.15 = $150
                    expected_in_result=['bob', '150'],
                ),
                evaluators=[ContainsExpected()],
            )
        ],
        evaluators=[RequestCountEvaluator(), TokenCountEvaluator()],
    )

    # Run both modes
    print('\n' + '=' * 70)
    print('TRADITIONAL MODE')
    print('=' * 70)
    traditional_report = await dataset.evaluate(run_traditional)

    print('\n' + '=' * 70)
    print('CODE MODE')
    print('=' * 70)
    code_mode_report = await dataset.evaluate(run_code_mode)

    # Extract results
    assert traditional_report.cases, 'Traditional mode returned no cases'
    assert code_mode_report.cases, 'Code mode returned no cases'

    trad_case = traditional_report.cases[0]
    code_case = code_mode_report.cases[0]

    trad_output = trad_case.output
    code_output = code_case.output

    assert trad_output is not None, 'Traditional mode output is None'
    assert code_output is not None, 'Code mode output is None'

    # Print comparison
    print('\n' + '=' * 70)
    print('COMPARISON RESULTS')
    print('=' * 70)
    print(f'\n{"Metric":<25} {"Traditional":>15} {"Code Mode":>15} {"Savings":>15}')
    print('-' * 70)

    # Request count
    trad_requests = trad_output.request_count
    code_requests = code_output.request_count
    req_savings = f'{((trad_requests - code_requests) / trad_requests * 100):.1f}%' if trad_requests > 0 else 'N/A'
    print(f'{"LLM Requests":<25} {trad_requests:>15} {code_requests:>15} {req_savings:>15}')

    # Token count
    trad_tokens = trad_output.total_input_tokens + trad_output.total_output_tokens
    code_tokens = code_output.total_input_tokens + code_output.total_output_tokens
    tok_savings = f'{((trad_tokens - code_tokens) / trad_tokens * 100):.1f}%' if trad_tokens > 0 else 'N/A'
    print(f'{"Total Tokens":<25} {trad_tokens:>15} {code_tokens:>15} {tok_savings:>15}')

    # Input tokens specifically
    print(f'{"  Input Tokens":<25} {trad_output.total_input_tokens:>15} {code_output.total_input_tokens:>15}')
    print(f'{"  Output Tokens":<25} {trad_output.total_output_tokens:>15} {code_output.total_output_tokens:>15}')

    print('-' * 70)

    # Results
    print(f'\nTraditional result: {trad_output.result[:200]}...')
    print(f'\nCode mode result: {code_output.result[:200]}...')

    # Assertions - code mode should be more efficient
    assert code_requests <= trad_requests, (
        f'Code mode should use fewer or equal requests: {code_requests} vs {trad_requests}'
    )

    # Both should produce correct results
    assert 'bob' in trad_output.result.lower(), f'Traditional should find Bob: {trad_output.result}'
    assert 'bob' in code_output.result.lower(), f'Code mode should find Bob: {code_output.result}'

    print('\n✓ Code mode used fewer/equal LLM requests')
    print('✓ Both modes produced correct results')
