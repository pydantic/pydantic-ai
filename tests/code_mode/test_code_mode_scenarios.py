"""Compelling code mode scenarios demonstrating complex tool orchestration.

These tests show how code mode enables patterns that would require many round-trips
in traditional tool calling.
"""

from __future__ import annotations

import os
from typing import TypedDict

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

pytestmark = [pytest.mark.anyio]


# =============================================================================
# Scenario 1: Cross-referencing / Join-like operation
# "Get team members, fetch each member's orders, find highest spender"
#
# Traditional mode: N+1 round trips (1 for team + N for each member's orders)
# Code mode: 1 round trip with loop + aggregation
# =============================================================================


class TeamMember(TypedDict):
    id: str
    name: str
    role: str


class Order(TypedDict):
    order_id: str
    user_id: str
    amount: float
    status: str


# Mock data
_TEAM_MEMBERS: dict[str, list[TeamMember]] = {
    'engineering': [
        {'id': 'u1', 'name': 'Alice', 'role': 'lead'},
        {'id': 'u2', 'name': 'Bob', 'role': 'senior'},
        {'id': 'u3', 'name': 'Carol', 'role': 'junior'},
    ],
    'sales': [
        {'id': 'u4', 'name': 'Dave', 'role': 'manager'},
        {'id': 'u5', 'name': 'Eve', 'role': 'rep'},
    ],
}

_USER_ORDERS: dict[str, list[Order]] = {
    'u1': [
        {'order_id': 'o1', 'user_id': 'u1', 'amount': 150.0, 'status': 'completed'},
        {'order_id': 'o2', 'user_id': 'u1', 'amount': 75.0, 'status': 'completed'},
    ],
    'u2': [
        {'order_id': 'o3', 'user_id': 'u2', 'amount': 500.0, 'status': 'completed'},
    ],
    'u3': [
        {'order_id': 'o4', 'user_id': 'u3', 'amount': 25.0, 'status': 'pending'},
    ],
    'u4': [
        {'order_id': 'o5', 'user_id': 'u4', 'amount': 1000.0, 'status': 'completed'},
        {'order_id': 'o6', 'user_id': 'u4', 'amount': 250.0, 'status': 'completed'},
    ],
    'u5': [],
}


def get_team_members(team_name: str) -> list[TeamMember]:
    """Get all members of a team.

    Args:
        team_name: Name of the team (e.g., 'engineering', 'sales')

    Returns:
        List of team members with their id, name, and role.
    """
    return _TEAM_MEMBERS.get(team_name, [])


def get_user_orders(user_id: str) -> list[Order]:
    """Get all orders for a specific user.

    Args:
        user_id: The user's unique identifier.

    Returns:
        List of orders with order_id, amount, and status.
    """
    return _USER_ORDERS.get(user_id, [])


# =============================================================================
# Scenario 2: Conditional branching with supplier selection
# "Check inventory. If low, compare supplier prices and pick cheapest"
#
# Traditional mode: Multiple round trips with model reasoning through branches
# Code mode: Clean if/else with nested tool calls in single execution
# =============================================================================


class InventoryStatus(TypedDict):
    product_id: str
    product_name: str
    quantity: int
    reorder_threshold: int


class SupplierQuote(TypedDict):
    supplier_name: str
    product_id: str
    unit_price: float
    lead_time_days: int
    minimum_order: int


_INVENTORY: dict[str, InventoryStatus] = {
    'SKU001': {'product_id': 'SKU001', 'product_name': 'Widget A', 'quantity': 5, 'reorder_threshold': 10},
    'SKU002': {'product_id': 'SKU002', 'product_name': 'Widget B', 'quantity': 100, 'reorder_threshold': 20},
    'SKU003': {'product_id': 'SKU003', 'product_name': 'Gadget X', 'quantity': 3, 'reorder_threshold': 15},
}

_SUPPLIER_QUOTES: dict[str, list[SupplierQuote]] = {
    'SKU001': [
        {'supplier_name': 'Acme Corp', 'product_id': 'SKU001', 'unit_price': 12.50, 'lead_time_days': 5, 'minimum_order': 50},
        {'supplier_name': 'Global Supply', 'product_id': 'SKU001', 'unit_price': 11.00, 'lead_time_days': 7, 'minimum_order': 100},
        {'supplier_name': 'Quick Parts', 'product_id': 'SKU001', 'unit_price': 14.00, 'lead_time_days': 2, 'minimum_order': 25},
    ],
    'SKU003': [
        {'supplier_name': 'Acme Corp', 'product_id': 'SKU003', 'unit_price': 45.00, 'lead_time_days': 5, 'minimum_order': 20},
        {'supplier_name': 'Tech Supplies', 'product_id': 'SKU003', 'unit_price': 42.00, 'lead_time_days': 10, 'minimum_order': 50},
    ],
}


def check_inventory(product_id: str) -> InventoryStatus:
    """Check current inventory level for a product.

    Args:
        product_id: The product SKU to check.

    Returns:
        Inventory status including quantity and reorder threshold.
    """
    return _INVENTORY.get(product_id, {
        'product_id': product_id,
        'product_name': 'Unknown',
        'quantity': 0,
        'reorder_threshold': 0,
    })


def get_supplier_quotes(product_id: str) -> list[SupplierQuote]:
    """Get price quotes from all suppliers for a product.

    Args:
        product_id: The product SKU to get quotes for.

    Returns:
        List of supplier quotes with prices, lead times, and minimum orders.
    """
    return _SUPPLIER_QUOTES.get(product_id, [])


# =============================================================================
# Test fixtures
# =============================================================================


@pytest.fixture
def anthropic_api_key() -> str:
    """Get Anthropic API key from environment."""
    key = os.getenv('ANTHROPIC_API_KEY')
    if not key:
        pytest.skip('ANTHROPIC_API_KEY not set')
    return key


# =============================================================================
# Tests
# =============================================================================


async def test_cross_reference_team_orders(anthropic_api_key: str):
    """Test cross-referencing: get team members then fetch each member's orders.

    This demonstrates a join-like operation:
    1. Fetch team members (1 tool call)
    2. For each member, fetch their orders (N tool calls)
    3. Aggregate to find highest spender

    In traditional mode: N+1 round trips
    In code mode: 1 round trip executing N+1 tool calls internally
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_team_members, takes_ctx=False)
    toolset.add_function(get_user_orders, takes_ctx=False)

    code_mode_toolset = CodeModeToolset(wrapped=toolset)
    agent: Agent[None, str] = Agent(model)

    async with code_mode_toolset:
        result = await agent.run(
            'Find the highest spender in the engineering team. '
            'Get all team members, then fetch each member\'s orders, '
            'sum their completed order amounts, and tell me who spent the most.',
            toolsets=[code_mode_toolset],
        )

    print(f'\n=== Cross-Reference Result ===\n{result.output}')
    print(f'\n=== Messages ===')
    for msg in result.all_messages():
        print(msg)

    # Bob (u2) has the highest spend: $500 from completed orders
    # Alice (u1) has $225, Carol (u3) has $0 completed
    assert result.output is not None
    assert 'bob' in result.output.lower() or '500' in result.output


async def test_conditional_inventory_reorder(anthropic_api_key: str):
    """Test conditional branching: check inventory and conditionally fetch supplier quotes.

    This demonstrates conditional logic:
    1. Check inventory level
    2. If below threshold: fetch supplier quotes and find cheapest
    3. If adequate: just report current stock

    In traditional mode: Model must reason through branches across multiple turns
    In code mode: Single execution with if/else logic
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(check_inventory, takes_ctx=False)
    toolset.add_function(get_supplier_quotes, takes_ctx=False)

    code_mode_toolset = CodeModeToolset(wrapped=toolset)
    agent: Agent[None, str] = Agent(model)

    async with code_mode_toolset:
        result = await agent.run(
            'Check the inventory for SKU001. '
            'If the quantity is below the reorder threshold, get supplier quotes '
            'and recommend the supplier with the lowest unit price. '
            'If inventory is adequate, just report the current stock level.',
            toolsets=[code_mode_toolset],
        )

    print(f'\n=== Conditional Inventory Result ===\n{result.output}')
    print(f'\n=== Messages ===')
    for msg in result.all_messages():
        print(msg)

    # SKU001 has quantity=5, threshold=10, so it's below threshold
    # Cheapest supplier is Global Supply at $11.00
    assert result.output is not None
    # Should mention low inventory and recommend Global Supply (cheapest)
    assert 'global' in result.output.lower() or '11' in result.output


async def test_adequate_inventory_no_quotes(anthropic_api_key: str):
    """Test the other branch: inventory is adequate, no supplier lookup needed.

    SKU002 has quantity=100, threshold=20, so no reorder needed.
    Code mode should skip the supplier quotes entirely.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(check_inventory, takes_ctx=False)
    toolset.add_function(get_supplier_quotes, takes_ctx=False)

    code_mode_toolset = CodeModeToolset(wrapped=toolset)
    agent: Agent[None, str] = Agent(model)

    async with code_mode_toolset:
        result = await agent.run(
            'Check the inventory for SKU002. '
            'If the quantity is below the reorder threshold, get supplier quotes '
            'and recommend the supplier with the lowest unit price. '
            'If inventory is adequate, just report the current stock level.',
            toolsets=[code_mode_toolset],
        )

    print(f'\n=== Adequate Inventory Result ===\n{result.output}')
    print(f'\n=== Messages ===')
    for msg in result.all_messages():
        print(msg)

    # SKU002 has quantity=100, threshold=20, so it's adequate
    assert result.output is not None
    # Should mention adequate stock, quantity 100
    assert '100' in result.output or 'adequate' in result.output.lower() or 'sufficient' in result.output.lower()
