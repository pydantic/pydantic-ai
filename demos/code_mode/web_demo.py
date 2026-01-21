"""Code Mode Web Demo

Run this demo to see code mode in action via the web UI.
The demo exposes mock business tools that the agent can call via generated Python code.

Usage:
    uv run python demos/code_mode_web_demo.py

Then open http://localhost:7932 in your browser.

Example prompts to try:
- "Analyze the engineering team's spending and find who saved the most from discounts"
- "Get all products and their inventory across warehouses"
- "List all orders for the engineering team and calculate total revenue"
"""

from __future__ import annotations

from typing import TypedDict

import logfire
import uvicorn

from pydantic_ai import Agent
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset


# =============================================================================
# Mock Data - Business domain data for the demo
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


# Teams
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

# Orders
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

# Discounts
_DISCOUNTS: dict[str, Discount] = {
    'p1': {'product_id': 'p1', 'discount_percent': 10.0, 'min_quantity': 2},
    'p3': {'product_id': 'p3', 'discount_percent': 15.0, 'min_quantity': 3},
    'p4': {'product_id': 'p4', 'discount_percent': 20.0, 'min_quantity': 5},
}

# Inventory
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
# Tool Functions
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
# Agent Setup
# =============================================================================


def create_toolset() -> FunctionToolset[None]:
    """Create the function toolset with all business tools."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_team_members, takes_ctx=False)
    toolset.add_function(get_user_orders, takes_ctx=False)
    toolset.add_function(get_product, takes_ctx=False)
    toolset.add_function(get_discount, takes_ctx=False)
    toolset.add_function(get_inventory, takes_ctx=False)
    return toolset


def create_traditional_agent() -> Agent[None, str]:
    """Create an agent with traditional tool calling."""
    toolset = create_toolset()

    agent: Agent[None, str] = Agent(
        'gateway/anthropic:claude-sonnet-4-5',
        toolsets=[toolset],
        system_prompt="""\
You are a helpful business analyst assistant. You have access to tools for querying
team members, orders, products, discounts, and inventory.

Use the available tools to fetch data and analyze it to answer user questions.
""",
    )

    return agent


def create_code_mode_agent() -> Agent[None, str]:
    """Create an agent configured with code mode."""
    toolset = create_toolset()
    code_toolset: CodeModeToolset[None] = CodeModeToolset(wrapped=toolset)

    agent: Agent[None, str] = Agent(
        'gateway/anthropic:claude-sonnet-4-5',
        toolsets=[code_toolset],
        system_prompt="""\
You are a helpful business analyst assistant. You have access to tools for querying
team members, orders, products, discounts, and inventory.

When asked to analyze data, write Python code that:
1. Fetches all necessary data using the available functions
2. Processes and aggregates the data as needed
3. Returns a structured result (dict or list)

Always complete the ENTIRE analysis in a single code execution.
""",
    )

    return agent


# =============================================================================
# Main
# =============================================================================


if __name__ == '__main__':
    import threading
    import webbrowser

    # Configure Logfire for tracing
    logfire.configure(service_name='code-mode-demo')
    logfire.instrument_pydantic_ai()

    # Create both agents
    traditional_agent = create_traditional_agent()
    code_mode_agent = create_code_mode_agent()

    # Create web apps for both
    traditional_app = traditional_agent.to_web(
        models=['gateway/anthropic:claude-sonnet-4-5'],
    )
    code_mode_app = code_mode_agent.to_web(
        models=['gateway/anthropic:claude-sonnet-4-5'],
    )

    TRADITIONAL_PORT = 7932
    CODE_MODE_PORT = 7933

    print('=' * 60)
    print('Code Mode vs Traditional Mode Demo')
    print('=' * 60)
    print()
    print(f'  Traditional mode: http://localhost:{TRADITIONAL_PORT}')
    print(f'  Code mode:        http://localhost:{CODE_MODE_PORT}')
    print()
    print('Open both URLs in separate browser tabs and compare!')
    print()
    print('=' * 60)
    print('TEST PROMPT (copy and paste into both):')
    print('=' * 60)
    print('''
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
- The product(s) that gave them the biggest discount
''')
    print('=' * 60)
    print('Expected answer: Bob with $150 savings (Monitor, 15% off)')
    print('=' * 60)
    print()
    print('View traces at https://logfire.pydantic.dev')
    print('=' * 60)

    # Run traditional mode server in a background thread
    def run_traditional():
        uvicorn.run(traditional_app, host='127.0.0.1', port=TRADITIONAL_PORT, log_level='warning')

    traditional_thread = threading.Thread(target=run_traditional, daemon=True)
    traditional_thread.start()

    # Run code mode server in main thread
    uvicorn.run(code_mode_app, host='127.0.0.1', port=CODE_MODE_PORT, log_level='info')
