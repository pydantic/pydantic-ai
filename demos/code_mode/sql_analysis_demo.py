"""CodeMode Demo: Nested Sales Report with Deep Dependency Chains.

This demo shows how code mode handles deep dependency chains where each query
depends on results from the previous query. Even with parallel tool calling,
traditional mode requires multiple sequential rounds.

Dependency chain (4 levels):
    get_regions() → get_top_sales(region) → get_sale_details(sale_id) → get_customer_info(customer_id)

Traditional mode (with parallel): 4+ rounds (must wait for each level)
Code mode: 1-2 rounds (all in nested loops)

Run:
    uv run python demos/code_mode/sql_analysis_demo.py
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any

import logfire

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic_ai.run import AgentRunResult
from pydantic_ai.runtime.monty import MontyRuntime
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.code_mode import CodeModeToolset

# =============================================================================
# Configuration
# =============================================================================

PROMPT = """
Generate a detailed sales report for the top 3 sales in each region.

Steps:
1. Get all regions
2. For each region, get the top 3 sales (returns sale_id and amount)
3. For each sale, get the sale details (returns product and customer_id)
4. For each sale, get the customer info (returns name and industry)
5. Return a list with each region's top sales including customer details
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 3

# =============================================================================
# Mock Database
# =============================================================================

_regions = ['West', 'East', 'North', 'South']

# Generate sales data
random.seed(42)
_sales: dict[str, list[dict[str, Any]]] = {}
_sale_details: dict[str, dict[str, Any]] = {}
_customers: dict[str, dict[str, Any]] = {}

# Customer pool
_customer_names = [
    ('C001', 'Acme Corp', 'Technology'),
    ('C002', 'GlobalTech', 'Technology'),
    ('C003', 'MegaRetail', 'Retail'),
    ('C004', 'HealthFirst', 'Healthcare'),
    ('C005', 'EduLearn', 'Education'),
    ('C006', 'FinanceHub', 'Finance'),
    ('C007', 'AutoDrive', 'Automotive'),
    ('C008', 'FoodCo', 'Food & Beverage'),
]

for cid, name, industry in _customer_names:
    _customers[cid] = {'customer_id': cid, 'name': name, 'industry': industry}

# Products
_products = ['Widget Pro', 'Gadget Plus', 'Smart Sensor', 'Power Bank', 'USB Hub']

# Generate sales per region
sale_counter = 1
for region in _regions:
    region_sales: list[dict[str, Any]] = []
    for _ in range(20):  # 20 sales per region
        sale_id = f'S{sale_counter:04d}'
        amount = round(random.uniform(500, 5000), 2)
        customer_id = random.choice([c[0] for c in _customer_names])
        product = random.choice(_products)

        region_sales.append({'sale_id': sale_id, 'amount': amount, 'region': region})
        _sale_details[sale_id] = {
            'sale_id': sale_id,
            'product': product,
            'customer_id': customer_id,
            'date': f'2024-{random.randint(10, 12):02d}-{random.randint(1, 28):02d}',
        }
        sale_counter += 1

    # Sort by amount descending
    region_sales.sort(key=lambda x: float(x['amount']), reverse=True)
    _sales[region] = region_sales


# =============================================================================
# Tools - Each level depends on the previous
# =============================================================================


def get_regions() -> list[str]:
    """Get list of all sales regions.

    Returns:
        List of region names.
    """
    return _regions.copy()


def get_top_sales(region: str, limit: int) -> list[dict[str, Any]]:
    """Get the top N sales for a region by amount.

    Args:
        region: The region name.
        limit: Maximum number of sales to return.

    Returns:
        List of sales with sale_id and amount, sorted by amount descending.
    """
    if region not in _sales:
        return []
    sales = _sales[region][:limit]
    return [{'sale_id': s['sale_id'], 'amount': s['amount']} for s in sales]


def get_sale_details(sale_id: str) -> dict[str, Any]:
    """Get detailed information for a sale.

    Args:
        sale_id: The sale ID.

    Returns:
        Sale details including product and customer_id.
    """
    if sale_id not in _sale_details:
        return {'error': f'Sale not found: {sale_id}'}
    return _sale_details[sale_id].copy()


def get_customer_info(customer_id: str) -> dict[str, Any]:
    """Get customer information.

    Args:
        customer_id: The customer ID.

    Returns:
        Customer info including name and industry.
    """
    if customer_id not in _customers:
        return {'error': f'Customer not found: {customer_id}'}
    return _customers[customer_id].copy()


def create_toolset() -> FunctionToolset[None]:
    """Create the toolset with 4-level dependency chain."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_regions)
    toolset.add_function(get_top_sales)
    toolset.add_function(get_sale_details)
    toolset.add_function(get_customer_info)
    return toolset


# =============================================================================
# Agent Factories
# =============================================================================


def create_traditional_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    return Agent(
        MODEL,
        toolsets=[toolset],
        system_prompt='You are a sales analyst. Use the available tools to generate reports.',
    )


def create_code_mode_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with CodeMode (tools as Python functions)."""
    runtime = MontyRuntime()
    code_toolset: CodeModeToolset[None] = CodeModeToolset(
        wrapped=toolset,
        max_retries=MAX_RETRIES,
        runtime=runtime,
    )
    return Agent(
        MODEL,
        toolsets=[code_toolset],
        system_prompt='You are a sales analyst. Use the available tools to generate reports.',
    )


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class RunMetrics:
    """Metrics collected from an agent run."""

    mode: str
    request_count: int
    input_tokens: int
    output_tokens: int
    retry_count: int
    output: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def extract_metrics(result: AgentRunResult[str], mode: str) -> RunMetrics:
    """Extract metrics from agent result."""
    request_count = 0
    input_tokens = 0
    output_tokens = 0
    retry_count = 0

    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            request_count += 1
            if msg.usage:
                input_tokens += msg.usage.input_tokens or 0
                output_tokens += msg.usage.output_tokens or 0
        for part in getattr(msg, 'parts', []):
            if isinstance(part, RetryPromptPart):
                retry_count += 1

    return RunMetrics(
        mode=mode,
        request_count=request_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retry_count=retry_count,
        output=result.output,
    )


# =============================================================================
# Run Functions
# =============================================================================


async def run_traditional(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with traditional tool calling."""
    with logfire.span('traditional_tool_calling'):
        agent = create_traditional_agent(toolset)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'traditional')


async def run_code_mode(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with CodeMode tool calling."""
    with logfire.span('code_mode_tool_calling'):
        agent = create_code_mode_agent(toolset)
        code_toolset = agent.toolsets[0]
        async with code_toolset:
            result = await agent.run(PROMPT)
    return extract_metrics(result, 'code_mode')


# =============================================================================
# Main Demo
# =============================================================================


def print_metrics(metrics: RunMetrics) -> None:
    """Print metrics in formatted table."""
    print(f'  LLM Requests:  {metrics.request_count}')
    print(f'  Input Tokens:  {metrics.input_tokens:,}')
    print(f'  Output Tokens: {metrics.output_tokens:,}')
    print(f'  Total Tokens:  {metrics.total_tokens:,}')
    print(f'  Retries:       {metrics.retry_count}')
    output_preview = metrics.output[:700] + '...' if len(metrics.output) > 700 else metrics.output
    print(f'\n  Output:\n  {output_preview}')


def print_data_stats() -> None:
    """Print statistics about the mock data."""
    total_sales = sum(len(s) for s in _sales.values())
    print(f'\n  Mock Data:')
    print(f'    Regions: {len(_regions)}')
    print(f'    Total Sales: {total_sales}')
    print(f'    Customers: {len(_customers)}')
    print(f'    Products: {len(_products)}')
    print(f'\n  Dependency Chain (4 levels):')
    print(f'    Level 0: get_regions() → {len(_regions)} regions')
    print(f'    Level 1: get_top_sales() × {len(_regions)} = {len(_regions) * 3} sales')
    print(f'    Level 2: get_sale_details() × {len(_regions) * 3} = {len(_regions) * 3} details')
    print(f'    Level 3: get_customer_info() × {len(_regions) * 3} = {len(_regions) * 3} customers')
    print(f'    Total tool calls: 1 + {len(_regions)} + {len(_regions) * 3} + {len(_regions) * 3} = {1 + len(_regions) + len(_regions) * 3 * 2}')


async def main() -> None:
    # Configure Logfire
    logfire.configure(service_name='code-mode-nested-demo')
    logfire.instrument_pydantic_ai()

    print('=' * 70)
    print('CodeMode Demo: Nested Sales Report with Deep Dependency Chains')
    print('=' * 70)
    print(f'\nModel: {MODEL}')
    print('Task: Generate report with 4-level dependency chain')

    print_data_stats()

    toolset = create_toolset()

    # Run Traditional
    print('\n' + '-' * 70)
    print('Running TRADITIONAL tool calling...')
    print('(Even with parallel calls, needs 4+ rounds due to dependencies)')
    print('-' * 70)

    with logfire.span('demo_traditional'):
        trad = await run_traditional(toolset)
    print_metrics(trad)

    # Run CodeMode
    print('\n' + '-' * 70)
    print('Running CODE MODE tool calling...')
    print('(All 29 tool calls in nested loops, 1-2 rounds)')
    print('-' * 70)

    with logfire.span('demo_code_mode'):
        code = await run_code_mode(toolset)
    print_metrics(code)

    # Comparison Summary
    print('\n' + '=' * 70)
    print('COMPARISON SUMMARY')
    print('=' * 70)

    request_reduction = trad.request_count - code.request_count
    if trad.request_count > 0:
        request_pct = request_reduction / trad.request_count * 100
    else:
        request_pct = 0

    token_diff = trad.total_tokens - code.total_tokens
    token_pct = (token_diff / trad.total_tokens * 100) if trad.total_tokens > 0 else 0

    print(
        f'\n  LLM Requests: {trad.request_count} → {code.request_count} '
        f'({request_reduction} fewer, {request_pct:.0f}% reduction)'
    )
    print(
        f'  Total Tokens: {trad.total_tokens:,} → {code.total_tokens:,} '
        f'({token_pct:+.1f}% {"savings" if token_diff > 0 else "increase"})'
    )

    print('\n  Key Insight: Deep dependency chains force sequential rounds.')
    print('               get_regions → get_top_sales → get_sale_details → get_customer_info')
    print('               Traditional mode MUST wait for each level, even with parallel calls.')
    print('               Code mode does all 29 calls in nested loops in ONE code block.')

    print('\n' + '=' * 70)
    print('View detailed traces: https://logfire.pydantic.dev')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
