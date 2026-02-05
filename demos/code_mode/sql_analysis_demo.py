"""CodeMode Demo: Regional Sales SQL Analysis.

This demo shows how code mode reduces context bloat when processing large datasets.
With 400+ sales records across 4 regions, traditional mode must send all data through
the LLM context. Code mode processes it in a loop, returning only summaries.

Traditional mode: All 400+ records flow through LLM context
Code mode: Loop processes records, returns only totals

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
Analyze Q4 2024 sales across all regions.

Steps:
1. Get all regions
2. For each region, query Q4 sales
3. Calculate total revenue by summing all transaction amounts
4. Count transactions
5. If revenue > 50000, check bonus rules
6. Append each region's summary to a results list
7. Return total revenue and the results list
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 3

# =============================================================================
# Mock SQL Database - Large Dataset
# =============================================================================

_regions = ['West', 'East', 'North', 'South']

# Bonus rules (only some regions have them)
_bonus_rules = {
    'West': {'threshold': 50000, 'bonus_pct': 0.05},
    'East': {'threshold': 50000, 'bonus_pct': 0.04},
}


def _generate_sales_data() -> dict[str, list[dict[str, Any]]]:
    """Generate large sales dataset - 100+ records per region."""
    random.seed(42)
    sales: dict[str, list[dict[str, Any]]] = {}

    # Revenue targets to ensure interesting results
    revenue_targets: dict[str, int] = {'West': 65000, 'East': 55000, 'North': 45000, 'South': 70000}

    for region in _regions:
        region_sales: list[dict[str, Any]] = []
        target = revenue_targets[region]
        current_total: float = 0
        sale_id = 1

        # Generate 100+ sales per region to bloat traditional mode context
        while len(region_sales) < 100 or current_total < target * 0.95:
            amount = round(random.uniform(100, 2000), 2)

            region_sales.append(
                {
                    'id': f'{region[0]}{sale_id:04d}',
                    'amount': amount,
                    'date': f'2024-{random.randint(10, 12):02d}-{random.randint(1, 28):02d}',
                }
            )

            current_total += amount
            sale_id += 1

        sales[region] = region_sales

    return sales


_sales = _generate_sales_data()


# =============================================================================
# Tools
# =============================================================================


def get_regions() -> list[str]:
    """Get list of all sales regions.

    Returns:
        List of region names.
    """
    return _regions.copy()


def query_sales(region: str, quarter: str) -> dict[str, Any]:
    """Query sales transactions for a region and quarter.

    Args:
        region: The region name.
        quarter: The quarter (e.g., "Q4").

    Returns:
        Dictionary with transaction records. Each has: id, amount, date.
    """
    if region not in _sales:
        return {'error': f'Unknown region: {region}', 'transactions': []}

    quarter_months = {
        'Q1': ['01', '02', '03'],
        'Q2': ['04', '05', '06'],
        'Q3': ['07', '08', '09'],
        'Q4': ['10', '11', '12'],
    }

    months = quarter_months.get(quarter, [])
    transactions = [s for s in _sales[region] if s['date'][5:7] in months]

    return {'region': region, 'quarter': quarter, 'transactions': transactions}


def get_bonus_rules(region: str) -> dict[str, Any] | None:
    """Get bonus eligibility rules for a region.

    Args:
        region: The region name.

    Returns:
        Bonus rules (threshold and bonus_pct) or None if no rules exist.
    """
    return _bonus_rules.get(region)


def create_toolset() -> FunctionToolset[None]:
    """Create the SQL analysis toolset."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_regions)
    toolset.add_function(query_sales)
    toolset.add_function(get_bonus_rules)
    return toolset


# =============================================================================
# Agent Factories
# =============================================================================


def create_tool_calling_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with standard tool calling."""
    return Agent(
        MODEL,
        toolsets=[toolset],
        system_prompt='You are a sales analyst. Use the available tools to analyze regional sales data.',
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
        system_prompt='You are a sales analyst. Use the available tools to analyze regional sales data.',
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


async def run_tool_calling(toolset: FunctionToolset[None]) -> RunMetrics:
    """Run with standard tool calling."""
    with logfire.span('tool_calling'):
        agent = create_tool_calling_agent(toolset)
        result = await agent.run(PROMPT)
    return extract_metrics(result, 'tool_calling')


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


def log_metrics(metrics: RunMetrics) -> None:
    """Log metrics to logfire."""
    logfire.info(
        '{mode} completed: {requests} requests, {tokens} tokens',
        mode=metrics.mode,
        requests=metrics.request_count,
        tokens=metrics.total_tokens,
        input_tokens=metrics.input_tokens,
        output_tokens=metrics.output_tokens,
        retries=metrics.retry_count,
    )


async def main() -> None:
    """Run the demo."""
    logfire.configure(service_name='code-mode-sql-demo')
    logfire.instrument_pydantic_ai()

    toolset = create_toolset()

    with logfire.span('demo_tool_calling'):
        trad = await run_tool_calling(toolset)
    log_metrics(trad)

    with logfire.span('demo_code_mode'):
        code = await run_code_mode(toolset)
    log_metrics(code)

    print('View traces: https://logfire.pydantic.dev')


if __name__ == '__main__':
    asyncio.run(main())
