"""CodeMode Demo: Team Expense Analysis.

This demo shows how code mode reduces token consumption when analyzing data.
With traditional tool calling, all expense data flows through the LLM context.
With code mode, aggregation happens in code and only summaries are returned.

Inspired by Anthropic's Programmatic Tool Calling example.

Run:
    uv run python demos/code_mode/expense_analysis_demo.py
"""

from __future__ import annotations

import asyncio
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
Analyze the Q3 travel expenses for the Engineering team.

For each team member:
1. Get their expenses for Q3 (category: travel)
2. Sum up their total
3. Check if they exceeded the standard budget of $5000
4. If over standard budget, check if they have a custom budget

Return:
- Total team members analyzed
- How many exceeded their budget
- List of those over budget with: name, total spent, their budget, amount over
"""

MODEL = 'gateway/anthropic:claude-sonnet-4-5'
MAX_RETRIES = 3

# =============================================================================
# Mock Expense Data
# =============================================================================

# Simulated team members
_team_members = [
    {'id': 1, 'name': 'Alice Chen'},
    {'id': 2, 'name': 'Bob Smith'},
    {'id': 3, 'name': 'Carol Jones'},
    {'id': 4, 'name': 'David Kim'},
    {'id': 5, 'name': 'Eve Wilson'},
]

# Simulated expense data (multiple line items per person to bloat traditional context)
_expenses = {
    1: [  # Alice - under budget
        {'date': '2024-07-15', 'amount': 450.00, 'description': 'Flight to NYC'},
        {'date': '2024-07-16', 'amount': 200.00, 'description': 'Hotel NYC'},
        {'date': '2024-07-17', 'amount': 85.00, 'description': 'Meals NYC'},
        {'date': '2024-08-20', 'amount': 380.00, 'description': 'Flight to Chicago'},
        {'date': '2024-08-21', 'amount': 175.00, 'description': 'Hotel Chicago'},
        {'date': '2024-09-05', 'amount': 520.00, 'description': 'Flight to Seattle'},
        {'date': '2024-09-06', 'amount': 225.00, 'description': 'Hotel Seattle'},
        {'date': '2024-09-07', 'amount': 95.00, 'description': 'Meals Seattle'},
    ],
    2: [  # Bob - over standard budget but has custom budget
        {'date': '2024-07-01', 'amount': 850.00, 'description': 'Flight to London'},
        {'date': '2024-07-02', 'amount': 450.00, 'description': 'Hotel London'},
        {'date': '2024-07-03', 'amount': 125.00, 'description': 'Meals London'},
        {'date': '2024-07-04', 'amount': 450.00, 'description': 'Hotel London'},
        {'date': '2024-07-05', 'amount': 120.00, 'description': 'Meals London'},
        {'date': '2024-08-10', 'amount': 780.00, 'description': 'Flight to Tokyo'},
        {'date': '2024-08-11', 'amount': 380.00, 'description': 'Hotel Tokyo'},
        {'date': '2024-08-12', 'amount': 380.00, 'description': 'Hotel Tokyo'},
        {'date': '2024-08-13', 'amount': 150.00, 'description': 'Meals Tokyo'},
        {'date': '2024-09-15', 'amount': 920.00, 'description': 'Flight to Singapore'},
        {'date': '2024-09-16', 'amount': 320.00, 'description': 'Hotel Singapore'},
        {'date': '2024-09-17', 'amount': 320.00, 'description': 'Hotel Singapore'},
        {'date': '2024-09-18', 'amount': 180.00, 'description': 'Meals Singapore'},
    ],
    3: [  # Carol - way over budget (no custom budget)
        {'date': '2024-07-08', 'amount': 1200.00, 'description': 'Flight to Paris'},
        {'date': '2024-07-09', 'amount': 550.00, 'description': 'Hotel Paris'},
        {'date': '2024-07-10', 'amount': 550.00, 'description': 'Hotel Paris'},
        {'date': '2024-07-11', 'amount': 550.00, 'description': 'Hotel Paris'},
        {'date': '2024-07-12', 'amount': 200.00, 'description': 'Meals Paris'},
        {'date': '2024-08-25', 'amount': 1100.00, 'description': 'Flight to Sydney'},
        {'date': '2024-08-26', 'amount': 480.00, 'description': 'Hotel Sydney'},
        {'date': '2024-08-27', 'amount': 480.00, 'description': 'Hotel Sydney'},
        {'date': '2024-08-28', 'amount': 480.00, 'description': 'Hotel Sydney'},
        {'date': '2024-08-29', 'amount': 220.00, 'description': 'Meals Sydney'},
        {'date': '2024-09-20', 'amount': 650.00, 'description': 'Flight to Denver'},
        {'date': '2024-09-21', 'amount': 280.00, 'description': 'Hotel Denver'},
    ],
    4: [  # David - slightly under budget
        {'date': '2024-07-22', 'amount': 420.00, 'description': 'Flight to Boston'},
        {'date': '2024-07-23', 'amount': 190.00, 'description': 'Hotel Boston'},
        {'date': '2024-07-24', 'amount': 75.00, 'description': 'Meals Boston'},
        {'date': '2024-08-05', 'amount': 510.00, 'description': 'Flight to Austin'},
        {'date': '2024-08-06', 'amount': 210.00, 'description': 'Hotel Austin'},
        {'date': '2024-08-07', 'amount': 90.00, 'description': 'Meals Austin'},
        {'date': '2024-09-12', 'amount': 480.00, 'description': 'Flight to Portland'},
        {'date': '2024-09-13', 'amount': 195.00, 'description': 'Hotel Portland'},
        {'date': '2024-09-14', 'amount': 85.00, 'description': 'Meals Portland'},
    ],
    5: [  # Eve - over standard budget (no custom budget)
        {'date': '2024-07-03', 'amount': 680.00, 'description': 'Flight to Miami'},
        {'date': '2024-07-04', 'amount': 320.00, 'description': 'Hotel Miami'},
        {'date': '2024-07-05', 'amount': 320.00, 'description': 'Hotel Miami'},
        {'date': '2024-07-06', 'amount': 145.00, 'description': 'Meals Miami'},
        {'date': '2024-08-18', 'amount': 750.00, 'description': 'Flight to San Diego'},
        {'date': '2024-08-19', 'amount': 290.00, 'description': 'Hotel San Diego'},
        {'date': '2024-08-20', 'amount': 290.00, 'description': 'Hotel San Diego'},
        {'date': '2024-08-21', 'amount': 130.00, 'description': 'Meals San Diego'},
        {'date': '2024-09-08', 'amount': 820.00, 'description': 'Flight to Las Vegas'},
        {'date': '2024-09-09', 'amount': 380.00, 'description': 'Hotel Las Vegas'},
        {'date': '2024-09-10', 'amount': 380.00, 'description': 'Hotel Las Vegas'},
        {'date': '2024-09-11', 'amount': 175.00, 'description': 'Meals Las Vegas'},
    ],
}

# Custom budgets (only Bob has one)
_custom_budgets = {
    2: {'amount': 7000.00, 'reason': 'International travel required'},
}


# =============================================================================
# Mock Expense Tools
# =============================================================================


def get_team_members(department: str) -> dict[str, Any]:
    """Get list of team members for a department.

    Args:
        department: The department name (e.g., "Engineering").

    Returns:
        Dictionary with list of team members.
    """
    return {'department': department, 'members': _team_members}


def get_expenses(user_id: int, quarter: str, category: str) -> dict[str, Any]:
    """Get expense line items for a user.

    Args:
        user_id: The user's ID.
        quarter: The quarter (e.g., "Q3").
        category: The expense category (e.g., "travel").

    Returns:
        Dictionary with expense items.
    """
    items = _expenses.get(user_id, [])
    return {'user_id': user_id, 'quarter': quarter, 'category': category, 'items': items}


def get_custom_budget(user_id: int) -> dict[str, Any] | None:
    """Get custom budget for a user if they have one.

    Args:
        user_id: The user's ID.

    Returns:
        Custom budget info or None if no custom budget.
    """
    if user_id in _custom_budgets:
        return _custom_budgets[user_id]
    return None


def create_toolset() -> FunctionToolset[None]:
    """Create the expense analysis toolset."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_team_members)
    toolset.add_function(get_expenses)
    toolset.add_function(get_custom_budget)
    return toolset


# =============================================================================
# Agent Factories
# =============================================================================


def create_traditional_agent(toolset: FunctionToolset[None]) -> Agent[None, str]:
    """Create agent with traditional tool calling."""
    return Agent(
        MODEL,
        toolsets=[toolset],
        system_prompt='You are a financial analyst. Use the available tools to analyze expenses.',
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
        system_prompt='You are a financial analyst. Use the available tools to analyze expenses.',
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
    print(
        f'\n  Output:\n  {metrics.output[:600]}...' if len(metrics.output) > 600 else f'\n  Output:\n  {metrics.output}'
    )


async def main() -> None:
    # Configure Logfire
    logfire.configure(service_name='code-mode-expense-demo')
    logfire.instrument_pydantic_ai()

    print('=' * 70)
    print('CodeMode Demo: Team Expense Analysis')
    print('=' * 70)
    print(f'\nModel: {MODEL}')
    print('Task: Analyze Q3 travel expenses for 5 team members')
    print('      (50+ expense line items to process)')

    toolset = create_toolset()

    # Run Traditional
    print('\n' + '-' * 70)
    print('Running TRADITIONAL tool calling...')
    print('(All expense data flows through LLM context)')
    print('-' * 70)

    with logfire.span('demo_traditional'):
        trad = await run_traditional(toolset)
    print_metrics(trad)

    # Run CodeMode
    print('\n' + '-' * 70)
    print('Running CODE MODE tool calling...')
    print('(Aggregation in code - only summaries returned)')
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
        f'\n  LLM Requests: {trad.request_count} → {code.request_count} ({request_reduction} fewer, {request_pct:.0f}% reduction)'
    )
    print(
        f'  Total Tokens: {trad.total_tokens:,} → {code.total_tokens:,} ({token_pct:+.1f}% {"savings" if token_diff > 0 else "increase"})'
    )

    print('\n  Key Insight: Traditional mode returns ALL expense line items to the LLM.')
    print('               Code mode sums them up in code, returning only totals.')

    print('\n' + '=' * 70)
    print('View detailed traces: https://logfire.pydantic.dev')
    print('=' * 70)


if __name__ == '__main__':
    asyncio.run(main())
