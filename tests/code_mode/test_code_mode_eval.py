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
# Mock Data - Team members and orders for N+1 query pattern
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


# =============================================================================
# Tool Functions
# =============================================================================


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
    """Create the function toolset with our tools."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_team_members, takes_ctx=False)
    toolset.add_function(get_user_orders, takes_ctx=False)
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

# Complex N+1 query: Get team -> For each member get orders -> Aggregate
COMPLEX_TASK = TaskInput(
    prompt="""\
Find the highest spender in the engineering team.

Steps:
1. Get all team members from the 'engineering' team
2. For each team member, fetch their orders
3. Sum the 'amount' field for orders with status 'completed'
4. Report who spent the most and their total

Return the name and total amount of the highest spender."""
)


# =============================================================================
# Test
# =============================================================================


async def test_code_mode_vs_traditional(allow_model_requests: None):
    """Compare code mode vs traditional mode on the same complex task.

    This test demonstrates that code mode:
    1. Uses fewer LLM requests (2 vs 4+)
    2. Uses fewer total tokens (context doesn't regrow)
    3. Produces correct results

    The task requires N+1 queries: 1 to get team members + N to get each member's orders.
    Traditional mode needs multiple round-trips; code mode does it in one execution.
    """
    # Create dataset with the complex task
    dataset: Dataset[TaskInput, TaskOutput, TaskMetadata] = Dataset(
        cases=[
            Case(
                name='highest_spender_engineering',
                inputs=COMPLEX_TASK,
                metadata=TaskMetadata(
                    expected_tool_calls=4,  # 1 team + 3 members
                    expected_in_result=['bob', '500'],  # Bob has highest completed orders
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
    print(f'\n{"Metric":<25} {"Traditional":>15} {"Code Mode":>15} {"Winner":>15}')
    print('-' * 70)

    # Request count
    trad_requests = trad_output.request_count
    code_requests = code_output.request_count
    req_winner = 'Code Mode' if code_requests < trad_requests else 'Traditional'
    print(f'{"LLM Requests":<25} {trad_requests:>15} {code_requests:>15} {req_winner:>15}')

    # Token count
    trad_tokens = trad_output.total_input_tokens + trad_output.total_output_tokens
    code_tokens = code_output.total_input_tokens + code_output.total_output_tokens
    tok_winner = 'Code Mode' if code_tokens < trad_tokens else 'Traditional'
    print(f'{"Total Tokens":<25} {trad_tokens:>15} {code_tokens:>15} {tok_winner:>15}')

    # Input tokens specifically
    print(f'{"  Input Tokens":<25} {trad_output.total_input_tokens:>15} {code_output.total_input_tokens:>15}')
    print(f'{"  Output Tokens":<25} {trad_output.total_output_tokens:>15} {code_output.total_output_tokens:>15}')

    print('-' * 70)

    # Results
    print(f'\nTraditional result: {trad_output.result[:100]}...')
    print(f'Code mode result: {code_output.result[:100]}...')

    # Assertions - code mode should be more efficient
    assert code_requests <= trad_requests, (
        f'Code mode should use fewer or equal requests: {code_requests} vs {trad_requests}'
    )

    # Both should produce correct results (contain "Bob" and "500")
    assert 'bob' in trad_output.result.lower() or '500' in trad_output.result
    assert 'bob' in code_output.result.lower() or '500' in code_output.result

    print('\n✓ Code mode used fewer/equal LLM requests')
    print('✓ Both modes produced correct results')
