# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pydantic-ai[logfire]",
# ]
# ///
"""Demo: Code Mode Inner Tool Approval with Logfire Tracing.

This script demonstrates how code mode surfaces inner tool approvals
and allows overriding args during the approval flow.

Usage:
    source .env && uv run python demos/code_mode/run_approval_demo.py
"""

from __future__ import annotations

import asyncio
import sys

import logfire

from pydantic_ai import Agent
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.tools import (
    DeferredToolApprovalResult,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
)
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

# =============================================================================
# Tool Functions (require approval)
# =============================================================================


def send_email(ctx: RunContext[None], to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        ctx: The run context with tool call approval status.
        to: Email recipient address
        subject: Email subject line
        body: Email body content
    """
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return f'Email sent to {to}: "{subject}"'


def delete_file(ctx: RunContext[None], path: str, force: bool = False) -> str:
    """Delete a file from the filesystem.

    Args:
        ctx: The run context with tool call approval status.
        path: Path to the file to delete
        force: If True, delete without confirmation
    """
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return f'Deleted file: {path} (force={force})'


def transfer_funds(ctx: RunContext[None], from_account: str, to_account: str, amount: float) -> str:
    """Transfer funds between accounts.

    Args:
        ctx: The run context with tool call approval status.
        from_account: Source account ID
        to_account: Destination account ID
        amount: Amount to transfer
    """
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return f'Transferred ${amount:.2f} from {from_account} to {to_account}'


# =============================================================================
# Demo Runner
# =============================================================================


async def run_demo():
    """Run the approval demo with auto-approval and override args."""
    # Setup toolset
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(send_email)
    toolset.add_function(delete_file)
    toolset.add_function(transfer_funds)

    code_mode = CodeModeToolset(wrapped=toolset)
    agent: Agent[None, str] = Agent('gateway/anthropic:claude-sonnet-4-5')

    prompt = """
    Please do the following:
    1. Send an email to bob@example.com with subject "Meeting" and body "See you at 3pm"
    2. Delete the temp file at /tmp/old_data.csv
    3. Transfer $100 from account ACC001 to ACC002

    Return a summary of all actions taken.
    """

    with logfire.span('approval_demo'):
        async with code_mode:
            # First run - will trigger approval requests
            result = await agent.run(prompt, toolsets=[code_mode], output_type=[str, DeferredToolRequests])

            iteration = 0
            while isinstance(result.output, DeferredToolRequests):
                iteration += 1
                deferred = result.output

                # Show what approval is being requested
                for approval in deferred.approvals:
                    print(f'\n[Iteration {iteration}] Approval requested for: {approval.tool_name}')
                    print(f'   Args: {approval.args}')

                # Auto-approve all with optional override
                approvals: dict[str, bool | DeferredToolApprovalResult] = {}
                for approval in deferred.approvals:
                    tool_call_id = approval.tool_call_id

                    # Demo: override args for transfer_funds to change amount
                    if approval.tool_name == 'transfer_funds':
                        print('   -> Overriding amount: $100 -> $50')
                        approvals[tool_call_id] = ToolApproved(
                            override_args={
                                'from_account': 'ACC001',
                                'to_account': 'ACC002',
                                'amount': 50.0,  # Override the amount!
                            }
                        )
                    else:
                        approvals[tool_call_id] = ToolApproved()

                # Resume with approvals
                result = await agent.run(
                    message_history=result.all_messages(),
                    toolsets=[code_mode],
                    output_type=[str, DeferredToolRequests],
                    deferred_tool_results=DeferredToolResults(approvals=approvals),
                )

    print('\n' + '=' * 60)
    print('FINAL RESULT')
    print('=' * 60)
    print(result.output)


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == '__main__':
    logfire.configure(service_name='code-mode-approval-demo')
    logfire.instrument_pydantic_ai()

    print('=' * 60)
    print('Code Mode Inner Tool Approval Demo')
    print('=' * 60)
    print('\nThis demo shows:')
    print('1. Inner tool approvals surfaced (not "run_code")')
    print('2. Tool args visible in approval request')
    print('3. Override args applied to inner tool')
    print('\nView traces at https://logfire.pydantic.dev')
    print('=' * 60)

    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        print('\nInterrupted')
        sys.exit(1)
