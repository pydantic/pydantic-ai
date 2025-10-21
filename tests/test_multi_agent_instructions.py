"""Test for issue #3207: Instructions ignored in back-to-back agent calls.

This test verifies that when running multiple agents sequentially with message_history,
each agent uses its own instructions rather than inheriting instructions from previous agents.
"""

from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelRequest, capture_run_messages
from pydantic_ai.models.test import TestModel


class Suggestion(BaseModel):
    """A suggestion for the user."""

    text: str = Field(description='The suggestion text')


def test_multi_agent_sequential_instructions_with_output_type():
    """Test that Agent2 uses its own instructions when called with Agent1's message history.

    This reproduces issue #3207 where Agent2's instructions were being ignored
    and Agent1's instructions were used instead.

    Scenario:
    1. Agent1 runs with instructions="Agent 1 instructions"
    2. Agent2 runs with message_history from Agent1 + output_type (structured output)
    3. Expected: Agent2's ModelRequest should have instructions="Agent 2 instructions"
    4. Bug: Agent2's ModelRequest incorrectly uses Agent1's instructions
    """
    # Create two agents with different instructions
    model1 = TestModel()
    agent1 = Agent(
        model1,
        instructions='Agent 1 instructions',
    )

    # Use a second TestModel instance to track what instructions it receives
    model2 = TestModel(custom_output_args={'text': 'Test suggestion'})
    agent2 = Agent(
        model2,
        instructions='Agent 2 instructions',
        output_type=Suggestion,
        output_retries=5,  # Allow more retries to capture the messages even if validation fails
    )

    # Run Agent1
    result1 = agent1.run_sync('Hello')

    # Run Agent2 with Agent1's message history, capturing messages
    # This is the scenario that triggers the bug in issue #3207
    with capture_run_messages() as agent2_messages:
        agent2.run_sync(message_history=result1.new_messages())

    # Find all ModelRequest messages created by Agent2
    agent2_requests = [msg for msg in agent2_messages if isinstance(msg, ModelRequest)]

    # We expect at least one ModelRequest from Agent2
    assert len(agent2_requests) > 0, 'Agent2 should have created at least one ModelRequest'

    # Check what instructions were used in Agent2's requests
    instructions_in_agent2_requests = [req.instructions for req in agent2_requests if req.instructions is not None]

    # Agent2 should use its own instructions, not Agent1's
    agent2_instructions_found = 'Agent 2 instructions' in instructions_in_agent2_requests

    # This assertion will FAIL with the bug (Agent1's instructions are used instead)
    assert agent2_instructions_found, (
        f'BUG REPRODUCED: Agent 2 instructions not found in requests created by Agent2.\n'
        f'Expected: "Agent 2 instructions"\n'
        f'Found: {instructions_in_agent2_requests}\n'
        f"This confirms issue #3207 - Agent1's instructions are leaking into Agent2's requests."
    )


def test_multi_agent_sequential_instructions_no_output_type():
    """Test multi-agent instructions without structured output.

    This is a simpler scenario that should also work correctly.
    """
    agent1 = Agent(
        TestModel(),
        instructions='Agent 1 instructions',
    )
    agent2 = Agent(
        TestModel(),
        instructions='Agent 2 instructions',
    )

    # Run Agent1
    result1 = agent1.run_sync('Hello')

    # Run Agent2 with Agent1's message history
    result2 = agent2.run_sync('Hello again', message_history=result1.new_messages())

    # Agent2's new requests should have Agent2's instructions
    agent2_new_requests = [msg for msg in result2.new_messages() if isinstance(msg, ModelRequest)]

    # At least one of Agent2's new requests should have its instructions
    agent2_instructions_found = any(
        req.instructions == 'Agent 2 instructions' for req in agent2_new_requests if req.instructions is not None
    )

    assert agent2_instructions_found, (
        f'Agent 2 instructions not found in new requests. '
        f'Instructions: {[req.instructions for req in agent2_new_requests if req.instructions is not None]}'
    )


def test_multi_agent_with_user_prompt_workaround():
    """Test that passing a user_prompt to Agent2 avoids the bug.

    This is the workaround mentioned in issue #3207.
    When a user_prompt is provided, Agent2 creates a fresh ModelRequest
    with its own instructions.
    """
    agent1 = Agent(
        TestModel(),
        instructions='Agent 1 instructions',
    )
    agent2 = Agent(
        TestModel(),
        instructions='Agent 2 instructions',
        output_type=Suggestion,
    )

    # Run Agent1
    result1 = agent1.run_sync('Hello')

    # Run Agent2 WITH a user_prompt (workaround)
    result2 = agent2.run_sync('Continue', message_history=result1.new_messages())

    # Get Agent2's new requests
    agent2_new_requests = [msg for msg in result2.new_messages() if isinstance(msg, ModelRequest)]

    # Should have Agent2's instructions
    agent2_instructions_found = any(
        req.instructions == 'Agent 2 instructions' for req in agent2_new_requests if req.instructions is not None
    )

    assert agent2_instructions_found, (
        f'Agent 2 instructions not found even with user_prompt workaround. '
        f'Instructions: {[req.instructions for req in agent2_new_requests if req.instructions is not None]}'
    )
