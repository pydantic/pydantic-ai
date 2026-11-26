"""`load_conversation` in a Temporal workflow, loading through an activity of the application's own.

The hook runs in the workflow before the run is assembled and can't be a durable operation, so a workflow loads
through an activity, as the docs say. In a module of its own, like the other hook tests, so the workflow's sandbox
imports only what it needs.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, Conversation, ConversationTypeAdapter
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models.test import TestModel

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.worker import Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-load-conversation')]


@activity.defn
async def load_stored_conversation(conversation_id: str) -> str:
    """Stands in for a store the worker reads, returning the conversation as JSON."""
    stored = Conversation(
        messages=[ModelRequest.user_text_prompt('My order is A-1.'), ModelResponse(parts=[TextPart('Noted.')])],
        conversation_id=conversation_id,
    )
    return ConversationTypeAdapter.dump_json(stored).decode()


class _LoadThroughAnActivity(AbstractCapability[Any]):
    """Load the conversation a run is given the `conversation_id` of, through the workflow's own activity.

    Example:
        agent = Agent(model, capabilities=[_LoadThroughAnActivity(), TemporalDurability()])
    """

    async def load_conversation(self, conversation_id: str, *, deps: Any) -> Conversation | None:
        stored = await workflow.execute_activity(
            load_stored_conversation, conversation_id, start_to_close_timeout=timedelta(seconds=10)
        )
        return ConversationTypeAdapter.validate_json(stored)


_agent = Agent(
    TestModel(custom_output_text='done'),
    name='load_conversation_agent',
    capabilities=[_LoadThroughAnActivity(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class LoadConversationWorkflow:
    @workflow.run
    async def run(self, conversation_id: str) -> list[str]:
        result = await _agent.run('What is my order?', conversation_id=conversation_id)
        return [message.kind for message in result.all_messages()] + [result.conversation_id]


async def test_load_conversation_loads_through_an_activity_in_a_workflow(client: Client):
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LoadConversationWorkflow],
        activities=[load_stored_conversation],
        plugins=[AgentPlugin(_agent)],
    ):
        kinds = await client.execute_workflow(
            LoadConversationWorkflow.run,
            args=['chat-1'],
            id=LoadConversationWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )

    assert kinds == ['request', 'response', 'request', 'response', 'chat-1']
