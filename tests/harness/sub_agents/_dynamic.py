"""A parent model that creates a `poet` with `create_agent`, delegates to it, then reports both results."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo
from tests.harness.sub_agents._sent import sent_texts

POET_INSTRUCTIONS = 'Write short poems.'
POET_RESULT = "Created sub-agent 'poet'. Delegate tasks to it with `delegate_task`. | a poem"
POET_CALLS: list[str] = []
"""The created `poet`'s model requests, one entry (the message it starts from) per request."""


def creator(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # The poet starts from the delegating agent's message: the instructions written for it, then its task.
    if (first := sent_texts(messages[:1])[-1]).startswith(POET_INSTRUCTIONS):
        POET_CALLS.append(first)
        return ModelResponse(parts=[TextPart('a poem')])
    returns = [
        part.model_response_str() for message in messages for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    if not returns:
        args = {'name': 'poet', 'description': 'Writes poems.', 'instructions': POET_INSTRUCTIONS}
        return ModelResponse(parts=[ToolCallPart('create_agent', args)])
    if len(returns) == 1:
        return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'poet', 'task': 'A poem, please.'})])
    return ModelResponse(parts=[TextPart(' | '.join(returns))])
