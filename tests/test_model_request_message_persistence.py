from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize('hook', ['before', 'wrap'])
@pytest.mark.parametrize('persistent', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
async def test_model_request_message_persistence_depends_on_context(
    hook: Literal['before', 'wrap'], persistent: bool, streaming: bool
) -> None:
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    model_messages: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        return ModelResponse(parts=[TextPart(content='done')])

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        model_messages.append(messages)
        yield 'done'

    def update_messages(ctx: RunContext[Any], request_context: ModelRequestContext) -> None:
        if persistent:
            ctx.messages.insert(0, marker)
        else:
            request_context.messages = [marker, *request_context.messages]

    @dataclass
    class RewriteMessages(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if hook == 'before':
                update_messages(ctx, request_context)
            return request_context

        async def wrap_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            handler: Any,
        ) -> ModelResponse:
            if hook == 'wrap':
                update_messages(ctx, request_context)
            return await handler(request_context)

    model = FunctionModel(model_function, stream_function=stream_function)
    agent = Agent(
        model,
        system_prompt='system prompt',
        capabilities=[RewriteMessages(), ReinjectSystemPrompt()],
    )
    if streaming:
        async with agent.run_stream('hello') as stream:
            assert await stream.get_output() == 'done'
            result_messages = stream.all_messages()
    else:
        result = await agent.run('hello')
        assert result.output == 'done'
        result_messages = result.all_messages()

    def contains_marker(messages: list[ModelMessage]) -> bool:
        return any(
            isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) and part.content == 'hook marker' for part in message.parts)
            for message in messages
        )

    assert contains_marker(model_messages[0]) is not persistent
    assert contains_marker(result_messages) is persistent
