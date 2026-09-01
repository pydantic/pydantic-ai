from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai.capabilities import ReinjectSystemPrompt
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


def _contains_marker(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(message, ModelRequest)
        and any(isinstance(part, UserPromptPart) and part.content == 'hook marker' for part in message.parts)
        for message in messages
    )


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

    assert _contains_marker(model_messages[0]) is not persistent
    assert _contains_marker(result_messages) is persistent


@pytest.mark.parametrize('hook', ['before', 'wrap'])
async def test_model_request_messages_allow_request_only_list_mutation(hook: Literal['before', 'wrap']) -> None:
    """The public list remains mutable while its top-level ownership is request-local."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    model_messages: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        return ModelResponse(parts=[TextPart(content='done')])

    def mutate_messages(request_context: ModelRequestContext) -> None:
        request_context.messages.append(marker)

    @dataclass
    class AppendMessages(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if hook == 'before':
                mutate_messages(request_context)
            return request_context

        async def wrap_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            handler: Any,
        ) -> ModelResponse:
            if hook == 'wrap':
                mutate_messages(request_context)
            return await handler(request_context)

    agent = Agent(FunctionModel(model_function), capabilities=[AppendMessages()])
    result = await agent.run('hello')
    assert result.output == 'done'

    # The in-place list edit reached the wire but not persistent history.
    assert _contains_marker(model_messages[0])
    assert not _contains_marker(result.all_messages())


async def test_reinject_system_prompt_preserves_the_persistent_existing_prompt_after_request_only_filtering() -> None:
    class FilterRequestPrompt(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            request_context.messages = [
                replace(message, parts=[part for part in message.parts if not isinstance(part, SystemPromptPart)])
                if isinstance(message, ModelRequest)
                else message
                for message in request_context.messages
            ]
            return request_context

    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('existing'), UserPromptPart('old')]),
        ModelResponse(parts=[TextPart('old response')]),
    ]
    result = await Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        system_prompt='new',
        capabilities=[FilterRequestPrompt(), ReinjectSystemPrompt()],
    ).run('hello', message_history=history)

    persistent_system_prompts = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    assert persistent_system_prompts == ['existing']


async def test_request_hook_can_clear_persistent_history() -> None:
    class ClearPersistentHistory(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            ctx.messages.clear()
            return request_context

    result = await Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[ClearPersistentHistory()],
    ).run('hello')

    assert result.output == 'done'
    assert len(result.all_messages()) == 1
    assert isinstance(result.all_messages()[0], ModelResponse)
