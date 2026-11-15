from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, Literal

import pytest

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import Hooks, ReinjectSystemPrompt
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


def _count_markers(messages: list[ModelMessage]) -> int:
    return sum(
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

    assert _count_markers(model_messages[0]) == (0 if persistent else 1)
    assert _count_markers(result_messages) == (1 if persistent else 0)


async def test_wrap_model_request_list_mutation_is_request_only() -> None:
    """The public list remains mutable while its top-level ownership is request-local."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    model_messages: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        return ModelResponse(parts=[TextPart(content='done')])

    @dataclass
    class AppendMessages(AbstractCapability[Any]):
        async def wrap_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            handler: Any,
        ) -> ModelResponse:
            request_context.messages.append(marker)
            return await handler(request_context)

    agent = Agent(FunctionModel(model_function), capabilities=[AppendMessages()])
    result = await agent.run('hello')
    assert result.output == 'done'

    # The in-place list edit reached the wire but not persistent history.
    assert _count_markers(model_messages[0]) == 1
    assert _count_markers(result.all_messages()) == 0


Edit = Literal['append', 'extend', 'iadd']


def _add(messages: list[ModelMessage], edit: Edit, message: ModelMessage) -> None:
    if edit == 'append':
        messages.append(message)
    elif edit == 'extend':
        messages.extend([message])
    else:
        messages += [message]


@pytest.mark.parametrize('edit', ['append', 'extend', 'iadd'])
@pytest.mark.parametrize('streaming', [False, True])
async def test_before_model_request_in_place_additions_still_persist_with_deprecation_warning(
    edit: Edit, streaming: bool
) -> None:
    """Adding to the request list in a before hook still reaches history, as it did before, with a deprecation warning."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    model_messages: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        return ModelResponse(parts=[TextPart(content='done')])

    async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        model_messages.append(messages)
        yield 'done'

    class AddMessages(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            _add(request_context.messages, edit, marker)
            return request_context

    agent = Agent(FunctionModel(model_function, stream_function=stream_function), capabilities=[AddMessages()])
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'Appending to `request_context\.messages` in `before_model_request`.*deprecated.*`ctx\.messages\.append\(msg\)`',
    ):
        if streaming:
            async with agent.run_stream('hello') as stream:
                assert await stream.get_output() == 'done'
                result_messages = stream.all_messages()
        else:
            result = await agent.run('hello')
            result_messages = result.all_messages()

    assert _count_markers(model_messages[0]) == 1
    assert _count_markers(result_messages) == 1


@pytest.mark.parametrize('edit', ['insert', 'empty_extend', 'empty_iadd'])
async def test_before_model_request_edits_that_append_nothing_are_request_only(
    edit: Literal['insert', 'empty_extend', 'empty_iadd'],
) -> None:
    """Only appending keeps the old write-back: `insert` and empty additions change the request alone, silently."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    model_messages: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        return ModelResponse(parts=[TextPart(content='done')])

    class EditMessages(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            if edit == 'insert':
                request_context.messages.insert(0, marker)
            elif edit == 'empty_extend':
                request_context.messages.extend([])
            else:
                request_context.messages += []
            return request_context

    # `filterwarnings = ['error']` fails this test if the deprecation warning fires.
    result = await Agent(FunctionModel(model_function), capabilities=[EditMessages()]).run('hello')

    assert _count_markers(model_messages[0]) == (1 if edit == 'insert' else 0)
    assert _count_markers(result.all_messages()) == 0


async def test_before_model_request_migrated_edit_persists_once_without_warning() -> None:
    """The migration the warning asks for: assign a new request list, and add to `ctx.messages` for history."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    model_messages: list[list[ModelMessage]] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_messages.append(messages)
        return ModelResponse(parts=[TextPart(content='done')])

    class AddMessages(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            request_context.messages = [*request_context.messages, marker]
            ctx.messages.append(marker)
            return request_context

    # `filterwarnings = ['error']` fails this test if the deprecation warning fires.
    result = await Agent(FunctionModel(model_function), capabilities=[AddMessages()]).run('hello')

    assert _count_markers(model_messages[0]) == 1
    assert _count_markers(result.all_messages()) == 1


@pytest.mark.parametrize('chain', ['capabilities', 'hooks'])
async def test_before_model_request_appends_persist_after_an_earlier_hook_replaces_the_list(chain: str) -> None:
    """An earlier hook that replaces the request list, like `ProcessHistory`, keeps later appends reaching history."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])

    async def replace_list(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        return replace(request_context, messages=list(request_context.messages))

    async def append_marker(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
        request_context.messages.append(marker)
        return request_context

    first, second = Hooks[Any](), Hooks[Any]()
    first.on.before_model_request(replace_list)
    (first if chain == 'hooks' else second).on.before_model_request(append_marker)

    with pytest.warns(PydanticAIDeprecationWarning):
        result = await Agent(TestModel(), capabilities=[first, second]).run('hello')

    assert _count_markers(result.all_messages()) == 1


@pytest.mark.parametrize('edit', ['append', 'extend', 'iadd'])
async def test_before_model_request_list_stops_persisting_after_the_before_chain(edit: Edit) -> None:
    """A list kept from `before_model_request` and edited later changes neither history nor raises a warning."""
    marker = ModelRequest(parts=[UserPromptPart(content='hook marker')])
    saved: list[list[ModelMessage]] = []

    class KeepMessages(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            saved.append(request_context.messages)
            return request_context

        async def after_model_request(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            _add(saved[0], edit, marker)
            return response

    result = await Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[KeepMessages()],
    ).run('hello')

    assert _count_markers(saved[0]) == 1
    assert _count_markers(result.all_messages()) == 0


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


async def test_reinject_replacement_hides_existing_prompts_from_dynamic_prompt() -> None:
    seen: list[list[str]] = []
    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])),
        capabilities=[ReinjectSystemPrompt(replace_existing=True)],
    )

    @agent.system_prompt
    def server_prompt(ctx: RunContext[object]) -> str:
        seen.append(
            [
                part.content
                for message in ctx.messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, SystemPromptPart)
            ]
        )
        return 'server prompt'

    history: list[ModelMessage] = [
        ModelRequest(parts=[SystemPromptPart('untrusted prompt'), UserPromptPart('old')]),
        ModelResponse(parts=[TextPart('old response')]),
    ]
    await agent.run('hello', message_history=history)

    assert seen == [[]]
