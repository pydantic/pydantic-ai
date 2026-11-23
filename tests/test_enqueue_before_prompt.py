"""Tests for `enqueue(..., placement='before_prompt')`: context placed before the user's prompt in the request being sent.

These use `FunctionModel` rather than recordings: what they pin is where content lands in message history, which
`EnqueuedMessagesEvent`s are emitted in what order, and which content is refused before any request is made, none of
which a provider recording can show. The wire order on real adapters is pinned in
`tests/harness/memory/test_snapshot_placement.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import _agent_graph
from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    DataCapabilityPart,
    EnqueuedMessagesEvent,
    InstructionCapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperatorSource,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ._inline_snapshot import snapshot
from .conftest import IsDatetime


@dataclass(kw_only=True)
class _ContextPart(DataCapabilityPart[str], namespace='test_before_prompt', name='context', tag='context'):
    pass


@dataclass(kw_only=True)
class _NotePart(InstructionCapabilityPart, namespace='test_before_prompt', name='note'):
    pass


def _context(text: str) -> _ContextPart:
    return _ContextPart(content=text, source=OperatorSource())


@dataclass
class _EnqueueBeforePrompt(AbstractCapability[Any]):
    """Enqueues `_ContextPart`s placed before the prompt from `before_model_request`, on every step."""

    texts: tuple[str, ...]

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        for text in self.texts:
            ctx.enqueue(_context(f'{text} {ctx.run_step}'), placement='before_prompt')
        return request_context


def _shape(messages: list[ModelMessage]) -> list[list[str]]:
    """Each request's parts as `<type>` or `<type>:<text>`, and each response as `response`."""
    shape: list[list[str]] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            shape.append(['response'])
            continue
        shape.append(
            [
                f'{type(part).__name__}:{part.content}'
                if isinstance(part, _ContextPart | UserPromptPart | SystemPromptPart)
                else type(part).__name__
                for part in message.parts
            ]
        )
    return shape


async def test_before_prompt_content_goes_into_the_prompt_request_in_enqueue_order():
    """Content from before the run and from a hook lands right before the prompt, after the standing system prompt.

    The app's content is delivered by the drain ahead of the hooks and the hook's by the one after them, so the two
    deliveries go into the same request one after the other. Each enqueue call gets an event, in history order,
    holding its parts as they sit in history.
    """
    received: list[list[ModelMessage]] = []
    events: list[EnqueuedMessagesEvent] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        received.append(messages)
        yield 'done'

    # Listening to events makes the run stream.
    @dataclass
    class Listen(AbstractCapability[Any]):
        @on_event(EnqueuedMessagesEvent)
        async def delivered(self, ctx: RunContext[Any], event: EnqueuedMessagesEvent) -> None:
            events.append(event)

    agent = Agent(
        FunctionModel(stream_function=stream_fn),
        system_prompt='Be brief.',
        capabilities=[_EnqueueBeforePrompt(texts=('memory', 'plan')), Listen()],
    )
    nodes: list[_agent_graph.ModelRequestNode[Any, Any]] = []
    async with agent.iter('What now?') as run:
        app_id = run.enqueue('The account is on the Team plan.', placement='before_prompt')
        async for node in run:
            if isinstance(node, _agent_graph.ModelRequestNode):
                nodes.append(node)
    assert run.result is not None
    history = run.result.all_messages()

    assert _shape(history) == snapshot(
        [
            [
                'SystemPromptPart:Be brief.',
                'UserPromptPart:The account is on the Team plan.',
                '_ContextPart:memory 1',
                '_ContextPart:plan 1',
                'UserPromptPart:What now?',
            ],
            ['response'],
        ]
    )
    # The model reads the question last, and the node's request is the copy in history.
    [[sent]] = received
    assert isinstance(sent, ModelRequest)
    assert sent.parts[-1] == UserPromptPart('What now?', timestamp=IsDatetime())
    assert nodes[0].request is history[0]
    assert run.result.new_messages() == history

    request = history[0]
    assert isinstance(request, ModelRequest)
    assert [event.enqueue_id == app_id for event in events] == [True, False, False]
    assert [event.parts for event in events] == [(part,) for part in request.parts[1:4]]
    assert [[message.parts for message in event.messages] for event in events] == [
        [[part]] for part in request.parts[1:4]
    ]


async def test_without_a_prompt_before_prompt_content_lands_as_append_would_put_it():
    """In the tool loop the request carries tool results, not a prompt: a hook's content is its own request, and a
    tool's goes right after its return."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[_EnqueueBeforePrompt(texts=('memory',))])

    @agent.tool
    def lookup(ctx: RunContext[Any]) -> str:
        ctx.enqueue(_context('found by the tool'), placement='before_prompt')
        return 'ok'

    result = await agent.run('What now?')
    assert _shape(result.all_messages()) == snapshot(
        [
            ['_ContextPart:memory 1', 'UserPromptPart:What now?'],
            ['response'],
            ['ToolReturnPart', '_ContextPart:found by the tool'],
            ['_ContextPart:memory 2'],
            ['response'],
        ]
    )


@pytest.mark.parametrize(
    'content',
    [
        pytest.param(SystemPromptPart('Be brief.'), id='system-prompt'),
        pytest.param(_NotePart(content='Be brief.', source=OperatorSource()), id='instruction-part'),
        pytest.param(ModelRequest(parts=[_context('context')]), id='complete-request'),
    ],
)
async def test_before_prompt_takes_only_data_and_user_content(content: SystemPromptPart | _NotePart | ModelRequest):
    """Instructions and system prompts stay after the user's words, and a complete message can't go into a request."""

    agent = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')])))
    async with agent.iter('What now?') as run:
        with pytest.raises(UserError, match="placement='before_prompt'"):
            run.enqueue(content, placement='before_prompt')
