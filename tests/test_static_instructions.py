"""Tests for `static=True` on the instruction decorators.

Whether an instruction block is [`dynamic`][pydantic_ai.messages.InstructionPart.dynamic] decides
which side of a provider's cache breakpoint it sits on, and it used to be inferred from the shape the
author reached for: literal text was static, a decorated function was always dynamic. An author who
writes a function purely because it is a nicer way to *produce* fixed text — from a template, a config
file, a loop over feature flags — was pushed outside the cacheable prefix for it.

`static=True` lets them say what they meant. It does two things, and the second is what makes the
first honest: the block sorts into the static group, *and* the function is called once for the run
rather than once per model request. A block that claims the cacheable prefix while being recomputed
every request would move the prefix under a static label — a cache trap rather than a cache fix.
"""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


def two_step_model(captured: list[list[InstructionPart]]) -> FunctionModel:
    """A model that records the blocks of each request, calls a tool once, then finishes.

    Two model requests in one run is the whole point: it is the only way to tell "once per run" from
    "once per request".
    """
    calls = 0

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        captured.append(list(info.model_request_parameters.instruction_parts or []))
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('ping', {}, tool_call_id='call')])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model_fn)


def build_agent(static: bool) -> tuple[Agent[None, str], list[int], list[list[InstructionPart]]]:
    """An agent whose one decorated instruction counts how often it is called."""
    calls: list[int] = []
    captured: list[list[InstructionPart]] = []
    agent = Agent(two_step_model(captured), instructions='Literal instructions.')

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    @agent.instructions(name='fixed', static=static)
    def fixed() -> str:
        calls.append(1)
        return 'Composed once.'

    return agent, calls, captured


async def test_static_function_is_called_once_per_run() -> None:
    agent, calls, captured = build_agent(static=True)
    await agent.run('Hello')

    assert len(captured) == 2, 'the run should make two model requests'
    assert len(calls) == 1
    # Same text on both requests, which is what the cacheable prefix depends on.
    assert [[part.content for part in request] for request in captured] == snapshot(
        [
            ['Literal instructions.', 'Composed once.'],
            ['Literal instructions.', 'Composed once.'],
        ]
    )


async def test_a_dynamic_function_is_still_called_every_request() -> None:
    agent, calls, captured = build_agent(static=False)
    await agent.run('Hello')

    assert len(captured) == 2
    assert len(calls) == 2


async def test_the_answer_is_not_reused_across_runs() -> None:
    # Cached on the run, not on the agent: a second run computes its own value, so a function reading
    # anything that changes between runs is still correct, just fixed within each one.
    agent, calls, _ = build_agent(static=True)
    await agent.run('Hello')
    await agent.run('Hello again')
    assert len(calls) == 2


async def test_static_sorts_into_the_cacheable_prefix() -> None:
    # `dynamic` is what puts a block behind the provider's cache breakpoint. Declared static, a
    # function's block keeps its place among the literal text; the dynamic one still sorts last, which
    # is what makes the ordering visible rather than incidental.
    captured: list[list[InstructionPart]] = []
    agent = Agent(two_step_model(captured), instructions='Literal instructions.')

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    @agent.instructions(name='recomputed')
    def recomputed() -> str:
        return 'Recomputed.'

    @agent.instructions(name='fixed', static=True)
    def fixed() -> str:
        return 'Composed once.'

    await agent.run('Hello')
    assert [(part.content, str(part.id), part.dynamic) for part in captured[0]] == snapshot(
        [
            ('Literal instructions.', 'agent', False),
            ('Composed once.', 'agent:fixed', False),
            ('Recomputed.', 'agent:recomputed', True),
        ]
    )


async def test_a_static_block_stays_addressable_by_its_name() -> None:
    agent, _, captured = build_agent(static=True)
    await agent.run('Hello')
    named = captured[0][-1]
    assert named.name == 'fixed'
    assert str(named.id) == 'agent:fixed'
    assert named.dynamic is False


async def test_capability_instructions_take_the_same_flag() -> None:
    calls: list[int] = []
    captured: list[list[InstructionPart]] = []
    capability = Capability[object](id='style')

    @capability.instructions(name='tone', static=True)
    def tone(_ctx: RunContext[object]) -> str:
        calls.append(1)
        return 'Be concise.'

    agent = Agent(two_step_model(captured), capabilities=[capability])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    await agent.run('Hello')

    assert len(captured) == 2
    assert len(calls) == 1
    part = captured[0][-1]
    assert (str(part.id), part.dynamic) == ('capability:style:tone', False)


async def test_two_static_functions_are_cached_apart() -> None:
    captured: list[list[InstructionPart]] = []
    agent = Agent(two_step_model(captured))

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    @agent.instructions(name='first', static=True)
    def first() -> str:
        return 'First.'

    @agent.instructions(name='second', static=True)
    def second() -> str:
        return 'Second.'

    await agent.run('Hello')
    assert [part.content for part in captured[1]] == ['First.', 'Second.']


async def test_a_static_function_returning_nothing_contributes_nothing() -> None:
    captured: list[list[InstructionPart]] = []
    calls: list[int] = []
    agent = Agent(two_step_model(captured), instructions='Literal instructions.')

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    @agent.instructions(static=True)
    def nothing() -> str | None:
        calls.append(1)
        return None

    await agent.run('Hello')
    # `None` is cached like any other answer, so the second request does not call the function again
    # just because there was nothing to show for the first.
    assert len(calls) == 1
    assert [[part.content for part in request] for request in captured] == [
        ['Literal instructions.'],
        ['Literal instructions.'],
    ]


async def test_static_functions_may_read_the_run_context() -> None:
    # Fixed *for the run*, not independent of it: `deps` do not change mid-run, so reading them is
    # exactly the case this is for.
    captured: list[list[InstructionPart]] = []
    agent = Agent[str, str](two_step_model(captured), deps_type=str)

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    @agent.instructions(static=True)
    def tenant(ctx: RunContext[str]) -> str:
        return f'You serve {ctx.deps}.'

    await agent.run('Hello', deps='ACME')
    assert [part.content for part in captured[1]] == ['You serve ACME.']
