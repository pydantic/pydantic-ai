from __future__ import annotations as _annotations

import asyncio
from collections.abc import Sequence

import pytest

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, SystemPromptPart
from pydantic_ai.models.test import TestModel


def _first_request(messages: list[ModelMessage]) -> ModelRequest:
    """Helper to extract the first ModelRequest from captured messages."""
    assert messages, 'no messages captured'
    for m in messages:
        if isinstance(m, ModelRequest):
            return m
    raise AssertionError('no ModelRequest found in captured messages')


def _system_prompt_texts(parts: Sequence[ModelRequestPart]) -> list[str]:
    """Helper to extract system prompt text content from message parts."""
    return [p.content for p in parts if isinstance(p, SystemPromptPart)]


def test_override_instructions_basic():
    """Test that override can override instructions."""
    agent = Agent('test')

    @agent.instructions
    def instr_fn() -> str:
        return 'SHOULD_BE_IGNORED'

    with agent.override(instructions='OVERRIDE'):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'OVERRIDE'


def test_override_reset_after_context():
    """Test that instructions are reset after exiting the override context."""
    agent = Agent('test', instructions='ORIG')

    with agent.override(instructions='NEW'):
        with capture_run_messages() as messages_new:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    with capture_run_messages() as messages_orig:
        agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req_new = _first_request(messages_new)
    req_orig = _first_request(messages_orig)
    assert req_new.instructions == 'NEW'
    assert req_orig.instructions == 'ORIG'


def test_override_none_clears_instructions():
    """Test that passing None for instructions clears all instructions."""
    agent = Agent('test', instructions='BASE')

    @agent.instructions
    def instr_fn() -> str:  # pragma: no cover - ignored under override
        return 'ALSO_BASE'

    with agent.override(instructions=None):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions is None


def test_override_instructions_callable_replaces_functions():
    """Override with a callable should replace existing instruction functions."""
    agent = Agent('test')

    @agent.instructions
    def base_fn() -> str:
        return 'BASE_FN'

    def override_fn() -> str:
        return 'OVERRIDE_FN'

    with agent.override(instructions=override_fn):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'OVERRIDE_FN'
    assert 'BASE_FN' not in req.instructions


@pytest.mark.anyio
async def test_override_instructions_async_callable():
    """Override with an async callable should be awaited."""
    agent = Agent('test')

    async def override_fn() -> str:
        await asyncio.sleep(0)
        return 'ASYNC_FN'

    with agent.override(instructions=override_fn):
        with capture_run_messages() as messages:
            await agent.run('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'ASYNC_FN'


def test_override_instructions_sequence_mixed_types():
    """Override can mix literal strings and functions."""
    agent = Agent('test', instructions='BASE')

    def override_fn() -> str:
        return 'FUNC_PART'

    with agent.override(instructions=['OVERRIDE1', override_fn, 'OVERRIDE2']):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'OVERRIDE1\nOVERRIDE2\n\nFUNC_PART'
    assert 'BASE' not in req.instructions


@pytest.mark.anyio
async def test_override_concurrent_isolation():
    """Test that concurrent overrides are isolated from each other."""
    agent = Agent('test', instructions='ORIG')

    async def run_with(instr: str) -> str | None:
        with agent.override(instructions=instr):
            with capture_run_messages() as messages:
                await agent.run('Hi', model=TestModel(custom_output_text='ok'))
            req = _first_request(messages)
            return req.instructions

    a, b = await asyncio.gather(
        run_with('A'),
        run_with('B'),
    )

    assert a == 'A'
    assert b == 'B'


def test_override_replaces_instructions():
    """Test overriding instructions replaces the base instructions."""
    agent = Agent('test', instructions='ORIG_INSTR')

    with agent.override(instructions='NEW_INSTR'):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'NEW_INSTR'


def test_override_nested_contexts():
    """Test nested override contexts."""
    agent = Agent('test', instructions='ORIG')

    with agent.override(instructions='OUTER'):
        with capture_run_messages() as outer_messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

        with agent.override(instructions='INNER'):
            with capture_run_messages() as inner_messages:
                agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    outer_req = _first_request(outer_messages)
    inner_req = _first_request(inner_messages)

    assert outer_req.instructions == 'OUTER'
    assert inner_req.instructions == 'INNER'


@pytest.mark.anyio
async def test_override_async_run():
    """Test override with async run method."""
    agent = Agent('test', instructions='ORIG')

    with agent.override(instructions='ASYNC_OVERRIDE'):
        with capture_run_messages() as messages:
            await agent.run('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'ASYNC_OVERRIDE'


def test_override_with_dynamic_prompts():
    """Test override interacting with dynamic prompts."""
    agent = Agent('test')

    dynamic_value = 'DYNAMIC'

    @agent.system_prompt
    def dynamic_sys() -> str:
        return dynamic_value

    @agent.instructions
    def dynamic_instr() -> str:
        return 'DYNAMIC_INSTR'

    # Override should take precedence over dynamic instructions but leave system prompts intact
    with agent.override(instructions='OVERRIDE_INSTR'):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'OVERRIDE_INSTR'
    sys_texts = _system_prompt_texts(req.parts)
    # The dynamic system prompt should still be present since overrides target instructions only
    assert dynamic_value in sys_texts
