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


def test_override_prompts_instructions_basic():
    """Test that override_prompts can override instructions."""
    agent = Agent('test')

    @agent.instructions
    def instr_fn() -> str:
        return 'SHOULD_BE_IGNORED'

    with agent.override_prompts(instructions='OVERRIDE'):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'OVERRIDE'


def test_override_prompts_system_prompts_basic():
    """Test that override_prompts can override system prompts."""
    agent = Agent('test', system_prompt=('ORIG1', 'ORIG2'))

    with agent.override_prompts(system_prompts=('NEW1', 'NEW2')):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    sys_texts = _system_prompt_texts(req.parts)
    assert sys_texts[:2] == ['NEW1', 'NEW2']
    assert 'ORIG1' not in sys_texts and 'ORIG2' not in sys_texts


def test_override_prompts_reset_after_context():
    """Test that prompts are reset after exiting the override context."""
    agent = Agent('test', system_prompt=('ORIG',))

    with agent.override_prompts(system_prompts=('NEW',)):
        with capture_run_messages() as messages_new:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    with capture_run_messages() as messages_orig:
        agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req_new = _first_request(messages_new)
    req_orig = _first_request(messages_orig)
    assert _system_prompt_texts(req_new.parts)[:1] == ['NEW']
    assert _system_prompt_texts(req_orig.parts)[:1] == ['ORIG']


def test_override_prompts_none_clears_instructions():
    """Test that passing None for instructions clears all instructions."""
    agent = Agent('test', instructions='BASE')

    @agent.instructions
    def instr_fn() -> str:  # pragma: no cover - ignored under override
        return 'ALSO_BASE'

    with agent.override_prompts(instructions=None):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions is None


@pytest.mark.anyio
async def test_override_prompts_concurrent_isolation():
    """Test that concurrent overrides are isolated from each other."""
    agent = Agent('test', system_prompt=('ORIG',))

    async def run_with(instr: str, sys_p: tuple[str, ...]):
        with agent.override_prompts(instructions=instr, system_prompts=sys_p):
            with capture_run_messages() as messages:
                await agent.run('Hi', model=TestModel(custom_output_text='ok'))
            req = _first_request(messages)
            return req.instructions, _system_prompt_texts(req.parts)[: len(sys_p)]

    a, b = await asyncio.gather(
        run_with('A', ('SA',)),
        run_with('B', ('SB1', 'SB2')),
    )

    assert a == ('A', ['SA'])
    assert b == ('B', ['SB1', 'SB2'])


def test_override_prompts_both_instructions_and_system():
    """Test overriding both instructions and system prompts simultaneously."""
    agent = Agent('test', instructions='ORIG_INSTR', system_prompt='ORIG_SYSTEM')

    with agent.override_prompts(instructions='NEW_INSTR', system_prompts=('NEW_SYS1', 'NEW_SYS2')):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'NEW_INSTR'
    sys_texts = _system_prompt_texts(req.parts)
    assert sys_texts[:2] == ['NEW_SYS1', 'NEW_SYS2']


def test_override_prompts_nested_contexts():
    """Test nested override contexts."""
    agent = Agent('test', system_prompt='ORIG')

    with agent.override_prompts(system_prompts=('OUTER',)):
        with capture_run_messages() as outer_messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

        with agent.override_prompts(system_prompts=('INNER',)):
            with capture_run_messages() as inner_messages:
                agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    outer_req = _first_request(outer_messages)
    inner_req = _first_request(inner_messages)

    assert _system_prompt_texts(outer_req.parts)[:1] == ['OUTER']
    assert _system_prompt_texts(inner_req.parts)[:1] == ['INNER']


def test_override_prompts_empty_tuple_system():
    """Test that empty tuple for system_prompts clears system prompts."""
    agent = Agent('test', system_prompt=('ORIG1', 'ORIG2'))

    with agent.override_prompts(system_prompts=()):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    sys_texts = _system_prompt_texts(req.parts)
    # Should not have the original system prompts, but might have other parts
    assert 'ORIG1' not in sys_texts
    assert 'ORIG2' not in sys_texts


@pytest.mark.anyio
async def test_override_prompts_async_run():
    """Test override_prompts with async run method."""
    agent = Agent('test', instructions='ORIG')

    with agent.override_prompts(instructions='ASYNC_OVERRIDE'):
        with capture_run_messages() as messages:
            await agent.run('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'ASYNC_OVERRIDE'


def test_override_prompts_with_dynamic_prompts():
    """Test override_prompts interacting with dynamic prompts."""
    agent = Agent('test')

    dynamic_value = 'DYNAMIC'

    @agent.system_prompt
    def dynamic_sys() -> str:
        return dynamic_value

    @agent.instructions
    def dynamic_instr() -> str:
        return 'DYNAMIC_INSTR'

    # Override should take precedence over dynamic prompts for instructions
    # For system prompts, overrides are added in addition to dynamic prompts
    with agent.override_prompts(instructions='OVERRIDE_INSTR', system_prompts=('OVERRIDE_SYS',)):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = _first_request(messages)
    assert req.instructions == 'OVERRIDE_INSTR'
    sys_texts = _system_prompt_texts(req.parts)
    # The override system prompt should be present
    assert 'OVERRIDE_SYS' in sys_texts
