"""`DurableRunContext` rebuilt from a JSON payload, as an engine with out-of-process units sees it."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    DurabilityEngineSpec,
    DurableRunContext,
    ToolsetCallToolParams,
    revalidate_function_call,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CustomEvent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage


@dataclass
class Progress(CustomEvent, name='durable_run_context_progress'):
    done: int = 0


SPEC = DurabilityEngineSpec(engine_name='Example', durable_unit_noun='job', durable_container_noun='pipeline')


def _round_trip(ctx: RunContext[Any]) -> dict[str, Any]:
    """Encode the serialized context the way a JSON-journal engine would, then decode it."""
    serialized = DurableRunContext.serialize_run_context(ctx)
    return JSON_CODEC.load(dict[str, Any], JSON_CODEC.dump(dict[str, Any], serialized))


async def test_restore_rebuilds_carried_fields_and_guards_the_rest():
    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=str)

    @agent.tool
    def capture(ctx: RunContext[str]) -> str:
        payloads.append(_round_trip(ctx))
        return 'ok'

    await agent.run('hello', deps='worker deps')
    [payload] = payloads

    ctx = DurableRunContext.restore(payload, deps='worker deps', agent=agent, engine_spec=SPEC)

    assert ctx.deps == 'worker deps'
    assert ctx.tool_name == 'capture'
    assert isinstance(ctx.usage, RunUsage)
    assert 'capture' in ctx.available_tool_names
    assert ctx.agent is agent
    assert ctx.root_capability is agent.root_capability
    assert ctx.tool_manager is None

    with pytest.raises(UserError, match="'messages' is not available on 'DurableRunContext' inside a durable unit"):
        ctx.messages
    with pytest.raises(UserError, match=r'`ctx.enqueue\(\)` is not supported inside a durable job'):
        ctx.enqueue('later')
    with pytest.raises(UserError, match='Emitting events from a tool or event stream handler is not supported'):
        await ctx.emit(Progress())


async def test_subclass_carries_extra_fields():
    class PromptRunContext(DurableRunContext[Any]):
        @classmethod
        def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
            return {**super().serialize_run_context(ctx), 'prompt': ctx.prompt}

    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=type(None))

    @agent.tool
    def capture(ctx: RunContext[None]) -> str:
        payloads.append(PromptRunContext.serialize_run_context(ctx))
        return 'ok'

    await agent.run('hello')
    ctx = PromptRunContext.restore(payloads[0], deps=None, agent=None, engine_spec=SPEC)

    assert ctx.prompt == 'hello'
    assert ctx.agent is None


async def test_validation_context_is_rebuilt_from_the_agent_on_first_read():
    calls: list[str | None] = []

    def build_validation_context(ctx: RunContext[str]) -> dict[str, str]:
        calls.append(ctx.tool_name)
        return {'tenant': ctx.deps}

    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=str, validation_context=build_validation_context)

    @agent.tool
    def capture(ctx: RunContext[str]) -> str:
        payloads.append(_round_trip(ctx))
        return 'ok'

    await agent.run('hello', deps='acme')
    calls.clear()

    ctx = DurableRunContext.restore(payloads[0], deps='acme', agent=agent, engine_spec=SPEC)
    assert calls == []
    assert ctx.validation_context == {'tenant': 'acme'}
    assert ctx.validation_context == {'tenant': 'acme'}
    assert calls == ['capture']

    without_agent = DurableRunContext.restore(payloads[0], deps='acme', agent=None, engine_spec=SPEC)
    with pytest.raises(UserError, match="'validation_context' is not available"):
        without_agent.validation_context


async def test_revalidate_function_call_rebuilds_the_tool_and_validated_arguments():
    payloads: list[dict[str, Any]] = []
    toolset = FunctionToolset[None]()

    @toolset.tool
    def add(ctx: RunContext[None], a: int, b: int) -> int:
        payloads.append(_round_trip(ctx))
        return a + b

    agent = Agent(TestModel(), name='example', deps_type=type(None), toolsets=[toolset])
    await agent.run('hello')
    ctx = DurableRunContext.restore(payloads[0], deps=None, agent=agent, engine_spec=SPEC)

    params = await revalidate_function_call(
        toolset, ToolsetCallToolParams('add', tool_args={'a': '1', 'b': 2}, ctx=ctx, tool=None), engine_spec=SPEC
    )
    assert params.tool is not None
    assert params.tool_args == {'a': 1, 'b': 2}

    with pytest.raises(UserError, match=r"Tool 'gone' not found in toolset .* not supported with Example"):
        await revalidate_function_call(
            toolset, ToolsetCallToolParams('gone', tool_args={}, ctx=ctx, tool=None), engine_spec=SPEC
        )


async def test_validation_context_rebuilt_on_a_shallow_copy_leaves_the_original_rebuildable():
    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=str, validation_context=lambda ctx: {'tenant': ctx.deps})

    @agent.tool
    def capture(ctx: RunContext[str]) -> str:
        payloads.append(_round_trip(ctx))
        return 'ok'

    await agent.run('hello', deps='acme')
    original = DurableRunContext.restore(payloads[0], deps='acme', agent=agent, engine_spec=SPEC)
    shallow = copy.copy(original)

    assert shallow.validation_context == {'tenant': 'acme'}
    assert original.validation_context == {'tenant': 'acme'}
