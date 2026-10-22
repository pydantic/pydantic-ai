"""`DurableRunContext` rebuilt from a JSON payload, as an engine with out-of-process units sees it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec import JSON_CODEC, DurabilityEngineSpec, DurableRunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CustomEvent
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


@dataclass
class Progress(CustomEvent):
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
