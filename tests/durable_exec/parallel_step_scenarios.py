"""Parallel tool calls in one run step over a run-held `per_run_step=False` dynamic toolset.

Shared by the engine suites. The run resolves the factory once and every durable unit reuses what it
resolved; the scenario's toolset records the step state each tool-call unit listed its tool under and
the one it called it under.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Literal

from pydantic_ai import Agent, FunctionToolset, ModelMessage, ModelResponse, RunContext, TextPart, ToolCallPart
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, ToolsetTool, WrapperToolset

Variant = Literal['copy', 'nested', 'inplace']
"""How the factory's toolset prepares each step.

`copy` returns a new instance from `for_run_step`. `nested` is a `per_run_step=True` `DynamicToolset`, as
`PrefixTools(Toolset(factory))` contributes from inside a dynamic capability. `inplace` updates itself and
returns `self`, as `docs/toolsets.md` describes.
"""

VARIANTS: tuple[Variant, ...] = ('copy', 'nested', 'inplace')

READINESS_WAIT_TIMEOUT = 10


@dataclass
class StepState:
    """The step state each tool-call unit listed its tool under, and the one it called it under."""

    listed: dict[str, int] = field(default_factory=dict[str, int])
    called: dict[str, int] = field(default_factory=dict[str, int])
    both_listed: asyncio.Event = field(default_factory=asyncio.Event)
    serials: Iterator[int] = field(default_factory=lambda: count(1))


@dataclass
class StepStateToolset(WrapperToolset[None]):
    """Returns a new instance from every `for_run_step`, carrying the step state its tools report."""

    state: StepState = field(kw_only=True)
    serial: int = field(default=0, kw_only=True)

    async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        wrapped = await self.wrapped.for_run_step(ctx)
        return StepStateToolset(wrapped, state=self.state, serial=next(self.state.serials))

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        tools = await self.wrapped.get_tools(ctx)
        if ctx.tool_call_id is not None:
            # Hold each call unit here until both have listed, as a listing that does I/O would.
            self.state.listed[ctx.tool_call_id] = self.serial
            if len(self.state.listed) == 2:
                self.state.both_listed.set()
            async with asyncio.timeout(READINESS_WAIT_TIMEOUT):
                await self.state.both_listed.wait()
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        assert ctx.tool_call_id is not None
        self.state.called[ctx.tool_call_id] = self.serial
        return await self.wrapped.call_tool(name, tool_args, ctx, tool)


@dataclass
class InPlaceStepStateToolset(StepStateToolset):
    """Updates its own step state and returns `self` from `for_run_step`."""

    async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        self.serial = next(self.state.serials)
        return self


def echo(text: str) -> str:
    return text


def parallel_calls_agent(
    name: str, variant: Variant, state: StepState, durability: AbstractCapability[None]
) -> Agent[None, str]:
    """An agent whose model calls `echo` twice in one response, from a run-held dynamic toolset."""

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        if variant == 'nested':
            return DynamicToolset(lambda ctx: StepStateToolset(FunctionToolset([echo]), state=state))
        if variant == 'inplace':
            return InPlaceStepStateToolset(FunctionToolset([echo]), state=state)
        return StepStateToolset(FunctionToolset([echo]), state=state)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[
                ToolCallPart('echo', {'text': 'a'}, tool_call_id='call-a'),
                ToolCallPart('echo', {'text': 'b'}, tool_call_id='call-b'),
            ]
        )

    return Agent(
        FunctionModel(model),
        deps_type=type(None),
        name=name,
        toolsets=[DynamicToolset(factory, per_run_step=False, id='step_state')],
        capabilities=[durability],
    )
