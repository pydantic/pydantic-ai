from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, replace
from typing import Any

from typing_extensions import Self

from .._instructions import normalize_toolset_instructions
from .._run_context import AgentDepsT, RunContext
from .._utils import gather
from ..exceptions import UserError
from ..messages import InstructionPart
from .abstract import AbstractToolset, ToolsetTool
from .wrapper import WrapperToolset


def _instruction_source(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
    """The toolset whose `id` keys the instruction blocks reached through `toolset`.

    A wrapper reports no `id` of its own — it isn't itself a registered toolset — but
    `WrapperToolset.get_instructions` keys what it passes through with the *wrapped* toolset's id,
    and a capability's contributed toolset always arrives inside a `CapabilityOwnedToolset`.
    Unwrapping is what makes one toolset reached through two different wrappers a single source
    rather than a false conflict: `Agent(toolsets=[shared.filtered(...), shared.prefixed('p')])`.
    """
    while isinstance(toolset, WrapperToolset):
        toolset = toolset.wrapped
    return toolset


@dataclass(kw_only=True)
class _CombinedToolsetTool(ToolsetTool[AgentDepsT]):
    """A tool definition for a combined toolset tools that keeps track of the source toolset and tool."""

    source_toolset: AbstractToolset[AgentDepsT]
    source_tool: ToolsetTool[AgentDepsT]


@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets.

    See [toolset docs](../toolsets.md#combining-toolsets) for more information.
    """

    toolsets: Sequence[AbstractToolset[AgentDepsT]]

    _exit_stack: AsyncExitStack | None = field(init=False, default=None)

    @property
    def id(self) -> str | None:
        return None

    @property
    def label(self) -> str:
        return f'{self.__class__.__name__}({", ".join(toolset.label for toolset in self.toolsets)})'  # pragma: no cover

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_toolsets = await gather(*(t.for_run(ctx) for t in self.toolsets))
        return replace(self, toolsets=new_toolsets)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        new_toolsets = await gather(*(t.for_run_step(ctx) for t in self.toolsets))
        if all(new is old for new, old in zip(new_toolsets, self.toolsets)):
            return self
        return replace(self, toolsets=new_toolsets)

    async def __aenter__(self) -> Self:
        async with AsyncExitStack() as exit_stack:
            for toolset in self.toolsets:
                await exit_stack.enter_async_context(toolset)
            self._exit_stack = exit_stack.pop_all()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        toolsets_tools = await gather(*(toolset.get_tools(ctx) for toolset in self.toolsets))
        all_tools: dict[str, ToolsetTool[AgentDepsT]] = {}

        for toolset, tools in zip(self.toolsets, toolsets_tools):
            for name, tool in tools.items():
                tool_toolset = tool.toolset
                if existing_tool := all_tools.get(name):
                    capitalized_toolset_label = tool_toolset.label[0].upper() + tool_toolset.label[1:]
                    raise UserError(
                        f'{capitalized_toolset_label} defines a tool whose name conflicts with existing tool from {existing_tool.toolset.label}: {name!r}. {toolset.tool_name_conflict_hint}'
                    )

                tool_def = tool.tool_def
                if tool_def.toolset_id is None and tool_toolset.id is not None:
                    tool_def = replace(tool_def, toolset_id=tool_toolset.id)

                all_tools[name] = _CombinedToolsetTool(
                    toolset=tool_toolset,
                    tool_def=tool_def,
                    max_retries=tool.max_retries,
                    args_validator=tool.args_validator,
                    args_validator_func=tool.args_validator_func,
                    source_toolset=toolset,
                    source_tool=tool,
                )
        return all_tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert isinstance(tool, _CombinedToolsetTool)
        return await tool.source_toolset.call_tool(name, tool_args, ctx, tool.source_tool)

    def apply(self, visitor: Callable[[AbstractToolset[AgentDepsT]], None]) -> None:
        for toolset in self.toolsets:
            toolset.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return replace(self, toolsets=[toolset.visit_and_replace(visitor) for toolset in self.toolsets])

    async def get_instructions(
        self, ctx: RunContext[AgentDepsT]
    ) -> str | InstructionPart | Sequence[str | InstructionPart] | None:
        results = await gather(*(ts.get_instructions(ctx) for ts in self.toolsets))
        parts: list[InstructionPart] = []
        # An `InstructionPart.id` has to name one block, so two toolsets contributing under one
        # `toolset:<id>` would leave an application unable to tell whose text it is addressing.
        # Checked as the blocks are keyed rather than when the agent is built: a shared id makes
        # nothing ambiguous until both toolsets actually contribute one, and most never do. That
        # also exempts the ids the framework assigns on the user's behalf for free -- `durable_exec`
        # feeds the agent's own function toolset back through `override(toolsets=...)` alongside the
        # one `_build_toolset_list` prepends, and neither carries instructions.
        sources_by_id: dict[str, AbstractToolset[AgentDepsT]] = {}
        for toolset, result in zip(self.toolsets, results):
            toolset_parts = normalize_toolset_instructions(result, toolset.id)
            source = _instruction_source(toolset)
            for part in toolset_parts:
                if part.id is None:
                    continue
                if (existing := sources_by_id.setdefault(part.id, source)) is not source:
                    raise UserError(
                        f'Two toolsets have the same `id` {existing.id!r} and both contribute instructions, '
                        f'so {part.id!r} would address blocks from each. '
                        'Toolset `id`s must be unique among all toolsets registered with the same agent.'
                    )
            parts.extend(toolset_parts)
        return parts or None
