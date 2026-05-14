from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace

from .._run_context import AgentDepsT, RunContext
from ..tools import ToolDefinition, ToolSelector, ToolsPrepareFunc, matches_tool_selector
from .abstract import AbstractToolset
from .prepared import PreparedToolset


@dataclass(init=False)
class SetDescriptionToolset(PreparedToolset[AgentDepsT]):
    """A toolset that surgically modifies the descriptions of selected tools.

    Pass exactly one of `replace`, `append`, or `prepend` — see
    [`SetToolDescription`][pydantic_ai.capabilities.SetToolDescription] for details.
    """

    prepare_func: ToolsPrepareFunc[AgentDepsT] = field(init=False, repr=False)
    replace: str | None
    append: str | None
    prepend: str | None

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        replace: str | None = None,
        append: str | None = None,
        prepend: str | None = None,
        tools: ToolSelector[AgentDepsT] = 'all',
    ) -> None:
        from ..capabilities.set_tool_description import _compose_description  # pyright: ignore[reportPrivateUsage]

        provided = [n for n, v in (('replace', replace), ('append', append), ('prepend', prepend)) if v is not None]
        if not provided:
            raise TypeError('`SetDescriptionToolset` requires exactly one of `replace`, `append`, or `prepend`.')
        if len(provided) > 1:
            joined = ', '.join(f'`{name}`' for name in provided)
            raise TypeError(f'`SetDescriptionToolset` cannot mix {joined} — pick exactly one.')

        self.replace = replace
        self.append = append
        self.prepend = prepend
        selector = tools

        async def _set_description(
            ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            resolved: list[ToolDefinition] = []
            for td in tool_defs:
                if await matches_tool_selector(selector, ctx, td):
                    td = _dc_replace(
                        td,
                        description=_compose_description(td.description, replace, append, prepend),
                    )
                resolved.append(td)
            return resolved

        super().__init__(wrapped=wrapped, prepare_func=_set_description)
