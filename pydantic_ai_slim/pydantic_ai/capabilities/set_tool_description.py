"""Capability that surgically modifies tool descriptions."""

from __future__ import annotations

from dataclasses import dataclass, replace

from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.tools import ToolDefinition, ToolSelector, matches_tool_selector
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from .abstract import AbstractCapability


@dataclass(init=False)
class SetToolDescription(AbstractCapability[AgentDepsT]):
    """Capability that surgically modifies the descriptions of selected tools.

    Pass exactly one of `replace`, `append`, or `prepend`:

    - `replace`: overwrite the existing description.
    - `append`: add text after the existing description (joined with two newlines).
    - `prepend`: add text before the existing description (joined with two newlines).

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import SetToolDescription

    # Replace the description on one tool.
    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            SetToolDescription(tools=['search'], replace='Search the knowledge base.'),
        ],
    )

    # Append a usage hint to every tool that has metadata `category='destructive'`.
    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            SetToolDescription(
                tools={'category': 'destructive'},
                append='Use with extreme care — this action cannot be undone.',
            ),
        ],
    )
    ```
    """

    tools: ToolSelector[AgentDepsT]
    replace: str | None
    append: str | None
    prepend: str | None

    def __init__(
        self,
        *,
        tools: ToolSelector[AgentDepsT] = 'all',
        replace: str | None = None,
        append: str | None = None,
        prepend: str | None = None,
    ) -> None:
        provided = [n for n, v in (('replace', replace), ('append', append), ('prepend', prepend)) if v is not None]
        if not provided:
            raise TypeError('`SetToolDescription` requires exactly one of `replace`, `append`, or `prepend`.')
        if len(provided) > 1:
            joined = ', '.join(f'`{name}`' for name in provided)
            raise TypeError(f'`SetToolDescription` cannot mix {joined} — pick exactly one.')
        self.tools = tools
        self.replace = replace
        self.append = append
        self.prepend = prepend

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'SetToolDescription'

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        selector = self.tools
        replace_with = self.replace
        append_with = self.append
        prepend_with = self.prepend

        async def _set_description(
            ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            resolved: list[ToolDefinition] = []
            for td in tool_defs:
                if await matches_tool_selector(selector, ctx, td):
                    td = replace(
                        td, description=_compose_description(td.description, replace_with, append_with, prepend_with)
                    )
                resolved.append(td)
            return resolved

        return PreparedToolset(toolset, _set_description)


def _compose_description(
    current: str | None,
    replace_with: str | None,
    append_with: str | None,
    prepend_with: str | None,
) -> str:
    """Build the new description from one of `replace`/`append`/`prepend`.

    `append` and `prepend` join with two newlines; if there's no existing description
    they fall back to the new fragment alone.
    """
    if replace_with is not None:
        return replace_with
    if append_with is not None:
        return f'{current}\n\n{append_with}' if current else append_with
    # prepend_with is not None (validated in __init__)
    assert prepend_with is not None
    return f'{prepend_with}\n\n{current}' if current else prepend_with
