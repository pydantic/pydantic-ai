from collections.abc import Sequence

from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebSearchTool
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc

from .abstract import AbstractCapability

_BUILTIN_WEB_SEARCH_TOOL = WebSearchTool()


class WebSearch(AbstractCapability[AgentDepsT]):
    """A capability that enables web search via builtin tools."""

    # def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
    #     return FunctionToolset([duckduckgo_search_tool()]).prepared(
    #         lambda ctx, tool_defs: [
    #             replace(tool_def, prefers_builtin=_BUILTIN_WEB_SEARCH_TOOL.unique_id) for tool_def in tool_defs
    #         ],
    #     )

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        return [_BUILTIN_WEB_SEARCH_TOOL]
