from collections.abc import Sequence

from pydantic_ai.builtin_tools import AbstractBuiltinTool, WebSearchTool
from pydantic_ai.tools import AgentDepsT, BuiltinToolFunc

from .abstract import AbstractCapability

_BUILTIN_WEB_SEARCH_TOOL = WebSearchTool()


class WebSearch(AbstractCapability[AgentDepsT]):
    """A capability that enables web search via builtin tools."""

    # TODO: Add toolset-based fallback for models without builtin web search (#3212)

    def get_builtin_tools(self) -> Sequence[AbstractBuiltinTool | BuiltinToolFunc[AgentDepsT]]:
        return [_BUILTIN_WEB_SEARCH_TOOL]
