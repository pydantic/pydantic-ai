from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.wrapper import WrapperToolset

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer


class HatchetMCPServer(WrapperToolset[AgentDepsT], ABC):
    """A wrapper for MCPServer that integrates with Hatchet, turning call_tool and get_tools to Hatchet tasks."""

    def __init__(
        self,
        wrapped: MCPServer,
    ):
        super().__init__(wrapped)

        pass
