from __future__ import annotations

from typing import TYPE_CHECKING

from ._agent import DBOSAgent, DBOSParallelExecutionMode
from ._model import DBOSModel
from ._utils import StepConfig

if TYPE_CHECKING:
    from ._mcp_server import DBOSMCPServer

__all__ = ['DBOSAgent', 'DBOSModel', 'DBOSMCPServer', 'DBOSParallelExecutionMode', 'StepConfig']


def __getattr__(name: str) -> object:
    if name == 'DBOSMCPServer':
        from ._mcp_server import DBOSMCPServer

        return DBOSMCPServer
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
