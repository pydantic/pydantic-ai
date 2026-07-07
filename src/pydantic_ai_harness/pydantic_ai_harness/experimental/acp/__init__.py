"""Expose a Pydantic AI agent over the Agent Client Protocol (ACP).

ACP lets terminal UIs and editors (such as Zed and Toad) drive a coding agent over stdio
JSON-RPC. [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent] adapts a
Pydantic AI [`Agent`][pydantic_ai.Agent] to that interface, and
[`run_acp_stdio`][pydantic_ai_harness.experimental.acp.run_acp_stdio] serves it over stdio.
"""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.acp._adapter import PydanticAIACPAgent
from pydantic_ai_harness.experimental.acp._client_toolsets import (
    AcpFileSystemToolset,
    AcpTerminalToolset,
    acp_filesystem,
    acp_terminal,
)
from pydantic_ai_harness.experimental.acp._permission import ToolCallPermission, default_permission_scope
from pydantic_ai_harness.experimental.acp._presentation import (
    ToolCallPresentation,
    chain_presenters,
    default_coding_presenter,
)
from pydantic_ai_harness.experimental.acp._server import run_acp_stdio, run_acp_stdio_sync
from pydantic_ai_harness.experimental.acp._session import AcpSession, AcpSessionConfig, McpServer
from pydantic_ai_harness.experimental.acp._store import InMemorySessionStore, SessionStore, StoredSession

warn_experimental('acp')

__all__ = [
    'AcpFileSystemToolset',
    'AcpSession',
    'AcpSessionConfig',
    'AcpTerminalToolset',
    'InMemorySessionStore',
    'McpServer',
    'PydanticAIACPAgent',
    'SessionStore',
    'StoredSession',
    'ToolCallPermission',
    'ToolCallPresentation',
    'acp_filesystem',
    'acp_terminal',
    'chain_presenters',
    'default_coding_presenter',
    'default_permission_scope',
    'run_acp_stdio',
    'run_acp_stdio_sync',
]
