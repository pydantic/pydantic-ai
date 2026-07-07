"""An ACP agent mounting client-backed fs/terminal toolsets, for the native-over-stdio test.

The model calls `read_file` then `run_command` on the first turn; the session config mounts the
ACP-client-backed toolsets, so those tool calls must route over the wire to the test's client.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from pydantic_ai_harness.experimental.acp import (
    AcpSession,
    AcpSessionConfig,
    acp_filesystem,
    acp_terminal,
    run_acp_stdio_sync,
)


async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
    if any(isinstance(part, ToolReturnPart) for message in messages for part in getattr(message, 'parts', [])):
        yield 'done'
        return
    yield {
        0: DeltaToolCall(name='read_file', json_args=json.dumps({'path': 'notes.txt'})),
        1: DeltaToolCall(name='run_command', json_args=json.dumps({'command': 'echo hi'})),
    }


def session_config(session: AcpSession) -> AcpSessionConfig[None]:
    toolsets = [toolset for toolset in (acp_filesystem(session), acp_terminal(session)) if toolset is not None]
    return AcpSessionConfig(deps=None, toolsets=toolsets)


if __name__ == '__main__':
    run_acp_stdio_sync(Agent(FunctionModel(stream_function=stream)), session_config=session_config)
