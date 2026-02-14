from __future__ import annotations as _annotations

import asyncio
import collections
import json
import logging
import os
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from pydantic_ai.messages import (
    ImageUrl,
    ModelMessage,
    UserContent,
)

try:
    if TYPE_CHECKING:
        import pjrpc  # type: ignore
        from pjrpc.server import AsyncDispatcher, AsyncMethodRegistry  # type: ignore
        from starlette.applications import Starlette  # type: ignore
        from starlette.routing import WebSocketRoute  # type: ignore
        from starlette.websockets import WebSocket, WebSocketDisconnect  # type: ignore
    else:
        import pjrpc  # type: ignore
        from pjrpc.server import AsyncDispatcher, AsyncMethodRegistry  # type: ignore
        from starlette.applications import Starlette  # type: ignore
        from starlette.routing import WebSocketRoute  # type: ignore
        from starlette.websockets import WebSocket, WebSocketDisconnect  # type: ignore
except ImportError as _import_error:
    raise ImportError(
        'Please install `pjrpc` and `starlette` to use `Agent.to_acp()` method, '
        'you can use the `acp` optional group — `pip install "pydantic-ai-slim[acp]"`'
    ) from _import_error

# Configure logging
logger = logging.getLogger('pydantic_ai.acp')

# --- ACP Pydantic Models ---


class TextContent(BaseModel):
    type: Literal['text'] = 'text'
    text: str


class ImageContent(BaseModel):
    type: Literal['image'] = 'image'
    image: str
    """Base64 encoded image data or URL."""


class ResourceLinkContent(BaseModel):
    type: Literal['resourceLink'] = 'resourceLink'
    uri: str


ContentBlock = Annotated[TextContent | ImageContent | ResourceLinkContent, Field(discriminator='type')]


class InitializeParams(BaseModel):
    client_name: str | None = None
    client_version: str | None = None
    protocol_version: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    _meta: dict[str, Any] | None = None


class InitializeResult(BaseModel):
    server_name: str
    server_version: str
    protocol_version: str | None = None
    capabilities: dict[str, Any]
    _meta: dict[str, Any] | None = None


class AuthenticateParams(BaseModel):
    method_id: str
    params: dict[str, Any]
    _meta: dict[str, Any] | None = None


class AuthenticateResult(BaseModel):
    success: bool
    data: dict[str, Any] | None = None
    _meta: dict[str, Any] | None = None


class SessionNewParams(BaseModel):
    mcpServers: dict[str, Any] | None = None
    _meta: dict[str, Any] | None = None


class SessionNewResult(BaseModel):
    sessionId: str
    _meta: dict[str, Any] | None = None


class SessionLoadParams(BaseModel):
    sessionId: str
    mcpServers: dict[str, Any] | None = None
    _meta: dict[str, Any] | None = None


class SessionLoadResult(BaseModel):
    sessionId: str
    history: list[dict[str, Any]]
    _meta: dict[str, Any] | None = None


class SessionSetModeParams(BaseModel):
    sessionId: str
    modeId: str
    _meta: dict[str, Any] | None = None


class SessionSetConfigOptionParams(BaseModel):
    sessionId: str
    key: str
    value: Any
    _meta: dict[str, Any] | None = None


class PermissionRequestParams(BaseModel):
    sessionId: str
    toolCall: dict[str, Any] | None = None
    options: list[dict[str, Any]] | None = None
    _meta: dict[str, Any] | None = None


class SessionPromptParams(BaseModel):
    sessionId: str
    prompt: str | list[ContentBlock]
    context: dict[str, Any] | None = None
    _meta: dict[str, Any] | None = None


class SessionCancelParams(BaseModel):
    sessionId: str
    _meta: dict[str, Any] | None = None


class SessionCancelResult(BaseModel):
    success: bool
    _meta: dict[str, Any] | None = None


# --- File System Params ---


class FsReadParams(BaseModel):
    sessionId: str
    path: str
    start_line: int | None = None
    end_line: int | None = None
    _meta: dict[str, Any] | None = None


class FsWriteParams(BaseModel):
    sessionId: str
    path: str
    content: str
    _meta: dict[str, Any] | None = None


# --- Terminal Params ---


class TerminalCreateParams(BaseModel):
    sessionId: str
    command: str
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None
    _meta: dict[str, Any] | None = None


class TerminalCreateResult(BaseModel):
    terminalId: str
    _meta: dict[str, Any] | None = None


class TerminalKillParams(BaseModel):
    sessionId: str
    terminalId: str
    _meta: dict[str, Any] | None = None


class TerminalKillResult(BaseModel):
    success: bool
    _meta: dict[str, Any] | None = None


class TerminalWaitForExitParams(BaseModel):
    sessionId: str
    terminalId: str
    _meta: dict[str, Any] | None = None


class TerminalWaitForExitResult(BaseModel):
    exitCode: int
    _meta: dict[str, Any] | None = None


class TerminalReleaseParams(BaseModel):
    sessionId: str
    terminalId: str
    _meta: dict[str, Any] | None = None


class TerminalOutputParams(BaseModel):
    sessionId: str
    terminalId: str
    _meta: dict[str, Any] | None = None


class TerminalOutputResult(BaseModel):
    output: str
    truncated: bool = False
    exitStatus: int | None = None
    _meta: dict[str, Any] | None = None


# --- Notifications ---


class AgentMessageChunk(BaseModel):
    type: Literal['agent_message_chunk'] = 'agent_message_chunk'
    delta: str


class AgentThoughtChunk(BaseModel):
    type: Literal['agent_thought_chunk'] = 'agent_thought_chunk'
    delta: str


class ToolCall(BaseModel):
    type: Literal['tool_call'] = 'tool_call'
    id: str
    name: str
    arguments: dict[str, Any]


class ToolCallUpdate(BaseModel):
    type: Literal['tool_call_update'] = 'tool_call_update'
    tool_call_id: str
    status: str


class TerminalOutput(BaseModel):
    type: Literal['terminal_output'] = 'terminal_output'
    terminalId: str
    output: str


# Strict Union for updates with all types
SessionUpdateUnion = Annotated[
    AgentMessageChunk | AgentThoughtChunk | ToolCall | ToolCallUpdate | TerminalOutput | dict[str, Any],
    Field(discriminator='type'),
]


class SessionUpdateNotification(BaseModel):
    sessionId: str
    update: SessionUpdateUnion
    _meta: dict[str, Any] | None = None


# --- Session Management ---


@dataclass
class SessionState:
    id: str
    history: list[ModelMessage] = field(default_factory=list)
    active_terminals: set[str] = field(default_factory=set)


# --- Terminal Management ---


@dataclass
class TerminalState:
    id: str
    process: asyncio.subprocess.Process
    output_buffer: collections.deque[str] = field(default_factory=lambda: collections.deque(maxlen=1000))
    """Buffer stores the last 1000 lines/chunks for polling."""


class TerminalManager:
    def __init__(self):
        self._terminals: dict[str, TerminalState] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    async def create(
        self,
        command: str,
        args: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        on_output: Callable[[str, str], Awaitable[None]],
    ) -> str:
        tid = str(uuid.uuid4())

        # Merge env
        param_env = os.environ.copy()
        if env:
            param_env.update(env)

        process = await asyncio.create_subprocess_exec(
            command, *args, cwd=cwd, env=param_env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

        self._terminals[tid] = TerminalState(id=tid, process=process)

        # Start monitoring task
        task = asyncio.create_task(self._monitor_output(tid, process, on_output))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return tid

    async def kill(self, tid: str):
        if tid in self._terminals:
            t = self._terminals[tid]
            try:
                t.process.kill()
                await t.process.wait()
            except ProcessLookupError:
                pass
            del self._terminals[tid]

    async def wait_for_exit(self, tid: str) -> int:
        if tid in self._terminals:
            t = self._terminals[tid]
            return await t.process.wait()
        return -1

    async def release(self, tid: str):
        await self.kill(tid)

    async def get_output(self, tid: str) -> tuple[str, int | None, bool]:
        output = ''
        exit_code = None
        truncated = False
        if tid in self._terminals:
            t = self._terminals[tid]
            if t.process.returncode is not None:
                exit_code = t.process.returncode

            # Join buffer
            output = ''.join(t.output_buffer)
            truncated = len(t.output_buffer) == t.output_buffer.maxlen

        return output, exit_code, truncated

    async def _monitor_output(
        self, tid: str, process: asyncio.subprocess.Process, on_output: Callable[[str, str], Awaitable[None]]
    ):
        if not process.stdout:
            return
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode('utf-8', errors='replace')

                # Append to buffer for polling
                if tid in self._terminals:
                    self._terminals[tid].output_buffer.append(text)

                # Stream via callback
                await on_output(tid, text)
        except Exception:
            pass
        finally:
            pass


# --- FastACP App ---


@dataclass
class FastACP(Starlette):
    """ASGI application and ACP logic that exposes an Agent via the Agent Client Protocol (ACP).

    Uses pjrpc for JSON-RPC 2.0 handling.
    """

    agent: Any
    sessions: dict[str, SessionState] = field(default_factory=dict)
    terminals: TerminalManager = field(default_factory=TerminalManager)
    root_dir: Path = field(default_factory=Path.cwd)

    def __init__(self, agent: Any, name: str | None = None, debug: bool = False, root_dir: Path | str | None = None):
        self.agent = agent
        self.name = name or agent.name or 'pydantic-ai-agent'
        self.sessions = {}
        self.terminals = TerminalManager()
        self.root_dir = Path(root_dir).resolve() if root_dir else Path.cwd()

        # JSON-RPC Registry
        self.registry: Any = AsyncMethodRegistry()
        self.registry.add(self._initialize, 'initialize')
        self.registry.add(self._session_new, 'session/new')
        self.registry.add(self._session_load, 'session/load')
        self.registry.add(self._session_prompt, 'session/prompt')
        self.registry.add(self._session_cancel, 'session/cancel')
        self.registry.add(self._session_set_mode, 'session/set_mode')
        self.registry.add(self._session_set_config_option, 'session/set_config_option')
        self.registry.add(self._session_request_permission, 'session/request_permission')
        self.registry.add(self._authenticate, 'authenticate')
        self.registry.add(self._ext_handler, 'ext/*')

        # File System & Terminal Methods
        self.registry.add(self._fs_read, 'fs/read_text_file')
        self.registry.add(self._fs_write, 'fs/write_text_file')
        self.registry.add(self._terminal_create, 'terminal/create')
        self.registry.add(self._terminal_kill, 'terminal/kill')
        self.registry.add(self._terminal_wait_for_exit, 'terminal/wait_for_exit')
        self.registry.add(self._terminal_release, 'terminal/release')
        self.registry.add(self._terminal_output, 'terminal/output')

        # Starlette init
        routes = [
            WebSocketRoute('/acp', self.handle_websocket),
        ]
        super().__init__(debug=debug, routes=routes)

    async def handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        dispatcher: Any = AsyncDispatcher(self.registry, params_by_name=True)

        async def send_notification(method: str, params: dict[str, Any]):
            # Pass None as id for Notifications
            notification = pjrpc.Notification(method=method, params=params)
            try:
                await websocket.send_text(json.dumps(notification.to_json()))
            except Exception:
                pass

        while True:
            try:
                message = await websocket.receive_text()
                response = await dispatcher.dispatch(message, context={'notify': send_notification})
                if response:
                    await websocket.send_text(response)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f'WebSocket Error: {e}')
                break

    def _validate_path(self, path_str: str) -> Path:
        """Securely resolve path against root_dir."""
        try:
            path = Path(path_str)
            if not path.is_absolute():
                path = (self.root_dir / path).resolve()
            else:
                path = path.resolve()

            if not str(path).startswith(str(self.root_dir)):
                raise ValueError('Path outside root directory')
            return path
        except Exception as e:
            raise pjrpc.exc.JsonRpcError(code=-32002, message=f'Invalid path: {e}')

    async def _initialize(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        p_ver = params.get('protocol_version') if params else None
        return InitializeResult(
            server_name=self.name,
            server_version='0.6.0',
            protocol_version=p_ver or '2024-11-05',
            capabilities={
                'prompts': True,
                'promptCapabilities': {
                    'text': True,
                    'image': True,
                    'resourceLink': True,
                    'audio': False,
                    'embeddedContext': False,
                },
                'streaming': True,
                'notificationCapabilities': {
                    'agent_message_chunk': True,
                    'terminal_output': True,
                    'tool_call': True,
                },
                'session': {
                    'cancel': True,
                    'loadSession': True,
                    'modes': True,
                },
                'tools': bool(self.agent._function_tools or self.agent._tools),
                'resources': True,
                'fs': {
                    'readTextFile': True,
                    'writeTextFile': True,
                },
                'terminal': True,
                'terminalCapabilities': {'create': True, 'output': True, 'poll': True, 'release': True},
                'authentication': {'basic': False},
                'mcpCapabilities': {'http': False, 'sse': False},
            },
        ).model_dump(exclude_none=True)

    async def _authenticate(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Authenticate stub. See schema for auth methods."""
        try:
            AuthenticateParams.model_validate(params or {})
        except ValidationError as e:
            raise pjrpc.exc.JsonRpcError(code=-32000, message=f'Auth Error: {e}')

        return AuthenticateResult(success=True).model_dump(exclude_none=True)

    async def _ext_handler(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Catch-all for extension methods."""
        logger.warning('Received unhandled extension request')
        raise pjrpc.exc.MethodNotFoundError()

    async def _session_new(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Create a new session."""
        try:
            SessionNewParams.model_validate(params or {})
        except ValidationError:
            pass

        session_id = str(uuid.uuid4())
        self.sessions[session_id] = SessionState(id=session_id)
        return SessionNewResult(sessionId=session_id).model_dump(exclude_none=True)

    async def _session_load(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Load an existing session and return history."""
        try:
            p = SessionLoadParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        if p.sessionId in self.sessions:
            session = self.sessions[p.sessionId]
            # Serialize history using TypeAdapter
            history = TypeAdapter(list[ModelMessage]).dump_python(session.history)
            return SessionLoadResult(sessionId=p.sessionId, history=history).model_dump(exclude_none=True)
        else:
            raise pjrpc.exc.JsonRpcError(code=-32001, message='Session not found')

    async def _session_set_mode(self, params: dict[str, Any], **kwargs: Any) -> dict[str, bool]:
        """Set agent mode (Stub)."""
        try:
            SessionSetModeParams.model_validate(params)
            return {'success': True}
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

    async def _session_set_config_option(self, params: dict[str, Any], **kwargs: Any) -> dict[str, bool]:
        """Set config option (Stub)."""
        try:
            SessionSetConfigOptionParams.model_validate(params)
            return {'success': True}
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

    async def _session_request_permission(self, params: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        """Inbound permission request stub (for testing or client-driven permission)."""
        try:
            PermissionRequestParams.model_validate(params or {})
            return {'outcome': 'selected', 'optionId': 'default'}
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

    async def _session_cancel(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any] | None:
        """Cancel a session.

        See https://agentclientprotocol.com/protocol/schema#session-cancel
        """
        try:
            p = SessionCancelParams.model_validate(params)
        except ValidationError as e:
            # If notification, error is logged by dispatcher, not returned
            logger.error(f'Invalid Session Cancel Params: {e}')
            return None

        if p.sessionId in self.sessions:
            session = self.sessions[p.sessionId]
            # Cleanup associated terminals
            for tid in list(session.active_terminals):
                await self.terminals.kill(tid)

            del self.sessions[p.sessionId]
            return SessionCancelResult(success=True).model_dump(exclude_none=True)
        else:
            return None

    async def _fs_read(self, params: dict[str, Any], **kwargs: Any) -> str:
        try:
            p = FsReadParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        path = self._validate_path(p.path)
        if not path.exists():
            raise pjrpc.exc.JsonRpcError(code=-32002, message='File not found')

        try:
            content = path.read_text(encoding='utf-8')
            lines = content.splitlines()
            start = (p.start_line or 1) - 1
            end = p.end_line or len(lines)
            return '\n'.join(lines[start:end])
        except Exception as e:
            raise pjrpc.exc.JsonRpcError(code=-32000, message=str(e))

    async def _fs_write(self, params: dict[str, Any], **kwargs: Any) -> dict[str, bool]:
        try:
            p = FsWriteParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        path = self._validate_path(p.path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(p.content, encoding='utf-8')
            return {'success': True}
        except Exception as e:
            raise pjrpc.exc.JsonRpcError(code=-32000, message=str(e))

    async def _terminal_create(self, params: dict[str, Any], context: dict[str, Any], **kwargs: Any) -> dict[str, str]:
        try:
            p = TerminalCreateParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        notify: Callable[[str, dict], Awaitable[None]] | None = context.get('notify')

        if p.sessionId not in self.sessions:
            raise pjrpc.exc.JsonRpcError(code=-32001, message='Session not found')

        session = self.sessions[p.sessionId]

        async def handle_output(tid: str, output: str):
            if notify:
                await notify(
                    'session/update',
                    SessionUpdateNotification(
                        sessionId=p.sessionId, update=TerminalOutput(terminalId=tid, output=output)
                    ).model_dump(exclude_none=True),
                )

        try:
            # Resolve CWD relative to root if provided, else root_dir
            if p.cwd:
                cwd = str(self._validate_path(p.cwd))
            else:
                cwd = str(self.root_dir)

            tid = await self.terminals.create(p.command, p.args, cwd, p.env, handle_output)
            session.active_terminals.add(tid)
            return TerminalCreateResult(terminalId=tid).model_dump(exclude_none=True)
        except Exception as e:
            raise pjrpc.exc.JsonRpcError(code=-32000, message=str(e))

    async def _terminal_kill(self, params: dict[str, Any], **kwargs: Any) -> dict[str, bool]:
        try:
            p = TerminalKillParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        if p.sessionId in self.sessions:
            self.sessions[p.sessionId].active_terminals.discard(p.terminalId)

        await self.terminals.kill(p.terminalId)
        return TerminalKillResult(success=True).model_dump(exclude_none=True)

    async def _terminal_wait_for_exit(self, params: dict[str, Any], **kwargs: Any) -> dict[str, int]:
        try:
            p = TerminalWaitForExitParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        exit_code = await self.terminals.wait_for_exit(p.terminalId)
        return TerminalWaitForExitResult(exitCode=exit_code).model_dump(exclude_none=True)

    async def _terminal_release(self, params: dict[str, Any], **kwargs: Any) -> dict[str, bool]:
        """Release terminal resources."""
        try:
            p = TerminalReleaseParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        await self.terminals.release(p.terminalId)
        if p.sessionId in self.sessions:
            self.sessions[p.sessionId].active_terminals.discard(p.terminalId)
        return {'success': True}

    async def _terminal_output(self, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Poll terminal output."""
        try:
            p = TerminalOutputParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        output, exit_code, truncated = await self.terminals.get_output(p.terminalId)
        return TerminalOutputResult(output=output, truncated=truncated, exitStatus=exit_code).model_dump(
            exclude_none=True
        )

    async def _session_prompt(self, params: dict[str, Any], context: dict[str, Any], **kwargs: Any) -> str:
        """Handle a prompt within a session, streaming results."""
        try:
            p = SessionPromptParams.model_validate(params)
        except ValidationError as e:
            raise pjrpc.exc.InvalidParamsError(str(e))

        if p.sessionId not in self.sessions:
            raise pjrpc.exc.JsonRpcError(code=-32001, message='Session not found')

        session = self.sessions[p.sessionId]
        notify: Callable[[str, dict], Awaitable[None]] | None = context.get('notify')

        # Convert ACP ContentBlocks to Pydantic AI UserContent
        user_content: str | Sequence[UserContent]
        if isinstance(p.prompt, str):
            user_content = p.prompt
        else:
            parts: list[UserContent] = []
            for block in p.prompt:
                if block.type == 'text':
                    parts.append(block.text)
                elif block.type == 'image':
                    parts.append(ImageUrl(url=block.image))
                elif block.type == 'resourceLink':
                    parts.append(f'User provided resource: {block.uri}')
            user_content = parts

        current_history = list(session.history)

        try:
            async with self.agent.run_stream(user_content, message_history=current_history) as result:
                async for chunk in result.stream_text(delta=True):
                    if p.sessionId not in self.sessions:
                        if notify:
                            await notify(
                                'session/update',
                                SessionUpdateNotification(
                                    sessionId=p.sessionId, update={'type': 'call_update', 'stopReason': 'cancelled'}
                                ).model_dump(exclude_none=True),
                            )
                        break

                    if notify:
                        await notify(
                            'session/update',
                            SessionUpdateNotification(
                                sessionId=p.sessionId, update=AgentMessageChunk(delta=chunk)
                            ).model_dump(exclude_none=True),
                        )

                if p.sessionId in self.sessions:
                    final_data = await result.get_output()
                    session.history = result.all_messages()
                    return str(final_data)
                else:
                    return 'Session Cancelled'
        except Exception as e:
            logger.error(f'Prompt Error: {e}')
            raise pjrpc.exc.JsonRpcError(code=-32000, message=str(e))


def agent_to_acp(
    agent: Any, *, name: str | None = None, debug: bool = False, root_dir: Path | str | None = None
) -> FastACP:
    """Create a FastACP server from an agent."""
    return FastACP(agent=agent, name=name, debug=debug, root_dir=root_dir)


# --- Stdio Transport Helper ---


async def run_stdio(agent: Any, name: str | None = None, root_dir: Path | str | None = None):
    """Run an ACP server for the agent using standard input/output transport."""
    acp_app = FastACP(agent, name=name, root_dir=root_dir)
    dispatcher = AsyncDispatcher(acp_app.registry, params_by_name=True)

    loop = asyncio.get_event_loop()

    def shutdown():
        pass

    try:
        loop.add_signal_handler(signal.SIGINT, shutdown)
        loop.add_signal_handler(signal.SIGTERM, shutdown)
    except NotImplementedError:
        pass

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    async def send_notification(method: str, params: dict):
        notification = pjrpc.Notification(method=method, params=params)
        msg = json.dumps(notification.to_json())
        sys.stdout.write(msg + '\n')
        sys.stdout.flush()

    while True:
        try:
            line = await reader.readline()
            if not line:
                break

            msg = line.decode('utf-8').strip()
            if not msg:
                continue

            # Context contains notifier
            response = await dispatcher.dispatch(msg, context={'notify': send_notification})

            if response:
                sys.stdout.write(response + '\n')
                sys.stdout.flush()
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f'Stdio Loop Error: {e}')
            pass
