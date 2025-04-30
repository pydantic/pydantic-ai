import ast
import inspect
import subprocess
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import time
from typing import Any, Callable, Literal, cast, override

import anyio
import httpx
import pydantic_core
from mcp import ClientSession, types as mcp_types
from mcp.shared.context import RequestContext
from pydantic import BaseModel, Json
from pydantic._internal._validate_call import ValidateCallWrapper  # pyright: ignore[reportPrivateImportUsage]
from typing_extensions import TypedDict

from .mcp import MCPServerHTTP, MCPServerStdio

__all__ = 'mcp_run_python_stdio', 'MCPRunPythonHTTP'

MCP_RUN_PYTHON_VERSION = '0.0.13'
Callback = Callable[..., Awaitable[Any]]


def mcp_run_python_stdio(callbacks: Sequence[Callback] = (), *, local_code: bool = False) -> MCPServerStdio:
    """Prepare a server server connection using `'stdio'` transport.

    Args:
        callbacks: A sequence of callback functions to be register on the server.
        local_code: Whether to run local `mcp-run-python` code, this is mostly used for development and testing.

    Returns:
        A server connection definition.
    """
    return MCPServerStdio(
        'deno',
        args=_deno_args('stdio', callbacks, local_code),
        cwd='mcp-run-python' if local_code else None,
        custom_sampling_callback=_PythonSamplingCallback(callbacks) if callbacks else None,
    )


@dataclass
class MCPRunPythonHTTP:
    """Setup for `mcp-run-python` running with HTTP transport."""

    callbacks: Sequence[Callback] = ()
    """Callbacks to be registered on the server."""
    port: int = 3001
    """Port to run the server on."""
    local_code: bool = False
    """Whether to run local `mcp-run-python` code, this is mostly used for development and testing."""

    @property
    def url(self) -> str:
        """URL the server will be run on."""
        return f'http://localhost:{self.port}/sse'

    def server_def(self, url: str | None = None) -> MCPServerHTTP:
        """Create a server definition to pass to a pydantic-ai [`Agent`][pydantic_ai.Agent]."""
        return MCPServerHTTP(
            url or self.url,
            custom_sampling_callback=_PythonSamplingCallback(self.callbacks) if self.callbacks else None,
        )

    def run(self) -> None:
        """Run the server and block until it is terminated."""
        try:
            subprocess.run(self._args(), cwd=self._cwd(), check=True)
        except KeyboardInterrupt:
            pass

    @asynccontextmanager
    async def run_context(self, server_wait_timeout: float | None = 2) -> AsyncIterator[None]:
        """Run the server as an async context manager.

        Args:
            server_wait_timeout: The timeout in seconds to wait for the server to start, or `None` to not wait.
        """
        p = await anyio.open_process(self._args(), cwd=self._cwd(), stdout=None, stderr=None)
        async with p:
            if server_wait_timeout:
                await self.wait_for_server(server_wait_timeout)
            yield
            p.terminate()

    async def wait_for_server(self, timeout: float = 2):
        """Wait for the server to be ready."""
        async with httpx.AsyncClient(timeout=0.01) as client:
            start = time()
            while True:
                try:
                    await client.head(self.url)
                except httpx.RequestError:
                    if time() - start > timeout:
                        raise TimeoutError(f'Server did not start within {timeout} seconds')
                    await anyio.sleep(0.1)
                else:
                    break

    def _args(self) -> list[str]:
        return ['deno'] + _deno_args('http', self.callbacks, self.local_code) + ['--port', str(self.port)]

    def _cwd(self) -> str | None:
        return 'mcp-run-python' if self.local_code else None


def _deno_args(mode: Literal['stdio', 'http'], callbacks: Sequence[Callback], local_code: bool) -> list[str]:
    args = [
        'run',
        '-N',
        '-R=node_modules',
        '-W=node_modules',
        '--node-modules-dir=auto',
        'src/main.ts' if local_code else f'jsr:@pydantic/mcp-run-python@{MCP_RUN_PYTHON_VERSION}',
        mode,
    ]

    if callbacks:
        sigs = '\n\n'.join(_callback_signature(cb) for cb in callbacks)
        args += ['--callbacks', sigs]
    return args


def _callback_signature(func: Callback) -> str:
    """Extract the signature of a function.

    This simply means getting the source code of the function, and removing the body of the function while keeping the docstring.
    """
    source = inspect.getsource(func)
    ast_mod = ast.parse(source)
    assert isinstance(ast_mod, ast.Module), f'Expected Module, got {type(ast_mod)}'
    assert len(ast_mod.body) == 1, f'Expected single function definition, got {len(ast_mod.body)}'
    f = ast_mod.body[0]
    assert isinstance(f, ast.AsyncFunctionDef), f'Expected an async function, got {type(func)}'
    lines = source.splitlines()
    e = f.body[0]
    # if the first expression is a docstring, keep it and no need for an ellipsis as the body
    if isinstance(e, ast.Expr) and isinstance(e.value, ast.Constant) and isinstance(e.value.value, str):
        e = f.body[1]
        lines = lines[: e.lineno - 1]
    else:
        lines = lines[: e.lineno - 1]
        lines.append(e.col_offset * ' ' + '...')

    # if the function has any decorators, this will remove them.
    if f.lineno != 1:
        lines = lines[f.lineno - 1 :]

    return '\n'.join(lines)


class _PythonSamplingCallback:
    def __init__(self, callbacks: Sequence[Callback]):
        self.function_lookup: dict[str, ValidateCallWrapper] = {}
        for callback in callbacks:
            name = callback.__name__
            if name in self.function_lookup:
                raise ValueError(f'Duplicate callback name: {name}')
            self.function_lookup[name] = ValidateCallWrapper(
                callback,  # pyright: ignore[reportArgumentType]
                None,
                False,
                None,
            )

    async def __call__(
        self,
        context: RequestContext[ClientSession, Any],
        params: mcp_types.CreateMessageRequestParams,
    ) -> mcp_types.CreateMessageResult | mcp_types.ErrorData | None:
        if not params.metadata or params.metadata.get('pydantic_custom_use') != '__python_function_call__':
            return None

        call_metadata = _PythonCallMetadata.model_validate(params.metadata)
        if function_wrapper := self.function_lookup.get(call_metadata.func):
            content: _CallSuccess | _CallError
            try:
                return_value = await function_wrapper.__pydantic_validator__.validate_python(call_metadata.args_kwargs)
            except ValueError as e:
                # special support for ValueError since it's commonly subclassed, and it's the parent of ValidationError
                # TODO we should probably have specific support for other common errors
                content = _CallError(exc_type='ValueError', message=str(e), kind='error')
            except Exception as e:
                content = _CallError(exc_type=e.__class__.__name__, message=str(e), kind='error')
            else:
                content = _CallSuccess(return_value=return_value, kind='success')

            content_text = pydantic_core.to_json(content, fallback=_json_fallback).decode()
            return mcp_types.CreateMessageResult(
                role='assistant', content=mcp_types.TextContent(type='text', text=content_text), model='python'
            )
        else:
            raise LookupError(f'Function `{call_metadata.func}` not found')

    @override
    def __repr__(self) -> str:
        return f'<_PythonSamplingCallback: {", ".join(map(repr, self.function_lookup))}>'


class _PythonCallMetadata(BaseModel):
    func: str
    args: Json[list[Any]] | None = None  # JSON
    kwargs: Json[dict[str, Any]] | None = None  # JSON

    @property
    def args_kwargs(self) -> pydantic_core.ArgsKwargs:
        return pydantic_core.ArgsKwargs(tuple(self.args or ()), self.kwargs)


class _CallSuccess(TypedDict):
    return_value: Any
    kind: Literal['success']


class _CallError(TypedDict):
    exc_type: str
    message: str
    kind: Literal['error']


def _json_fallback(value: Any) -> Any:
    tp = cast(Any, type(value))
    if tp.__module__ == 'numpy':
        if tp.__name__ in {'ndarray', 'matrix'}:
            return value.tolist()
        else:
            return value.item()
    else:
        return repr(value)
