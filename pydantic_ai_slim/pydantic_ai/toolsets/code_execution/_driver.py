"""Stdio-based sandbox driver for CPython runtimes.

Self-contained script with zero dependencies beyond the Python 3.10+ stdlib.
Runs inside any sandbox (Docker, E2B, Modal, etc.) and communicates with the
host via NDJSON (newline-delimited JSON) over stdin/stdout.

Protocol:
    Host -> Driver: init, result, error
    Driver -> Host: call, calls_ready, complete, error

This file is both a module (importable for path resolution) and an executable
script (python -u _driver.py).
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
import traceback
from typing import Any

# Protocol channel — overridden in __main__ with an fd-level redirect.
_real_stdout = sys.stdout


def _write_msg(msg: dict[str, Any]) -> None:
    """Write a single NDJSON message to the real stdout."""
    _real_stdout.write(json.dumps(msg, default=str) + '\n')
    _real_stdout.flush()


def _transform_last_expr(code: str) -> str:
    """If the last statement is an expression, wrap it in a return statement.

    Uses AST parsing so multiline expressions (dicts, lists, etc.) are handled
    correctly. String-based approaches break on edge cases.
    """
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1]
        ret = ast.Return(value=last_expr.value)
        ast.copy_location(ret, last_expr)
        tree.body[-1] = ret
        ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _build_proxy(
    name: str,
    loop: asyncio.AbstractEventLoop,
    call_counter: list[int],
    pending_futures: dict[int, asyncio.Future[Any]],
    result_cache: dict[str, Any],
    calls_ready_handle: list[asyncio.Handle | None],
) -> Any:
    """Build an eager proxy function for a declared tool.

    Calling the proxy immediately sends a call message and returns a future.
    `await` just waits for the result. This enables fire-then-await parallelism:
    `f1 = tool_a(); f2 = tool_b(); r1 = await f1; r2 = await f2` fires both
    calls instantly.

    NOT async — returns a future directly. If this were `async def`, calling
    without `await` would return an unstarted coroutine (no parallelism).

    After sending a call message, a `calls_ready` boundary message is
    scheduled via `loop.call_soon`. Each proxy call cancels the previous
    handle, so exactly one `calls_ready` is emitted after the last
    synchronous proxy call in a batch — when the event loop runs on the
    next `await`.
    """

    def proxy(*args: Any, **kwargs: Any) -> asyncio.Future[Any]:
        call_counter[0] += 1
        cid = call_counter[0]

        if str(cid) in result_cache:
            f: asyncio.Future[Any] = loop.create_future()
            f.set_result(result_cache[str(cid)])
            return f

        future: asyncio.Future[Any] = loop.create_future()
        pending_futures[cid] = future
        _write_msg({'type': 'call', 'id': cid, 'function': name, 'args': list(args), 'kwargs': kwargs})

        # Schedule a calls_ready fence to fire when the event loop runs next.
        if calls_ready_handle[0] is not None:
            calls_ready_handle[0].cancel()
        calls_ready_handle[0] = loop.call_soon(_write_msg, {'type': 'calls_ready'})

        return future

    return proxy


def _compile_code(code: str, code_globals: dict[str, Any]) -> Any | None:
    """Parse, transform, and compile LLM code into an async function.

    Returns the compiled async function, or None if a syntax error was
    reported (error message already sent to host).
    """
    try:
        transformed = _transform_last_expr(code)
    except SyntaxError as e:
        _write_msg({'type': 'error', 'error': str(e), 'error_type': 'syntax'})
        return None

    func_code = 'async def __code__():\n'
    for line in transformed.splitlines():
        func_code += f'    {line}\n'

    try:
        compiled = compile(func_code, '<code>', 'exec')
    except SyntaxError as e:  # pragma: no cover
        _write_msg({'type': 'error', 'error': str(e), 'error_type': 'syntax'})
        return None

    exec(compiled, code_globals)
    return code_globals['__code__']


async def _stdin_reader(
    reader: asyncio.StreamReader,
    pending_futures: dict[int, asyncio.Future[Any]],
) -> None:
    """Read result/error messages from stdin and resolve corresponding futures."""
    while True:
        raw = await reader.readline()
        if not raw:
            break
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            sys.stderr.write(f'Warning: malformed JSON from host: {raw[:200]!r}\n')
            continue

        msg_type = msg.get('type')
        cid = msg.get('id')

        if msg_type == 'result' and cid is not None:
            future = pending_futures.pop(cid, None)
            if future is not None and not future.done():
                future.set_result(msg.get('result'))
        elif msg_type == 'error' and cid is not None:
            future = pending_futures.pop(cid, None)
            if future is not None and not future.done():
                future.set_exception(RuntimeError(msg.get('error', 'Tool error')))


async def _execute(init_msg: dict[str, Any], reader: asyncio.StreamReader) -> None:
    """Parse code, build proxies, execute, and report the result."""
    code: str = init_msg.get('code', '')
    functions: list[str] = init_msg.get('functions', [])
    # TODO(sequential): Read `sequential_functions` from init_msg and build sync proxies
    # for them. Sequential proxies should either (a) be async coroutines that await their
    # result future inline rather than returning it eagerly, or (b) send a special
    # `sync_call` message type where the driver blocks until the result arrives before
    # continuing code execution. See MontyEnvironment._execution_loop for reference.
    result_cache: dict[str, Any] = init_msg.get('result_cache', {})

    if not code.strip():
        _write_msg({'type': 'complete', 'result': None})
        return

    loop = asyncio.get_running_loop()
    call_counter: list[int] = [0]
    pending_futures: dict[int, asyncio.Future[Any]] = {}
    calls_ready_handle: list[asyncio.Handle | None] = [None]

    code_globals: dict[str, Any] = {'__builtins__': __builtins__, 'asyncio': asyncio}
    for name in functions:
        code_globals[name] = _build_proxy(name, loop, call_counter, pending_futures, result_cache, calls_ready_handle)

    code_fn = _compile_code(code, code_globals)
    if code_fn is None:
        return

    stdin_task = asyncio.create_task(_stdin_reader(reader, pending_futures))

    try:
        result = await code_fn()
        _write_msg({'type': 'complete', 'result': result})
    except SyntaxError as e:
        _write_msg({'type': 'error', 'error': str(e), 'error_type': 'syntax'})
    except Exception:
        _write_msg({'type': 'error', 'error': traceback.format_exc(), 'error_type': 'runtime'})
    finally:
        stdin_task.cancel()
        try:
            await stdin_task
        except asyncio.CancelledError:
            pass


async def _main(proto_stdin: Any) -> None:
    """Entry point: connect stdin, read init message, execute code."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, proto_stdin)

    line = await reader.readline()
    if not line:  # pragma: lax no cover
        _write_msg({'type': 'error', 'error': 'No init message received', 'error_type': 'runtime'})
        return

    try:
        init_msg = json.loads(line)
    except json.JSONDecodeError as e:  # pragma: lax no cover
        _write_msg({'type': 'error', 'error': f'Invalid init message: {e}', 'error_type': 'runtime'})
        return

    if init_msg.get('type') != 'init':  # pragma: lax no cover
        _write_msg(
            {
                'type': 'error',
                'error': f'Expected init message, got: {init_msg.get("type")}',
                'error_type': 'runtime',
            }
        )
        return

    await _execute(init_msg, reader)


if __name__ == '__main__':
    import os

    # Flush Python-level buffers before touching the underlying fds.
    sys.stdout.flush()
    sys.stderr.flush()

    # Save protocol fds to new fd numbers, then redirect fd 0/1 so LLM code
    # (os.write, os.read, subprocesses, C extensions) cannot read/corrupt
    # the protocol channel. The duped fds survive the redirect.
    _proto_stdin_fd = os.dup(0)
    _proto_stdout_fd = os.dup(1)

    os.dup2(2, 1)  # fd 1 → stderr
    _devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(_devnull, 0)  # fd 0 → /dev/null
    os.close(_devnull)

    _proto_stdin = os.fdopen(_proto_stdin_fd, 'rb', buffering=0)
    _real_stdout = os.fdopen(_proto_stdout_fd, 'w', buffering=1)

    asyncio.run(_main(_proto_stdin))
