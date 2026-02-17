"""Tests for the stdio driver protocol.

Tests _driver.py as a local subprocess (no Docker, no runtime class needed).
Communicates directly via NDJSON over stdin/stdout pipes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from pydantic_ai.runtime._driver import (
    _build_proxy,  # pyright: ignore[reportPrivateUsage]
    _compile_code,  # pyright: ignore[reportPrivateUsage]
    _execute,  # pyright: ignore[reportPrivateUsage]
    _StderrRedirect,  # pyright: ignore[reportPrivateUsage]
    _stdin_reader,  # pyright: ignore[reportPrivateUsage]
    _transform_last_expr,  # pyright: ignore[reportPrivateUsage]
)

DRIVER_PATH = Path(__file__).parents[2] / 'pydantic_ai_slim' / 'pydantic_ai' / 'runtime' / '_driver.py'

pytestmark = pytest.mark.anyio


async def start_driver(
    code: str,
    functions: list[str] | None = None,
    result_cache: dict[str, object] | None = None,
) -> asyncio.subprocess.Process:
    """Start the driver subprocess and send the init message."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        '-u',
        str(DRIVER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    init_msg: dict[str, object] = {
        'type': 'init',
        'code': code,
        'functions': functions or [],
    }
    if result_cache is not None:
        init_msg['result_cache'] = result_cache
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(init_msg).encode() + b'\n')
    await proc.stdin.drain()
    return proc


async def read_msg(proc: asyncio.subprocess.Process) -> dict[str, object]:
    """Read a single NDJSON message from the driver's stdout."""
    assert proc.stdout is not None
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=10.0)
    assert line, 'Driver produced no output (EOF)'
    return json.loads(line)


async def send_msg(proc: asyncio.subprocess.Process, msg: dict[str, object]) -> None:
    """Send a single NDJSON message to the driver's stdin."""
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(msg).encode() + b'\n')
    await proc.stdin.drain()


async def test_simple_tool_call():
    """Driver sends call message, receives result, returns complete."""
    proc = await start_driver('await add(x=1, y=2)', functions=['add'])

    # Read call message
    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['function'] == 'add'
    assert msg['kwargs'] == {'x': 1, 'y': 2}
    assert msg['id'] == 1

    # Read calls_ready fence (batch boundary)
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    # Send result
    await send_msg(proc, {'type': 'result', 'id': 1, 'result': 3})

    # Read completion
    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 3

    proc.kill()
    await proc.wait()


async def test_parallel_tool_calls():
    """Fire-then-await pattern sends multiple calls before awaiting."""
    code = 'f1 = add(x=1, y=2)\nf2 = add(x=3, y=4)\nr1 = await f1\nr2 = await f2\nr1 + r2'
    proc = await start_driver(code, functions=['add'])

    # Read 2 call messages (both sent before any result)
    msg1 = await read_msg(proc)
    msg2 = await read_msg(proc)
    assert msg1['type'] == 'call'
    assert msg1['id'] == 1
    assert msg2['type'] == 'call'
    assert msg2['id'] == 2

    # Read calls_ready fence (batch boundary after last synchronous call)
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    # Send both results
    await send_msg(proc, {'type': 'result', 'id': 1, 'result': 3})
    await send_msg(proc, {'type': 'result', 'id': 2, 'result': 7})

    # Read completion
    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 10

    proc.kill()
    await proc.wait()


async def test_result_cache_all_hits():
    """With full result cache, no call messages emitted."""
    proc = await start_driver(
        'r = await add(x=1, y=2)\nr',
        functions=['add'],
        result_cache={'1': 3},
    )

    # Should go straight to complete (no call message)
    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 3

    proc.kill()
    await proc.wait()


async def test_result_cache_partial_hit():
    """Partial cache: cached calls skip RPC, uncached go through."""
    code = 'f1 = add(x=1, y=2)\nf2 = add(x=3, y=4)\nr1 = await f1\nr2 = await f2\nr1 + r2'
    proc = await start_driver(code, functions=['add'], result_cache={'1': 3})

    # Only 1 call message (call 2, since call 1 is cached)
    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['id'] == 2

    # Read calls_ready fence
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    # Send result for call 2
    await send_msg(proc, {'type': 'result', 'id': 2, 'result': 7})

    # Read completion
    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 10

    proc.kill()
    await proc.wait()


async def test_syntax_error():
    """Syntax errors in code produce error message with type=syntax."""
    proc = await start_driver('def while invalid')

    msg = await read_msg(proc)
    assert msg['type'] == 'error'
    assert msg['error_type'] == 'syntax'

    proc.kill()
    await proc.wait()


async def test_runtime_error():
    """Runtime exceptions produce error message with type=runtime."""
    proc = await start_driver('1 / 0')

    msg = await read_msg(proc)
    assert msg['type'] == 'error'
    assert msg['error_type'] == 'runtime'
    assert 'ZeroDivisionError' in str(msg['error'])

    proc.kill()
    await proc.wait()


async def test_empty_code():
    """Empty code returns None."""
    proc = await start_driver('')

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] is None

    proc.kill()
    await proc.wait()


async def test_no_function_calls():
    """Code that doesn't call functions executes locally."""
    proc = await start_driver('1 + 2')

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 3

    proc.kill()
    await proc.wait()


async def test_last_expression_multiline_dict():
    """Multiline dict as last expression is correctly returned."""
    proc = await start_driver('{"a": 1, "b": 2}')

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == {'a': 1, 'b': 2}

    proc.kill()
    await proc.wait()


async def test_print_goes_to_stderr():
    """print() in code goes to stderr, not stdout (protocol protected)."""
    proc = await start_driver('print("hello")\n42')

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 42

    # Check stderr contains the print output
    assert proc.stderr is not None
    # Give the process a moment to finish writing stderr
    proc.kill()
    await proc.wait()
    stderr_output = await proc.stderr.read()
    assert b'hello' in stderr_output


async def test_tool_error_propagated():
    """Host error message surfaces as exception in code."""
    code = 'result = "no error"\ntry:\n    await bad_tool()\nexcept Exception as e:\n    result = str(e)\nresult'
    proc = await start_driver(code, functions=['bad_tool'])

    # Read call message
    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['function'] == 'bad_tool'
    call_id = msg['id']

    # Read calls_ready fence
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    # Send error response
    await send_msg(proc, {'type': 'error', 'id': call_id, 'error': 'tool failed'})

    # Code catches the exception and returns the error string
    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert 'tool failed' in str(msg['result'])

    proc.kill()
    await proc.wait()


async def test_undefined_function():
    """Calling undefined function raises runtime error."""
    proc = await start_driver('await nonexistent()')

    msg = await read_msg(proc)
    assert msg['type'] == 'error'
    assert msg['error_type'] == 'runtime'
    assert 'nonexistent' in str(msg['error'])

    proc.kill()
    await proc.wait()


async def test_sequential_call_ids():
    """Call IDs are sequential across different proxy functions."""
    code = (
        'f1 = tool_a(x=1)\n'
        'f2 = tool_b(y=2)\n'
        'f3 = tool_a(x=3)\n'
        'r1 = await f1\n'
        'r2 = await f2\n'
        'r3 = await f3\n'
        '[r1, r2, r3]'
    )
    proc = await start_driver(code, functions=['tool_a', 'tool_b'])

    # Read 3 call messages — IDs should be 1, 2, 3
    msg1 = await read_msg(proc)
    msg2 = await read_msg(proc)
    msg3 = await read_msg(proc)
    assert msg1['id'] == 1
    assert msg1['function'] == 'tool_a'
    assert msg2['id'] == 2
    assert msg2['function'] == 'tool_b'
    assert msg3['id'] == 3
    assert msg3['function'] == 'tool_a'

    # Read calls_ready fence (one per batch, after all 3 synchronous calls)
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    # Send all results
    await send_msg(proc, {'type': 'result', 'id': 1, 'result': 'a1'})
    await send_msg(proc, {'type': 'result', 'id': 2, 'result': 'b1'})
    await send_msg(proc, {'type': 'result', 'id': 3, 'result': 'a2'})

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == ['a1', 'b1', 'a2']

    proc.kill()
    await proc.wait()


async def test_positional_args():
    """Positional args are sent in the call message's args field."""
    proc = await start_driver('await add(1, 2)', functions=['add'])

    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['function'] == 'add'
    assert msg['args'] == [1, 2]
    assert msg['kwargs'] == {}

    # Read calls_ready fence
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    await send_msg(proc, {'type': 'result', 'id': 1, 'result': 3})

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 3

    proc.kill()
    await proc.wait()


async def test_mixed_positional_and_keyword_args():
    """Mixed positional and keyword args are split correctly."""
    proc = await start_driver('await add(1, y=2)', functions=['add'])

    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['function'] == 'add'
    assert msg['args'] == [1]
    assert msg['kwargs'] == {'y': 2}

    # Read calls_ready fence
    msg = await read_msg(proc)
    assert msg['type'] == 'calls_ready'

    await send_msg(proc, {'type': 'result', 'id': 1, 'result': 3})

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == 3

    proc.kill()
    await proc.wait()


async def test_stderr_redirect_all_properties():
    """StderrRedirect stream properties are accessible from user code."""
    code = (
        'import sys\n'
        '{\n'
        '    "isatty": sys.stdout.isatty(),\n'
        '    "writable": sys.stdout.writable(),\n'
        '    "readable": sys.stdout.readable(),\n'
        '    "closed": sys.stdout.closed,\n'
        '    "fileno_works": isinstance(sys.stdout.fileno(), int),\n'
        '    "has_encoding": isinstance(sys.stdout.encoding, str),\n'
        '}'
    )
    proc = await start_driver(code)
    msg = await read_msg(proc)
    assert msg == {
        'type': 'complete',
        'result': {
            'isatty': False,
            'writable': True,
            'readable': False,
            'closed': False,
            'fileno_works': True,
            'has_encoding': True,
        },
    }
    proc.kill()
    await proc.wait()


async def test_code_ending_with_assignment():
    """Code ending with assignment returns None (not the assigned value)."""
    proc = await start_driver('x = 42')
    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] is None
    proc.kill()
    await proc.wait()


async def test_runtime_syntax_error_in_eval():
    """SyntaxError raised at runtime (e.g. in eval) is reported as syntax error."""
    proc = await start_driver('eval("def while")')
    msg = await read_msg(proc)
    assert msg['type'] == 'error'
    assert msg['error_type'] == 'syntax'
    proc.kill()
    await proc.wait()


async def _start_raw_driver() -> asyncio.subprocess.Process:
    """Start driver without sending init — for testing init error paths."""
    return await asyncio.create_subprocess_exec(
        sys.executable,
        '-u',
        str(DRIVER_PATH),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def test_main_error_paths():
    """Driver handles missing/invalid/wrong-type init messages."""
    # No init (close stdin)
    proc = await _start_raw_driver()
    assert proc.stdin is not None
    proc.stdin.close()
    msg = await read_msg(proc)
    assert 'No init message' in str(msg['error'])
    proc.kill()
    await proc.wait()

    # Invalid JSON
    proc = await _start_raw_driver()
    assert proc.stdin is not None
    proc.stdin.write(b'not json\n')
    await proc.stdin.drain()
    msg = await read_msg(proc)
    assert 'Invalid init message' in str(msg['error'])
    proc.kill()
    await proc.wait()

    # Wrong type
    proc = await _start_raw_driver()
    assert proc.stdin is not None
    proc.stdin.write(json.dumps({'type': 'wrong'}).encode() + b'\n')
    await proc.stdin.drain()
    msg = await read_msg(proc)
    assert 'Expected init message' in str(msg['error'])
    proc.kill()
    await proc.wait()


# =============================================================================
# In-process unit tests — import driver functions directly so coverage tracks them
# =============================================================================


class TestStderrRedirect:
    """Test _StderrRedirect stream methods via direct import."""

    def test_write(self, capsys: pytest.CaptureFixture[str]) -> None:
        r = _StderrRedirect()
        r.write('hello')
        assert capsys.readouterr().err == 'hello'

    def test_fileno(self) -> None:
        r = _StderrRedirect()
        assert r.fileno() == sys.stderr.fileno()

    def test_isatty(self) -> None:
        assert _StderrRedirect().isatty() is False

    def test_writable(self) -> None:
        assert _StderrRedirect().writable() is True

    def test_readable(self) -> None:
        assert _StderrRedirect().readable() is False

    def test_encoding(self) -> None:
        assert _StderrRedirect().encoding == sys.stderr.encoding

    def test_errors(self) -> None:
        assert _StderrRedirect().errors == sys.stderr.errors

    def test_closed(self) -> None:
        assert _StderrRedirect().closed is False


def test_transform_last_expr_assignment():
    """Code ending in assignment has no return added."""
    result = _transform_last_expr('x = 42')
    assert 'return' not in result


def test_compile_code_syntax_error(monkeypatch: pytest.MonkeyPatch):
    """_compile_code with invalid code sends error and returns None."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)
    result = _compile_code('def while', {})
    assert result is None
    assert messages[0]['error_type'] == 'syntax'


async def test_build_proxy_cache_hit():
    """Proxy with a cache hit returns an already-resolved future."""
    loop = asyncio.get_running_loop()
    call_counter: list[int] = [0]
    pending: dict[int, asyncio.Future[object]] = {}
    cache: dict[str, object] = {'1': 42}
    handle: list[asyncio.Handle | None] = [None]

    proxy = _build_proxy('tool', loop, call_counter, pending, cache, handle)
    future = proxy()
    assert future.done()
    assert future.result() == 42


async def test_stdin_reader_eof():
    """EOF on reader exits cleanly."""
    reader = asyncio.StreamReader()
    reader.feed_eof()
    pending: dict[int, asyncio.Future[object]] = {}
    await _stdin_reader(reader, pending)  # should return without error


async def test_stdin_reader_malformed_json(capsys: pytest.CaptureFixture[str]):
    """Malformed JSON produces a warning and continues to EOF."""
    reader = asyncio.StreamReader()
    reader.feed_data(b'not json\n')
    reader.feed_eof()
    pending: dict[int, asyncio.Future[object]] = {}
    await _stdin_reader(reader, pending)
    assert 'malformed JSON' in capsys.readouterr().err


async def test_stdin_reader_error_message():
    """Error message sets exception on the pending future."""
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({'type': 'error', 'id': 1, 'error': 'fail'}).encode() + b'\n')
    reader.feed_eof()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    pending: dict[int, asyncio.Future[object]] = {1: future}
    await _stdin_reader(reader, pending)
    with pytest.raises(RuntimeError, match='fail'):
        future.result()


async def test_stdin_reader_result_message():
    """Result message sets the value on the pending future."""
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({'type': 'result', 'id': 1, 'result': 99}).encode() + b'\n')
    reader.feed_eof()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    pending: dict[int, asyncio.Future[object]] = {1: future}
    await _stdin_reader(reader, pending)
    assert future.result() == 99


async def test_execute_empty_code(monkeypatch: pytest.MonkeyPatch):
    """Empty code sends complete with None."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)
    reader = asyncio.StreamReader()
    reader.feed_eof()
    await _execute({'code': '', 'functions': []}, reader)
    assert messages == [{'type': 'complete', 'result': None}]


async def test_execute_syntax_error(monkeypatch: pytest.MonkeyPatch):
    """Syntax error in code sends error and returns."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)
    reader = asyncio.StreamReader()
    reader.feed_eof()
    await _execute({'code': 'def while', 'functions': []}, reader)
    assert messages[0]['error_type'] == 'syntax'


async def test_execute_runtime_error(monkeypatch: pytest.MonkeyPatch):
    """Runtime error sends error with traceback."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)
    reader = asyncio.StreamReader()
    reader.feed_eof()
    await _execute({'code': '1 / 0', 'functions': []}, reader)
    assert messages[0]['error_type'] == 'runtime'
    assert 'ZeroDivisionError' in str(messages[0]['error'])


async def test_execute_runtime_syntax_error(monkeypatch: pytest.MonkeyPatch):
    """SyntaxError at runtime (e.g. eval) is reported as syntax."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)
    reader = asyncio.StreamReader()
    reader.feed_eof()
    await _execute({'code': 'eval("def while")', 'functions': []}, reader)
    assert messages[0]['error_type'] == 'syntax'


async def test_build_proxy_normal_call(monkeypatch: pytest.MonkeyPatch):
    """Proxy without cache hit creates a pending future and writes call + calls_ready messages."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)

    loop = asyncio.get_running_loop()
    call_counter: list[int] = [0]
    pending: dict[int, asyncio.Future[object]] = {}
    cache: dict[str, object] = {}
    handle: list[asyncio.Handle | None] = [None]

    proxy = _build_proxy('add', loop, call_counter, pending, cache, handle)
    future = proxy(x=1, y=2)

    # Future is pending (not resolved from cache)
    assert not future.done()
    assert 1 in pending
    assert pending[1] is future

    # Call message was written
    assert messages[0] == {'type': 'call', 'id': 1, 'function': 'add', 'args': [], 'kwargs': {'x': 1, 'y': 2}}

    # Let the event loop process the calls_ready callback
    await asyncio.sleep(0)
    assert messages[1] == {'type': 'calls_ready'}

    # Clean up: cancel the handle
    if handle[0] is not None:  # pragma: no cover
        handle[0].cancel()


async def test_build_proxy_batched_calls(monkeypatch: pytest.MonkeyPatch):
    """Two proxy calls without awaiting cancel the first calls_ready handle, emitting only one."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)

    loop = asyncio.get_running_loop()
    call_counter: list[int] = [0]
    pending: dict[int, asyncio.Future[object]] = {}
    cache: dict[str, object] = {}
    handle: list[asyncio.Handle | None] = [None]

    proxy = _build_proxy('add', loop, call_counter, pending, cache, handle)
    proxy(x=1, y=2)
    proxy(x=3, y=4)

    # Both call messages written, but only one calls_ready after the loop tick
    await asyncio.sleep(0)
    assert messages == [
        {'type': 'call', 'id': 1, 'function': 'add', 'args': [], 'kwargs': {'x': 1, 'y': 2}},
        {'type': 'call', 'id': 2, 'function': 'add', 'args': [], 'kwargs': {'x': 3, 'y': 4}},
        {'type': 'calls_ready'},
    ]

    if handle[0] is not None:  # pragma: no cover
        handle[0].cancel()


async def test_stdin_reader_result_no_pending_future():
    """Result for an unknown call ID is silently skipped."""
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({'type': 'result', 'id': 999, 'result': 42}).encode() + b'\n')
    reader.feed_eof()
    pending: dict[int, asyncio.Future[object]] = {}
    await _stdin_reader(reader, pending)
    # No crash — unknown ID is ignored


async def test_stdin_reader_unknown_msg_type():
    """Unknown message type is silently skipped."""
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({'type': 'heartbeat'}).encode() + b'\n')
    reader.feed_eof()
    pending: dict[int, asyncio.Future[object]] = {}
    await _stdin_reader(reader, pending)
    # No crash — unknown type is ignored


async def test_stdin_reader_error_no_pending_future():
    """Error for an unknown call ID is silently skipped."""
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({'type': 'error', 'id': 999, 'error': 'fail'}).encode() + b'\n')
    reader.feed_eof()
    pending: dict[int, asyncio.Future[object]] = {}
    await _stdin_reader(reader, pending)
    # No crash — unknown ID is ignored


async def test_stdin_reader_result_for_done_future():
    """Result for an already-resolved future does not overwrite it."""
    reader = asyncio.StreamReader()
    reader.feed_data(json.dumps({'type': 'result', 'id': 1, 'result': 99}).encode() + b'\n')
    reader.feed_eof()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    future.set_result('already done')
    pending: dict[int, asyncio.Future[object]] = {1: future}
    await _stdin_reader(reader, pending)
    assert future.result() == 'already done'


async def test_execute_with_function_call(monkeypatch: pytest.MonkeyPatch):
    """_execute with functions builds proxies, calls them, and completes successfully."""
    messages: list[dict[str, object]] = []
    monkeypatch.setattr('pydantic_ai.runtime._driver._write_msg', messages.append)

    reader = asyncio.StreamReader()

    async def provide_result():
        # Wait for the calls_ready message
        while not any(m.get('type') == 'calls_ready' for m in messages):
            await asyncio.sleep(0.01)
        # Feed result for call id 1
        reader.feed_data(json.dumps({'type': 'result', 'id': 1, 'result': 42}).encode() + b'\n')
        reader.feed_eof()

    provider = asyncio.create_task(provide_result())

    await _execute(
        {'code': 'await add(x=1, y=2)', 'functions': ['add']},
        reader,
    )
    await provider

    call_msgs = [m for m in messages if m.get('type') == 'call']
    assert len(call_msgs) == 1
    assert call_msgs[0]['function'] == 'add'

    complete_msgs = [m for m in messages if m.get('type') == 'complete']
    assert len(complete_msgs) == 1
    assert complete_msgs[0]['result'] == 42
