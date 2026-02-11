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

DRIVER_PATH = Path(__file__).parents[2] / 'pydantic_ai_slim' / 'pydantic_ai' / 'runtime' / '_driver.py'

pytestmark = pytest.mark.anyio


async def start_driver(
    code: str,
    functions: list[str] | None = None,
    result_cache: dict[str, object] | None = None,
) -> asyncio.subprocess.Process:
    """Start the driver subprocess and send the init message."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, '-u', str(DRIVER_PATH),
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
    code = (
        'f1 = add(x=1, y=2)\n'
        'f2 = add(x=3, y=4)\n'
        'r1 = await f1\n'
        'r2 = await f2\n'
        'r1 + r2'
    )
    proc = await start_driver(code, functions=['add'])

    # Read 2 call messages (both sent before any result)
    msg1 = await read_msg(proc)
    msg2 = await read_msg(proc)
    assert msg1['type'] == 'call'
    assert msg1['id'] == 1
    assert msg2['type'] == 'call'
    assert msg2['id'] == 2

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
    code = (
        'f1 = add(x=1, y=2)\n'
        'f2 = add(x=3, y=4)\n'
        'r1 = await f1\n'
        'r2 = await f2\n'
        'r1 + r2'
    )
    proc = await start_driver(code, functions=['add'], result_cache={'1': 3})

    # Only 1 call message (call 2, since call 1 is cached)
    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['id'] == 2

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
    code = (
        'result = "no error"\n'
        'try:\n'
        '    await bad_tool()\n'
        'except Exception as e:\n'
        '    result = str(e)\n'
        'result'
    )
    proc = await start_driver(code, functions=['bad_tool'])

    # Read call message
    msg = await read_msg(proc)
    assert msg['type'] == 'call'
    assert msg['function'] == 'bad_tool'

    # Send error response
    await send_msg(proc, {'type': 'error', 'id': msg['id'], 'error': 'tool failed'})

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

    # Send all results
    await send_msg(proc, {'type': 'result', 'id': 1, 'result': 'a1'})
    await send_msg(proc, {'type': 'result', 'id': 2, 'result': 'b1'})
    await send_msg(proc, {'type': 'result', 'id': 3, 'result': 'a2'})

    msg = await read_msg(proc)
    assert msg['type'] == 'complete'
    assert msg['result'] == ['a1', 'b1', 'a2']

    proc.kill()
    await proc.wait()
