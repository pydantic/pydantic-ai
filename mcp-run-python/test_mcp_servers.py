from __future__ import annotations as _annotations

import asyncio
import os
import subprocess
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import pytest
from httpx import AsyncClient, HTTPError
from inline_snapshot import snapshot
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

if TYPE_CHECKING:
    from mcp import ClientSession

CLI_JS_PATH = 'mcp-run-python/cli.js'
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture(name='mcp_session', params=['stdio', 'sse'])
async def fixture_mcp_session(request: pytest.FixtureRequest) -> AsyncIterator[ClientSession]:
    if request.param == 'stdio':
        server_params = StdioServerParameters(command='node', args=[CLI_JS_PATH, 'stdio'])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
    else:
        port = 3101

        env = dict(os.environ)
        env['PORT'] = str(port)
        p = subprocess.Popen(['node', CLI_JS_PATH, 'sse'], env=env)
        try:
            url = f'http://localhost:{port}'
            async with AsyncClient() as client:
                for _ in range(5):
                    try:
                        await client.get(url, timeout=0.01)
                    except HTTPError:
                        await asyncio.sleep(0.1)
                    else:
                        break

            async with sse_client(f'{url}/sse') as (read, write):
                async with ClientSession(read, write) as session:
                    yield session
        finally:
            p.terminate()
            exit_code = p.wait()
            if exit_code > 0:
                pytest.fail(f'Process exited with code {exit_code}')


async def test_stdio_list_tools(mcp_session: ClientSession) -> None:
    await mcp_session.initialize()
    tools = await mcp_session.list_tools()
    assert len(tools.tools) == 1
    tool = tools.tools[0]
    assert tool.name == 'run_python_code'
    assert tool.description
    assert tool.description.startswith('Tool to execute Python code and return stdout, stderr, and return value.')
    assert tool.inputSchema['properties'] == snapshot(
        {'python_code': {'type': 'string', 'description': 'Python code to run'}}
    )


@pytest.mark.parametrize(
    'code,expected_output',
    [
        pytest.param(
            [
                'x = 4',
                "print(f'{x=}')",
                'x',
            ],
            snapshot("""\
<status>success</status>
<output>
x=4
</output>
<return_value>
4
</return_value>\
"""),
            id='basic_code',
        ),
        pytest.param(
            [
                'import numpy',
                'numpy.array([1, 2, 3])',
            ],
            snapshot("""\
<status>success</status>
<dependencies>["numpy"]</dependencies>
<return_value>
[
  1,
  2,
  3
]
</return_value>\
"""),
            id='import_numpy',
        ),
        pytest.param(
            [
                'print(unknown)',
            ],
            snapshot("""\
<status>run-error</status>
<error>
Traceback (most recent call last):
  File "main.py", line 1, in <module>
    print(unknown)
          ^^^^^^^
NameError: name 'unknown' is not defined

</error>\
"""),
            id='undefined_variable',
        ),
    ],
)
async def test_stdio_run_python_code(mcp_session: ClientSession, code: list[str], expected_output: str) -> None:
    await mcp_session.initialize()
    result = await mcp_session.call_tool('run_python_code', {'python_code': '\n'.join(code)})
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    assert content.text == expected_output
