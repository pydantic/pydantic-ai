from __future__ import annotations as _annotations

import asyncio
import re
import secrets
import subprocess
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import contextmanager
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest
from httpx import AsyncClient, HTTPError
from inline_snapshot import snapshot
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

if TYPE_CHECKING:
    from mcp import ClientSession

pytestmark = pytest.mark.anyio
DENO_ARGS = [
    'run',
    '-N',
    f'-R=mcp-run-python/node_modules,{tempfile.gettempdir()}',
    f'-W=mcp-run-python/node_modules,{tempfile.gettempdir()}',
    '--node-modules-dir=auto',
    'mcp-run-python/src/main.ts',
]


class McpTools(StrEnum):
    RUN_PYTHON_CODE = 'run_python_code'
    UPLOAD_FILE_FROM_URI = 'upload_file_from_uri'
    RETRIEVE_FILE = 'retrieve_file'


LOREM_IPSUM = """But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was born and I will give you a complete account of the system, and expound the actual teachings of the great explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleasure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally encounter consequences that are extremely painful. Nor again is there anyone who loves or pursues or desires to obtain pain of itself, because it is pain, but because occasionally circumstances occur in which toil and pain can procure him some great pleasure. To take a trivial example, which of us ever undertakes laborious physical exercise, except to obtain some advantage from it? But who has any right to find fault with a man who chooses to enjoy a pleasure that has no annoying consequences, or one who avoids a pain that produces no resultant pleasure?

On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains.
"""


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def server_type(request: pytest.FixtureRequest) -> Literal['stdio', 'sse', 'streamable_http']:
    """Indirect fixture to accept server type parametrization."""
    return request.param


@pytest.fixture
def mount(request: pytest.FixtureRequest) -> bool | str:
    """Indirect fixture to accept mount parametrization."""
    return request.param


@pytest.fixture(name='mcp_session', scope='function')  # Function scope to ensure no files are stored
async def fixture_mcp_session(
    server_type: Literal['stdio', 'sse', 'streamable_http'],
    mount: bool | str,
) -> AsyncIterator[ClientSession]:
    # Build mount parameter
    if isinstance(mount, bool):
        mount_param = ['--mount'] if mount else []
    else:
        mount_param = [f'--mount={mount}']

    match server_type:
        case 'stdio':
            server_params = StdioServerParameters(command='deno', args=[*DENO_ARGS, 'stdio'] + mount_param)
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    yield session

        case 'streamable_http':
            port = 3101
            cmd = ['deno', *DENO_ARGS, 'streamable_http', f'--port={port}'] + mount_param
            print(f'Running command: {" ".join(cmd)}')
            p = subprocess.Popen(cmd)
            try:
                url = f'http://localhost:{port}/mcp'

                async with AsyncClient() as client:
                    for _ in range(10):
                        try:
                            await client.get(url, timeout=0.01)
                        except HTTPError:
                            await asyncio.sleep(0.1)
                        else:
                            break

                async with streamablehttp_client(url) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        yield session

            finally:
                p.terminate()
                exit_code = p.wait()
                if exit_code > 0:
                    pytest.fail(f'Process exited with code {exit_code}')

        case 'sse':
            port = 3101
            cmd = ['deno', *DENO_ARGS, 'sse', f'--port={port}'] + mount_param
            print(f'Running command: {" ".join(cmd)}')
            p = subprocess.Popen(cmd)
            try:
                url = f'http://localhost:{port}'
                async with AsyncClient() as client:
                    for _ in range(10):
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


@pytest.mark.parametrize('server_type', ['stdio', 'sse', 'streamable_http'], indirect=True)
@pytest.mark.parametrize('mount', [False, './storage/'], indirect=True)
class TestMcp:
    async def test_list_tools(
        self, mcp_session: ClientSession, server_type: Literal['stdio', 'sse', 'streamable_http'], mount: bool | str
    ) -> None:
        await mcp_session.initialize()
        tools = await mcp_session.list_tools()
        if mount is False:
            assert len(tools.tools) == 1
            tool = tools.tools[0]
            assert tool.name == McpTools.RUN_PYTHON_CODE
            assert tool.description
            assert tool.description.startswith(
                'Tool to execute Python code and return stdout, stderr, and return value.'
            )
            assert tool.inputSchema['properties'] == snapshot(
                {'python_code': {'type': 'string', 'description': 'Python code to run'}}
            )
        else:
            # Check tools
            assert len(tools.tools) == 3
            # sort tools by their name
            sorted_tools = sorted(tools.tools, key=lambda t: t.name)
            tool = sorted_tools[1]
            assert tool.name == McpTools.RUN_PYTHON_CODE
            assert tool.description
            assert tool.description.startswith(
                'Tool to execute Python code and return stdout, stderr, and return value.'
            )
            assert tool.inputSchema['properties'] == snapshot(
                {'python_code': {'type': 'string', 'description': 'Python code to run'}}
            )
            # Check resources (no file uploaded)
            resources = await mcp_session.list_resources()
            assert len(resources.resources) == 0

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
                id='basic-code',
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
                id='import-numpy',
            ),
            pytest.param(
                [
                    '# /// script',
                    '# dependencies = ["pydantic", "email-validator"]',
                    '# ///',
                    'import pydantic',
                    'class Model(pydantic.BaseModel):',
                    '    email: pydantic.EmailStr',
                    "Model(email='hello@pydantic.dev')",
                ],
                snapshot("""\
<status>success</status>
<dependencies>["pydantic","email-validator"]</dependencies>
<return_value>
{
  "email": "hello@pydantic.dev"
}
</return_value>\
"""),
                id='magic-comment-import',
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
                id='undefined-variable',
            ),
        ],
    )
    async def test_run_python_code(
        self,
        mcp_session: ClientSession,
        server_type: Literal['stdio', 'sse', 'streamable_http'],
        mount: bool | str,
        code: list[str],
        expected_output: str,
    ) -> None:
        await mcp_session.initialize()
        result = await mcp_session.call_tool(McpTools.RUN_PYTHON_CODE, {'python_code': '\n'.join(code)})
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, types.TextContent)
        assert content.text == expected_output

    def get_dir_from_instructions(self, instructions: str | None) -> Path:
        if instructions and (match := re.search(r'Persistent storage is mounted at:\s*"([^"]+)"', instructions)):
            return Path(match.group(1))
        else:
            raise ValueError(f'Could not parse directory from initialize instruction: {instructions}')

    @contextmanager
    def serve_file_once(self, data: bytes):
        """
        Context manager that exposes `filepath` at a localhost URL.
        Yields the URL as a string. When the context exits, the server stops.

        Args:
            filepath: Path to the file to serve.
            single_use: If True, the server shuts down after the first successful GET.
        """

        token = secrets.token_urlsafe(16)  # secret path component: /<token>
        ctype = 'application/octet-stream'
        length = len(data)

        class _BytesHandler(BaseHTTPRequestHandler):
            # Silence default logging
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(length))
                self.end_headers()

                # Stream in chunks so it works for large data
                chunk_size = 64 * 1024
                for i in range(0, length, chunk_size):
                    self.wfile.write(data[i : i + chunk_size])
                    self.wfile.flush()

                threading.Thread(target=self.server.shutdown, daemon=True).start()

        httpd = ThreadingHTTPServer(('127.0.0.1', 0), _BytesHandler)
        # Avoid "address already in use" on quick reuse
        httpd.daemon_threads = True

        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            port = httpd.server_address[1]
            url = f'http://127.0.0.1:{port}/{token}'
            yield url
        finally:
            # Safe shutdown even if already stopped by single_use
            try:
                httpd.shutdown()
            except Exception:
                pass
            httpd.server_close()
            t.join(timeout=2)

    @pytest.mark.parametrize('uri_type', ['http', 'file'])
    async def test_upload_files(
        self,
        mcp_session: ClientSession,
        server_type: Literal['stdio', 'sse', 'streamable_http'],
        mount: bool | str,
        uri_type: Literal['http', 'file'],
        tmp_path: Path,
    ) -> None:
        if mount is False:
            pytest.skip('No directory mounted.')
        result = await mcp_session.initialize()

        # Extract directory from response
        storageDir = self.get_dir_from_instructions(result.instructions)
        assert storageDir.is_dir()

        match uri_type:
            case 'file':
                filename = 'lorem.txt'
                lorem_file = tmp_path / filename
                lorem_file.write_text(LOREM_IPSUM)

                result = await mcp_session.call_tool(
                    McpTools.UPLOAD_FILE_FROM_URI, {'uri': f'file://{str(lorem_file)}', 'filename': filename}
                )

            case 'http':
                filename = 'lorem.txt'
                with self.serve_file_once(LOREM_IPSUM.encode()) as url:
                    result = await mcp_session.call_tool(
                        McpTools.UPLOAD_FILE_FROM_URI, {'uri': url, 'filename': filename}
                    )

        assert result.isError is False
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, types.ResourceLink)
        assert str(content.uri) == f'file:///{filename}'
        assert content.name == filename
        assert content.mimeType is not None
        assert content.mimeType.startswith('text/plain')

        createdFile = storageDir / filename
        assert createdFile.exists()
        assert createdFile.is_file()
        assert createdFile.read_text() == LOREM_IPSUM


async def test_install_run_python_code() -> None:
    node_modules = Path(__file__).parent / 'node_modules'
    if node_modules.exists():
        # shutil.rmtree can't delete node_modules :-(
        subprocess.run(['rm', '-r', node_modules], check=True)

    logs: list[str] = []

    async def logging_callback(params: types.LoggingMessageNotificationParams) -> None:
        logs.append(f'{params.level}: {params.data}')

    server_params = StdioServerParameters(command='deno', args=[*DENO_ARGS, 'stdio'])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, logging_callback=logging_callback) as mcp_session:
            await mcp_session.initialize()
            await mcp_session.set_logging_level('debug')
            result = await mcp_session.call_tool(
                'run_python_code', {'python_code': 'import numpy\nnumpy.array([1, 2, 3])'}
            )
            assert len(result.content) == 1
            content = result.content[0]
            assert isinstance(content, types.TextContent)
            expected_output = """\
<status>success</status>
<dependencies>["numpy"]</dependencies>
<return_value>
[
  1,
  2,
  3
]
</return_value>\
"""
            assert content.text == expected_output
            assert len(logs) >= 18
            assert re.search(
                r"debug: Didn't find package numpy\S+?\.whl locally, attempting to load from", '\n'.join(logs)
            )
