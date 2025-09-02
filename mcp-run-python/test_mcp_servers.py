from __future__ import annotations as _annotations

import asyncio
import base64
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
from pydantic import FileUrl

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
    DELETE_FILE = 'delete_file'


CSV_DATA = """Name,Age,Department,Salary
Alice,25,Engineering,60000
Bob,32,Marketing,52000
Charlie,29,Engineering,70000
Diana,45,HR,65000
Ethan,35,Marketing,58000
Fiona,28,Engineering,72000
George,40,HR,64000
Hannah,31,Engineering,68000
Ian,38,Marketing,61000
Julia,27,HR,59000
"""

BASE_64_IMAGE = 'iVBORw0KGgoAAAANSUhEUgAAAQAAAAEACAIAAADTED8xAAAEX0lEQVR4nOzdO8vX9R/HcS/56f8PWotGQkPBBUWESCQYNJR0GjIn6UBTgUMZTiGE4ZgRVKNkuDSEFtgBQqIiKunkEFdkWLmEBQUWiNUQYd2KNwTPx+MGvD7Tk/f2/S7O7tmyatKnJx8b3f/p6EOj+5euu2Z0/+Sxt0f3N++9fHR/+57/j+7vuPuT0f3Vo+vwHycA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQtDr561+gDpzf9PLp/4eNzo/uXzv41uv/BM0+O7h9/bsPo/vqPdo3u7965GN13AUgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSlh5ce+XoA9+eODK6v3r7naP7b31zaHT/4p+3jO4f2/Tb6P7K41tH9zff+8LovgtAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkLb09ZmLow8sb1ke3d92YXR+1dO7PhzdX7f2xtH9Q5fN/t/g2j9eHt3/cc350X0XgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBNAKQJgDQBkCYA0gRAmgBIEwBpAiBtcf3eW0cfePTE7Pf1D9yxMrq/4YrR+VWvnN84uv/lvs2j+2v3nx3dv3rT/0b3XQDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmAtKWrzq0ffeD312f339h5ZnT/npsPj+7//cPDo/un739idP/Xg5+P7j/y/G2j+y4AaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQNpi/5FfRh94753XRvcP7F0zuv/V7e+O7t906v3R/WdP/zO6f9/ixdH9G3Z/NrrvApAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkCYA0AZAmANIEQJoASBMAaQIgTQCkLb25vDL6wLoHjo7ur7z03ej++u+fGt0/vm/2+/dfHF4e3d9xauPo/taN20b3XQDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmANAGQJgDSBECaAEgTAGkCIE0ApAmAtH8DAAD//9drYGg9ROu9AAAAAElFTkSuQmCC'


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
            assert len(tools.tools) == 4
            # sort tools by their name
            sorted_tools = sorted(tools.tools, key=lambda t: t.name)

            # Check tool names
            assert set(tool.name for tool in tools.tools) == set(McpTools)

            # Check run python tool
            tool = sorted_tools[2]
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
    ...<9 lines>...
    .run_async(globals, locals)
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
                filename = 'data.csv'
                data_file = tmp_path / filename
                data_file.write_text(CSV_DATA)

                result = await mcp_session.call_tool(
                    McpTools.UPLOAD_FILE_FROM_URI, {'uri': f'file://{str(data_file)}', 'filename': filename}
                )

            case 'http':
                filename = 'data.csv'
                with self.serve_file_once(CSV_DATA.encode()) as url:
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
        assert content.mimeType.startswith('text/csv')

        createdFile = storageDir / filename
        assert createdFile.exists()
        assert createdFile.is_file()
        assert createdFile.read_text() == CSV_DATA

    @pytest.mark.parametrize('content_type', ['bytes', 'text'])
    async def test_download_files(
        self,
        mcp_session: ClientSession,
        server_type: Literal['stdio', 'sse', 'streamable_http'],
        mount: bool | str,
        content_type: Literal['bytes', 'text'],
    ):
        if mount is False:
            pytest.skip('No directory mounted.')
        result = await mcp_session.initialize()

        # Extract directory from response
        storageDir = self.get_dir_from_instructions(result.instructions)
        assert storageDir.is_dir()

        match content_type:
            case 'bytes':
                filename = 'image.png'
                ctype = 'image/png'
                file_path = storageDir / filename
                file_path.write_bytes(base64.b64decode(BASE_64_IMAGE))

            case 'text':
                filename = 'data.csv'
                ctype = 'text/csv'
                file_path = storageDir / filename
                file_path.write_text(CSV_DATA)

        result = await mcp_session.call_tool(McpTools.RETRIEVE_FILE, {'filename': filename})
        assert result.isError is False
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, types.ResourceLink)
        assert str(content.uri) == f'file:///{filename}'
        assert content.name == filename
        assert content.mimeType is not None
        assert content.mimeType.startswith(ctype)

        result = await mcp_session.list_resources()

        assert len(result.resources) == 1
        resource = result.resources[0]
        assert resource.name == filename
        assert resource.mimeType is not None
        assert resource.mimeType.startswith(ctype)
        assert str(resource.uri) == f'file:///{filename}'

        result = await mcp_session.read_resource(FileUrl(f'file:///{filename}'))

        assert len(result.contents) == 1
        resource = result.contents[0]
        assert str(resource.uri) == f'file:///{filename}'
        assert resource.mimeType is not None
        assert resource.mimeType.startswith(ctype)

        match content_type:
            case 'bytes':
                assert isinstance(resource, types.BlobResourceContents)
                assert resource.blob == BASE_64_IMAGE

            case 'text':
                assert isinstance(resource, types.TextResourceContents)
                assert resource.text == CSV_DATA

    async def test_delete_file(
        self,
        mcp_session: ClientSession,
        server_type: Literal['stdio', 'sse', 'streamable_http'],
        mount: bool | str,
    ):
        if mount is False:
            pytest.skip('No directory mounted.')
        result = await mcp_session.initialize()

        # Extract directory from response
        storageDir = self.get_dir_from_instructions(result.instructions)
        assert storageDir.is_dir()

        filename = 'data.csv'
        file_path = storageDir / filename
        file_path.write_text(CSV_DATA)

        result = await mcp_session.call_tool(McpTools.DELETE_FILE, {'filename': filename})
        assert result.isError is False
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, types.TextContent)
        assert content.text.endswith('deleted successfully')
        assert not file_path.exists()

    # Code pieces use hardcoded values of mount
    @pytest.mark.parametrize(
        'code,expected_output,content_type',
        [
            pytest.param(
                [
                    '# /// script',
                    '# dependencies = ["pillow"]',
                    '# ///',
                    'from PIL import Image, ImageFilter',
                    'img = Image.open("storage/image.png")',
                    'gray_img = img.convert("L")',
                    'gray_img.save("storage/image-gray.png")',
                    'print(f"Image size: {img.size}")',
                ],
                snapshot("""\
<status>success</status>
<dependencies>["pillow"]</dependencies>
<output>
Image size: (256, 256)
</output>\
"""),
                'bytes',
                id='image-transform',
            ),
            pytest.param(
                [
                    '# /// script',
                    '# dependencies = ["pandas"]',
                    '# ///',
                    'import pandas as pd',
                    'df = pd.read_csv("storage/data.csv")',
                    'df["Age_in_10_years"] = df["Age"] + 10',
                    'df.to_csv("storage/data-processed.csv", index=False)',
                    'print(df.describe())',
                ],
                snapshot("""\
<status>success</status>
<dependencies>["pandas"]</dependencies>
<output>
             Age        Salary  Age_in_10_years
count  10.000000     10.000000        10.000000
mean   33.000000  62900.000000        43.000000
std     6.394442   6100.091074         6.394442
min    25.000000  52000.000000        35.000000
25%    28.250000  59250.000000        38.250000
50%    31.500000  62500.000000        41.500000
75%    37.250000  67250.000000        47.250000
max    45.000000  72000.000000        55.000000
</output>\
"""),
                'text',
                id='dataframe-manipulation',
            ),
        ],
    )
    async def test_run_python_code_with_file(
        self,
        mcp_session: ClientSession,
        server_type: Literal['stdio', 'sse', 'streamable_http'],
        mount: bool | str,
        code: list[str],
        expected_output: str,
        content_type: Literal['bytes', 'text'],
        tmp_path: Path,
    ):
        if mount is False:
            pytest.skip('No directory mounted.')
        await mcp_session.initialize()

        match content_type:
            case 'bytes':
                filename = 'image.png'
                output_file = 'image-gray.png'
                ctype = 'image/png'
                data_file = tmp_path / filename
                data_file.write_bytes(base64.b64decode(BASE_64_IMAGE))

            case 'text':
                filename = 'data.csv'
                output_file = 'data-processed.csv'
                ctype = 'text/csv'
                data_file = tmp_path / filename
                data_file.write_text(CSV_DATA)

        result = await mcp_session.call_tool(
            McpTools.UPLOAD_FILE_FROM_URI, {'uri': f'file://{str(data_file)}', 'filename': filename}
        )
        assert result.isError is False

        result = await mcp_session.call_tool(McpTools.RUN_PYTHON_CODE, {'python_code': '\n'.join(code)})
        assert result.isError is False
        assert len(result.content) == 1
        content = result.content[0]
        assert isinstance(content, types.TextContent)
        assert content.text == expected_output

        result = await mcp_session.read_resource(FileUrl(f'file:///{output_file}'))
        assert len(result.contents) == 1
        resource = result.contents[0]
        assert resource.mimeType is not None
        assert resource.mimeType.startswith(ctype)
        assert (
            isinstance(resource, types.BlobResourceContents)
            if content_type == 'bytes'
            else isinstance(resource, types.TextResourceContents)
        )


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
            assert len(logs) >= 16
            assert re.search(
                r"debug: loadPackage: Didn't find package numpy\S*\.whl locally, attempting to load from",
                '\n'.join(logs),
            )
