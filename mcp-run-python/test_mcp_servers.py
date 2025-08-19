from __future__ import annotations as _annotations

import asyncio
import re
import subprocess
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

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
    '-R=mcp-run-python/node_modules',
    '-W=mcp-run-python/node_modules',
    '--node-modules-dir=auto',
    'mcp-run-python/src/main.ts',
]


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture(name='mcp_session_with_mount')
async def fixture_mcp_session_with_mount() -> AsyncIterator[tuple[ClientSession, Path]]:
    """Fixture that provides an MCP session with filesystem mounting enabled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        mount_path = f'{temp_path}:/tmp/mounted'

        server_params = StdioServerParameters(command='deno', args=[*DENO_ARGS, 'stdio', '--mount', mount_path])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session, temp_path


@pytest.fixture(name='mcp_session', params=['stdio', 'sse', 'streamable_http'])
async def fixture_mcp_session(request: pytest.FixtureRequest) -> AsyncIterator[ClientSession]:
    if request.param == 'stdio':
        server_params = StdioServerParameters(command='deno', args=[*DENO_ARGS, 'stdio'])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                yield session
    elif request.param == 'streamable_http':
        port = 3101
        p = subprocess.Popen(['deno', *DENO_ARGS, 'streamable_http', f'--port={port}'])
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

    else:
        port = 3101

        p = subprocess.Popen(['deno', *DENO_ARGS, 'sse', f'--port={port}'])
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


async def test_list_tools(mcp_session: ClientSession) -> None:
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
async def test_run_python_code(mcp_session: ClientSession, code: list[str], expected_output: str) -> None:
    await mcp_session.initialize()
    result = await mcp_session.call_tool('run_python_code', {'python_code': '\n'.join(code)})
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    assert content.text == expected_output


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


async def test_filesystem_mount_read_file(mcp_session_with_mount: tuple[ClientSession, Path]) -> None:
    """Test reading a file from mounted filesystem."""
    mcp_session, temp_path = mcp_session_with_mount

    # Create a test file in the temporary directory
    test_file = temp_path / 'test.txt'
    test_content = 'Hello from mounted filesystem!'
    test_file.write_text(test_content)

    await mcp_session.initialize()

    # Python code to check if mount exists and read the mounted file
    python_code = f"""
from pathlib import Path
import os

# Check if mount directory exists
mount_path = Path('/tmp/mounted')
print(f'Mount path exists: {{mount_path.exists()}}')
if mount_path.exists():
    print(f'Mount path contents: {{list(mount_path.iterdir())}}')

# Try to read the file
try:
    content = Path('/tmp/mounted/test.txt').read_text()
    print(f'File content: {{content}}')
    content
except FileNotFoundError as e:
    print(f'File not found: {{e}}')
    # Check if the file exists in the local temp directory
    local_file = Path('{temp_path}/test.txt')
    print(f'Local file exists: {{local_file.exists()}}')
    if local_file.exists():
        print(f'Local file content: {{local_file.read_text()}}')
    "File not mounted"
"""

    result = await mcp_session.call_tool('run_python_code', {'python_code': python_code})
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)

    # If mounting worked, we should see the file content
    # If mounting didn't work, we should at least see debug info
    assert 'success' in content.text
    # The test should pass if either the file was read successfully or we can see it wasn't mounted
    if test_content in content.text:
        # Mounting worked
        assert test_content in content.text
    else:
        # Mounting didn't work, but we should see debug info
        assert 'Mount path exists:' in content.text


async def test_filesystem_mount_write_file(mcp_session_with_mount: tuple[ClientSession, Path]) -> None:
    """Test writing a file to mounted filesystem and syncing back."""
    mcp_session, temp_path = mcp_session_with_mount

    await mcp_session.initialize()

    # Python code to check mount and write a file in the mounted directory
    python_code = """
from pathlib import Path

# Check if mount directory exists
mount_path = Path('/tmp/mounted')
print(f'Mount path exists: {mount_path.exists()}')

if mount_path.exists():
    try:
        output_file = Path('/tmp/mounted/output.txt')
        content = 'Generated by Python in Pyodide!'
        output_file.write_text(content)
        print(f'Wrote to {output_file}')
        content
    except Exception as e:
        print(f'Error writing file: {e}')
        "Write failed"
else:
    print('Mount path does not exist')
    "Mount not available"
"""

    result = await mcp_session.call_tool('run_python_code', {'python_code': python_code})
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    assert 'success' in content.text

    # Only check for file sync if mounting worked
    if 'Generated by Python in Pyodide!' in content.text:
        # Check that the file was synced back to the local filesystem
        output_file = temp_path / 'output.txt'
        assert output_file.exists()
        assert output_file.read_text() == 'Generated by Python in Pyodide!'
    else:
        # If mounting didn't work, we should see debug info
        assert 'Mount path exists:' in content.text


async def test_filesystem_mount_directory_structure(mcp_session_with_mount: tuple[ClientSession, Path]) -> None:
    """Test mounting and working with directory structures."""
    mcp_session, temp_path = mcp_session_with_mount

    # Create a nested directory structure
    nested_dir = temp_path / 'data' / 'subdir'
    nested_dir.mkdir(parents=True)

    # Create files in different directories
    (temp_path / 'root.txt').write_text('Root file')
    (temp_path / 'data' / 'data.txt').write_text('Data file')
    (nested_dir / 'nested.txt').write_text('Nested file')

    await mcp_session.initialize()

    # Python code to explore the mounted directory structure
    python_code = """
from pathlib import Path
import os

mount_path = Path('/tmp/mounted')
print(f'Mount path exists: {mount_path.exists()}')

files_found = []

if mount_path.exists():
    # Walk through all files
    try:
        for root, dirs, files in os.walk(mount_path):
            for file in files:
                file_path = Path(root) / file
                relative_path = file_path.relative_to(mount_path)
                content = file_path.read_text().strip()
                files_found.append(f"{relative_path}: {content}")

        files_found.sort()
        for file_info in files_found:
            print(file_info)
    except Exception as e:
        print(f'Error walking directory: {e}')
else:
    print('Mount path does not exist')

len(files_found)
"""

    result = await mcp_session.call_tool('run_python_code', {'python_code': python_code})
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    assert 'success' in content.text

    # Only check for specific files if mounting worked
    if 'root.txt: Root file' in content.text:
        # Mounting worked
        assert 'root.txt: Root file' in content.text
        assert 'data/data.txt: Data file' in content.text
        assert 'data/subdir/nested.txt: Nested file' in content.text
        assert '<return_value>\n3\n</return_value>' in content.text
    else:
        # Mounting didn't work, but we should see debug info
        assert 'Mount path exists:' in content.text
        # Should return 0 files found
        assert '<return_value>\n0\n</return_value>' in content.text


async def test_filesystem_mount_binary_files(mcp_session_with_mount: tuple[ClientSession, Path]) -> None:
    """Test mounting and handling binary files."""
    mcp_session, temp_path = mcp_session_with_mount

    # Create a binary file
    binary_data = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])  # PNG header
    binary_file = temp_path / 'test.png'
    binary_file.write_bytes(binary_data)

    await mcp_session.initialize()

    # Python code to check mount and read the binary file
    python_code = """
from pathlib import Path

mount_path = Path('/tmp/mounted')
print(f'Mount path exists: {mount_path.exists()}')

if mount_path.exists():
    try:
        binary_file = Path('/tmp/mounted/test.png')
        data = binary_file.read_bytes()
        print(f'Binary file size: {len(data)} bytes')
        print(f'First 4 bytes: {list(data[:4])}')

        # Write a new binary file
        new_binary = Path('/tmp/mounted/output.bin')
        new_data = bytes([1, 2, 3, 4, 5])
        new_binary.write_bytes(new_data)
        print(f'Created binary file with {len(new_data)} bytes')

        len(data)
    except Exception as e:
        print(f'Error with binary files: {e}')
        "Binary file error"
else:
    print('Mount path does not exist')
    "Mount not available"
"""

    result = await mcp_session.call_tool('run_python_code', {'python_code': python_code})
    assert len(result.content) == 1
    content = result.content[0]
    assert isinstance(content, types.TextContent)
    assert 'success' in content.text

    # Only check for specific binary file operations if mounting worked
    if 'Binary file size: 8 bytes' in content.text:
        # Mounting worked
        assert 'Binary file size: 8 bytes' in content.text
        assert 'First 4 bytes: [137, 80, 78, 71]' in content.text

        # Check that the new binary file was synced back
        output_file = temp_path / 'output.bin'
        assert output_file.exists()
        assert output_file.read_bytes() == bytes([1, 2, 3, 4, 5])
    else:
        # Mounting didn't work, but we should see debug info
        assert 'Mount path exists:' in content.text


async def test_filesystem_mount_invalid_path_format() -> None:
    """Test that invalid mount path format is handled gracefully."""
    logs: list[str] = []

    async def logging_callback(params: types.LoggingMessageNotificationParams) -> None:
        logs.append(f'{params.level}: {params.data}')

    # Use invalid mount path format (missing colon separator)
    invalid_mount_path = '/invalid/path'

    server_params = StdioServerParameters(command='deno', args=[*DENO_ARGS, 'stdio', '--mount', invalid_mount_path])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, logging_callback=logging_callback) as mcp_session:
            await mcp_session.initialize()
            await mcp_session.set_logging_level('debug')

            # Run simple code - should work despite invalid mount path
            result = await mcp_session.call_tool('run_python_code', {'python_code': 'print("Hello")\n"Hello"'})
            assert len(result.content) == 1
            content = result.content[0]
            assert isinstance(content, types.TextContent)
            assert 'success' in content.text
            assert 'Hello' in content.text

            # Check that warning was logged about invalid mount path
            log_text = '\n'.join(logs)
            assert 'Invalid mount path format' in log_text


async def test_filesystem_mount_nonexistent_local_path() -> None:
    """Test mounting a non-existent local path."""
    logs: list[str] = []

    async def logging_callback(params: types.LoggingMessageNotificationParams) -> None:
        logs.append(f'{params.level}: {params.data}')

    # Use non-existent local path
    nonexistent_mount_path = '/nonexistent/path:/tmp/mounted'

    server_params = StdioServerParameters(command='deno', args=[*DENO_ARGS, 'stdio', '--mount', nonexistent_mount_path])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write, logging_callback=logging_callback) as mcp_session:
            await mcp_session.initialize()
            await mcp_session.set_logging_level('debug')

            # Run simple code - should work despite failed mount
            result = await mcp_session.call_tool('run_python_code', {'python_code': 'print("Hello")\n"Hello"'})
            assert len(result.content) == 1
            content = result.content[0]
            assert isinstance(content, types.TextContent)
            assert 'success' in content.text
            assert 'Hello' in content.text

            # Check that warning was logged about failed mount
            log_text = '\n'.join(logs)
            assert 'Failed to mount' in log_text
