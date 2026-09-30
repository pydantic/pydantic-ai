"""`CodeMode(workspace_files=True)` routes sandboxed `pathlib` and `open()` calls to the run's workspace."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_monty import MountDir, OSAccess

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.workspaces import FileEntry
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode._eager import EagerCodeModeToolset

from ...workspace_fakes import FakeWorkspace


async def run_snippets(workspace: FakeWorkspace | None, *codes: str, capability: CodeMode[object]) -> list[object]:
    """Run each snippet as one `run_code` call and return what the model saw for each: a value or a retry message."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = sum(isinstance(p, ToolReturnPart | RetryPromptPart) for m in messages for p in m.parts)
        if answered < len(codes):
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': codes[answered]})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[capability])
    result = await agent.run('go', workspace=workspace)
    return [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart | RetryPromptPart)
    ]


async def run_one(workspace: FakeWorkspace, code: str) -> object:
    [content] = await run_snippets(workspace, code, capability=CodeMode(workspace_files=True, max_retries=1))
    return content


async def test_sandbox_reads_and_writes_the_workspace() -> None:
    workspace = FakeWorkspace('files', {'/workspace/data.csv': b'a,b\n1,2\n'})
    result = await run_one(
        workspace,
        "from pathlib import Path\nPath('report.md').write_text(Path('data.csv').read_text().upper())\n"
        "[Path('/workspace/report.md').read_text(), str(Path('report.md').absolute())]",
    )
    assert result == ['A,B\n1,2\n', '/workspace/report.md']
    assert workspace.files['/workspace/report.md'] == b'A,B\n1,2\n'


async def test_writes_persist_across_run_code_calls() -> None:
    workspace = FakeWorkspace('persist')
    results = await run_snippets(
        workspace,
        "from pathlib import Path\nPath('note.txt').write_bytes(b'kept')",
        "Path('note.txt').read_bytes()",
        capability=CodeMode(workspace_files=True),
    )
    assert results == [4, b'kept']


async def test_open_modes() -> None:
    workspace = FakeWorkspace('open', {'/workspace/old.txt': b'old'})
    result = await run_one(
        workspace,
        "f = open('log.txt', 'a')\nf.write('one\\n')\nf.close()\n"
        "f = open('log.txt', 'a')\nf.write('two\\n')\nf.close()\n"
        "f = open('old.txt', 'w')\nf.close()\n"
        "[open('log.txt').read(), open('old.txt').read()]",
    )
    assert result == ['one\ntwo\n', '']


@pytest.mark.parametrize(
    ('code', 'error'),
    [
        ("open('missing.txt')", 'FileNotFoundError'),
        ("open('sub', 'a')", 'IsADirectoryError'),
        ("from pathlib import Path\nPath('missing.txt').read_text()", 'FileNotFoundError'),
        ("from pathlib import Path\nPath('sub').mkdir()", 'FileExistsError'),
        ("from pathlib import Path\nPath('nope/deeper').mkdir()", 'FileNotFoundError'),
        ("from pathlib import Path\nPath('file.txt/child').mkdir()", 'NotADirectoryError'),
        ("from pathlib import Path\nPath('sub').unlink()", 'IsADirectoryError'),
        ("from pathlib import Path\nPath('file.txt').rmdir()", 'NotADirectoryError'),
        ("from pathlib import Path\nPath('sub').rmdir()", 'Directory not empty'),
        ("from pathlib import Path\nPath('file.txt').rename('moved.txt')", '`Path.rename` is not supported'),
        ("from pathlib import Path\nPath('missing/new.txt').write_text('x')", 'FileNotFoundError'),
        ("from pathlib import Path\nPath('file.txt/new.txt').write_bytes(b'x')", 'NotADirectoryError'),
        ("from pathlib import Path\nPath('missing/new.txt').append_text('x')", 'FileNotFoundError'),
        ("open('missing/new.txt', 'w')", 'FileNotFoundError'),
    ],
)
async def test_errors_reach_the_sandbox(code: str, error: str) -> None:
    workspace = FakeWorkspace('errors', {'/workspace/file.txt': b'x', '/workspace/sub/inner.txt': b'y'})
    result = await run_one(workspace, code)
    assert isinstance(result, str)
    assert error in result


async def test_directories_and_metadata() -> None:
    workspace = FakeWorkspace('dirs', {'/workspace/file.txt': b'hello'})
    result = await run_one(
        workspace,
        'from pathlib import Path\n'
        "Path('a/b').mkdir(parents=True)\n"
        "Path('a').mkdir(exist_ok=True)\n"
        "Path('c').mkdir()\n"
        "appended = [Path('c/log.txt').append_text('x'), Path('c/log.txt').append_bytes(b'yz')]\n"
        "listed = sorted(str(p) for p in Path('.').iterdir())\n"
        "logged = Path('c/log.txt').read_text()\n"
        "Path('c/log.txt').unlink()\n"
        "Path('c').rmdir()\n"
        "[listed, appended, logged, Path('c').exists(), Path('file.txt').stat().st_size, Path('a').stat().st_size,\n"
        " Path('file.txt').is_file(), Path('a').is_file(), Path('a').is_dir(), Path('missing').is_dir(),\n"
        " Path('file.txt').is_symlink(), Path('/').is_symlink(), str(Path('a/../file.txt').resolve())]",
    )
    assert result == [
        ['a', 'c', 'file.txt'],
        [1, 2],
        'xyz',
        False,
        5,
        0,
        True,
        False,
        True,
        False,
        False,
        False,
        '/workspace/file.txt',
    ]


class _UnsizedWorkspace(FakeWorkspace):
    """A backend that, like some shell fallbacks, reports no file size."""

    async def stat(self, path: str) -> FileEntry:
        entry = await super().stat(path)
        return FileEntry(name=entry.name, path=entry.path, is_dir=entry.is_dir, size=None)


async def test_stat_measures_a_file_the_backend_does_not_size() -> None:
    workspace = _UnsizedWorkspace('unsized', {'/workspace/file.txt': b'abc'})
    assert await run_one(workspace, "from pathlib import Path\nPath('file.txt').stat().st_size") == 3


async def test_mounted_paths_still_reach_the_host(tmp_path: Path) -> None:
    (tmp_path / 'host.txt').write_text('from host')
    workspace = FakeWorkspace('mixed', {'/workspace/ws.txt': b'from workspace'})
    [result] = await run_snippets(
        workspace,
        "from pathlib import Path\n[Path('/host/host.txt').read_text(), Path('ws.txt').read_text()]",
        capability=CodeMode(workspace_files=True, mount=[MountDir(virtual_path='/host', host_path=str(tmp_path))]),
    )
    assert result == ['from host', 'from workspace']


async def test_os_access_still_answers_environment_but_not_files() -> None:
    workspace = FakeWorkspace('env', {'/workspace/ws.txt': b'from workspace'})
    os_access = OSAccess(environ={'TOKEN': 'secret'})
    [result] = await run_snippets(
        workspace,
        'import os\nimport datetime\nfrom pathlib import Path\n'
        "[os.getenv('TOKEN'), dict(os.environ), datetime.datetime.now().year > 2000, Path('ws.txt').read_text()]",
        capability=CodeMode(workspace_files=True, os_access=os_access),
    )
    assert result == ['secret', {'TOKEN': 'secret'}, True, 'from workspace']


def test_eager_code_mode_forwards_workspace_files() -> None:
    toolset = CodeMode[object](workspace_files=True, eager=True).get_wrapper_toolset(FunctionToolset[object]())
    assert isinstance(toolset, EagerCodeModeToolset)
    assert toolset.workspace_files


async def test_run_without_a_workspace_fails_at_start() -> None:
    with pytest.raises(UserError, match=r'`CodeMode\(workspace_files=True\)` needs a workspace'):
        await run_snippets(None, '1', capability=CodeMode(workspace_files=True))


@pytest.mark.parametrize(
    ('capability', 'expected', 'absent'),
    [
        (CodeMode[object](workspace_files=True), ['Workspace filesystem', 'No environment or clock'], 'mount'),
        (
            CodeMode[object](workspace_files=True, os_access=OSAccess()),
            ['Workspace filesystem', 'routed to the OS handler'],
            'mount point',
        ),
        (
            CodeMode[object](workspace_files=True, mount=MountDir(virtual_path='/host', host_path='.')),
            ['Workspace filesystem', 'Paths under the configured mount point(s) are routed to the host instead.'],
            'Configured OS access',
        ),
    ],
)
async def test_description_advertises_the_workspace(
    capability: CodeMode[object], expected: Sequence[str], absent: str
) -> None:
    descriptions: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        descriptions.extend(tool.description or '' for tool in info.function_tools if tool.name == 'run_code')
        return ModelResponse(parts=[TextPart('done')])

    await Agent(FunctionModel(model_fn), capabilities=[capability]).run('go', workspace=FakeWorkspace('describe'))
    [description] = descriptions
    for text in expected:
        assert text in description
    assert absent not in description
