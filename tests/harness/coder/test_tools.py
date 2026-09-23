"""Coder's tool surface as the model sees it; the tools themselves are tested with `FileSystem` and `Shell`."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

from .._recording_durability import RecordingDurability
from .._tool_calls import call_tool

pytestmark = pytest.mark.anyio


async def call(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
    unrestricted_filesystem: bool = False,
) -> str:
    coder = Coder[None](tmp_path, unrestricted_filesystem=unrestricted_filesystem)
    return await call_tool([coder, *capabilities], name, arguments)


class TestCoder:
    @pytest.mark.parametrize('extra_limits', [False, True])
    async def test_durable_binding(self, tmp_path: Path, extra_limits: bool) -> None:
        durability = RecordingDurability()
        capabilities: list[AbstractCapability[object]] = [Coder(tmp_path), durability]
        if extra_limits:
            capabilities.append(ToolOutputLimits())
        agent = Agent(TestModel(call_tools=[], custom_output_text='done'), name='coder', capabilities=capabilities)
        result = await agent.run('Inspect tools')
        assert result.output == 'done'
        assert [name for name, _ in durability.calls] == ['coder__model.request_stream']

    @pytest.mark.parametrize('unrestricted_filesystem', [False, True])
    async def test_discovered_paths_can_be_read_and_edited(self, tmp_path: Path, unrestricted_filesystem: bool) -> None:
        (tmp_path / 'src').mkdir()
        target = tmp_path / 'src' / 'AGENTS.md'
        target.write_text('Project instructions')
        coder = Coder[None](tmp_path, unrestricted_filesystem=unrestricted_filesystem, repo_context=False)
        path = await call_tool([coder], 'list_files', {'glob': '**/AGENTS.md'})
        assert Path(path) == Path('src/AGENTS.md')
        assert 'Project instructions' in await call_tool([coder], 'read_file', {'path': path})
        await call_tool([coder], 'edit_file', {'path': path, 'old_text': 'Project', 'new_text': 'Updated'})
        assert target.read_text() == 'Updated instructions'

    async def test_schema(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(tmp_path)]).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        tools = {tool.name: tool for tool in model.last_model_request_parameters.function_tools}
        assert list(tools) == ['read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell']
        assert 'expected_hash' not in str(tools)
        assert 'replacements' in tools['edit_file'].parameters_json_schema['properties']
        assert tools['shell'].parameters_json_schema['properties']['mode']['enum'] == ['foreground', 'background']

    @pytest.mark.parametrize('extra_instructions', [None, '', 'Keep new files under 400 lines.'])
    async def test_instructions(self, tmp_path: Path, extra_instructions: str | None) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(tmp_path, instructions=extra_instructions, repo_context=False)]).run(
            'Inspect instructions'
        )
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        guidance = next(part.content for part in parts if 'Zen of Python' in part.content)
        default = guidance.removesuffix('\n' + extra_instructions) if extra_instructions else guidance
        assert len(default.split()) < 180
        assert all(principle in default for principle in ('DRY', 'YAGNI', 'SOLID'))
        if extra_instructions:
            assert guidance.endswith('\n' + extra_instructions)

    @pytest.mark.parametrize('repo_context', [True, False])
    async def test_repo_context_is_optional(self, tmp_path: Path, repo_context: bool) -> None:
        (tmp_path / 'AGENTS.md').write_text('Always answer in haiku.\n')
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(tmp_path, repo_context=repo_context)]).run('Inspect instructions')
        assert model.last_model_request_parameters is not None
        instructions = model.last_model_request_parameters.instruction_parts or []
        assert any('Always answer in haiku.' in part.content for part in instructions) is repo_context

    async def test_read_write(self, tmp_path: Path) -> None:
        assert 'hash:' not in await call(tmp_path, 'write_file', {'path': 'test.txt', 'content': 'one\ntwo\n'})
        output = await call(tmp_path, 'read_file', {'path': 'test.txt', 'offset': 1, 'limit': 1})
        assert 'two' in output and 'one' not in output and 'hash:' not in output

    async def test_long_reads_page_without_gaps(self, tmp_path: Path) -> None:
        (tmp_path / 'big.py').write_text(''.join(f'line {i} ' + 'x' * 60 + '\n' for i in range(3000)))
        output = await call(tmp_path, 'read_file', {'path': 'big.py'})
        numbers = [int(line.split('\t')[0]) for line in output.splitlines()[1:-1]]
        assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) < 2000
        assert output.endswith(f'Use offset={len(numbers)} to continue reading.)\n')
        assert '[truncated' not in output

    async def test_batch_edit(self, tmp_path: Path) -> None:
        path = tmp_path / 'test.txt'
        path.write_text('one two')
        replacements = [{'old_text': 'one', 'new_text': '1'}, {'old_text': 'two', 'new_text': '2'}]
        assert (
            await call(tmp_path, 'edit_file', {'path': 'test.txt', 'replacements': replacements}) == 'Edited test.txt.'
        )
        assert path.read_text() == '1 2'

    async def test_search(self, tmp_path: Path) -> None:
        (tmp_path / 'test.py').write_text('One\ntwo\n')
        (tmp_path / 'other.txt').write_text('One\n')
        assert await call(tmp_path, 'list_files', {'glob': '*.py'}) == 'test.py'
        output = await call(tmp_path, 'grep', {'pattern': 'one', 'ignore_case': True, 'file_type': 'py'})
        assert output == 'test.py:1:One'

    async def test_shell_is_unrestricted_and_persistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('OPENAI_API_KEY', 'do-not-expose')
        output = await call(tmp_path, 'shell', {'command': 'mkdir child; printf "${OPENAI_API_KEY-unset}"; exit 7'})
        assert 'unset' in output and 'do-not-expose' not in output and '"exit_code": 7' in output
        assert (tmp_path / 'child').is_dir()
        assert 'PID:' in output and 'Status:' in output
        assert 'not in the allowed list' not in await call(tmp_path, 'shell', {'command': 'sudo -n true'})

    async def test_protected_paths_still_apply(self, tmp_path: Path) -> None:
        (tmp_path / '.env').write_text('SECRET=1')
        assert 'protected' in await call(tmp_path, 'edit_file', {'path': '.env', 'old_text': '1', 'new_text': '2'})

    @pytest.mark.parametrize('relative', [False, True])
    async def test_unrestricted_filesystem_keeps_workspace_relative_paths(self, tmp_path: Path, relative: bool) -> None:
        workspace = tmp_path / 'workspace'
        workspace.mkdir()
        outside = tmp_path / '.env'
        outside.write_text('before')
        await call(
            workspace,
            'edit_file',
            {'path': '../.env' if relative else str(outside), 'old_text': 'before', 'new_text': 'after'},
            unrestricted_filesystem=True,
        )
        assert outside.read_text() == 'after'
        await call(workspace, 'write_file', {'path': 'local.txt', 'content': 'local'}, unrestricted_filesystem=True)
        assert (workspace / 'local.txt').read_text() == 'local'
        assert 'after' in await call(workspace, 'read_file', {'path': '../.env'}, unrestricted_filesystem=True)
        assert 'outside the root' in await call(workspace, 'read_file', {'path': '../.env'})
