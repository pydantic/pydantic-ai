"""Coder's tool surface as the model sees it; the tools themselves are tested with `FileSystem` and `Shell`."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, ValidatedToolArgs
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceBackend
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.coder._capability import MAX_FILE_TOOL_RETRIES
from pydantic_ai_harness.tool_output_limits import ToolOutputLimits

from ...workspace_fakes import FakeWorkspace, FilesystemOnlyWorkspaceBackend
from .._recording_durability import RecordingDurability
from .._tool_calls import call_tool, call_tools


@dataclass
class _ToolLog(AbstractCapability[object]):
    """A host capability bound next to `Coder`, recording every tool call it sees."""

    seen: list[str] = field(default_factory=list[str])

    async def before_tool_execute(
        self, ctx: RunContext[object], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        self.seen.append(call.tool_name)
        return args


def _delegate_a_command(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The parent run delegates, the delegate runs one command, and both then finish."""
    first = messages[0].parts[-1]
    prompt = first.content if isinstance(first, UserPromptPart) else None
    if len(messages) > 1:
        return ModelResponse(parts=[TextPart('done')])
    if prompt == 'Delegate it':
        return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': 'self', 'task': 'Run it'})])
    return ModelResponse(parts=[ToolCallPart('shell', {'command': 'echo delegated'})])


async def call(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
    unrestricted_filesystem: bool = False,
) -> str:
    coder = Coder[None](unrestricted_filesystem=unrestricted_filesystem)
    return await call_tool(
        [coder, *capabilities], name, arguments, workspace=LocalWorkspaceBackend(working_dir=tmp_path)
    )


class TestCoder:
    def test_deprecated_workspace_is_ignored(self) -> None:
        with pytest.warns(Warning, match='workspace'):
            coder = Coder(workspace='elsewhere')
        assert coder is not None

    @pytest.mark.parametrize('extra_limits', [False, True])
    async def test_durable_binding(self, tmp_path: Path, extra_limits: bool) -> None:
        durability = RecordingDurability()
        capabilities: list[AbstractCapability[object]] = [
            LocalWorkspace(tmp_path),
            Coder(),
            durability,
        ]
        if extra_limits:
            capabilities.append(ToolOutputLimits())
        agent = Agent(TestModel(call_tools=[], custom_output_text='done'), name='coder', capabilities=capabilities)
        result = await agent.run('Inspect tools')
        assert result.output == 'done'
        assert [(name, getattr(args[0], 'method', None)) for name, args in durability.calls] == [
            ('coder__capability__workspace.call', 'ensure'),
            ('coder__capability__workspace.call', 'stat'),
            ('coder__capability__workspace.call', 'stat'),
            ('coder__model.request_stream', None),
        ]

    @pytest.mark.parametrize('unrestricted_filesystem', [False, True])
    async def test_discovered_paths_can_be_read_and_edited(self, tmp_path: Path, unrestricted_filesystem: bool) -> None:
        (tmp_path / 'src').mkdir()
        target = tmp_path / 'src' / 'AGENTS.md'
        target.write_text('Project instructions')
        coder = Coder[None](unrestricted_filesystem=unrestricted_filesystem, repo_context=False)
        workspace = LocalWorkspaceBackend(working_dir=tmp_path)
        path = await call_tool([coder], 'list_files', {'glob': '**/AGENTS.md'}, workspace=workspace)
        assert Path(path) == Path('src/AGENTS.md')
        assert 'Project instructions' in await call_tool([coder], 'read_file', {'path': path}, workspace=workspace)
        await call_tool(
            [coder], 'edit_file', {'path': path, 'old_text': 'Project', 'new_text': 'Updated'}, workspace=workspace
        )
        assert target.read_text() == 'Updated instructions'

    async def test_schema(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder()]).run(
            'Inspect tools', workspace=LocalWorkspaceBackend(working_dir=tmp_path)
        )
        assert model.last_model_request_parameters is not None
        tools = {tool.name: tool for tool in model.last_model_request_parameters.function_tools}
        assert list(tools) == ['read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell', 'delegate_task']
        assert 'expected_hash' not in str(tools)
        assert 'replacements' in tools['edit_file'].parameters_json_schema['properties']
        assert tools['shell'].parameters_json_schema['properties']['mode']['enum'] == ['foreground', 'background']

    async def test_sub_agents_can_be_left_out(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[LocalWorkspace(tmp_path), Coder(sub_agents=False)]).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        tools = [tool.name for tool in model.last_model_request_parameters.function_tools]
        assert tools == ['read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell']

    async def test_the_delegate_carries_capabilities_bound_next_to_coder(self, tmp_path: Path) -> None:
        """The delegate is the agent `Coder` is bound to, so a host hook sees the commands it runs."""
        log = _ToolLog()
        capabilities: list[AbstractCapability[object]] = [LocalWorkspace(tmp_path), Coder(repo_context=False), log]
        agent = Agent(FunctionModel(_delegate_a_command), capabilities=capabilities)
        assert (await agent.run('Delegate it')).output == 'done'
        assert log.seen == ['delegate_task', 'shell']

    async def test_passing_coder_to_run_is_refused(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[LocalWorkspace(tmp_path)])
        with pytest.raises(UserError, match='only carries what is bound to the `Agent`'):
            await agent.run('Inspect tools', capabilities=[Coder()])
        result = await agent.run('Inspect tools', capabilities=[Coder(sub_agents=False)])
        assert result.output == 'success (no tool calls)'

    @pytest.mark.parametrize('extra_instructions', [None, '', 'Keep new files under 400 lines.'])
    async def test_instructions(self, tmp_path: Path, extra_instructions: str | None) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(instructions=extra_instructions, repo_context=False)]).run(
            'Inspect instructions', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        guidance = next(part.content for part in parts if 'Zen of Python' in part.content)
        default = guidance.removesuffix('\n' + extra_instructions) if extra_instructions else guidance
        assert len(default.split()) < 180
        assert all(principle in default for principle in ('DRY', 'YAGNI', 'SOLID'))
        assert 'delete any scratch files you created' in default
        if extra_instructions:
            assert guidance.endswith('\n' + extra_instructions)

    async def test_file_tool_mistakes_do_not_end_the_run(self, tmp_path: Path) -> None:
        """Consecutive denied writes come back as retries until `MAX_FILE_TOOL_RETRIES` is spent."""
        denied: tuple[str, dict[str, object]] = ('write_file', {'path': '/elsewhere/probe.py', 'content': 'x'})
        coder = Coder[None](repo_context=False, sub_agents=False)
        workspace = LocalWorkspaceBackend(working_dir=tmp_path)
        results = await call_tools([coder], [denied] * MAX_FILE_TOOL_RETRIES, workspace=workspace)
        assert len(results) == MAX_FILE_TOOL_RETRIES
        assert all('the file tools only work inside it' in result for result in results)
        with pytest.raises(UnexpectedModelBehavior, match=f'exceeded max retries count of {MAX_FILE_TOOL_RETRIES}'):
            await call_tools([coder], [denied] * (MAX_FILE_TOOL_RETRIES + 1), workspace=workspace)

    @pytest.mark.parametrize(
        ('unrestricted', 'files'),
        [(False, 'the file tools only accept paths inside it'), (True, 'relative file paths resolve from it')],
    )
    async def test_instructions_name_the_project_once(self, tmp_path: Path, unrestricted: bool, files: str) -> None:
        model = TestModel(call_tools=[])
        coder = Coder(unrestricted_filesystem=unrestricted)
        await Agent(model, capabilities=[coder]).run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        notes = [part.content for part in parts if 'Your project is' in part.content]
        assert notes == [f'Your project is `{tmp_path.resolve()}`. Shell commands start there, and {files}.']

    async def test_lazy_sandbox_instructions_leave_out_the_project(self, tmp_path: Path) -> None:
        # Without `RepoContext` nothing touches the workspace at run start, so neither do the instructions.
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(repo_context=False)]).run(
            'go', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        parts = model.last_model_request_parameters.instruction_parts or []
        assert not any('Your project is' in part.content for part in parts)

    @pytest.mark.parametrize('repo_context', [True, False])
    async def test_repo_context_is_optional(self, tmp_path: Path, repo_context: bool) -> None:
        (tmp_path / 'AGENTS.md').write_text('Always answer in haiku.\n')
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder(repo_context=repo_context)]).run(
            'Inspect instructions', workspace=LocalWorkspaceBackend(working_dir=tmp_path)
        )
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

    @pytest.mark.parametrize('read_only', [True, False], ids=['read-only', 'filesystem-only'])
    async def test_search_without_commands(self, tmp_path: Path, read_only: bool) -> None:
        """A reviewer on a read-only or filesystem-only workspace can still list and search files."""
        (tmp_path / 'notes.txt').write_text('needle\n')
        workspace: WorkspaceBackend = (
            ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
            if read_only
            else FilesystemOnlyWorkspaceBackend(FakeWorkspace('files', {'/workspace/notes.txt': b'needle\n'}))
        )
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Coder()]).run('Inspect', workspace=workspace)
        assert model.last_model_request_parameters is not None
        names = {tool.name for tool in model.last_model_request_parameters.function_tools}
        assert {'list_files', 'grep'} <= names
        assert 'shell' not in names
        assert await call_tool([Coder[None]()], 'grep', {'pattern': 'needle'}, workspace=workspace) == (
            'notes.txt:1:needle'
        )
        assert await call_tool([Coder[None]()], 'list_files', {}, workspace=workspace) == 'notes.txt'

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
        refused = await call(workspace, 'read_file', {'path': '../.env'})
        assert f'`{outside}` is outside the project root `{workspace.resolve()}`' in refused
        assert 'Create or clone it inside the project, or use a shell tool if you have one.' in refused
