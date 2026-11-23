"""How `Shell` and `FileSystem` name the working directory in their instructions.

Not VCR tests: what is checked is the instruction text each request carries, and whether building it
started the workspace, which `TestModel` and `FunctionModel` expose directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import InstructionDeltaPart, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, SupportsKnownWorkingDir, WorkspaceRef
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.shell import Shell


class LazyBackend(LocalWorkspaceBackend, SupportsKnownWorkingDir):
    """A local directory that, like a sandbox, has no ref until an operation starts it."""

    def __init__(self, working_dir: Path, *, configured: bool) -> None:
        super().__init__(working_dir)
        self._configured = str(working_dir.resolve()) if configured else None
        self.started = False

    @property
    def known_working_dir(self) -> str | None:
        return self._configured

    @property
    def ref(self) -> WorkspaceRef | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().ref if self.started else None

    async def working_dir(self) -> str:
        self.started = True
        return await super().working_dir()


def _instructions(model: TestModel, path: Path) -> str:
    assert model.last_model_request_parameters is not None
    parts = model.last_model_request_parameters.instruction_parts or []
    return '\n\n'.join(part.content for part in parts).replace(str(path.resolve()), '<dir>')


class TestWorkingDirInstructions:
    @pytest.mark.parametrize(
        ('root_dir', 'expected'),
        [
            (
                None,
                snapshot("""\
<pyai:context id="capability:shell">
Shell commands start in `<dir>`.
</pyai:context>

<pyai:context id="capability:file_system">
Relative file paths resolve from `<dir>`, and the file tools only accept paths inside it.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
"""),
            ),
            (
                '/',
                snapshot("""\
<pyai:context id="capability:shell">
Shell commands start in `<dir>`.
</pyai:context>

<pyai:context id="capability:file_system">
Relative file paths resolve from `<dir>`.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
"""),
            ),
        ],
    )
    async def test_a_local_workspace_is_named(self, tmp_path: Path, root_dir: str | None, expected: str) -> None:
        model = TestModel(call_tools=[])
        agent = Agent(model, capabilities=[Shell(), FileSystem(root_dir=root_dir)])
        await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path))
        assert _instructions(model, tmp_path) == expected

    async def test_a_configured_sandbox_is_named_without_starting_it(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        backend = LazyBackend(tmp_path, configured=True)
        await Agent(model, capabilities=[Shell(), FileSystem()]).run('go', workspace=backend)
        assert '`<dir>`' in _instructions(model, tmp_path)
        assert not backend.started

    async def test_an_unstarted_sandbox_is_named_after_its_first_tool_call(self, tmp_path: Path) -> None:
        """Until a tool starts the sandbox, the lines don't name it; then the path is appended, not rewritten."""
        (tmp_path / 'a.txt').write_text('a')
        backend = LazyBackend(tmp_path, configured=False)
        first: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                first.append(info.instructions or '')
                return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'a.txt'})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(model_fn), capabilities=[Shell(), FileSystem()])
        result = await agent.run('go', workspace=backend)

        assert first == snapshot(
            [
                """\
<pyai:context id="capability:shell">
Shell commands start in the working directory; use relative paths, not guessed absolute ones.
</pyai:context>

<pyai:context id="capability:file_system">
Relative file paths resolve from the working directory, and the file tools only accept paths inside it. Use them, not guessed absolute paths.
</pyai:context>
A later <pyai:context> element with the same id replaces the earlier one and stays in effect until replaced again.\
"""
            ]
        )
        deltas = [
            str(p.content).replace(str(tmp_path.resolve()), '<dir>')
            for m in result.all_messages()
            for p in m.parts
            if isinstance(p, InstructionDeltaPart)
        ]
        assert deltas == snapshot(
            [
                'Shell commands start in `<dir>`.',
                'Relative file paths resolve from `<dir>`, and the file tools only accept paths inside it.',
            ]
        )
