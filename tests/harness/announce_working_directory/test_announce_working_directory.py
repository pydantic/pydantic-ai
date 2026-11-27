"""Test that `AnnounceWorkingDirectory` names the workspace's working directory once, whoever brings it.

Not VCR tests: what is checked is the instruction text each request carries, which `TestModel` exposes directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import AnnounceWorkingDirectory, FileSystem, Shell


def _lines(model: TestModel) -> list[str]:
    assert model.last_model_request_parameters is not None
    parts = model.last_model_request_parameters.instruction_parts or []
    return [part.content for part in parts if 'working directory is' in part.content]


class TestAnnounceWorkingDirectory:
    @pytest.mark.parametrize(
        'capabilities',
        [
            pytest.param(lambda: [AnnounceWorkingDirectory[None]()], id='own-tools'),
            pytest.param(lambda: [Shell[None]()], id='shell'),
            pytest.param(lambda: [FileSystem[None]()], id='file-system'),
            pytest.param(lambda: [Shell[None](), FileSystem[None](), AnnounceWorkingDirectory[None]()], id='all'),
        ],
    )
    async def test_the_directory_is_named_once(
        self, tmp_path: Path, capabilities: Callable[[], list[AbstractCapability[None]]]
    ) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, deps_type=type(None), capabilities=capabilities()).run(
            'go', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert _lines(model) == [f'The working directory is `{tmp_path.resolve()}`.']

    async def test_a_workspace_that_is_gone_is_left_out(self, tmp_path: Path) -> None:
        """A run that calls no tool doesn't fail because its workspace is gone; a tool call would report it."""
        model = TestModel(call_tools=[])
        agent = Agent(model, deps_type=type(None), capabilities=[AnnounceWorkingDirectory[None]()])
        await agent.run('go', workspace=LocalWorkspaceBackend(tmp_path / 'gone'))
        assert _lines(model) == []

    @pytest.mark.parametrize(
        'capability',
        [
            pytest.param(lambda: Shell[None](id='shell', defer_loading=True), id='shell'),
            pytest.param(lambda: FileSystem[None](id='file_system', defer_loading=True), id='file-system'),
        ],
    )
    async def test_a_deferred_capability_does_not_name_it_before_it_loads(
        self, tmp_path: Path, capability: Callable[[], AbstractCapability[None]]
    ) -> None:
        """Naming the directory starts a sandbox, which a deferred capability that's never loaded doesn't need."""
        model = TestModel(call_tools=[])
        await Agent(model, deps_type=type(None), capabilities=[capability()]).run(
            'go', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert _lines(model) == []

    async def test_the_workspace_is_asked_once_per_run(self, tmp_path: Path) -> None:
        """A later request reuses the line, so a workspace that goes away mid-run can't drop it from the prefix."""

        class Counting(LocalWorkspaceBackend):
            asked = 0

            async def working_dir(self) -> str:
                self.asked += 1
                return await super().working_dir()

        lines: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            parts = info.model_request_parameters.instruction_parts or []
            lines.append([part.content for part in parts if 'working directory is' in part.content])
            if len(lines) == 1:
                return ModelResponse(parts=[ToolCallPart('probe', {})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[AnnounceWorkingDirectory[None]()])

        @agent.tool_plain
        def probe() -> str:
            return 'ok'

        first, second = Counting(tmp_path), Counting(tmp_path)
        await agent.run('go', workspace=first)
        await agent.run('again', workspace=second)
        line = f'The working directory is `{tmp_path.resolve()}`.'
        assert lines == [[line], [line], [line]]
        assert (first.asked, second.asked) == (1, 1)
