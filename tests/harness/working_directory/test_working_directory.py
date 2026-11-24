"""Test that `WorkingDirectory` names the workspace's working directory once, whoever brings it.

Not VCR tests: what is checked is the instruction text each request carries, which `TestModel` exposes directly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import FileSystem, Shell, WorkingDirectory


def _lines(model: TestModel) -> list[str]:
    assert model.last_model_request_parameters is not None
    parts = model.last_model_request_parameters.instruction_parts or []
    return [part.content for part in parts if 'working directory is' in part.content]


class TestWorkingDirectory:
    @pytest.mark.parametrize(
        'capabilities',
        [
            pytest.param(lambda: [WorkingDirectory[None]()], id='own-tools'),
            pytest.param(lambda: [Shell[None]()], id='shell'),
            pytest.param(lambda: [FileSystem[None]()], id='file-system'),
            pytest.param(lambda: [Shell[None](), FileSystem[None](), WorkingDirectory[None]()], id='all'),
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
        agent = Agent(model, deps_type=type(None), capabilities=[WorkingDirectory[None]()])
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
