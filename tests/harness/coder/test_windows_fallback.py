"""`Coder` gives a run on Windows that no capability gives a workspace a `WindowsWorkspace('.')`.

The platform is patched, with `LocalWorkspace` standing in for `WindowsWorkspace`.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness import Coder


@pytest.fixture
def on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Pretend to run on Windows, in `tmp_path`."""
    monkeypatch.setattr('pydantic_ai_harness.coder._capability.sys', SimpleNamespace(platform='win32'))
    monkeypatch.setattr('pydantic_ai_harness.coder._capability.WindowsWorkspace', LocalWorkspace)
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestCoderWindowsFallback:
    async def test_a_run_without_a_workspace_gets_the_current_directory(self, on_windows: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Coder(repo_context=False, sub_agents=False)])
        result = await agent.run('hello')
        assert result.workspace is not None
        assert result.workspace.ref == WorkspaceRef(provider='local', id=str(on_windows))

    async def test_another_workspace_wins(self, on_windows: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
        other = tmp_path_factory.mktemp('other')
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[Coder(repo_context=False, sub_agents=False), LocalWorkspace(other)],
        )
        result = await agent.run('hello')
        assert result.workspace is not None
        assert result.workspace.ref == WorkspaceRef(provider='local', id=str(other))

    async def test_only_on_windows(self) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Coder(repo_context=False, sub_agents=False)])
        with pytest.raises(UserError, match='needs a workspace'):
            await agent.run('hello')
