"""`SessionRuntime`: one live session built over the `ClaiProcess` it shares stores with."""

import io
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime.process import ClaiProcess
from pydantic_clai2.runtime.session_runtime import create_shell


async def test_a_runtime_without_a_terminal_serves_input_over_the_process_stores(tmp_path: Path) -> None:
    """With no terminal, `send_input` feeds the read loop; the turn is saved in the process's conversation store."""
    process = ClaiProcess.open(SettingsStore(tmp_path / 'config.db'))
    (tmp_path / 'project').mkdir()
    output = io.StringIO()
    runtime = create_shell(
        Agent(TestModel(), deps_type=type(None)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, force_terminal=False, width=120),
        settings=Settings(model='test'),
        store=None,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=tmp_path / 'project',
        process=process,
    )
    assert (runtime.process, runtime.context.store, runtime.sessions.store) == (
        process,
        process.store,
        process.conversations,
    )

    runtime.send_input('hello')
    runtime.send_input('/exit')

    assert await runtime.run() == 'exit'
    assert '> hello\n\nsuccess (no tool calls)' in output.getvalue()
    saved = await process.conversations.get(conversation_id=runtime.session.summary.id)
    assert saved.summary.workspace == str((tmp_path / 'project').resolve())


def test_a_runtime_takes_its_store_from_the_process(tmp_path: Path) -> None:
    """`store` and `process` name the settings twice, so passing both is refused rather than one dropped."""
    store = SettingsStore(tmp_path / 'config.db')
    with pytest.raises(TypeError, match='`store` or `process`, not both'):
        create_shell(
            Agent(TestModel(), deps_type=type(None)),
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO()),
            settings=None,
            store=store,
            builtin_plugins=(),
            project=ProjectSettings(),
            headless=True,
            process=ClaiProcess.open(store),
        )
