"""The effort shortcut shares the model settings editor's controls and storage."""

from io import StringIO
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.completion import CompleteEvent, Document

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_clai2._app import create_shell
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource


@pytest.mark.parametrize(
    'model,key',
    [
        ('openai-codex:gpt-6-astra', 'openai_reasoning_effort'),
        ('openai@work:gpt-5', 'openai_reasoning_effort'),
        ('anthropic:claude-sonnet-4-6', 'anthropic_effort'),
        ('anthropic:claude-opus-4-6', 'anthropic_effort'),
        ('vllm:glm-5.3', 'glm_reasoning_effort'),
        ('test', None),
        ('google-gla:gemini-3-pro-preview', None),
    ],
)
async def test_effort_command(tmp_path: Path, model: str, key: str | None) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    saved: dict[str, JsonValue] = {'future_setting': {'keep': True}, 'custom_params': {'unrelated': 1}}
    if key == 'openai_reasoning_effort':
        saved['service_tier'] = 'priority'
    store.save_model_settings(model, saved)
    shell = create_shell(
        Agent(TestModel(model_name=model)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model=None),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert not shell.commands.runs_during_turn('/effort')
    assert '/effort:' in await shell.commands.execute_async('/help')
    with pytest.raises(ValueError, match='Usage: /effort'):
        await shell.commands.execute_async('/effort high low')
    source = ModelSettingsSource(store, model)
    row = source.effort_row()
    completions = [c.text for c in shell.commands.get_completions(Document('/effort '), CompleteEvent())]
    assert list(shell.commands.get_completions(Document('/effort high '), CompleteEvent())) == []
    if key is None:
        assert row is None and completions == []
        for command in ('/effort', '/effort high', '/effort reset'):
            assert 'No reasoning effort control' in await shell.commands.execute_async(command)
        assert store.model_settings(model) == saved
        return
    assert row is not None and row.key == key
    assert completions == [*row.choices, 'reset']
    assert f': {source.current(row)} ({key})' in await shell.commands.execute_async('/effort')
    assert store.model_settings(model) == saved  # Viewing never materializes defaults.
    for value in row.choices:
        assert 'Saved' in await shell.commands.execute_async(f'/effort {value}')
        assert SettingsStore(store.path).model_settings(model) == {**saved, key: value}
        assert f': {value} ({key})' in await shell.commands.execute_async('/effort')
    for invalid in ('banana', 'null', 'true'):
        with pytest.raises(ValueError, match='Choose'):
            await shell.commands.execute_async(f'/effort {invalid}')
        assert store.model_settings(model) == {**saved, key: row.choices[-1]}
    assert 'Reset' in await shell.commands.execute_async('/effort reset')
    assert store.model_settings(model) == saved
    await shell.commands.execute_async('/set model openai:gpt-5')
    assert 'openai:gpt-5' in await shell.commands.execute_async('/effort')
    await shell.commands.execute_async('/effort high')
    assert store.model_settings('openai:gpt-5')['openai_reasoning_effort'] == 'high'
    assert store.model_settings(model) == saved


async def test_effort_uses_mapped_controls_and_validation(tmp_path: Path) -> None:
    model = 'custom:claude-opus-5'
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=Settings(model=model),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    shell.context.settings_model = lambda name: name.replace('custom:', 'anthropic:')
    saved: dict[str, JsonValue] = {'anthropic_thinking_mode': 'disabled', 'future_setting': 1}
    store.save_model_settings(model, saved)
    assert 'requires thinking' in await shell.commands.execute_async('/effort max')
    assert store.model_settings(model) == saved
    assert 'Saved' in await shell.commands.execute_async('/effort low')
    assert store.model_settings(model) == {**saved, 'anthropic_effort': 'low'}
    assert store.model_settings('anthropic:claude-opus-5') == {}
    assert (shell.context.model_settings(model) or {}).get('anthropic_effort') == 'low'
    # An obsolete override does not make an unsupported control available.
    await shell.commands.execute_async('/set model test')
    store.save_model_settings('test', {'openai_reasoning_effort': 'high'})
    assert 'No reasoning effort control' in await shell.commands.execute_async('/effort')
