"""The effort shortcut shares the model settings editor's controls and storage."""

from io import StringIO
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.completion import CompleteEvent, Document

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.openai import OpenAIProvider
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
    store.save_model_settings(model, {'custom_params': {'output_config.effort': 'low'}})
    assert 'Custom output_config.effort="low"' in await shell.commands.execute_async('/effort high')
    assert (shell.context.model_overrides(model) or {}).get('extra_body') == {'output_config': {'effort': 'low'}}
    # An obsolete override does not make an unsupported control available.
    await shell.commands.execute_async('/set model test')
    store.save_model_settings('test', {'openai_reasoning_effort': 'high'})
    assert 'No reasoning effort control' in await shell.commands.execute_async('/effort')


@pytest.mark.parametrize('model_class', [OpenAIChatModel, OpenAIResponsesModel])
async def test_effort_for_supplied_model_instance(
    tmp_path: Path, model_class: type[OpenAIChatModel] | type[OpenAIResponsesModel]
) -> None:
    model = model_class('gpt-5', provider=OpenAIProvider(api_key='test'))
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        Agent(model),
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
    assert 'openai_reasoning_effort' in await shell.commands.execute_async('/effort')
    assert 'high' in [c.text for c in shell.commands.get_completions(Document('/effort '), CompleteEvent())]
    assert 'Saved' in await shell.commands.execute_async('/effort high')
    assert store.model_settings('gpt-5') == {'openai_reasoning_effort': 'high'}
    assert store.model_settings('openai:gpt-5') == {}
    assert (shell.context.model_overrides('gpt-5') or {}).get('openai_reasoning_effort') == 'high'
    # Chat Completions and Responses consume different custom effort parameters.
    custom: dict[str, JsonValue] = (
        {'reasoning_effort': 'low'} if model_class is OpenAIChatModel else {'reasoning.effort': 'low'}
    )
    store.save_model_settings('gpt-5', {'custom_params': custom})
    assert 'overrides configured effort' in await shell.commands.execute_async('/effort high')
    assert store.model_settings('gpt-5') == {'custom_params': custom}
    await shell.commands.execute_async('/set model test')
    assert 'No reasoning effort control for test' in await shell.commands.execute_async('/effort')


@pytest.mark.parametrize(
    'model,key,custom,body,conflict',
    [
        ('vllm:glm-5.3', 'glm_reasoning_effort', {'reasoning_effort': 'low'}, {'reasoning_effort': 'low'}, True),
        (
            'openai-chat:gpt-5',
            'openai_reasoning_effort',
            {'reasoning_effort': 'low'},
            {'reasoning_effort': 'low'},
            True,
        ),
        (
            'openai:gpt-5',
            'openai_reasoning_effort',
            {'reasoning.effort': 'low'},
            {'reasoning': {'effort': 'low'}},
            True,
        ),
        (
            'openai-codex:gpt-5',
            'openai_reasoning_effort',
            {'reasoning': {'effort': 'low'}},
            {'reasoning': {'effort': 'low'}},
            True,
        ),
        ('openai:gpt-5', 'openai_reasoning_effort', {'reasoning': None}, {'reasoning': None}, True),
        (
            'anthropic:claude-opus-5',
            'anthropic_effort',
            {'output_config.effort': 'low'},
            {'output_config': {'effort': 'low'}},
            True,
        ),
        (
            'anthropic:claude-opus-5',
            'anthropic_effort',
            {'output_config': {'effort': 'low'}},
            {'output_config': {'effort': 'low'}},
            True,
        ),
        (
            'openai:gpt-5',
            'openai_reasoning_effort',
            {'reasoning.summary': 'auto'},
            {'reasoning': {'summary': 'auto'}},
            False,
        ),
        ('anthropic:claude-opus-5', 'anthropic_effort', {'output_config': {}}, {'output_config': {}}, False),
        (
            'openai-chat:gpt-5',
            'openai_reasoning_effort',
            {'reasoning.effort': 'low'},
            {'reasoning': {'effort': 'low'}},
            False,
        ),
        ('openai:gpt-5', 'openai_reasoning_effort', {'reasoning_effort': 'low'}, {'reasoning_effort': 'low'}, False),
        # Expansion order matters: a later parent can remove a dotted effort override.
        (
            'openai:gpt-5',
            'openai_reasoning_effort',
            {'reasoning.effort': 'low', 'reasoning': {'summary': 'auto'}},
            {'reasoning': {'summary': 'auto'}},
            False,
        ),
    ],
)
async def test_effort_custom_parameter_precedence(
    tmp_path: Path, model: str, key: str, custom: dict[str, JsonValue], body: dict[str, JsonValue], conflict: bool
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    saved: dict[str, JsonValue] = {key: 'medium', 'custom_params': custom}
    store.save_model_settings(model, saved)
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
    result = await shell.commands.execute_async('/effort high')
    if conflict:
        assert 'overrides configured effort' in result
        assert 'Remove it with /model settings' in result
        assert 'low' in result or 'null' in result
        assert store.model_settings(model) == saved
        assert 'overrides configured effort' in await shell.commands.execute_async('/effort')
    else:
        assert 'Saved' in result
        assert store.model_settings(model) == {**saved, key: 'high'}
    assert (shell.context.model_overrides(model) or {}).get('extra_body') == body
    reset = await shell.commands.execute_async('/effort reset')
    assert 'Reset' in reset
    assert ('overrides configured effort' in reset) is conflict
    assert store.model_settings(model) == {'custom_params': custom}
    assert (shell.context.model_overrides(model) or {}).get('extra_body') == body
