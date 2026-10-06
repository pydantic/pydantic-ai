"""Family defaults and model-settings navigation, independent of the active model."""

from pathlib import Path

import pytest
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.model_options import model_options
from pydantic_clai2.models.model_settings import model_defaults
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.model_menu import (
    ModelSettingsSource,
    build_model_settings_picker,
    model_settings_command,
    model_settings_summary,
)
from tests.clai2.menu_script import Script, make_context, pick


@pytest.mark.parametrize('provider', ['openai', 'openai-codex', 'openai-chat', 'azure', 'openrouter', 'custom'])
@pytest.mark.parametrize('name', ['gpt-6', 'gpt-6-astra', 'gpt-6.1', 'gpt-5.6', 'gpt-5.6-pro', 'openai/gpt-5.6:free'])
def test_defaults_apply_without_saved_preferences(tmp_path: Path, provider: str, name: str) -> None:
    context, _ = make_context(tmp_path)
    model = f'{provider}:{name}'
    assert context.model_settings(model) == {
        'thinking': True,
        'service_tier': 'default',
        'openai_reasoning_effort': 'medium',
        'openai_reasoning_context': 'all_turns',
        'openai_reasoning_mode': 'standard',
        'openai_reasoning_summary': 'detailed',
        'openai_text_verbosity': 'low',
    }
    assert context.store.model_settings(model) == {}


@pytest.mark.parametrize('model', ['openai:gpt-5.5', 'openai:gpt-5.60', 'gpt-60', 'gpt-5.6ish', 'test', ''])
def test_other_models_unchanged(model: str) -> None:
    assert model_defaults(model=model) == {}


@pytest.mark.parametrize(('provider', 'ttl'), [('anthropic', '5m'), ('gateway/anthropic', '5m'), ('claude-code', '1h')])
def test_anthropic_cache_defaults(tmp_path: Path, provider: str, ttl: str) -> None:
    context, _ = make_context(tmp_path)
    model = f'{provider}:claude-sonnet-4-6'
    expected = {
        'anthropic_cache': ttl,
        'anthropic_cache_instructions': ttl,
        'anthropic_cache_tool_definitions': ttl,
    }
    assert context.model_settings(model) == expected
    assert context.store.model_settings(model) == {}
    source = ModelSettingsSource(context.store, model)
    menu = FieldMenu(source, searchable=False)
    for key in expected:
        row = menu.row_for(key)
        assert row is not None
        assert source.current(row) == ttl
    context.store.save_model_settings(model, {key: False for key in expected})
    assert context.model_settings(model) == {key: False for key in expected}


@pytest.mark.parametrize('provider', ['anthropic', 'claude-code'])
async def test_cache_defaults_reach_anthropic_request(
    tmp_path: Path, provider: str, allow_model_requests: None
) -> None:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai import Agent, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from tests.models.test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

    context, _ = make_context(tmp_path)
    client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(type='text', text='Done')], BetaUsage(input_tokens=1, output_tokens=1))
    )
    model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model, instructions='Static instructions')

    @agent.tool_plain
    def lookup() -> str:  # pragma: no cover
        """Look up information."""
        return 'result'

    await agent.run(
        'Continue',
        model_settings=context.model_settings(f'{provider}:claude-sonnet-4-6'),
        message_history=[
            ModelRequest(parts=[UserPromptPart('Start')]),
            ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='call')]),
            ModelRequest(parts=[ToolReturnPart('lookup', 'result', tool_call_id='call')]),
        ],
    )
    request = get_mock_chat_completion_kwargs(client)[0]
    expected = {'type': 'ephemeral', 'ttl': '1h' if provider == 'claude-code' else '5m'}
    assert request['cache_control'] == expected
    assert request['system'][-1]['cache_control'] == expected
    assert request['tools'][-1]['cache_control'] == expected
    assert model.resolve_cache_retention(None) is None  # Library defaults remain opt-in.


def test_bare_identity_and_override_reset(tmp_path: Path) -> None:
    assert model_defaults(model='gpt-6') == model_defaults(model='custom:gpt-6')
    context, _ = make_context(tmp_path)
    model = 'openai:gpt-6-astra'
    source = ModelSettingsSource(context.store, model)
    menu = FieldMenu(source, searchable=False)
    effort = menu.row_for('openai_reasoning_effort')
    assert effort is not None
    assert source.current(effort) == effort.default == 'medium'
    assert 'Reasoning Effort' in menu.details(MenuItem('', value=effort.key))
    assert not effort.allow_custom
    assert source.apply(effort, 'high').startswith('Saved')
    assert context.store.model_settings(model) == {'openai_reasoning_effort': 'high'}
    assert context.model_settings(model) == {**model_defaults(model=model), 'openai_reasoning_effort': 'high'}
    source.reset(effort)
    assert source.current(effort) == 'medium'
    context.store.save_model_settings(model, {'thinking': False})
    assert context.model_settings(model) == {**model_defaults(model=model), 'thinking': False}


async def test_picker_edits_multiple_models_without_selecting_them(tmp_path: Path) -> None:
    context, applied = make_context(tmp_path)
    names = ['openai:gpt-6-astra', 'openai:gpt-5.6']
    for name in names:
        context.store.add_model(name=name)
    selected = context.settings.model
    script = Script(
        lists=[
            pick(names[0]),
            pick('openai_reasoning_effort'),
            MenuResult(cancelled=True),
            pick(names[1]),
            pick('openai_text_verbosity'),
            MenuResult(cancelled=True),
            MenuResult(cancelled=True),
        ],
        choices=[pick('high'), pick('medium')],
        texts=[],
    )
    result = await model_settings_command(context, [], runners=script.runners)
    assert 'Saved openai_reasoning_effort' in result and 'Saved openai_text_verbosity' in result
    assert context.store.model_settings(names[0]) == {'openai_reasoning_effort': 'high'}
    assert context.store.model_settings(names[1]) == {'openai_text_verbosity': 'medium'}
    assert context.settings.model == selected and applied == []


@pytest.mark.parametrize('result', [MenuResult(cancelled=True), MenuResult(), pick(123), pick('missing')])
async def test_picker_exit(tmp_path: Path, result: MenuResult) -> None:
    context, _ = make_context(tmp_path)
    assert (
        await model_settings_command(context, [], runners=Script(lists=[result], choices=[], texts=[]).runners)
        == 'No changes.'
    )


def test_picker_builds_without_active_model(tmp_path: Path) -> None:
    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'empty.db'),
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
    )
    menu = build_model_settings_picker(context=context, current=None)
    assert menu.highlighted is not None and menu.highlighted.disabled
    context.store.add_model(name='openai:gpt-6')
    menu = build_model_settings_picker(context=context, current='openai:gpt-6')
    assert menu.highlighted is not None and menu.highlighted.value == 'openai:gpt-6'


@pytest.mark.parametrize('model', ['openai:gpt-6', 'openai:gpt-5.6', 'openrouter:openai/gpt-6'])
def test_reasoning_models_hide_irrelevant_controls(model: str) -> None:
    assert (
        not {'temperature', 'top_p', 'top_k', 'seed', 'timeout', 'presence_penalty', 'frequency_penalty'}
        & model_options(model=model).keys()
    )


@pytest.mark.parametrize('model', ['google:gemini-3-pro', 'google:gemini-2.5-pro'])
def test_gemini_thinking(model: str) -> None:
    assert 'thinking' in model_options(model=model)


@pytest.mark.parametrize('model', ['custom:unknown', 'google:gemini-2.0-flash'])
def test_unknown_and_nonreasoning_fallbacks(model: str) -> None:
    assert 'thinking' not in model_options(model=model)


def test_model_preview(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    assert 'No custom settings' in model_settings_summary(store=context.store, model='test')
    context.store.save_model_settings('openai:gpt-6', {'openai_reasoning_effort': 'high'})
    text = model_settings_summary(store=context.store, model='openai:gpt-6')
    assert 'Reasoning Effort: high' in text and 'Verbosity: low' in text


def test_nonreasoning_openai_hides_verbosity() -> None:
    assert 'openai_text_verbosity' not in model_options(model='openai:gpt-4o')


@pytest.mark.parametrize('provider', ['openrouter', 'vllm'])
def test_chat_compatible_reasoning_defaults_can_be_overridden(tmp_path: Path, provider: str) -> None:
    context, _ = make_context(tmp_path)
    name = f'{provider}:openai/gpt-6'
    source = ModelSettingsSource(context.store, name)
    menu = FieldMenu(source)
    row = menu.row_for('openai_reasoning_effort')
    assert row is not None
    assert source.current(row) == 'medium'
    assert source.apply(row, 'high').startswith('Saved')
    settings = context.model_settings(name)
    assert settings == {**model_defaults(model=name), 'openai_reasoning_effort': 'high'}
    assert menu.row_for('service_tier') is not None
    assert menu.row_for('openai_reasoning_mode') is None
    assert menu.row_for('openai_reasoning_context') is None
    assert menu.row_for('openai_reasoning_summary') is None
    source.reset(row)
    assert source.current(row) == 'medium'


@pytest.mark.parametrize('model', ['custom:gpt-6', 'custom:o3'])
def test_unknown_provider_does_not_claim_chat_protocol_support(model: str) -> None:
    assert set(model_options(model=model)) == {'max_tokens', 'thinking', 'custom_params'}
