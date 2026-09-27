"""Claude Fable 5.1 thinking controls and GLM thinking controls."""

from pathlib import Path

import pytest
from menu_script import make_context

from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.model_menu import ModelSettingsSource
from pydantic_clai2.model_options import model_options, validate_model_options
from pydantic_clai2.model_settings import ModelSettingsForm


def _form(**values: object) -> ModelSettingsForm:
    return ModelSettingsForm.model_validate(values)


def _settings(**values: object) -> dict[str, object]:
    return dict(_form(**values).to_model_settings() or {})


@pytest.mark.parametrize(
    'model,display,preserved,interleaved',
    [
        ('anthropic:claude-fable-5-1', True, True, False),
        ('anthropic:claude-fable-5.1', True, True, False),
        ('anthropic:claude-opus-4-6', False, True, False),
        ('anthropic:claude-sonnet-4-5', False, False, True),
        ('anthropic:claude-opus-4-1', False, False, True),
        ('anthropic:claude-opus-3', False, False, False),
    ],
)
def test_claude_control_gating(*, model: str, display: bool, preserved: bool, interleaved: bool) -> None:
    options = model_options(model=model)
    assert ('anthropic_thinking_display' in options) is display
    assert ('anthropic_preserved_thinking' in options) is preserved
    assert ('anthropic_interleaved_thinking' in options) is interleaved
    if display:
        assert options['anthropic_thinking_display'] == ('updates', 'summarized')


@pytest.mark.parametrize(
    'model,thinking,clear,effort',
    [
        ('openrouter:z-ai/glm-5.2', True, True, True),
        ('openrouter:z-ai/glm-4.10', True, True, False),
        ('vllm:glm-4.5', True, True, False),
        ('openrouter:z-ai/glm-4', False, False, False),
        ('openrouter:z-ai/chatglm-9', False, False, False),
        ('openai:gpt-6-astra', False, False, False),
    ],
)
def test_glm_control_gating(*, model: str, thinking: bool, clear: bool, effort: bool) -> None:
    options = model_options(model=model)
    assert ('glm_thinking' in options) is thinking
    assert ('glm_clear_thinking' in options) is clear
    assert ('glm_reasoning_effort' in options) is effort


def test_fable_thinking_display_and_preserved_thinking() -> None:
    settings = _settings(
        anthropic_thinking_mode='adaptive',
        anthropic_thinking_display='updates',
        anthropic_preserved_thinking='drop_block',
    )
    assert settings['anthropic_thinking'] == {
        'type': 'adaptive',
        'display': 'updates',
        'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
    }
    assert settings['extra_headers'] == {'anthropic-beta': 'thinking-display-updates-2026-08-18'}


def test_display_alone_chooses_adaptive_and_summarized_needs_no_beta() -> None:
    settings = _settings(anthropic_thinking_display='summarized')
    assert settings['anthropic_thinking'] == {'type': 'adaptive', 'display': 'summarized'}
    assert 'extra_headers' not in settings


def test_preserved_thinking_alone_chooses_adaptive() -> None:
    settings = _settings(anthropic_preserved_thinking='error')
    assert settings['anthropic_thinking'] == {
        'type': 'adaptive',
        'block_binding': {'prefix_mismatch_behavior': 'error'},
    }


def test_classic_thinking_keeps_its_budget_and_room_to_answer() -> None:
    settings = _settings(
        anthropic_thinking_mode='enabled',
        anthropic_thinking_budget=2000,
        anthropic_preserved_thinking='drop_block',
    )
    assert settings['anthropic_thinking'] == {
        'type': 'enabled',
        'budget_tokens': 2000,
        'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
    }
    assert settings['max_tokens'] == 2000 + 4096


def test_interleaved_thinking_asks_for_its_beta() -> None:
    assert _settings(anthropic_interleaved_thinking=True)['extra_headers'] == {
        'anthropic-beta': 'interleaved-thinking-2025-05-14'
    }
    assert _settings(anthropic_interleaved_thinking=False) == {}


def test_both_betas_are_sent_together() -> None:
    settings = _settings(anthropic_interleaved_thinking=True, anthropic_thinking_display='updates')
    assert settings['extra_headers'] == {
        'anthropic-beta': 'interleaved-thinking-2025-05-14,thinking-display-updates-2026-08-18'
    }


def test_disabled_thinking_ignores_display_and_binding() -> None:
    settings = _settings(
        anthropic_thinking_mode='disabled',
        anthropic_thinking_display='updates',
        anthropic_preserved_thinking='drop_block',
        anthropic_interleaved_thinking=True,
    )
    assert settings['anthropic_thinking'] == {'type': 'disabled'}
    assert settings['extra_headers'] == {'anthropic-beta': 'interleaved-thinking-2025-05-14'}


@pytest.mark.parametrize(
    'model,values,message',
    [
        (
            'anthropic:claude-fable-5-1',
            {'anthropic_thinking_mode': 'disabled', 'anthropic_thinking_display': 'updates'},
            'display or binding',
        ),
        (
            'anthropic:claude-fable-5-1',
            {'anthropic_thinking_mode': 'disabled', 'anthropic_preserved_thinking': 'error'},
            'display or binding',
        ),
        ('openrouter:z-ai/glm-5.2', {'glm_thinking': 'disabled', 'glm_reasoning_effort': 'max'}, 'reasoning effort'),
    ],
)
def test_dependent_controls_require_thinking(*, model: str, values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_model_options(model=model, form=_form(**values))


def test_glm_reasoning_effort_without_thinking_mode_is_allowed() -> None:
    validate_model_options(model='openrouter:z-ai/glm-5.2', form=_form(glm_reasoning_effort='low'))
    assert _settings(glm_reasoning_effort='low')['extra_body'] == {'reasoning_effort': 'low'}


def test_glm_body_only_carries_what_was_set() -> None:
    assert _settings(glm_clear_thinking=True)['extra_body'] == {'thinking': {'clear_thinking': True}}
    assert _settings(glm_thinking='enabled')['extra_body'] == {'thinking': {'type': 'enabled'}}
    assert _settings() == {}
    assert _settings(glm_thinking='disabled', glm_reasoning_effort='max')['extra_body'] == {
        'thinking': {'type': 'disabled'}
    }


def test_glm_full_body() -> None:
    settings = _settings(glm_thinking='enabled', glm_clear_thinking=False, glm_reasoning_effort='max')
    assert settings['extra_body'] == {
        'thinking': {'type': 'enabled', 'clear_thinking': False},
        'reasoning_effort': 'max',
    }


def test_custom_params_win_over_glm_controls() -> None:
    settings = _settings(
        glm_thinking='enabled',
        glm_clear_thinking=True,
        custom_params={'thinking': {'type': 'disabled'}, 'reasoning_effort': 'none'},
    )
    assert settings['extra_body'] == {
        'thinking': {'type': 'disabled'},
        'reasoning_effort': 'none',
    }


def test_custom_params_win_over_claude_controls() -> None:
    settings = _settings(anthropic_thinking_mode='disabled', custom_params={'thinking': {'type': 'enabled'}})
    assert settings['anthropic_thinking'] == {'type': 'disabled'}
    assert settings['extra_body'] == {'thinking': {'type': 'enabled'}}


def test_new_controls_stay_hidden_elsewhere() -> None:
    options = model_options(model='openai:gpt-6-astra')
    assert not {key for key in options if key.startswith(('glm_', 'anthropic_'))}
    assert not {key for key in model_options(model='google:gemini-3-pro') if key.startswith(('glm_', 'anthropic_'))}


def test_claude_rows_appear_in_the_editor(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    rows = {row.key: row for row in ModelSettingsSource(context.store, 'anthropic:claude-fable-5-1').rows()}
    assert rows['anthropic_thinking_display'].label == 'Thinking Display'
    assert rows['anthropic_preserved_thinking'].label == 'Preserved Thinking'
    assert rows['anthropic_thinking_display'].choices == ('updates', 'summarized')
    assert rows['anthropic_preserved_thinking'].choices == ('error', 'drop_block')
    assert not rows['anthropic_preserved_thinking'].allow_custom


def test_resetting_thinking_mode_clears_its_dependents(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    model = 'anthropic:claude-fable-5-1'
    context.store.save_model_settings(
        model,
        {
            'anthropic_thinking_mode': 'adaptive',
            'anthropic_thinking_display': 'updates',
            'anthropic_thinking_budget': 2000,
        },
    )
    source = ModelSettingsSource(context.store, model)
    row = FieldMenu(source, searchable=False).row_for('anthropic_thinking_mode')
    assert row is not None
    source.reset(row)
    assert context.store.model_settings(model) == {}
