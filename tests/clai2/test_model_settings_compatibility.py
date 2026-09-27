"""Shared model preferences survive switching between settings-schema versions."""

from pathlib import Path

import pytest
from menu_script import make_context
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from pydantic_clai2 import model_settings
from pydantic_clai2.custom_params import CustomParamsMenu
from pydantic_clai2.field_menu import FieldMenu, FieldRow
from pydantic_clai2.model_menu import ModelSettingsSource
from pydantic_clai2.model_settings import ModelSettingsForm, model_settings_from_json


def test_legacy_reader_ignores_custom_params(monkeypatch: pytest.MonkeyPatch) -> None:
    class LegacyForm(BaseModel):
        model_config = ConfigDict(extra='forbid', strict=True)
        temperature: float | None = None

    # Model the older branch from the traceback, which has no custom_params field.
    monkeypatch.setattr(model_settings, 'ModelSettingsForm', LegacyForm)
    saved: dict[str, JsonValue] = {
        'temperature': 0.5,
        'custom_params': {'chat_template_kwargs.reasoning_level': 30},
    }
    assert model_settings_from_json(saved).model_dump() == {'temperature': 0.5}
    assert saved['custom_params'] == {'chat_template_kwargs.reasoning_level': 30}


def test_reads_keep_known_overrides_and_ignore_future_fields(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    saved: dict[str, JsonValue] = {'temperature': 0.5, 'future_setting': {'nested': [1, None, False]}}
    context.store.save_model_settings('test', saved)
    assert context.model_settings('test') == {'temperature': 0.5}
    assert context.store.model_settings('test') == saved
    context.store.save_model_settings('test', {'future_setting': True})
    assert context.model_settings('test') is None


def test_known_values_and_new_edits_still_validate() -> None:
    with pytest.raises(ValidationError, match='temperature'):
        model_settings_from_json({'temperature': 'invalid', 'future_setting': True})
    with pytest.raises(ValidationError, match='Extra inputs'):
        ModelSettingsForm.model_validate({'future_setting': True})


def test_edits_and_resets_preserve_unrecognized_preferences(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    model = 'openai:gpt-4o'
    future: dict[str, JsonValue] = {'future_setting': {'nested': [1, None, False]}}
    context.store.save_model_settings(model, future)
    source = ModelSettingsSource(context.store, model)
    menu = FieldMenu(source)
    row = menu.row_for('temperature')
    assert row is not None
    assert menu.row_for('future_setting') is None
    assert source.problem(row, '0.5') is None
    assert source.apply(row, '0.5').startswith('Saved')
    assert context.store.model_settings(model) == {**future, 'temperature': 0.5}
    assert source.apply(row, 'null').startswith('Saved')
    assert context.store.model_settings(model) == future
    source.apply(row, '0.8')
    source.reset(row)
    assert context.store.model_settings(model) == future
    assert source.problem(row, 'invalid') is not None
    assert not source.apply(row, 'invalid').startswith('Saved')
    assert context.store.model_settings(model) == future


def test_new_unknown_fields_cannot_be_written_via_editor(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    source = ModelSettingsSource(context.store, 'openai:gpt-4o')
    row = FieldRow(key='future_setting', description='', default='')
    assert source.problem(row, 'true') == 'Extra inputs are not permitted'
    assert not source.apply(row, 'true').startswith('Saved')
    assert context.store.model_settings('openai:gpt-4o') == {}


def test_custom_params_edit_keeps_unknown_fields(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    model = 'openai:gpt-4o'
    context.store.save_model_settings(model, {'future_setting': True})
    menu = CustomParamsMenu(store=context.store, model=model)
    menu.save(pairs={'chat_template_kwargs.reasoning_level': 30})
    assert context.store.model_settings(model) == {
        'future_setting': True,
        'custom_params': {'chat_template_kwargs.reasoning_level': 30},
    }
    menu.save(pairs={})
    assert context.store.model_settings(model) == {'future_setting': True}


@pytest.mark.parametrize('key', ['', 'a..b', ' .x', 'x.'])
def test_invalid_custom_keys_are_validation_errors(key: str) -> None:
    with pytest.raises(ValidationError, match='nonempty dot-separated'):
        model_settings_from_json({'custom_params': {key: 1}})


def test_empty_custom_parameters_are_valid() -> None:
    assert ModelSettingsForm(custom_params=None).to_model_settings() is None
    assert ModelSettingsForm(custom_params={}).to_model_settings() is None
