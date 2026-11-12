"""A model's `api_key` setting names the `/keys` entry its provider is built with."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers import Provider
from pydantic_clai2.config.api_keys import save_key
from pydantic_clai2.config.credential_store import save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import key_profiles
from pydantic_clai2.models.model_settings import ModelSettingsForm
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.plugins import ModelProvider
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SettingsStore:
    """The environment's key, a saved `/keys` entry, and a `work` profile with a key of its own."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-env')
    save_key(name='TEAM_KEY', value='sk-team')
    save_codex_credentials(account='openai@work', value=json.dumps({'token': 'sk-work'}))
    return SettingsStore(tmp_path / 'config.db')


def resolver(store: SettingsStore, providers: dict[str, ModelProvider] | None = None) -> ModelResolver:
    return ModelResolver(console=Console(file=io.StringIO()), plugins=lambda: providers or {}, store=store)


def key(model: Model | str) -> str:
    assert isinstance(model, OpenAIResponsesModel)
    return model.client.api_key


async def test_a_saved_key_replaces_the_environment_key(store: SettingsStore) -> None:
    models = resolver(store)
    assert await models.resolve('openai:gpt-5') == 'openai:gpt-5'  # Core's own, with `OPENAI_API_KEY`.
    store.save_model_settings('openai:gpt-5', {'api_key': 'TEAM_KEY'})
    assert key(await models.resolve('openai:gpt-5')) == 'sk-team'
    # The default account is the environment's, so it takes the plain model's key; a profile keeps its own.
    assert key(await models.resolve('openai@default:gpt-5')) == 'sk-team'
    assert key(await models.resolve('openai@work:gpt-5')) == 'sk-work'
    assert key(await models.resolve('openai@*:gpt-5')) == 'sk-work'  # The one signed-in account.
    store.save_model_settings('openai@work:gpt-5', {'api_key': 'TEAM_KEY'})
    assert key(await models.resolve('openai@work:gpt-5')) == 'sk-team'


async def test_a_deleted_key_fails_closed(store: SettingsStore) -> None:
    store.save_model_settings('openai:gpt-5', {'api_key': 'GONE_KEY'})
    with pytest.raises(UserError, match='Saved API key GONE_KEY is missing'):
        await resolver(store).resolve('openai:gpt-5')


async def test_models_that_take_their_key_elsewhere_fail_clearly(store: SettingsStore) -> None:
    models = resolver(store, {'single': ModelProvider(prefix='single', resolve=lambda name: TestModel())})
    store.save_chain(name='both', models=['openai:gpt-5', 'openai:gpt-4.1'])
    for name, provider in (
        ('openrouter:openai/gpt-5', 'openrouter'),
        ('openai-codex:gpt-5', 'openai-codex'),
        ('single:x', 'single'),
        ('chain:both', 'chain:both'),
    ):
        store.save_model_settings(name, {'api_key': 'TEAM_KEY'})
        with pytest.raises(UserError, match=f'{provider} takes no API key from /model settings'):
            await models.resolve(name)


def test_the_row_offers_saved_keys_and_names_the_environment_key(
    store: SettingsStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = ModelSettingsSource(store, 'openai:gpt-5')
    row = FieldMenu(source).row_for('api_key')
    assert row is not None
    assert (row.label, row.choices, row.note) == ('API Key', ('TEAM_KEY',), 'from OPENAI_API_KEY in the environment')
    assert source.problem(row, 'OTHER_KEY') == 'OTHER_KEY is not in /keys. Save the key there first.'
    assert source.apply(row, 'TEAM_KEY') == 'Saved api_key for openai:gpt-5. Applies when this model is selected.'
    assert ModelSettingsForm(api_key='TEAM_KEY').to_model_settings() is None
    row = FieldMenu(source).row_for('api_key')
    assert row is not None and row.note == ''
    work = FieldMenu(ModelSettingsSource(store, 'openai@work:gpt-5')).row_for('api_key')
    assert work is not None and work.note == 'from the openai@work sign-in'
    monkeypatch.delenv('OPENAI_API_KEY')
    unset = FieldMenu(ModelSettingsSource(store, 'openai:gpt-4.1')).row_for('api_key')
    assert unset is not None and unset.note == ''
    for name, settings_as in (('openrouter:openai/gpt-5', None), ('single:x', None), ('chain:both', 'openai:gpt-5')):
        assert FieldMenu(ModelSettingsSource(store, name, settings_as=settings_as)).row_for('api_key') is None


def test_each_provider_names_its_key_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    assert [
        key_profiles.key_variable(provider) for provider in ('openai', 'cohere', 'system-one', 'gateway/openai')
    ] == [
        'OPENAI_API_KEY',
        'CO_API_KEY',
        'SYSTEM_ONE_API_KEY',
        'PYDANTIC_AI_GATEWAY_API_KEY',
    ]
    assert key_profiles.takes_key('gateway/openai')

    def missing_extra(provider: str) -> type[Provider[object]]:
        raise ImportError(f'install the `{provider}` extra')

    monkeypatch.setattr(key_profiles, 'infer_provider_class', missing_extra)
    assert key_profiles.takes_key('mistral')
