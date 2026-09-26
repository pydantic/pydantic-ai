"""Tests for ACP model selection through session config options."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Literal

import acp
import pytest
from acp import schema
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.acp import InMemorySessionStore, PydanticAIACPAgent
from pydantic_ai_harness.experimental.acp._adapter import _all_known_model_names
from tests.experimental.acp._acp_clients import RecordingClient  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


def test_all_known_model_names_are_strings() -> None:
    # Guards `models='all'` advertising only string model ids from Pydantic AI's public
    # enumeration API.
    names = _all_known_model_names()
    assert len(names) > 100
    assert all(isinstance(name, str) for name in names)


def _adapter(
    *, models: Sequence[KnownModelName | str] | Literal['all'] | None = None, store: InMemorySessionStore | None = None
) -> PydanticAIACPAgent[None, str]:
    return PydanticAIACPAgent(Agent(TestModel(custom_output_text='hi')), models=models, session_store=store)


def _text_from(client: RecordingClient) -> str:
    return ''.join(
        str(getattr(getattr(update, 'content', None), 'text', ''))
        for update in client.updates
        if getattr(update, 'session_update', '') == 'agent_message_chunk'
    )


def _model_option(
    response: schema.NewSessionResponse | schema.LoadSessionResponse | schema.SetSessionConfigOptionResponse,
) -> schema.SessionConfigOptionSelect:
    options = response.config_options
    assert options is not None
    [option] = options
    assert isinstance(option, schema.SessionConfigOptionSelect)
    return option


def _select_options(option: schema.SessionConfigOptionSelect) -> list[schema.SessionConfigSelectOption]:
    options: list[schema.SessionConfigSelectOption] = []
    for item in option.options:
        assert isinstance(item, schema.SessionConfigSelectOption)
        options.append(item)
    return options


async def _started(adapter: PydanticAIACPAgent[None, str]) -> str:
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    return (await adapter.new_session(cwd='/ws')).session_id


async def test_models_all_advertises_every_known_model() -> None:
    adapter = _adapter(models='all')
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    response = await adapter.new_session(cwd='/ws')
    option = _model_option(response)
    ids = [model.value for model in _select_options(option)]
    assert len(ids) > 100  # the whole known set, not a curated handful
    assert 'openai:gpt-4o' in ids
    assert option.current_value == ids[0]  # first known model is the default


async def test_new_session_advertises_configured_models() -> None:
    adapter = _adapter(models=['openai:gpt-4o', 'test'])
    adapter.on_connect(RecordingClient())
    await adapter.initialize(protocol_version=1)
    response = await adapter.new_session(cwd='/ws')
    option = _model_option(response)
    assert option.id == 'model'
    assert option.name == 'Model'
    assert [model.value for model in _select_options(option)] == ['openai:gpt-4o', 'test']
    assert option.current_value == 'openai:gpt-4o'  # the first configured model is the default


async def test_new_session_without_models_advertises_none() -> None:
    adapter = _adapter()
    assert (await adapter.new_session(cwd='/ws')).config_options is None


async def test_set_model_config_updates_the_session_and_persists() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(models=['openai:gpt-4o', 'test'], store=store)
    session_id = await _started(adapter)
    response = await adapter.set_config_option(config_id='model', value='test', session_id=session_id)
    assert response is not None
    option = _model_option(response)
    assert option.current_value == 'test'
    assert adapter._sessions[session_id].model == 'test'  # pyright: ignore[reportPrivateUsage]
    stored = await store.load(session_id)
    assert stored is not None and stored.model == 'test'


async def test_selected_model_applies_to_a_run() -> None:
    # The agent's own model is canned to answer 'hi'; the 'test' override resolves to a default
    # TestModel whose canned answer differs, so the override observably reached the run.
    client = RecordingClient()
    adapter = _adapter(models=['test'])
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session_id = (await adapter.new_session(cwd='/ws')).session_id
    response = await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session_id)
    assert response.stop_reason == 'end_turn'
    assert _text_from(client) == 'success (no tool calls)'  # TestModel's default, not the agent model's 'hi'


async def test_model_resolver_applies_selected_model_to_a_run() -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'resolved model'

    seen: list[str] = []
    resolved_model = FunctionModel(stream_function=stream)

    def resolve(model_id: str) -> FunctionModel:
        seen.append(model_id)
        return resolved_model

    client = RecordingClient()
    adapter: PydanticAIACPAgent[None, str] = PydanticAIACPAgent(
        Agent(TestModel(custom_output_text='agent model')),
        models=['test', 'host:gpt-custom'],
        model_resolver=resolve,
    )
    adapter.on_connect(client)
    await adapter.initialize(protocol_version=1)
    session_id = (await adapter.new_session(cwd='/ws')).session_id
    await adapter.set_config_option(config_id='model', value='host:gpt-custom', session_id=session_id)

    response = await adapter.prompt(prompt=[acp.text_block('hi')], session_id=session_id)

    assert response.stop_reason == 'end_turn'
    assert seen == ['host:gpt-custom']
    assert _text_from(client) == 'resolved model'


async def test_selected_model_survives_reload() -> None:
    store = InMemorySessionStore()
    adapter = _adapter(models=['openai:gpt-4o', 'test'], store=store)
    session_id = await _started(adapter)
    await adapter.set_config_option(config_id='model', value='test', session_id=session_id)
    response = await adapter.load_session(cwd='/ws', session_id=session_id)
    assert adapter._sessions[session_id].model == 'test'  # pyright: ignore[reportPrivateUsage]
    assert response is not None
    option = _model_option(response)
    assert option.current_value == 'test'


async def test_session_model_option_returns_available_and_current_models() -> None:
    adapter = _adapter(models=['openai:gpt-4o', 'test'])
    session_id = await _started(adapter)
    await adapter.set_config_option(config_id='model', value='test', session_id=session_id)

    state = adapter.session_model_option(session_id)

    assert state is not None
    assert [model.value for model in _select_options(state)] == ['openai:gpt-4o', 'test']
    assert state.current_value == 'test'
    assert adapter.session_model_option('no-such-session') is None


async def test_session_model_option_without_models_returns_none() -> None:
    adapter = _adapter()
    session_id = await _started(adapter)

    assert adapter.session_model_option(session_id) is None


async def test_set_unknown_model_is_rejected() -> None:
    adapter = _adapter(models=['test'])
    session_id = await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_config_option(config_id='model', value='not-a-model', session_id=session_id)


async def test_set_model_config_for_unknown_session_is_rejected() -> None:
    adapter = _adapter(models=['test'])
    await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_config_option(config_id='model', value='test', session_id='no-such-session')


async def test_unknown_config_option_is_rejected() -> None:
    adapter = _adapter(models=['test'])
    session_id = await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_config_option(config_id='theme', value='test', session_id=session_id)


async def test_set_model_config_without_configured_models_is_rejected() -> None:
    adapter = _adapter()
    session_id = await _started(adapter)
    with pytest.raises(acp.RequestError):
        await adapter.set_config_option(config_id='model', value='test', session_id=session_id)
