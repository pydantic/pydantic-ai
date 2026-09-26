"""Model-aware choices, native request settings, and custom parameter editing."""

from pathlib import Path

import httpx2 as httpx
import pytest
from menu_script import Script, make_context, pick, typed
from pydantic import JsonValue, TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.textinput import TextInputResult  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2.custom_params import CustomParamsMenu, DeleteParam, expand_params, parse_pair
from pydantic_clai2.field_menu import FieldMenu
from pydantic_clai2.model_menu import ModelSettingsSource, model_settings_command, run_model_settings
from pydantic_clai2.model_options import model_options, validate_model_options
from pydantic_clai2.model_settings import ModelSettingsForm, model_defaults, model_settings_from_json


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize(
    ('model', 'expected', 'absent'),
    [
        ('openai:gpt-5.6', 'openai_reasoning_mode', 'anthropic_thinking_mode'),
        ('openai:gpt-5.4', 'openai_reasoning_context', 'openai_reasoning_mode'),
        ('openai-chat:gpt-5.6', 'openai_reasoning_effort', 'openai_reasoning_context'),
        ('anthropic:claude-fable-5-1', 'anthropic_effort', 'temperature'),
        ('anthropic:claude-sonnet-4-5', 'anthropic_thinking_budget', 'anthropic_effort'),
        ('google:gemini-3-pro', 'custom_params', 'openai_reasoning_effort'),
        ('openai:gpt-4o', 'temperature', 'openai_reasoning_effort'),
    ],
)
def test_model_fields(model: str, expected: str, absent: str) -> None:
    options = model_options(model=model)
    assert expected in options and absent not in options
    assert not any('retry' in key or 'retries' in key for key in options)


@pytest.mark.parametrize(
    ('name', 'included', 'excluded'),
    [
        ('gpt-5', 'minimal', 'max'),
        ('gpt-5.1', 'none', 'xhigh'),
        ('gpt-5.1-codex-max', 'xhigh', 'max'),
        ('gpt-5.2', 'xhigh', 'max'),
        ('gpt-5.6', 'max', 'minimal'),
        ('gpt-5.2-chat-latest', 'medium', 'high'),
        ('o3', 'high', 'max'),
        ('o3', 'high', 'minimal'),
        ('gpt-5.1', 'none', 'minimal'),
    ],
)
def test_efforts(name: str, included: str, excluded: str) -> None:
    efforts = model_options(model=f'openai:{name}')['openai_reasoning_effort']
    assert included in efforts and excluded not in efforts


def test_anthropic_modes_and_effort() -> None:
    classic = model_options(model='anthropic:claude-sonnet-4-5')
    assert classic['anthropic_thinking_mode'] == ('enabled', 'disabled')
    adaptive = model_options(model='anthropic:claude-opus-4-6')
    assert adaptive['anthropic_thinking_mode'] == ('adaptive', 'disabled')
    assert 'xhigh' not in adaptive['anthropic_effort']
    assert 'xhigh' in model_options(model='anthropic:claude-opus-4-7')['anthropic_effort']
    assert 'anthropic_thinking_budget' in model_options(model='anthropic:unknown')
    for name in ('claude-opus-4-5', 'claude-sonnet-4-6'):
        assert 'max' not in model_options(model=f'anthropic:{name}')['anthropic_effort']


@pytest.mark.parametrize(
    ('model', 'values', 'error'),
    [
        ('openai:gpt-5', {'openai_reasoning_effort': 'max'}, 'choose'),
        ('openai-chat:gpt-5.6', {'openai_reasoning_context': 'all_turns'}, 'not supported'),
        ('anthropic:claude-sonnet-4-5', {'anthropic_thinking_budget': 2048}, 'before setting'),
        ('anthropic:claude-sonnet-4-5', {'anthropic_thinking_mode': 'enabled', 'max_tokens': 10000}, 'less than'),
        ('anthropic:claude-sonnet-4-5', {'anthropic_thinking_mode': 'enabled', 'temperature': 0.5}, 'temperature'),
        ('anthropic:claude-sonnet-4-5', {'anthropic_thinking_mode': 'enabled', 'top_p': 0.8}, 'temperature'),
        (
            'anthropic:claude-opus-5',
            {'anthropic_thinking_mode': 'disabled', 'anthropic_effort': 'max'},
            'requires thinking',
        ),
    ],
)
def test_invalid_combinations(model: str, values: dict[str, JsonValue], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        validate_model_options(model=model, form=model_settings_from_json(values))


def test_native_settings_conversion() -> None:
    openai: dict[str, JsonValue] = {
        'openai_reasoning_effort': 'max',
        'openai_reasoning_context': 'all_turns',
        'openai_reasoning_mode': 'pro',
        'openai_reasoning_summary': 'detailed',
        'openai_text_verbosity': 'low',
    }
    assert model_settings_from_json(openai).to_model_settings() == openai
    for mode in ('adaptive', 'disabled'):
        form = model_settings_from_json({'anthropic_thinking_mode': mode, 'anthropic_effort': 'high'})
        assert form.to_model_settings() == {'anthropic_thinking': {'type': mode}, 'anthropic_effort': 'high'}
    for budget in (None, 2048):
        form = ModelSettingsForm(anthropic_thinking_mode='enabled', anthropic_thinking_budget=budget)
        assert form.to_model_settings() == {
            'anthropic_thinking': {'type': 'enabled', 'budget_tokens': budget or 10000},
            'max_tokens': (budget or 10000) + 4096,
        }
        validate_model_options(model='anthropic:claude-sonnet-4-5', form=form)


def test_dotted_pairs_and_types() -> None:
    for raw, value in [
        ('true', True),
        ('2', 2),
        ('0.5', 0.5),
        ('null', None),
        ('medium', 'medium'),
        ('[1]', [1]),
        ('"2"', '2'),
    ]:
        assert parse_pair(text=f'a.b = {raw}') == ('a.b', value)
    pairs: dict[str, JsonValue] = {'a': {'old': 1}, 'a.b': 2, 'c': 3, 'c.d': False}
    assert expand_params(pairs=pairs) == {'a': {'old': 1, 'b': 2}, 'c': {'d': False}}
    assert pairs['a'] == {'old': 1}
    assert model_settings_from_json({'custom_params': pairs}).to_model_settings() == {
        'extra_body': expand_params(pairs=pairs)
    }
    for invalid in ('', 'a', '= x', 'a =', 'a..b = 2'):
        with pytest.raises(ValueError):
            parse_pair(text=invalid)


async def test_direct_command_persists_without_switching(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    context.store.add_model(name='openai:gpt-5.6')
    selected = context.settings.model
    script = Script(
        lists=[pick('openai_reasoning_effort'), MenuResult(cancelled=True)], choices=[pick('max')], texts=[]
    )
    assert 'Saved openai_reasoning_effort' in await model_settings_command(
        context, ['openai:gpt-5.6'], runners=script.runners
    )
    assert context.settings.model == selected
    assert context.model_settings('openai:gpt-5.6') == {
        **model_defaults(model='openai:gpt-5.6'),
        'openai_reasoning_effort': 'max',
    }
    assert (
        await model_settings_command(
            context, [], runners=Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[]).runners
        )
        == 'No changes.'
    )
    for args in (['a', 'b'], ['missing']):
        with pytest.raises(ValueError):
            await model_settings_command(context, args)
    context.settings = context.settings.model_copy(update={'model': None})
    assert (
        await model_settings_command(
            context, [], runners=Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[]).runners
        )
        == 'No changes.'
    )


def test_editor_validation_and_reset(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    source = ModelSettingsSource(context.store, 'openai:gpt-5')
    row = next(row for row in source.rows() if row.key == 'openai_reasoning_effort')
    assert 'choose' in (source.problem(row, 'max') or '')
    assert 'choose' in source.apply(row, 'max')
    assert context.store.model_settings(source.model) == {}
    context.store.save_model_settings(source.model, {'anthropic_effort': 'high'})
    stale = next(row for row in source.rows() if row.key == 'anthropic_effort')
    source.reset(stale)
    assert context.store.model_settings(source.model) == {}
    claude = ModelSettingsSource(context.store, 'anthropic:claude-sonnet-4-5')
    context.store.save_model_settings(
        claude.model, {'anthropic_thinking_mode': 'enabled', 'anthropic_thinking_budget': 2048}
    )
    claude.reset(next(row for row in claude.rows() if row.key == 'anthropic_thinking_mode'))
    assert context.store.model_settings(claude.model) == {}


def test_custom_menu_add_edit_rename_delete_cancel(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    model = 'openai:gpt-5.6'
    context.store.save_model_settings(model, {'temperature': 1.0})
    custom = CustomParamsMenu(store=context.store, model=model)
    assert custom.delete_marker(None, MenuItem('Add', value=False)) is None
    marker = custom.delete_marker(None, MenuItem('a', value='a'))
    assert marker is not None and marker.item is not None and marker.item.value == DeleteParam(key='a')
    script = Script(
        lists=[
            pick(False),
            pick('a'),
            pick('a'),
            pick('b'),
            marker,
            pick(DeleteParam(key='b')),
            MenuResult(cancelled=True),
        ],
        choices=[],
        texts=[typed('a = true'), TextInputResult(cancelled=True), typed('b = 3'), typed('b = 4')],
    )
    assert len(custom.run(runners=script.runners)) == 5
    assert context.store.model_settings(model) == {'temperature': 1.0}
    custom.save(pairs={'nested.a': True})
    menu = FieldMenu(ModelSettingsSource(context.store, model))
    row = menu.row_for('custom_params')
    assert row is not None
    reset = menu.reset_marker(None, MenuItem('custom_params', value='custom_params'))
    script = Script(
        lists=[pick('custom_params'), MenuResult(cancelled=True), reset, MenuResult(cancelled=True)],
        choices=[],
        texts=[],
    )
    assert run_model_settings(store=context.store, model=model, runners=script.runners) == [
        f'Reset custom_params for {model}.'
    ]
    assert context.model_settings(model) == {**model_defaults(model=model), 'temperature': 1.0}


def test_custom_editor_validates_before_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    custom = CustomParamsMenu(store=context.store, model='test')
    keys = iter(['enter', 'a', ' ', '=', ' ', '2', 'enter'])
    monkeypatch.setattr('pydantic_clai2.custom_params.menu_key', lambda: next(keys))
    assert custom.editor(key=None).run().value == 'a = 2'


async def test_saved_controls_and_custom_overrides_reach_openai_body(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    name = 'openai:gpt-5.6'
    context.store.save_model_settings(
        name,
        {
            'openai_reasoning_effort': 'high',
            'openai_reasoning_context': 'all_turns',
            'openai_reasoning_mode': 'standard',
            'custom_params': {'reasoning.effort': 'max', 'custom.flag': True},
        },
    )
    bodies: list[dict[str, JsonValue]] = []
    adapter = TypeAdapter(dict[str, JsonValue])

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(adapter.validate_json(request.content))
        return httpx.Response(
            200,
            json={
                'id': 'resp_test',
                'object': 'response',
                'created_at': 0,
                'status': 'completed',
                'model': 'gpt-5.6',
                'output': [
                    {
                        'type': 'message',
                        'id': 'msg_test',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'done', 'annotations': []}],
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = OpenAIResponsesModel('gpt-5.6', provider=OpenAIProvider(api_key='test', http_client=client))
        with override_allow_model_requests(True):
            saved = context.store.model_settings(name)
            custom = saved.pop('custom_params')
            context.store.save_model_settings(name, saved)
            await Agent(model).run('hello', model_settings=context.model_settings(name))
            context.store.save_model_settings(name, {**saved, 'custom_params': custom})
            result = await Agent(model).run('hello', model_settings=context.model_settings(name))
    assert result.output == 'done'
    assert bodies[0]['reasoning'] == {
        'effort': 'high',
        'context': 'all_turns',
        'mode': 'standard',
        'summary': 'detailed',
    }
    assert bodies[1]['reasoning'] == {'effort': 'max'}
    assert bodies[1]['custom'] == {'flag': True}


def test_classic_budget_keeps_explicit_output_limit() -> None:
    form = ModelSettingsForm(anthropic_thinking_mode='enabled', anthropic_thinking_budget=2048, max_tokens=8192)
    validate_model_options(model='anthropic:claude-sonnet-4-5', form=form)
    assert form.to_model_settings() == {
        'max_tokens': 8192,
        'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 2048},
    }


def test_settings_search_does_not_reset_on_lowercase_r(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    source = ModelSettingsSource(context.store, 'openai:gpt-5.6')
    menu = FieldMenu(source)
    keys = iter([*'reasoning', 'enter', 'R'])
    monkeypatch.setattr('pydantic_clai2.field_menu.menu_key', lambda: next(keys))
    widget = menu.build()
    result = widget.run()
    assert result.item is not None and result.item.value == 'openai_reasoning_effort'
    assert widget.run() == menu.reset_marker(None, result.item)


@pytest.mark.parametrize(('name', 'mode'), [('claude-sonnet-4-5', 'enabled'), ('claude-fable-5-1', 'adaptive')])
async def test_claude_native_thinking_reaches_request(name: str, mode: str) -> None:
    bodies: list[dict[str, JsonValue]] = []
    adapter = TypeAdapter(dict[str, JsonValue])

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(adapter.validate_json(request.content))
        return httpx.Response(
            200,
            json={
                'id': 'msg_test',
                'type': 'message',
                'role': 'assistant',
                'model': name,
                'content': [{'type': 'text', 'text': 'done'}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 1, 'output_tokens': 1},
            },
        )

    form = model_settings_from_json({'anthropic_thinking_mode': mode})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        model = AnthropicModel(name, provider=AnthropicProvider(api_key='test', http_client=client))
        with override_allow_model_requests(True):
            assert (await Agent(model).run('hello', model_settings=form.to_model_settings())).output == 'done'
    assert bodies[0]['thinking'] == (
        {'type': 'adaptive'} if mode == 'adaptive' else {'type': 'enabled', 'budget_tokens': 10000}
    )
    if mode == 'enabled':
        assert bodies[0]['max_tokens'] == 14096
