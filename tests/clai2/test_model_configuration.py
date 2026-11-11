"""Model-aware choices, native request settings, and custom parameter editing."""

import json
from pathlib import Path

import httpx2 as httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from termflow.tui import MenuItem
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_clai2.models.custom_params import expand_params, parse_pair
from pydantic_clai2.models.model_options import model_options, validate_model_options
from pydantic_clai2.models.model_settings import ModelSettingsForm, model_defaults, model_settings_from_json
from pydantic_clai2.ui.menus.custom_params import (
    _problem,  # pyright: ignore[reportPrivateUsage]
    edit_custom_params,
)
from pydantic_clai2.ui.menus.dialogs import run_settings_menu
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource, model_settings_command, model_settings_menu
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs, Type
from tests.clai2.menu_script import Script, make_context, pick


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
    assert 'max' not in model_options(model='anthropic:claude-opus-4-5')['anthropic_effort']
    assert 'max' in model_options(model='anthropic:claude-sonnet-4-6')['anthropic_effort']


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


async def test_custom_params_add_edit_rename_remove_cancel(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    model = 'openai:gpt-5.6'
    context.store.save_model_settings(model, {'temperature': 1.0})
    dialogs = ScriptedDialogs(
        [
            Pick(value=''),
            Type(text='a = true'),
            Pick(value='a'),
            Back(),
            Pick(value='a'),
            Type(text='b = 3'),
            Pick(value=''),
            Type(text=''),
            Pick(value='b'),
            Type(text=''),
            Back(),
        ]
    )
    assert (
        await edit_custom_params(dialogs, store=context.store, model=model) == [f'Saved custom params for {model}.'] * 3
    )
    assert [asked.initial for asked in dialogs.asked if asked.kind == 'text'] == [
        '',
        'a = true',
        'a = true',
        '',
        'b = 3',
    ]
    assert [option.label for option in dialogs.asked[4].options] == ['a = true', '+ Add param...']
    assert context.store.model_settings(model) == {'temperature': 1.0}
    assert _problem('a = 2') is None and _problem('a') == 'Expected format: key = value'


async def test_model_settings_menu_opens_custom_params_and_resets_them(tmp_path: Path) -> None:
    context, _ = make_context(tmp_path)
    model = 'openai:gpt-5.6'
    context.store.save_model_settings(model, {'temperature': 1.0, 'custom_params': {'nested.a': True}})
    menu = FieldMenu(ModelSettingsSource(context.store, model))
    reset = menu.reset_marker(None, MenuItem('custom_params', value='custom_params'))
    script = Script(
        lists=[pick('custom_params'), reset, MenuResult(cancelled=True)], choices=[MenuResult(cancelled=True)], texts=[]
    )
    assert await run_settings_menu(model_settings_menu(context.store, model), script.runners) == [
        f'Reset custom_params for {model}.'
    ]
    assert context.model_settings(model) == {**model_defaults(model=model), 'temperature': 1.0}


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
    monkeypatch.setattr('pydantic_clai2.ui.menus.field_menu.menu_key', lambda: next(keys))
    widget = menu.build()
    result = widget.run()
    assert result.item is not None and result.item.value == 'openai_reasoning_effort'
    assert widget.run() == menu.reset_marker(None, result.item)


@pytest.mark.parametrize(('name', 'mode'), [('claude-sonnet-4-5', 'enabled'), ('claude-fable-5-1', 'adaptive')])
async def test_claude_native_thinking_reaches_request(name: str, mode: str) -> None:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    bodies: list[dict[str, JsonValue]] = []
    adapter = TypeAdapter(dict[str, JsonValue])

    def respond(request: httpx.Request) -> httpx.Response:
        body = adapter.validate_json(request.content)
        bodies.append(body)
        # The default `max_tokens` is the model's maximum output, above the SDK's non-streaming limit, so the
        # request streams.
        assert body['stream'] is True
        message: dict[str, JsonValue] = {
            'id': 'msg_test',
            'type': 'message',
            'role': 'assistant',
            'model': name,
            'content': [],
            'stop_reason': None,
            'usage': {'input_tokens': 1, 'output_tokens': 0},
        }
        events: list[dict[str, JsonValue]] = [
            {'type': 'message_start', 'message': message},
            {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'done'}},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 1}},
            {'type': 'message_stop'},
        ]
        return httpx.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            content=''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events),
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
        assert bodies[0]['max_tokens'] == 64000
