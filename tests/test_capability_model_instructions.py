"""The `ModelInstructions` capability.

Offline tests with `FunctionModel`: which text a model gets depends only on its name and the run context, not on the
provider, and scripted models can fail on cue to make `ModelFallback` move on. The route test reads the names real
model classes report, which needs no request.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Literal

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    InstructionDeltaPart,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.capabilities import ModelFallback, ModelInstructions
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from .conftest import try_import

with try_import() as openai_available:
    import openai  # noqa: F401  # pyright: ignore[reportUnusedImport]


def unwrap(content: str) -> str:
    """The block's own text: an appending block is sent as a `<pyai:context>` element."""
    match = re.fullmatch(r'<pyai:context id="[^"]+">\n(.*)\n</pyai:context>(?:\n.*)?', content, re.DOTALL)
    return match.group(1) if match else content


class Recorder:
    """A model function that records the instruction parts of every request it gets."""

    def __init__(self) -> None:
        self.requests: list[list[InstructionPart]] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.requests.append(list(info.model_request_parameters.instruction_parts or []))
        return ModelResponse(parts=[TextPart('ok')])

    @property
    def texts(self) -> list[list[str]]:
        return [[unwrap(part.content) for part in parts] for parts in self.requests]


def model_text(parts: Sequence[InstructionPart]) -> str | None:
    """The content of the `ModelInstructions` block, if the request had one."""
    return next((unwrap(part.content) for part in parts if str(part.id) == 'capability:model_instructions:text'), None)


def instructions_for(model_name: str, instructions: dict[str, str]) -> str | None:
    recorder = Recorder()
    agent = Agent(FunctionModel(recorder, model_name=model_name), capabilities=[ModelInstructions(instructions)])
    request = agent.run_sync('Hi').all_messages()[0]
    assert isinstance(request, ModelRequest)
    return request.instructions


@pytest.mark.parametrize(
    ('model_name', 'expected'),
    [
        ('gpt-6-luna', 'GPT-6'),
        ('openai/gpt-6-luna', 'GPT-6'),
        ('claude-opus-5-5', 'Claude'),
        ('anthropic/claude-opus-5-5', 'Claude'),
        ('gemini-3-pro', 'Anything else'),
    ],
)
def test_patterns_match_the_model_name(model_name: str, expected: str) -> None:
    """A pattern without `:` matches the model name, or the part after its last `/`; `'*'` matches the rest."""
    instructions = {'gpt-6*': 'GPT-6', 'claude-*': 'Claude', '*': 'Anything else'}
    assert instructions_for(model_name, instructions) == expected


def test_first_match_wins() -> None:
    assert instructions_for('gpt-6-luna', {'gpt-6-l*': 'Luna', 'gpt-6*': 'GPT-6'}) == 'Luna'
    assert instructions_for('gpt-6-luna', {'gpt-6*': 'GPT-6', 'gpt-6-l*': 'Luna'}) == 'GPT-6'


def test_an_unmatched_model_gets_no_block() -> None:
    recorder = Recorder()
    agent = Agent(
        FunctionModel(recorder, model_name='gemini-3-pro'),
        instructions='Be brief.',
        capabilities=[ModelInstructions({'gpt-6*': 'GPT-6'})],
    )
    result = agent.run_sync('Hi')
    assert recorder.texts == [['Be brief.']]
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == 'Be brief.'


def test_a_pattern_with_a_colon_matches_the_model_id() -> None:
    """`FunctionModel`'s provider is `function`, so its model ID is `function:<name>`."""
    instructions = {'other:gpt-6*': 'Other route', 'function:gpt-6*': 'Function route'}
    assert instructions_for('gpt-6-luna', instructions) == 'Function route'
    assert instructions_for('gpt-6-luna', {'other:gpt-6*': 'Other route'}) is None


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize(
    ('model_id', 'expected'),
    [
        ('openai:gpt-6-luna', 'OpenAI'),
        ('gateway/openai:gpt-6-luna', 'OpenAI'),
        ('azure:gpt-6-luna', 'Azure'),
        ('openrouter:openai/gpt-6-luna', 'GPT-6'),
    ],
)
async def test_routes(model_id: str, expected: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """One name matches a model on every route; a provider-qualified pattern tells them apart.

    A unit test of the matching: these are the names real model classes report, and no request is needed to read them.
    """
    for name, value in {
        'OPENAI_API_KEY': 'test',
        'AZURE_OPENAI_API_KEY': 'test',
        'AZURE_OPENAI_ENDPOINT': 'https://example.openai.azure.com',
        'OPENAI_API_VERSION': '2025-01-01',
        'OPENROUTER_API_KEY': 'test',
        'PYDANTIC_AI_GATEWAY_API_KEY': 'pylf_v1_us_test',
    }.items():
        monkeypatch.setenv(name, value)
    capability = ModelInstructions[None]({'openai:gpt-6*': 'OpenAI', 'azure:*': 'Azure', 'gpt-6*': 'GPT-6'})
    ctx = RunContext(deps=None, model=infer_model(model_id), usage=RunUsage())
    part = await capability._instruction(ctx)  # pyright: ignore[reportPrivateUsage]
    assert part.content == expected


def test_mapping_block_is_static_and_stable() -> None:
    """The mapping's text depends only on the model, so it sits in the cached prefix and doesn't change."""
    recorder = Recorder()
    agent = Agent(instructions='Be brief.', capabilities=[ModelInstructions({'gpt-6*': 'GPT-6'})])

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    calls = iter([True, False])

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        recorder(messages, info)
        return ModelResponse(parts=[ToolCallPart('look_up')] if next(calls) else [TextPart('done')])

    first = agent.run_sync('Hi', model=FunctionModel(respond, model_name='gpt-6-luna'))
    agent.run_sync(
        'Again', message_history=first.all_messages(), model=FunctionModel(recorder, model_name='gpt-6-luna')
    )

    assert [
        [(str(part.id), unwrap(part.content), part.dynamic, part.on_change) for part in parts]
        for parts in recorder.requests
    ] == snapshot(
        [
            [
                ('agent', 'Be brief.', False, 'rewrite'),
                ('capability:model_instructions:text', 'GPT-6', False, 'rewrite'),
            ],
            [
                ('agent', 'Be brief.', False, 'rewrite'),
                ('capability:model_instructions:text', 'GPT-6', False, 'rewrite'),
            ],
            [
                ('agent', 'Be brief.', False, 'rewrite'),
                ('capability:model_instructions:text', 'GPT-6', False, 'rewrite'),
            ],
        ]
    )


def test_a_callable() -> None:
    def per_model(ctx: RunContext[str]) -> str | None:
        assert isinstance(ctx.model, FunctionModel)
        if ctx.model.profile.get('delegates_unprompted', False):
            return f"Don't spawn sub-agents unless {ctx.deps} asks."
        return None

    recorder = Recorder()
    agent = Agent(FunctionModel(recorder), deps_type=str, capabilities=[ModelInstructions(per_model)])
    agent.run_sync('Hi', deps='the user', model=FunctionModel(recorder, profile={'delegates_unprompted': True}))
    agent.run_sync('Hi', deps='the user')

    assert [
        [(str(part.id), unwrap(part.content), part.dynamic) for part in parts] for parts in recorder.requests
    ] == snapshot(
        [
            [('capability:model_instructions:text', "Don't spawn sub-agents unless the user asks.", True)],
            [],
        ]
    )


async def test_an_async_callable() -> None:
    async def per_model(ctx: RunContext[None]) -> str:
        return f'You are {ctx.model.model_name}.'

    recorder = Recorder()
    agent = Agent(
        FunctionModel(recorder, model_name='gpt-6-luna'),
        deps_type=type(None),
        capabilities=[ModelInstructions(per_model)],
    )
    await agent.run('Hi')
    assert recorder.texts == [['You are gpt-6-luna.']]


def test_added_after_the_agent_instructions_in_capability_order() -> None:
    recorder = Recorder()

    agent = Agent(
        FunctionModel(recorder, model_name='gpt-6-luna'),
        instructions='Be brief.',
        capabilities=[
            ModelInstructions({'gpt-6*': 'First.'}, id='first'),
            ModelInstructions({'*': 'Second.'}, id='second'),
        ],
    )
    agent.run_sync('Hi')
    assert [[(str(part.id), unwrap(part.content)) for part in parts] for parts in recorder.requests] == snapshot(
        [[('agent', 'Be brief.'), ('capability:first:text', 'First.'), ('capability:second:text', 'Second.')]]
    )


@pytest.mark.parametrize(
    ('kind', 'on_change'),
    [(None, 'rewrite'), ('tuning', 'rewrite'), ('rule', 'append')],
)
def test_model_fallback_resolves_each_model(kind: Literal['tuning', 'rule'] | None, on_change: str) -> None:
    """The model `ModelFallback` moves a request on to gets its own text; `'tuning'` is the default, and rewrites."""
    tried: list[str | None] = []

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tried.append(model_text(info.model_request_parameters.instruction_parts or []))
        raise ModelAPIError('gpt-6-luna', 'Overloaded')

    instructions = {'gpt-6*': 'GPT-6', 'claude-*': 'Claude'}
    recorder = Recorder()
    agent = Agent(
        FunctionModel(primary, model_name='gpt-6-luna'),
        capabilities=[
            ModelInstructions(instructions) if kind is None else ModelInstructions(instructions, kind=kind),
            ModelFallback(FunctionModel(recorder, model_name='claude-opus-5-5')),
        ],
    )
    result = agent.run_sync('Hi')

    assert tried == ['GPT-6']
    assert [[(unwrap(part.content), part.on_change) for part in parts] for parts in recorder.requests] == [
        [('Claude', on_change)]
    ]
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == 'Claude'


def switch_models(kind: Literal['tuning', 'rule'], *, to: str) -> tuple[str | None, list[tuple[str, str | None]]]:
    """Run on a GPT model, continue on `to`, and return the instructions the second run sent and the changes it appended."""
    agent = Agent(capabilities=[ModelInstructions({'gpt-6*': 'GPT-6', 'claude-*': 'Claude'}, kind=kind)])
    first = agent.run_sync('Hi', model=FunctionModel(Recorder(), model_name='gpt-6-luna'))
    second = agent.run_sync(
        'Again', message_history=first.all_messages(), model=FunctionModel(Recorder(), model_name=to)
    )
    request = second.new_messages()[0]
    assert isinstance(request, ModelRequest)
    changes = [(str(part.id), part.content) for part in request.parts if isinstance(part, InstructionDeltaPart)]
    return request.instructions, changes


def test_a_rule_appends_the_change() -> None:
    """A rule keeps the earlier text in the prefix and appends the change, or its withdrawal for an unmatched model."""
    assert switch_models('rule', to='claude-opus-5-5') == ('GPT-6', [('capability:model_instructions:text', 'Claude')])
    assert switch_models('rule', to='gemini-3-pro') == ('GPT-6', [('capability:model_instructions:text', None)])


@pytest.mark.moves_cache_prefix(reason='tuning rewrites the instructions when the conversation moves to another model')
def test_tuning_rewrites_the_block() -> None:
    """Tuning sends the new model what a fresh run on it would, with nothing appended, or no block when unmatched."""
    assert switch_models('tuning', to='claude-opus-5-5') == ('Claude', [])
    assert switch_models('tuning', to='gemini-3-pro') == (None, [])


def test_without_an_id_changes_rewrite() -> None:
    """An unaddressable block can't append its changes, so even a rule rewrites them."""
    recorder = Recorder()
    agent = Agent(
        FunctionModel(recorder, model_name='gpt-6-luna'),
        capabilities=[ModelInstructions({'*': 'Any.'}, kind='rule', id=None)],
    )
    agent.run_sync('Hi')
    assert [[(unwrap(part.content), part.on_change) for part in parts] for parts in recorder.requests] == [
        [('Any.', 'rewrite')]
    ]


@pytest.mark.parametrize('instrument', [False, True])
def test_fallback_model_warns_and_sends_nothing(instrument: bool) -> None:
    recorder = Recorder()
    model: Model = FallbackModel(FunctionModel(recorder, model_name='gpt-6-luna'))
    if instrument:
        model = InstrumentedModel(model)
    agent = Agent(model, instructions='Be brief.', capabilities=[ModelInstructions({'*': 'Any.'})])
    with pytest.warns(UserWarning, match="can't tell which model a `FallbackModel` will use"):
        agent.run_sync('Hi')
    assert recorder.texts == [['Be brief.']]


def test_fallback_model_does_not_call_the_function() -> None:
    def per_model(ctx: RunContext[None]) -> str:
        raise AssertionError('not called on a `FallbackModel`')  # pragma: no cover

    agent = Agent(FallbackModel(TestModel()), deps_type=type(None), capabilities=[ModelInstructions(per_model)])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = agent.run_sync('Hi')
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions is None


def test_two_with_one_id_and_different_instructions_raise() -> None:
    with pytest.raises(
        UserError, match="used by multiple `ModelInstructions` capabilities that disagree on 'instructions'"
    ):
        Agent(TestModel(), capabilities=[ModelInstructions({'*': 'One.'}), ModelInstructions({'*': 'Two.'})])
    with pytest.raises(UserError, match="disagree on 'kind'"):
        Agent(
            TestModel(), capabilities=[ModelInstructions({'*': 'One.'}), ModelInstructions({'*': 'One.'}, kind='rule')]
        )


def test_two_with_one_id_and_the_same_instructions_are_one() -> None:
    recorder = Recorder()
    agent = Agent(
        FunctionModel(recorder),
        capabilities=[ModelInstructions({'*': 'One.'}), ModelInstructions({'*': 'One.'})],
    )
    agent.run_sync('Hi')
    assert recorder.texts == [['One.']]


def test_from_spec() -> None:
    """Agent specs take the mapping; a function can only be passed in code."""
    recorder = Recorder()
    agent = Agent.from_spec(
        {'capabilities': [{'ModelInstructions': {'instructions': {'gpt-6*': 'GPT-6'}}}]},
    )
    agent.run_sync('Hi', model=FunctionModel(recorder, model_name='gpt-6-luna'))
    assert recorder.texts == [['GPT-6']]
