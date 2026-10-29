"""Prefect retry identities through the public agent API with a local test model."""

from __future__ import annotations

import asyncio
import threading
import uuid

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from ..conftest import try_import

with try_import() as imports_successful:
    from prefect import flow

    from pydantic_ai.durable_exec.prefect import PrefectDurability

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='prefect not installed'),
    pytest.mark.usefixtures('prefect_test_server'),
    pytest.mark.xdist_group(name='prefect'),
]


@pytest.fixture(autouse=True)
def blockbuster_excluded_modules() -> tuple[str, ...]:
    """Prefect inspects flow source synchronously while constructing its tasks."""
    return ('pydantic_ai.durable_exec.prefect',)


@pytest.mark.parametrize(
    ('input_kind', 'follow_up'),
    [
        ('prompt', False),
        ('prompt', True),
        ('deps', True),
        ('metadata', True),
        ('history', True),
        ('spec_metadata_factory', True),
    ],
)
async def test_flow_retry_preserves_default_ids_after_branch_reordering(input_kind: str, follow_up: bool) -> None:
    model_calls: list[None] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_calls.append(None)
        return ModelResponse(parts=[TextPart('answer')])

    agent = Agent(
        FunctionModel(model_fn), deps_type=str, name='reordered_default_ids', capabilities=[PrefectDurability[str]()]
    )
    attempts: list[dict[str, tuple[str, str]]] = []

    @flow(retries=1)
    async def reordered() -> None:
        ids: dict[str, tuple[str, str]] = {}
        attempts.append(ids)
        first, second = ('left', 'right') if len(attempts) == 1 else ('right', 'left')
        first_done = asyncio.Event()

        async def branch(label: str, after: asyncio.Event | None) -> None:
            if after is not None:
                await after.wait()
            history: list[ModelMessage] | None = None
            if input_kind == 'history':
                history = [ModelRequest(parts=[UserPromptPart(label)]), ModelResponse(parts=[TextPart('previous')])]
            branch_metadata: dict[str, str] | None = {'branch': label} if input_kind == 'metadata' else None
            branch_spec: dict[str, object] | None = (
                {'metadata': {'branch': label}} if input_kind == 'spec_metadata_factory' else None
            )

            def metadata_fn(ctx: RunContext[str]) -> dict[str, object]:
                return {'derived_run_id': ctx.run_id}

            result = await agent.run(
                label if input_kind == 'prompt' else 'same prompt',
                deps=label if input_kind == 'deps' else 'same deps',
                metadata=metadata_fn if input_kind == 'spec_metadata_factory' else branch_metadata,
                message_history=history,
                spec=branch_spec,
            )
            ids[label] = result.run_id, result.conversation_id
            if follow_up and label == 'left':
                continued = await agent.run('follow up', deps='same deps', message_history=result.all_messages())
                assert continued.conversation_id == result.conversation_id
                assert continued.run_id != result.run_id
                assert continued.new_messages()
                ids['follow_up'] = continued.run_id, continued.conversation_id
            if after is None:
                first_done.set()

        async with asyncio.TaskGroup() as group:
            group.create_task(branch(first, None))
            group.create_task(branch(second, first_done))
        if len(attempts) == 1:
            raise RuntimeError('retry after both branches finish')

    await reordered()
    original, retried = attempts
    assert original['left'] != original['right']
    assert retried == original
    assert len(model_calls) == 2 + int(follow_up)


async def test_identical_projected_inputs_get_separate_retry_stable_ids() -> None:
    agent = Agent(TestModel(), name='identical_default_ids', capabilities=[PrefectDurability[object]()])
    attempts: list[list[str]] = []

    @flow(retries=1)
    async def repeated() -> None:
        ids: list[str] = []
        for prompt in ('same', 'different', 'same', 'same'):
            # Live resources do not distinguish projected inputs, even when recreated for each call.
            deps: dict[str, object] = {'resource': threading.Lock()}
            ids.append((await agent.run(prompt, deps=deps)).run_id)
        attempts.append(ids)
        if len(attempts) == 1:
            raise RuntimeError('retry')

    await repeated()
    original, retried = attempts
    assert original == retried
    assert len(set(original)) == len(original)
    assert original[2:] == [f'{original[0]}:1', f'{original[0]}:2']


@pytest.mark.parametrize('identity', ['history', 'shared_conversation', 'explicit_conversation'])
async def test_flow_retry_preserves_incoming_conversation_identity(identity: str) -> None:
    model_calls: list[None] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_calls.append(None)
        return ModelResponse(parts=[TextPart('answer')])

    agent = Agent(
        FunctionModel(model_fn), deps_type=str, name='incoming_identity', capabilities=[PrefectDurability[str]()]
    )
    attempts: list[dict[str, tuple[str, str]]] = []

    @flow(retries=1)
    async def reordered() -> None:
        ids: dict[str, tuple[str, str]] = {}
        attempts.append(ids)
        order = ('left', 'right') if len(attempts) == 1 else ('right', 'left')
        for label in order:
            if identity == 'explicit_conversation':
                result = await agent.run('same prompt', deps='same', conversation_id=label)
                assert result.conversation_id == label
            else:
                parent = await agent.run(
                    'same parent prompt',
                    deps=label,
                    conversation_id='shared' if identity == 'shared_conversation' else None,
                )
                ids[f'{label}:parent'] = parent.run_id, parent.conversation_id
                result = await agent.run('same follow-up', deps='same', message_history=parent.all_messages())
                assert result.conversation_id == parent.conversation_id
                assert result.new_messages()
            ids[label] = result.run_id, result.conversation_id
        if len(attempts) == 1:
            raise RuntimeError('retry')

    await reordered()
    original, retried = attempts
    assert original['left'][0] != original['right'][0]
    assert retried == original
    # Logical run identity separates these calls without changing cross-run model-result reuse.
    assert len(model_calls) == (1 if identity == 'explicit_conversation' else 3)


async def test_default_id_static_metadata_precedence_and_factory_lifecycle() -> None:
    agent = Agent(
        TestModel(),
        name='static_metadata_ids',
        metadata={'base': 'kept', 'shared': 'base'},
        capabilities=[PrefectDurability[object]()],
    )
    factory_run_ids: list[str | None] = []

    def metadata_factory(ctx: RunContext[object]) -> dict[str, object]:
        factory_run_ids.append(ctx.run_id)
        return {'derived_run_id': ctx.run_id}

    @flow
    async def compare() -> None:
        merged = await agent.run('same', metadata={'shared': 'run'})
        with agent.override(metadata={'base': 'kept', 'shared': 'run'}):
            overridden = await agent.run('same', metadata={'ignored': True})
        assert overridden.run_id == f'{merged.run_id}:1'

        with agent.override(metadata=metadata_factory):
            dynamic = await agent.run('same', metadata={'ignored': True})
        assert factory_run_ids and set(factory_run_ids) == {dynamic.run_id}
        assert dynamic.metadata == {'derived_run_id': dynamic.run_id}
        with agent.override(metadata={}):
            without_metadata = await agent.run('same')
        assert without_metadata.run_id == f'{dynamic.run_id}:1'
        assert dynamic.run_id != merged.run_id

    await compare()


@pytest.mark.parametrize('provided', [None, 'custom-prefect-id'])
async def test_prefect_legacy_default_run_id_override(provided: str | None) -> None:
    class LegacyPrefect(PrefectDurability[object]):
        def get_default_run_id(self) -> str | None:
            return provided

    agent = Agent(TestModel(), name='legacy_default_id', capabilities=[LegacyPrefect()])
    result = await agent.run('outside a flow')
    if provided is not None:
        assert result.run_id == provided
    else:
        assert ':' not in result.run_id


async def test_prefect_legacy_default_run_id_override_delegates_to_super() -> None:
    """An existing override can retain the public hook's counter-based policy."""

    class LegacyPrefect(PrefectDurability[object]):
        def get_default_run_id(self) -> str | None:
            return super().get_default_run_id()

    agent = Agent(TestModel(), name='delegating_default_id', capabilities=[LegacyPrefect()])
    outside = await agent.run('outside')
    assert uuid.UUID(outside.run_id).version == 7

    @flow
    async def delegated() -> None:
        first = await agent.run('first')
        second = await agent.run('second')
        assert first.run_id.endswith(':0')
        assert second.run_id == first.run_id.removesuffix(':0') + ':1'

    await delegated()
