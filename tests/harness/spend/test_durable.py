from __future__ import annotations

import uuid
from collections.abc import Generator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

try:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from pydantic_ai.durable_exec.dbos import DBOSDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('dbos not installed', allow_module_level=True)

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness.spend import Budget, InMemorySpendStore, SpendEntry, SpendLimits, Spent


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture
def dbos(tmp_path: Path) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'durable_spend_limits',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


class AdvancingClock:
    """Return a different day on every call so an unjournaled replay changes its key."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> datetime:
        value = datetime(2026, 8, 1, tzinfo=timezone.utc) + timedelta(days=self.calls)
        self.calls += 1
        return value


class RecordingStore(InMemorySpendStore):
    """Keep the entries that reached the store so replay behavior is observable."""

    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[SpendEntry, ...]] = []

    async def add_many(self, entries: Sequence[SpendEntry]) -> Mapping[str, Spent]:
        self.batches.append(tuple(entries))
        return await super().add_many(entries)


_model_calls = 0


async def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del messages, info
    global _model_calls
    _model_calls += 1
    return ModelResponse(
        parts=[TextPart('done')],
        model_name='test',
        usage=RequestUsage(input_tokens=10, output_tokens=5),
    )


_clock = AdvancingClock()
_store = RecordingStore()
_limits = SpendLimits[None](
    budgets=[Budget(window='day')],
    store=_store,
    clock=_clock,
    price=lambda response: Decimal('1'),
)
_agent: Agent[None, str] = Agent(
    FunctionModel(_respond),
    name='durable_spend',
    deps_type=type(None),
    capabilities=[_limits, DBOSDurability[None]()],
)


@DBOS.workflow(name='durable_spend')
async def _workflow() -> str:
    return (await _agent.run('count this')).output


def test_default_id_is_stable_and_composes_with_durability() -> None:
    assert SpendLimits[None]().id == 'spend_limits'
    assert SpendLimits[None].from_spec().id == 'spend_limits'
    agent: Agent[None, str] = Agent(
        FunctionModel(_respond),
        name='default_spend_id',
        deps_type=type(None),
        capabilities=[SpendLimits[None](), DBOSDurability[None]()],
    )
    assert agent.name == 'default_spend_id'


@pytest.mark.anyio
async def test_dbos_journals_clock_reads_and_accrual(dbos: DBOS) -> None:
    global _model_calls
    _model_calls = 0
    _clock.calls = 0
    _store.batches.clear()
    workflow_id = str(uuid.uuid4())

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'done'

    first_clock_calls = _clock.calls
    first_batches = tuple(_store.batches)
    assert len(first_batches) == 1
    assert first_batches[0][0].requests == 1

    with SetWorkflowID(workflow_id):
        assert await _workflow() == 'done'

    assert _model_calls == 1
    assert _clock.calls == first_clock_calls
    assert tuple(_store.batches) == first_batches
    key = first_batches[0][0].key
    assert (await _store.get_many([key]))[key].requests == 1

    steps = await dbos.list_workflow_steps_async(workflow_id)
    names = {step['function_name'] for step in steps}
    assert 'durable_spend__capability__spend_limits.now' in names
    assert 'durable_spend__capability__spend_limits.read' in names
    assert 'durable_spend__capability__spend_limits.accrue' in names


@pytest.mark.anyio
async def test_store_token_deduplicates_without_consulting_the_journal() -> None:
    source = RecordingStore()
    limits = SpendLimits[None](
        budgets=[Budget(window='total')],
        store=source,
        price=lambda response: Decimal('1'),
    )
    agent: Agent[None, str] = Agent(FunctionModel(_respond), deps_type=type(None), capabilities=[limits])
    await agent.run('capture one accrual', run_id='stable-run')
    entry = source.batches[0][0]
    assert entry.token is not None

    replacement = InMemorySpendStore()
    first = await replacement.add_many([entry])
    second = await replacement.add_many([entry])

    assert first == second
    assert second[entry.key].requests == 1
