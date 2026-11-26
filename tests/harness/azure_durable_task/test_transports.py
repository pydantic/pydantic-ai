import asyncio
import json
import sys
from collections.abc import Generator
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import partial
from importlib import import_module
from typing import Annotated, Any, Self
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Field, Json, JsonValue, SecretStr, computed_field, model_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    Choice,
    Choices,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability, Hooks, RawToolArgs, ValidatedToolArgs, durable_operation
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolKind,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    import pydantic_ai_harness
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

    from .worker import (
        InMemoryWorker,
        completed,
        registered_agent,
        run_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_import_without_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    package = 'pydantic_ai_harness.azure_durable_task'
    monkeypatch.delattr(pydantic_ai_harness, 'azure_durable_task', raising=False)
    for name in [name for name in sys.modules if name == package or name.startswith(f'{package}.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, 'pydantic_ai.mcp', None)

    durability_type = import_module(package).AzureDurableTaskDurability
    durability = durability_type.from_agent(Agent(TestModel(), name='no-mcp', capabilities=[durability_type()]))
    assert durability is not None
    assert durability.get_durable_operation_backend().registrations()


@pytest.mark.parametrize('non_utf8', [False, True])
def test_serialization_failure_is_terminal_with_native_retries(
    durable_worker: 'InMemoryWorker', non_utf8: bool
) -> None:
    effects: list[str] = []
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='bad-json',
        capabilities=[
            AzureDurableTaskDurability(
                tool_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
            )
        ],
    )

    @agent.tool_plain
    def write() -> object:
        effects.append('written')
        return b'\xff' if non_utf8 else object()

    with registered_agent(durable_worker, agent):
        handle = schedule_agent(durable_worker, agent, 'Go', deps=None)
        durable_worker.run_until_idle()
        assert effects == ['written']
        assert durable_worker.timers == []
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            completed(durable_worker, handle)
    assert effects == ['written']


def test_typed_tool_arguments_are_restored_from_json(durable_worker: 'InMemoryWorker') -> None:
    class Payload(BaseModel):
        value: int
        timestamp: datetime
        identifier: UUID
        encoded: Json[list[int]]

    args = {
        'value': 42,
        'timestamp': '2026-01-01T00:00:00Z',
        'identifier': '12345678-1234-5678-1234-567812345678',
        'encoded': '[1,2]',
    }
    observed: list[int] = []
    validated: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart('consume', json.dumps(args))] if len(messages) == 1 else [TextPart('done')]
        )

    agent = Agent(FunctionModel(model), deps_type=int, name='typed-args', capabilities=[AzureDurableTaskDurability()])

    def validate(ctx: RunContext[int], payload: Payload) -> None:
        assert isinstance(payload, Payload)
        assert ctx.deps == 7
        validated.append(payload.value)

    @agent.tool_plain(args_validator=validate)
    def consume(payload: Payload) -> str:
        assert isinstance(payload, Payload)
        assert payload.timestamp.year == 2026
        assert payload.identifier.hex == '12345678123456781234567812345678'
        assert payload.encoded == [1, 2]
        observed.append(payload.value)
        return 'typed'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=7).output == 'done'
    assert observed == [42]
    assert validated == [42]


def test_aliased_tool_arguments_are_restored_from_json(durable_worker: 'InMemoryWorker') -> None:
    """The activity input carries validated arguments by Python name, while the tool's schema reads `guestName`."""
    observed: list[tuple[str, int]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        args = {'guestName': 'Ada', 'nights': 2}
        return ModelResponse(parts=[ToolCallPart('book', args)] if len(messages) == 1 else [TextPart('done')])

    agent = Agent(FunctionModel(model), name='aliased-args', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def book(guest: Annotated[str, Field(alias='guestName')], nights: int) -> str:
        observed.append((guest, nights))
        return 'booked'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert observed == [('Ada', 2)]


class Venue(TypedDict):
    room: Annotated[str, Field(alias='roomName')]


@dataclass
class Place:
    city: Annotated[str, Field(alias='cityName')]


class SerializedAs(BaseModel):
    room: str = Field(serialization_alias='roomOut')


class Counted(BaseModel):
    model_config = ConfigDict(extra='forbid')
    qty: int

    @computed_field
    @property
    def doubled(self) -> int:
        return self.qty * 2


def test_tool_arguments_are_validated_from_the_model_arguments_in_the_activity(
    durable_worker: 'InMemoryWorker',
) -> None:
    """The activity validates the arguments the model sent, as the orchestrator did, rather than a JSON dump of the validated ones.

    The dump writes a `TypedDict` and a stdlib dataclass by field name while their schema reads the alias, masks
    the `SecretStr`, and writes the parsed `Json` value while the schema expects a string.
    """
    observed: list[tuple[Venue, Place, str, list[int], SerializedAs, Counted]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        args = {
            'venue': {'roomName': 'A1'},
            'place': {'cityName': 'Paris'},
            'token': 'hunter2',
            'nums': '[1, 2]',
            'out': {'room': 'A1'},
            'counted': {'qty': 2},
        }
        return ModelResponse(parts=[ToolCallPart('book', args)] if len(messages) == 1 else [TextPart('done')])

    agent = Agent(FunctionModel(model), name='model-args', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def book(
        venue: Venue, place: Place, token: SecretStr, nums: Json[list[int]], out: SerializedAs, counted: Counted
    ) -> str:
        observed.append((venue, place, token.get_secret_value(), nums, out, counted))
        return 'booked'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert observed == [
        ({'room': 'A1'}, Place(city='Paris'), 'hunter2', [1, 2], SerializedAs(room='A1'), Counted(qty=2))
    ]


class Folded(BaseModel):
    price: int
    qty: int = 1

    @model_validator(mode='before')
    @classmethod
    def fold_quantity(cls, data: dict[str, int]) -> dict[str, int]:
        # Mutates its input: run twice on one dict, the second run finds no `qty`.
        data['price'] = data.pop('price') * data.pop('qty')
        return data


class PriceOptions(TypedDict, total=False):
    max_price: Annotated[float, Field(alias='maxPrice')]


class Order(BaseModel):
    qty: int


def test_arguments_changed_by_hooks_reach_the_activity_as_changed(durable_worker: 'InMemoryWorker') -> None:
    """Arguments a hook replaced, mutated or repaired don't validate back from the model's, so the activity gets them dumped."""
    observed: list[object] = []
    hooks = Hooks[None]()

    @hooks.on.before_tool_validate(tools=['stay'])
    async def repair(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        return {'nights': 2}

    @hooks.on.after_tool_validate(tools=['order'])
    async def replace_note(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        return {**args, 'note': 'hook'}

    @hooks.on.before_tool_execute(tools=['order'])
    async def mutate_order(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        args['order'].qty = 99
        return args

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[
                ToolCallPart('order', {'order': {'qty': 1}, 'note': 'model'}),
                ToolCallPart('stay', '{"nights": "two"}'),
            ]
        )

    agent = Agent(
        FunctionModel(model), name='hook-args', deps_type=type(None), capabilities=[hooks, AzureDurableTaskDurability()]
    )

    @agent.tool_plain
    def order(order: Order, note: str) -> str:
        observed.append((order, note))
        return 'ordered'

    @agent.tool_plain
    def stay(nights: int) -> str:
        observed.append(nights)
        return 'stayed'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert observed == [(Order(qty=99), 'hook'), 2]


@dataclass
class Account:
    name: str
    tenant_id: int = field(compare=False)


def test_field_a_hook_pinned_reaches_the_activity_as_pinned_where_eq_ignores_it(
    durable_worker: 'InMemoryWorker',
) -> None:
    """A field a hook pinned reaches the activity as pinned, even where `==` ignores it.

    `token` makes the dump lossy, so the model's arguments would be sent if they validated to the hook's; they do
    under `==`, which skips `tenant_id`. Their JSON form doesn't, so the call keeps the dump, as on main, and the
    `SecretStr` arrives masked.
    """
    observed: list[tuple[int, str]] = []
    hooks = Hooks[None]()

    @hooks.on.after_tool_validate(tools=['act'])
    async def pin_tenant(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        return {**args, 'account': replace(args['account'], tenant_id=1)}

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[ToolCallPart('act', {'account': {'name': 'acme', 'tenant_id': 2}, 'token': 'hunter2'})]
        )

    agent = Agent(
        FunctionModel(model),
        name='pinned-tenant',
        deps_type=type(None),
        capabilities=[hooks, AzureDurableTaskDurability()],
    )

    @agent.tool_plain
    def act(account: Account, token: SecretStr) -> str:
        observed.append((account.tenant_id, token.get_secret_value()))
        return 'acted'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert observed == [(1, '**********')]


def test_validator_that_mutates_its_input_runs_once_in_the_activity(
    durable_worker: 'InMemoryWorker',
) -> None:
    """Checking which arguments to send runs the tool's validators on copies, so the activity validates each input once.

    `fold` crosses as its dump, which the activity validates once. The orchestrator has already folded
    `fold_with_options`'s own arguments in place, so validating them again raises there; that call keeps the dump
    too, whose `PriceOptions` field arrives without its aliased key.
    """
    observed: list[object] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(
            parts=[
                ToolCallPart('fold', {'filters': {'price': 3, 'qty': 2}}),
                ToolCallPart('fold_with_options', {'filters': {'price': 3, 'qty': 2}, 'opts': {'maxPrice': 10}}),
            ]
        )

    agent = Agent(FunctionModel(model), name='folding-args', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def fold(filters: Folded) -> str:
        observed.append(filters)
        return 'folded'

    @agent.tool_plain
    def fold_with_options(filters: Folded, opts: PriceOptions) -> str:
        observed.append((filters, opts))
        return 'folded'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    # The model's arguments were folded in place, so this call keeps the dump and loses `opts` as on main (#13440).
    assert observed == [Folded(price=6, qty=1), (Folded(price=6, qty=1), {})]


def test_choice_action_output_and_partial_output(durable_worker: 'InMemoryWorker') -> None:
    choices = Choices({'answer': Choice(value=lambda: 42)})
    choice_agent = Agent(TestModel(), output_type=choices, name='choice', capabilities=[AzureDurableTaskDurability()])
    with registered_agent(durable_worker, choice_agent):
        assert run_agent(durable_worker, choice_agent, 'Hi', deps=None).output == 42

    def output(prefix: str, value: str) -> str:
        return f'{prefix}:{value}'

    partial_agent = Agent(
        TestModel(custom_output_args={'value': 'x'}),
        output_type=partial(output, 'prefix'),
        name='partial',
        capabilities=[AzureDurableTaskDurability()],
    )
    with registered_agent(durable_worker, partial_agent):
        assert run_agent(durable_worker, partial_agent, 'Hi', deps=None).output == 'prefix:x'


def test_strict_and_json_outputs_round_trip(durable_worker: 'InMemoryWorker') -> None:
    class StrictOutput(BaseModel):
        model_config = ConfigDict(strict=True)
        timestamp: datetime

    class JsonOutput(BaseModel):
        data: Json[list[int]]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"timestamp":"2026-01-01T00:00:00Z"}')])

    strict = Agent(
        FunctionModel(model),
        output_type=StrictOutput,
        name='strict',
        capabilities=[AzureDurableTaskDurability()],
    )
    expected = strict.run_sync('Hi').output
    with registered_agent(durable_worker, strict):
        assert run_agent(durable_worker, strict, 'Hi', deps=None).output == expected

    json_agent = Agent(
        TestModel(custom_output_args={'data': '[1,2]'}),
        output_type=JsonOutput,
        name='json',
        capabilities=[AzureDurableTaskDurability()],
    )
    expected_json = json_agent.run_sync('Hi').output
    with registered_agent(durable_worker, json_agent):
        assert run_agent(durable_worker, json_agent, 'Hi', deps=None).output == expected_json


def test_incompatible_persisted_tool_definition_is_rejected(durable_worker: 'InMemoryWorker') -> None:
    agent = Agent(TestModel(), name='removed', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def original() -> str:
        # The invalid persisted name must be rejected before this tool runs.
        return 'original'  # pragma: no cover

    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        for _ in range(20):
            if not (durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call_tool')):
                assert durable_worker.step()
        assert durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call_tool')
        instance_id, name, task_id, encoded = durable_worker.activity_queue.popleft()
        assert encoded is not None
        payload = json.loads(encoded)
        payload['parameters']['name'] = 'removed'
        payload['parameters']['original_name'] = 'removed'
        payload['parameters']['tool_def']['name'] = 'removed'
        durable_worker.activity_queue.appendleft((instance_id, name, task_id, json.dumps(payload)))
        with pytest.raises(OrchestrationFailedError, match='is no longer available'):
            completed(durable_worker, run)


def test_replaced_activity_context_preserves_tool_kind_snapshot(durable_worker: 'InMemoryWorker') -> None:
    lookup_kind = ToolKind('azure_activity_context', 'lookup')
    names: list[str | None] = []
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='context-copy',
        capabilities=[AzureDurableTaskDurability()],
    )

    @agent.tool(tool_kind=lookup_kind)
    async def lookup(ctx: RunContext[object]) -> str:
        copied = replace(ctx, tool_call_id='copied-call')
        assert copied is not ctx
        assert copied.tool_call_id == 'copied-call'
        assert ctx.tool_name_for_kind(lookup_kind) == 'lookup'
        names.append(copied.tool_name_for_kind(lookup_kind))
        assert copied.tool_name_for_kind('azure_activity_context.missing') is None
        return 'found'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert names == ['lookup']
    assert 'context--copy__function_toolset__-x3C-agent-x3E-.call_tool' in durable_worker.activity_calls


@pytest.mark.parametrize('stage', ['envelope', 'parameters', 'dependencies'])
def test_invalid_activity_input_is_terminal(durable_worker: 'InMemoryWorker', stage: str) -> None:
    agent = Agent(
        TestModel(),
        deps_type=str,
        name='invalid-wire',
        capabilities=[
            AzureDurableTaskDurability(
                tool_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
            )
        ],
    )
    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps='valid')
        assert durable_worker.step()
        instance_id, name, task_id, encoded = durable_worker.activity_queue.popleft()
        assert encoded is not None
        payload = json.loads(encoded)
        if stage == 'envelope':
            payload = {}
        elif stage == 'parameters':
            payload['parameters'] = None
        else:
            payload['deps_input'] = {'not': 'a string'}
        durable_worker.activity_queue.append((instance_id, name, task_id, json.dumps(payload)))
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            completed(durable_worker, run)
    assert durable_worker.timers == []
    assert durable_worker.activity_calls == ['invalid--wire__model.request']


@pytest.mark.parametrize('kind', ['function', 'mcp'])
@pytest.mark.parametrize('cancel', [False, True])
def test_activity_toolset_scope_closes_after_failure_or_cancellation(
    durable_worker: 'InMemoryWorker', kind: str, cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifetimes: list[str] = []

    def fail() -> None:
        assert lifetimes[-1] == 'enter'
        if cancel:
            raise asyncio.CancelledError('cancel tool')
        raise RuntimeError('failed tool')

    toolset: AbstractToolset[object]
    if kind == 'function':

        class Functions(FunctionToolset[object]):
            async def __aenter__(self) -> Self:
                lifetimes.append('enter')
                return await super().__aenter__()

            async def __aexit__(self, *args: Any) -> None:
                try:
                    await super().__aexit__(*args)
                finally:
                    lifetimes.append('exit')

        toolset = Functions([fail], id='tools')
    else:
        with try_import() as mcp_imports:
            from fastmcp import FastMCP

            from pydantic_ai.mcp import MCPToolset
        if not mcp_imports():
            # Optional-extra probes can skip here; all-extras coverage does not.
            pytest.skip('mcp not installed')  # pragma: lax no cover

        server: FastMCP[None] = FastMCP()
        server.tool(fail)

        class MCP(MCPToolset[object]):
            async def __aenter__(self) -> Self:
                entered = await super().__aenter__()
                lifetimes.append('enter')
                return entered

            async def __aexit__(self, *args: Any) -> bool | None:
                try:
                    return await super().__aexit__(*args)
                finally:
                    lifetimes.append('exit')

            async def call_tool(
                self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
            ) -> Any:
                fail()

        toolset = MCP(server, id='tools')

    agent = Agent(TestModel(), name='lifetimes', toolsets=[toolset], capabilities=[AzureDurableTaskDurability()])
    activity_turn = durable_worker.activity_turn

    def checked_activity_turn(instance_id: str, name: str, task_id: int, encoded: str | None) -> None:
        activity_turn(instance_id, name, task_id, encoded)
        assert lifetimes.count('enter') == lifetimes.count('exit')

    monkeypatch.setattr(durable_worker, 'activity_turn', checked_activity_turn)
    with registered_agent(durable_worker, agent):
        with pytest.raises(OrchestrationFailedError) as error:
            run_agent(durable_worker, agent, 'Hi', deps=None)
    if not cancel:
        assert 'failed tool' in str(error.value)
    assert 'enter' in lifetimes


@pytest.mark.parametrize('kind', ['collision', 'missing-format', 'codec-failure', 'non-utf8'])
def test_activity_result_wire_contract(durable_worker: 'InMemoryWorker', kind: str) -> None:
    lookalike: JsonValue = {
        'format': 'pydantic-ai-durabletask-result-v1',
        'value': 'user data',
        'error': 'not a failure',
    }
    returned: list[object] = []

    class Producer(AbstractCapability[object]):
        id = 'producer'

        async def before_run(self, ctx: RunContext[object]) -> None:
            returned.append(await self.produce(ctx))

        @durable_operation('produce')
        async def produce(self, ctx: RunContext[object]) -> object:
            if kind == 'codec-failure':
                return object()
            if kind == 'non-utf8':
                return b'\xff'
            return lookalike

    agent = Agent(TestModel(), name='wire', capabilities=[Producer(), AzureDurableTaskDurability()])
    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        if kind == 'collision':
            completed(durable_worker, run)
            # A value shaped like the result envelope comes back as the value, not as an error.
            assert returned and all(value == lookalike for value in returned)
            return
        if kind == 'missing-format':
            assert durable_worker.step()  # Schedule the capability operation.
            assert durable_worker.step()  # Run it.
            event = durable_worker.instances[run].new_events[0]
            recorded = json.loads(event.taskCompleted.result.value)
            del recorded['format']
            event.taskCompleted.result.value = json.dumps(recorded)
        with pytest.raises(
            OrchestrationFailedError,
            match='Field required' if kind == 'missing-format' else 'Durable Task serialization failed',
        ):
            completed(durable_worker, run)


def test_model_compaction_restores_the_model(durable_worker: 'InMemoryWorker') -> None:
    class CompactModel(TestModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            assert request_context.model is self
            assert request_context.model_settings == {'temperature': 0.2}
            assert instructions == 'Keep decisions'
            return ModelResponse(parts=[TextPart('compacted')])

    model = CompactModel()
    agent = Agent(model, name='compact', capabilities=[AzureDurableTaskDurability()])
    capability = AzureDurableTaskDurability.from_agent(agent)
    assert capability is not None

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, bool]:
        async def body() -> bool:
            request = ModelRequestContext(
                model=model,
                messages=[],
                model_settings={'temperature': 0.2},
                model_request_parameters=ModelRequestParameters(),
            )

            async def handler(context: ModelRequestContext) -> ModelResponse:
                return await context.model.compact_messages(context, instructions='Keep decisions')

            result = await capability.wrap_model_request(
                RunContext(deps=None, model=model, usage=RunUsage()), request_context=request, handler=handler
            )
            return result.parts == [TextPart('compacted')]

        return (yield from run_durable(context, body))

    with registered_agent(durable_worker, agent):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.schedule(orchestrator)
        state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() is True
    assert durable_worker.activity_calls == ['compact__model.compact_messages']


@pytest.mark.parametrize('restore', [True, False])
def test_workspace_is_reconstructed_in_activities(durable_worker: 'InMemoryWorker', restore: bool) -> None:
    provider = InMemoryProvider()
    durability = AzureDurableTaskDurability[object]()

    class Workspaces(AbstractCapability[object]):
        def get_workspace(self, ctx: RunContext[object], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            return provider.backend(ref) if ref is None or restore or durability.in_durable_context else None

        async def before_run(self, ctx: RunContext[object]) -> None:
            await ctx.workspace.write_text('hook.txt', 'written in hook')

    agent = Agent(
        TestModel(custom_output_text='done'),
        name='workspace',
        capabilities=[Workspaces(), durability],
    )

    @agent.tool
    async def read(ctx: RunContext[object]) -> str:
        assert await ctx.workspace.read_text('hook.txt') == 'written in hook'
        return 'read in activity'

    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        if not restore:
            with pytest.raises(OrchestrationFailedError, match='No capability can supply the workspace'):
                completed(durable_worker, run)
            return
        assert completed(durable_worker, run) == 'done'
    assert 'workspace__capability__workspace.call' in durable_worker.activity_calls
    assert len(provider.environments) == 1


def test_mcp_discovery_instructions_and_calls(durable_worker: 'InMemoryWorker') -> None:
    with try_import() as mcp_imports:
        from fastmcp import FastMCP

        from pydantic_ai.mcp import MCPToolset
    if not mcp_imports():
        # Optional-extra probes can skip here; all-extras coverage does not.
        pytest.skip('mcp not installed')  # pragma: lax no cover

    server: FastMCP[None] = FastMCP(name='calc', instructions='Add numbers.')
    calls: list[int] = []

    @server.tool
    def add(a: int, b: int) -> int:
        calls.append(a + b)
        return a + b

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('add', {'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model),
        name='mcp',
        toolsets=[MCPToolset[object](server, id='calc', include_instructions=True)],
        capabilities=[AzureDurableTaskDurability()],
    )
    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert calls == [5]
    assert durable_worker.activity_calls == [
        'mcp__mcp_server__calc.get_tools',
        'mcp__mcp_server__calc.get_instructions',
        'mcp__model.request',
        'mcp__mcp_server__calc.call_tool',
        'mcp__mcp_server__calc.get_instructions',
        'mcp__model.request',
    ]


def test_nested_aliased_tool_arguments_cross_the_activity_input_by_alias(durable_worker: 'InMemoryWorker') -> None:
    """By Python name, the required `roomName` would fail revalidation and `maxPrice` would revalidate to its default.

    Not a VCR test: the claim is about the arguments crossing into the activity input, and no provider request is
    involved.
    """

    class Room(BaseModel):
        name: str = Field(alias='roomName')
        max_price: float | None = Field(None, alias='maxPrice')

    @pydantic_dataclass
    class Guest:
        name: Annotated[str, Field(alias='guestName')]

    observed: list[tuple[str, float | None, str]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        args = {'room': {'roomName': 'A1', 'maxPrice': 10}, 'guest': {'guestName': 'Ada'}}
        return ModelResponse(parts=[ToolCallPart('book', args)] if len(messages) == 1 else [TextPart('done')])

    agent = Agent(FunctionModel(model), name='nested-aliased-args', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def book(room: Room, guest: Guest) -> str:
        observed.append((room.name, room.max_price, guest.name))
        return 'booked'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert observed == [('A1', 10.0, 'Ada')]
