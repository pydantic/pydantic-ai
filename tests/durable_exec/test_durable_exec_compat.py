from __future__ import annotations

from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FunctionToolset,
    ModelResponse,
    RunContext,
    TextPart,
    Tool,
    durable_exec,
)
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.durable_exec import (
    IDENTITY_CODEC,
    JSON_CODEC,
    BaseDurabilityCapability,
    CallableOperationBackend,
    CapabilityOperationId,
    DurabilityEngineSpec,
    DurableOperationId,
    EventStreamHandlerId,
    JournalOperationNamer,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    OperationConfigRole,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityOperationResult,
    ModelRequestContextProjection,
    capability_operation_result_type,
)
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    ToolConfig,
    _ApprovalRequired,  # pyright: ignore[reportPrivateUsage]
    _CallDeferred,  # pyright: ignore[reportPrivateUsage]
    _ModelRetry,  # pyright: ignore[reportPrivateUsage]
    _ToolFailed,  # pyright: ignore[reportPrivateUsage]
    _ToolReturn,  # pyright: ignore[reportPrivateUsage]
    _ValidationError,  # pyright: ignore[reportPrivateUsage]
    _ValidationErrorDetail,  # pyright: ignore[reportPrivateUsage]
    run_args_validator,
    unwrap_recorded_tool_call_result,
    unwrap_tool_call_result,
    validate_tool_args,
    wrap_tool_call_result,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RunUsage


def test_public_engine_builder_exports() -> None:
    assert durable_exec.__all__ == [
        'BaseDurabilityCapability',
        'ToolsetCallToolId',
        'CallableOperationBackend',
        'ModelCancelSuspendedResponseId',
        'CapabilityOperationId',
        'ModelCompactMessagesId',
        'DurabilityCodec',
        'DurabilityEngineSpec',
        'DurableOperationBackend',
        'DurableOperationId',
        'DurableOperationNamer',
        'EventStreamHandlerId',
        'ToolsetGetInstructionsId',
        'ToolsetGetToolsId',
        'IDENTITY_CODEC',
        'JSON_CODEC',
        'JournalCallableOperationBackend',
        'JournalOperationNamer',
        'ModelRequestId',
        'OperationConfigRole',
        'RegisteredOperationBackend',
        'RoleBasedOperationConfig',
        'ToolsetKind',
        'ToolsetValidateToolArgumentsId',
    ]
    assert all(getattr(durable_exec, name) is not None for name in durable_exec.__all__)


JOURNAL_OPERATION_NAMES = {
    'compat__model.request',
    'compat__model.request.registered',
    'compat__model.request_stream',
    'compat__model.request_stream.registered',
    'compat__model.cancel_suspended_response',
    'compat__model.cancel_suspended_response.registered',
    'compat__model.compact_messages',
    'compat__model.compact_messages.registered',
    'compat__event_stream_handler',
    'compat__function_toolset__functions.call_tool:function_tool',
    'compat__function_toolset__functions.validate_args',
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool:dynamic_tool',
    'compat__dynamic_toolset__dynamic.validate_args',
    'compat__capability__compat.operation',
}

PREFECT_OPERATION_NAMES = {
    'Model Request: test',
    'Model Request (Streaming): test',
    'Cancel Suspended Response: test',
    'Compact Messages: test',
    'Handle Stream Event',
    'Get MCP Tools: mcp',
    'Get MCP Instructions: mcp',
    'Discover Tools: dynamic',
    'Call Tool: function_tool',
    'Validate Tool Args: function_tool',
    'Call MCP Tool: mcp_tool',
    'Call Tool: dynamic_tool',
    'Validate Tool Args: dynamic_tool',
    'Capability: compat.operation',
}

TEMPORAL_ACTIVITY_NAMES = {
    'agent__compat__model_request',
    'agent__compat__model_request_stream',
    'agent__compat__model_cancel_suspended_response',
    'agent__compat__model_compact_messages',
    'agent__compat__event_stream_handler',
    'agent__compat__toolset__<agent>__call_tool',
    'agent__compat__toolset__<agent>__validate_args',
    'agent__compat__toolset__functions__call_tool',
    'agent__compat__toolset__functions__validate_args',
    'agent__compat__mcp_server__mcp__get_tools',
    'agent__compat__mcp_server__mcp__get_instructions',
    'agent__compat__mcp_server__mcp__call_tool',
    'agent__compat__dynamic_toolset__dynamic__get_tools',
    'agent__compat__dynamic_toolset__dynamic__call_tool',
    'agent__compat__dynamic_toolset__dynamic__validate_args',
    'agent__compat__capability__compat__operation',
}

DBOS_OPERATION_NAMES = {
    'compat__model.request',
    'compat__model.request_stream',
    'compat__model.cancel_suspended_response',
    'compat__model.compact_messages',
    'compat__event_stream_handler',
    'compat__mcp_server__mcp.get_tools',
    'compat__mcp_server__mcp.get_instructions',
    'compat__mcp_server__mcp.call_tool',
    'compat__dynamic_toolset__dynamic.get_tools',
    'compat__dynamic_toolset__dynamic.call_tool',
    'compat__dynamic_toolset__dynamic.validate_args',
    'compat__capability__compat.operation',
}


def _operation_ids() -> list[DurableOperationId]:
    return [
        ModelRequestId(None, streaming=False, model_name='test'),
        ModelRequestId('registered', streaming=False, model_name='test'),
        ModelRequestId(None, streaming=True, model_name='test'),
        ModelRequestId('registered', streaming=True, model_name='test'),
        ModelCancelSuspendedResponseId(None, model_name='test'),
        ModelCancelSuspendedResponseId('registered', model_name='test'),
        ModelCompactMessagesId(None, model_name='test'),
        ModelCompactMessagesId('registered', model_name='test'),
        EventStreamHandlerId(),
        ToolsetCallToolId('function', toolset_id='functions'),
        ToolsetValidateToolArgumentsId('function', toolset_id='functions'),
        ToolsetGetToolsId('mcp', toolset_id='mcp'),
        ToolsetGetInstructionsId('mcp'),
        ToolsetCallToolId('mcp', toolset_id='mcp'),
        ToolsetGetToolsId('dynamic', toolset_id='dynamic'),
        ToolsetCallToolId('dynamic', toolset_id='dynamic'),
        ToolsetValidateToolArgumentsId('dynamic', toolset_id='dynamic'),
        CapabilityOperationId('compat', operation='operation'),
    ]


def _operation_label(operation_id: DurableOperationId) -> str | None:
    if isinstance(operation_id, (ToolsetCallToolId, ToolsetValidateToolArgumentsId)):
        return {'function': 'function_tool', 'mcp': 'mcp_tool', 'dynamic': 'dynamic_tool'}[operation_id.toolset_kind]
    return None


class CompatCapability(AbstractCapability[Any]):
    id = 'compat'

    @durable_operation('operation')
    async def operation(self, ctx: RunContext[Any]) -> None:
        pass


class _JournalConfig:
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> ToolConfig:
        return {}

    def for_tool(
        self, role: OperationConfigRole, operation_id: DurableOperationId, tool: object | None, tool_name: str
    ) -> ToolConfig:
        return {}


class _JournalBackend(CallableOperationBackend[ToolConfig]):
    def __init__(self, durability: JournalDurability) -> None:
        super().__init__(namer=JournalOperationNamer(durability.name), config=_JournalConfig())
        self._durability = durability

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        self._durability.recorded_names.append(name)
        return await body()


class JournalDurability(BaseDurabilityCapability[Any]):
    engine_spec = DurabilityEngineSpec(
        engine_name='Journal compatibility stub',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
    )

    @property
    def in_durable_context(self) -> bool:
        return True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorded_names: list[str] = []

    def get_durable_operation_backend(self) -> durable_exec.DurableOperationBackend[ToolConfig]:
        return _JournalBackend(self)


async def test_journal_operation_name_assembly_sequence() -> None:
    async def function_tool() -> str:
        return 'function'

    async def dynamic_tool() -> str:
        return 'dynamic'

    function_toolset = FunctionToolset(tools=[function_tool], id='functions')
    dynamic_toolset = DynamicToolset(lambda _: FunctionToolset(tools=[dynamic_tool]), id='dynamic')
    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=[function_toolset, dynamic_toolset],
        capabilities=[CompatCapability(), JournalDurability(event_stream_handler=_event_handler)],
    )

    result = await agent.run('Call every tool')

    assert result.output == '{"function_tool":"function","dynamic_tool":"dynamic"}'
    durability = JournalDurability.from_agent(agent)
    assert durability is not None
    assert durability.recorded_names == [
        'compat__dynamic_toolset__dynamic.get_tools',
        'compat__model.request_stream',
        'compat__event_stream_handler',
        'compat__event_stream_handler',
        'compat__function_toolset__functions.call_tool:function_tool',
        'compat__dynamic_toolset__dynamic.call_tool:dynamic_tool',
        'compat__event_stream_handler',
        'compat__event_stream_handler',
        'compat__dynamic_toolset__dynamic.get_tools',
        'compat__model.request_stream',
    ]
    assert set(durability.recorded_names) <= JOURNAL_OPERATION_NAMES


def test_default_journal_operation_name_matrix() -> None:
    namer = JournalOperationNamer('compat')
    names = {
        namer.invocation_name(operation_id, label=_operation_label(operation_id)).operation_name
        for operation_id in _operation_ids()
    }
    assert names == JOURNAL_OPERATION_NAMES


def test_prefect_operation_name_matrix() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec.prefect._operation_names import PrefectOperationNamer

    namer = PrefectOperationNamer()
    names = {
        namer.invocation_name(operation_id, label=_operation_label(operation_id)).operation_name
        for operation_id in _operation_ids()
    }
    assert names == PREFECT_OPERATION_NAMES


def test_prefect_operation_name_assembly_completeness() -> None:
    pytest.importorskip('prefect')
    from pydantic_ai.durable_exec._toolset import DurableDynamicToolset, DurableFunctionToolset, DurableMCPToolset
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.prefect._operation_names import PrefectOperationNamer

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[CompatCapability(), PrefectDurability(event_stream_handler=_event_handler)],
    )
    durability = PrefectDurability.from_agent(agent)
    assert durability is not None
    assert {type(toolset) for toolset in durability._toolsets_by_id.values()} >= {  # pyright: ignore[reportPrivateUsage]
        DurableFunctionToolset,
        DurableMCPToolset,
        DurableDynamicToolset,
    }
    namer = PrefectOperationNamer()
    assembled_names = {
        namer.invocation_name(operation_id, label=_operation_label(operation_id)).operation_name
        for operation_id in _operation_ids()
    }
    assert assembled_names == PREFECT_OPERATION_NAMES


def test_dbos_operation_name_matrix_and_assembly_completeness() -> None:
    pytest.importorskip('dbos')
    from pydantic_ai.durable_exec._toolset import DurableDynamicToolset, DurableMCPToolset
    from pydantic_ai.durable_exec.dbos import DBOSDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[CompatCapability(), DBOSDurability(event_stream_handler=_event_handler)],
    )
    durability = DBOSDurability.from_agent(agent)
    assert durability is not None
    assert {type(toolset) for toolset in durability._toolsets_by_id.values()} >= {  # pyright: ignore[reportPrivateUsage]
        DurableMCPToolset,
        DurableDynamicToolset,
    }
    backend = durability._operation_backend  # pyright: ignore[reportPrivateUsage]
    assert backend is not None
    registered_names = {cast(Any, registration).dbos_function_name for registration in backend.registrations()}
    assert registered_names == DBOS_OPERATION_NAMES


def _synthetic_toolsets() -> tuple[FunctionToolset[Any], DynamicToolset[Any], Any]:
    pytest.importorskip('mcp')
    from fastmcp.client.transports import StdioTransport

    from pydantic_ai.mcp import MCPToolset

    # Assembly inspects this tool's definition but never executes its body.
    async def function_tool() -> None: """Declaration used only for toolset assembly."""  # fmt: skip  # pragma: no branch

    function_toolset = FunctionToolset(id='functions')
    function_toolset.add_function(function_tool)
    dynamic_toolset = DynamicToolset(lambda _: FunctionToolset(tools=[function_tool]), id='dynamic')
    mcp_toolset = MCPToolset(StdioTransport(command='python', args=['-m', 'tests.mcp_server']), id='mcp')
    return function_toolset, dynamic_toolset, mcp_toolset


async def _event_handler(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        pass


def test_temporal_activity_name_matrix_and_assembly_completeness() -> None:
    pytest.importorskip('temporalio')
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.durable_exec.temporal import TemporalDurability

    agent = Agent(
        TestModel(),
        name='compat',
        toolsets=list(_synthetic_toolsets()),
        capabilities=[CompatCapability(), TemporalDurability(event_stream_handler=_event_handler)],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    names = {
        ActivityDefinition.must_from_callable(item).name  # pyright: ignore[reportUnknownMemberType]
        for item in durability.temporal_activities
    }
    assert names == TEMPORAL_ACTIVITY_NAMES


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (_ToolReturn('ok'), {'result': 'ok', 'kind': 'tool_return'}),
        (_ApprovalRequired({'scope': 'write'}), {'metadata': {'scope': 'write'}, 'kind': 'approval_required'}),
        (_CallDeferred({'ticket': 7}), {'metadata': {'ticket': 7}, 'kind': 'call_deferred'}),
        (_ModelRetry('retry me'), {'message': 'retry me', 'kind': 'model_retry'}),
        (
            _ValidationError(
                'int',
                errors=[_ValidationErrorDetail('int_parsing', loc=['value'], msg='bad integer', input='x')],
            ),
            {
                'title': 'int',
                'errors': [{'type': 'int_parsing', 'loc': ['value'], 'msg': 'bad integer', 'input': 'x'}],
                'kind': 'validation_error',
            },
        ),
        (_ToolFailed('failed'), {'message': 'failed', 'kind': 'tool_failed'}),
    ],
)
def test_call_tool_result_json_payload_goldens(value: CallToolResult, expected: dict[str, Any]) -> None:
    assert JSON_CODEC.dump(CallToolResult, value) == expected
    assert IDENTITY_CODEC.dump(CallToolResult, value) is value
    assert IDENTITY_CODEC.load(CallToolResult, value) is value


@pytest.mark.parametrize(
    ('tp', 'value', 'expected'),
    [
        (
            ModelResponse,
            ModelResponse(parts=[TextPart('hello')]),
            {
                'parts': [
                    {
                        'content': 'hello',
                        'id': None,
                        'provider_name': None,
                        'provider_details': None,
                        'part_kind': 'text',
                    }
                ],
                'usage': {
                    'input_tokens': 0,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 0,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                    'cost': None,
                },
                'model_name': None,
                'timestamp': '2020-01-01T00:00:00Z',
                'kind': 'response',
                'provider_name': None,
                'provider_url': None,
                'provider_details': None,
                'provider_response_id': None,
                'finish_reason': None,
                'run_id': None,
                'conversation_id': None,
                'metadata': None,
                'state': 'complete',
            },
        ),
        (
            dict[str, ToolDefinition],
            {'tool': ToolDefinition(name='tool', description='Do it', parameters_json_schema={'type': 'object'})},
            {
                'tool': {
                    'name': 'tool',
                    'parameters_json_schema': {'type': 'object'},
                    'description': 'Do it',
                    'outer_typed_dict_key': None,
                    'strict': None,
                    'sequential': False,
                    'kind': 'function',
                    'metadata': None,
                    'timeout': None,
                    'defer_loading': False,
                    'unless_native': None,
                    'with_native': None,
                    'tool_kind': None,
                    'return_schema': None,
                    'include_return_schema': None,
                    'toolset_id': None,
                    'capability_id': None,
                }
            },
        ),
        (str | None, 'instructions', 'instructions'),
        (str | None, None, None),
        (type(None), None, None),
    ],
)
def test_json_and_identity_codec_payload_goldens(tp: Any, value: Any, expected: Any) -> None:
    if isinstance(value, ModelResponse):
        value.timestamp = value.timestamp.replace(year=2020, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    assert JSON_CODEC.dump(tp, value) == expected
    assert IDENTITY_CODEC.dump(tp, value) is value
    assert IDENTITY_CODEC.load(tp, value) is value


def test_capability_operation_result_payload_golden() -> None:
    delta = RunUsage(requests=1, tool_calls=2, input_tokens=3, details={'cached': 4})
    result = CapabilityOperationResult(5, usage_delta=delta)
    result_type = capability_operation_result_type(int)

    assert JSON_CODEC.dump(result_type, result) == {
        'value': 5,
        'usage_delta': {
            'input_tokens': 3,
            'cache_write_tokens': 0,
            'cache_read_tokens': 0,
            'output_tokens': 0,
            'input_audio_tokens': 0,
            'cache_audio_read_tokens': 0,
            'output_audio_tokens': 0,
            'details': {'cached': 4},
            'cost': None,
            'requests': 1,
            'tool_calls': 2,
        },
    }
    assert IDENTITY_CODEC.dump(result_type, result) is result


def test_model_request_context_projection_payload_golden() -> None:
    projection = ModelRequestContextProjection(
        [],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        model_id='restricted',
        streaming=False,
    )

    assert JSON_CODEC.dump(ModelRequestContextProjection, projection) == {
        'messages': [],
        'model_settings': None,
        'model_request_parameters': {
            'function_tools': [],
            'native_tools': [],
            'tool_visibility': None,
            'revealed_tool_names': [],
            'deferred_capability_ids': [],
            'output_mode': 'text',
            'output_object': None,
            'output_tools': [],
            'prompted_output_template': None,
            'allow_text_output': True,
            'allow_image_output': False,
            'instruction_parts': None,
            'thinking': None,
        },
        'model_id': 'restricted',
        'streaming': False,
    }


def test_pre_wrapper_tool_result_upgrade_paths() -> None:
    raw_payload = {'answer': 42}
    assert unwrap_recorded_tool_call_result(raw_payload) is raw_payload
    with pytest.raises(ValidationError):
        JSON_CODEC.load(CallToolResult, raw_payload)


async def test_validation_error_crosses_call_tool_result_boundary() -> None:
    async def invalid_tool_body() -> None:
        TypeAdapter(int).validate_python('invalid-from-tool-body')

    payload = await wrap_tool_call_result(invalid_tool_body())
    assert isinstance(payload, _ValidationError)
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)

    async def typed(value: int) -> None:
        pass

    toolset = FunctionToolset(tools=[typed])
    tool = (await toolset.get_tools(RunContext(deps=None, model=TestModel(), usage=RunUsage())))['typed']
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    payload = await wrap_tool_call_result(validate_tool_args(tool, {'value': 'not-an-int'}, ctx))
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)

    def non_serializable_input(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python(object())

    unsafe_toolset = FunctionToolset(tools=[Tool(typed, args_validator=non_serializable_input)])
    unsafe_tool = (await unsafe_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(unsafe_tool, {'value': 1}, ctx))
    dumped = JSON_CODEC.dump(CallToolResult, payload)
    sanitized = dumped['errors'][0]['input']
    assert sanitized['type'] == 'builtins.object'
    assert sanitized['repr'].startswith('<object object at 0x')

    class BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError('broken repr')

    def broken_repr_input(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python(BrokenRepr())

    broken_toolset = FunctionToolset(tools=[Tool(typed, args_validator=broken_repr_input)])
    broken_tool = (await broken_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(broken_tool, {'value': 1}, ctx))
    dumped = JSON_CODEC.dump(CallToolResult, payload)
    assert dumped['errors'][0]['input'] == {
        'type': f'{BrokenRepr.__module__}.{BrokenRepr.__qualname__}',
        'repr': '<repr failed>',
    }

    def invalid_args_validator(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python('invalid-from-args-validator')

    validated_toolset = FunctionToolset(tools=[Tool(typed, args_validator=invalid_args_validator)])
    validated_tool = (await validated_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(validated_tool, {'value': 1}, ctx))
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)

    async def invalid_async_args_validator(ctx: RunContext[None], value: int) -> None:
        TypeAdapter(int).validate_python('invalid-from-async-args-validator')

    async_validated_toolset = FunctionToolset(tools=[Tool(typed, args_validator=invalid_async_args_validator)])
    async_validated_tool = (await async_validated_toolset.get_tools(ctx))['typed']
    payload = await wrap_tool_call_result(run_args_validator(async_validated_tool, {'value': 1}, ctx))
    with pytest.raises(ValidationError, match='valid integer'):
        unwrap_tool_call_result(payload)
