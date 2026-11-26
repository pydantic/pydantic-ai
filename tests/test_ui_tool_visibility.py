"""Per-tool UI visibility: what the UI adapters send a client for each tool, live and from `dump_messages`.

These run the real adapters on a `FunctionModel`: what is under test is the adapters' projection of tool calls and
returns, which no provider response changes, so there is nothing for a cassette to record.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import pytest

from pydantic_ai import Agent, AgentRunResult, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, Capability, PrepareTools, WrapModelRequestHandler
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, SkipModelRequest, ToolFailed
from pydantic_ai.messages import (
    CustomEvent as PydanticAICustomEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, BuiltinToolCallsReturns, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, RunContext, Tool, ToolDefinition, ToolUIVisibility
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset

from ._inline_snapshot import snapshot
from .conftest import legacy_retry_prompt_part, try_import

with try_import() as imports_successful:
    from ag_ui.core import CustomEvent, RunAgentInput, UserMessage
    from starlette.requests import Request
    from starlette.responses import StreamingResponse

    from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream
    from pydantic_ai.ui.ag_ui._utils import ACTIVITY_EVENTS_VERSION, DEFAULT_AG_UI_VERSION, parse_ag_ui_version
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage
    from pydantic_ai.ui.vercel_ai.response_types import DataChunk

pytestmark = pytest.mark.skipif(not imports_successful(), reason='ag-ui-protocol or starlette not installed')

ProtocolName = Literal['ag_ui', 'vercel_ai']
PROTOCOLS: list[ProtocolName] = ['ag_ui', 'vercel_ai']

SECRET = 'internal-margin-42'
"""What the server-side tools return: it must never reach the client unless the tool is public."""


def calling(*calls: tuple[str, str]) -> FunctionModel:
    """A model whose first response makes `calls`, as `(tool name, JSON args)`, and whose next one answers `'Done.'`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and not any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield {
                index: DeltaToolCall(name=name, json_args=args, tool_call_id=f'call_{index}')
                for index, (name, args) in enumerate(calls)
            }
        else:
            yield 'Done.'

    return FunctionModel(stream_function=stream)


@dataclass
class Run:
    """What a client got from one adapter run, and the history the run produced."""

    protocol: ProtocolName
    events: list[dict[str, Any]]
    messages: list[ModelMessage]

    @property
    def wire(self) -> str:
        return json.dumps(self.events)

    def tool_events(self) -> list[dict[str, Any]]:
        """The run's tool events, with the fields this file asserts on."""
        if self.protocol == 'ag_ui':
            keep = ('type', 'toolCallId', 'toolCallName', 'delta', 'content', 'activityType', 'name', 'value')
            return [
                {key: event[key] for key in keep if key in event}
                for event in self.events
                if event['type'].startswith('TOOL_CALL') or event['type'] in ('ACTIVITY_SNAPSHOT', 'CUSTOM')
            ]
        keep = ('type', 'toolCallId', 'toolName', 'inputTextDelta', 'input', 'output', 'errorText', 'data')
        return [
            {key: event[key] for key in keep if key in event}
            for event in self.events
            if event['type'].startswith('tool-') or event['type'].startswith('data-')
        ]


async def run_adapter(
    protocol: ProtocolName,
    agent: Agent[None, Any],
    *,
    manage_message_history: Literal['client', 'server'] = 'server',
    **adapter_kwargs: Any,
) -> Run:
    """Run `agent` through the protocol's adapter for one user message, decoding the SSE wire it writes.

    The history is held on the server, as non-public tools need it to be.
    """
    adapter_kwargs['manage_message_history'] = manage_message_history
    messages: list[ModelMessage] = []

    async def keep_history(result: AgentRunResult[Any]) -> None:
        messages.extend(result.all_messages())

    if protocol == 'ag_ui':
        run_input = RunAgentInput(
            thread_id='thread_1',
            run_id='run_1',
            messages=[UserMessage(id='msg_1', content='Where is my order?')],
            state={},
            tools=[],
            context=[],
            forwarded_props={},
        )
        ag_ui_adapter = AGUIAdapter(agent, run_input=run_input, **adapter_kwargs)
        wire = [
            chunk async for chunk in ag_ui_adapter.encode_stream(ag_ui_adapter.run_stream(on_complete=keep_history))
        ]
    else:
        request = SubmitMessage(
            trigger='submit-message',
            id='chat_1',
            messages=[UIMessage(id='msg_1', role='user', parts=[TextUIPart(text='Where is my order?')])],
        )
        vercel_adapter = VercelAIAdapter(agent, run_input=request, sdk_version=6, **adapter_kwargs)
        wire = [
            chunk async for chunk in vercel_adapter.encode_stream(vercel_adapter.run_stream(on_complete=keep_history))
        ]

    events = [
        json.loads(data)
        for chunk in wire
        for line in chunk.splitlines()
        if line.startswith('data: ') and (data := line.removeprefix('data: ')) != '[DONE]'
    ]
    return Run(protocol, events, messages)


def dump(
    protocol: ProtocolName, messages: Sequence[ModelMessage], default_ui_visibility: ToolUIVisibility = 'public'
) -> str:
    """What a "load chat" endpoint would send for `messages`, as JSON."""
    if protocol == 'ag_ui':
        dumped = AGUIAdapter.dump_messages(messages, default_ui_visibility=default_ui_visibility)
    else:
        dumped = VercelAIAdapter.dump_messages(messages, default_ui_visibility=default_ui_visibility)
    return json.dumps([message.model_dump(mode='json', by_alias=True, exclude_none=True) for message in dumped])


def stamps(messages: Sequence[ModelMessage]) -> list[tuple[str, str, ToolUIVisibility | None]]:
    """Each tool part in `messages`, as `(part kind, tool name, ui_visibility stamp)`."""
    return [
        (part.part_kind, part.tool_name, part.ui_visibility)
        for message in messages
        for part in message.parts
        if isinstance(part, (ToolCallPart, ToolReturnPart, NativeToolCallPart, NativeToolReturnPart))
    ]


def secret_agent(ui_visibility: ToolUIVisibility | None, **agent_kwargs: Any) -> Agent[None, Any]:
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')), **agent_kwargs)

    @agent.tool_plain(ui_visibility=ui_visibility)
    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    return agent


EXPECTED_REDACTED_EVENTS: dict[ProtocolName, Any] = {
    'ag_ui': snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'lookup_order'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
            {'type': 'TOOL_CALL_RESULT', 'toolCallId': 'call_0', 'content': '[REDACTED]'},
        ]
    ),
    'vercel_ai': snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'lookup_order'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'lookup_order',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-output-available', 'toolCallId': 'call_0', 'output': '[REDACTED]'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_redacted_tool_keeps_the_call_and_replaces_the_result(protocol: ProtocolName):
    run = await run_adapter(protocol, secret_agent('redacted'))

    assert SECRET not in run.wire
    assert run.tool_events() == EXPECTED_REDACTED_EVENTS[protocol]
    assert SECRET not in dump(protocol, run.messages)
    assert '[REDACTED]' in dump(protocol, run.messages)

    # The stored history keeps the real return; only what goes to the client is projected.
    tool_return = next(part for part in run.messages[2].parts if isinstance(part, ToolReturnPart))
    assert tool_return.content == {'order_id': 'o_1', 'margin': SECRET}
    assert stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-return', 'lookup_order', 'redacted'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_private_tool_sends_neither_call_nor_result(protocol: ProtocolName):
    run = await run_adapter(protocol, secret_agent('private'))

    assert 'lookup_order' not in run.wire
    assert 'call_0' not in run.wire
    assert run.tool_events() == []
    dumped = dump(protocol, run.messages)
    assert 'lookup_order' not in dumped
    assert 'Done.' in dumped
    assert stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'private'),
        ('tool-return', 'lookup_order', 'private'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_public_tool_is_unchanged(protocol: ProtocolName):
    """A tool with no setting under the default adapter is sent in full, and its parts stay unstamped."""
    run = await run_adapter(protocol, secret_agent(None))

    assert SECRET in run.wire
    assert SECRET in dump(protocol, run.messages)
    assert stamps(run.messages) == [('tool-call', 'lookup_order', None), ('tool-return', 'lookup_order', None)]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_adapter_default_applies_to_tools_without_a_setting(protocol: ProtocolName):
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}'), ('order_status', '{"order_id": "o_1"}')))

    @agent.tool_plain
    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    @agent.tool_plain(ui_visibility='public')
    def order_status(order_id: str) -> str:
        return 'shipped'

    run = await run_adapter(protocol, agent, default_ui_visibility='redacted')

    assert SECRET not in run.wire
    assert 'shipped' in run.wire
    # The default is written into the stamp, so a reload without it projects the same way.
    assert stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-call', 'order_status', 'public'),
        ('tool-return', 'lookup_order', 'redacted'),
        ('tool-return', 'order_status', 'public'),
    ]
    dumped = dump(protocol, run.messages)
    assert SECRET not in dumped
    assert 'shipped' in dumped


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_dump_reads_the_stamp_not_the_tool_definitions(protocol: ProtocolName):
    """A reload projects from the stamps alone: the tool can be renamed or removed, and a dump default doesn't override a stamp."""
    run = await run_adapter(protocol, secret_agent('redacted'))

    assert SECRET not in dump(protocol, run.messages)
    assert 'lookup_order' in dump(protocol, run.messages, default_ui_visibility='private')

    # The stamps survive the stored history's own round trip.
    stored = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(run.messages))
    assert stamps(stored) == stamps(run.messages)
    assert SECRET not in dump(protocol, stored)

    # The protocols don't carry them, so a history the client sends back has none, and only the placeholder.
    if protocol == 'ag_ui':
        client_copy = AGUIAdapter.load_messages(AGUIAdapter.dump_messages(run.messages))
    else:
        client_copy = VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(run.messages))
    assert [stamp for *_, stamp in stamps(client_copy)] == [None, None]
    assert SECRET not in dump(protocol, client_copy)

    # History with no stamps, such as one from an earlier version, takes the dump's default.
    unstamped = (await run_adapter(protocol, secret_agent(None))).messages
    assert SECRET in dump(protocol, unstamped)
    assert SECRET not in dump(protocol, unstamped, default_ui_visibility='redacted')
    assert 'lookup_order' not in dump(protocol, unstamped, default_ui_visibility='private')


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_external_and_output_tools_stay_public(protocol: ProtocolName):
    """Results of external (frontend) tools come from the client, and output tool args are the run's answer."""
    frontend = ExternalToolset(
        [ToolDefinition(name='show_map', parameters_json_schema={'type': 'object', 'properties': {}})]
    )
    agent = Agent(
        calling(('show_map', '{}')),
        toolsets=[frontend],
        output_type=[ToolOutput(str, name='final_result'), DeferredToolRequests],
    )

    run = await run_adapter(protocol, agent, default_ui_visibility='private')

    assert 'show_map' in run.wire
    assert stamps(run.messages) == [('tool-call', 'show_map', 'public')]

    answer = Agent(calling(('final_result', '{"response": "It shipped."}')), output_type=ToolOutput(str))
    answered = await run_adapter(protocol, answer, default_ui_visibility='private')

    assert 'It shipped.' in answered.wire
    assert stamps(answered.messages) == [
        ('tool-call', 'final_result', 'public'),
        ('tool-return', 'final_result', 'public'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_approval_required_private_tool_is_shown_redacted(protocol: ProtocolName):
    """The user can't approve a call they can't see, so a `'private'` tool that needs approval is `'redacted'`."""
    agent = Agent(calling(('refund', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True, ui_visibility='private')
    def refund(order_id: str) -> str:
        return SECRET  # pragma: no cover

    run = await run_adapter(protocol, agent)

    assert 'refund' in run.wire
    assert stamps(run.messages) == [('tool-call', 'refund', 'redacted')]
    assert 'refund' in dump(protocol, run.messages)


DEFERRALS: dict[str, Callable[[], CallDeferred | ApprovalRequired]] = {
    'CallDeferred': CallDeferred,
    'ApprovalRequired': ApprovalRequired,
    'question': lambda: CallDeferred(kind='question'),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('deferral', list(DEFERRALS))
async def test_private_tool_that_defers_at_runtime_is_shown_redacted(protocol: ProtocolName, deferral: str):
    """The call was hidden when it streamed; once it defers, the client has to see what it is answering."""
    agent = Agent(calling(('escalate', '{"order_id": "o_1"}')), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(ui_visibility='private')
    def escalate(order_id: str) -> str:
        raise DEFERRALS[deferral]()

    run = await run_adapter(protocol, agent)

    assert run.tool_events() == EXPECTED_DEFERRED_EVENTS[(protocol, deferral)]
    # The response in history holds a `'redacted'` copy of the call; the part that was streamed is left as it was.
    assert stamps(run.messages) == [('tool-call', 'escalate', 'redacted')]
    assert 'escalate' in dump(protocol, run.messages)


EXPECTED_DEFERRED_EVENTS: dict[tuple[ProtocolName, str], Any] = {
    ('ag_ui', 'question'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    ('vercel_ai', 'question'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
        ]
    ),
    ('ag_ui', 'CallDeferred'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    ('ag_ui', 'ApprovalRequired'): snapshot(
        [
            {'type': 'TOOL_CALL_START', 'toolCallId': 'call_0', 'toolCallName': 'escalate'},
            {'type': 'TOOL_CALL_ARGS', 'toolCallId': 'call_0', 'delta': '{"order_id": "o_1"}'},
            {'type': 'TOOL_CALL_END', 'toolCallId': 'call_0'},
        ]
    ),
    ('vercel_ai', 'CallDeferred'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
        ]
    ),
    ('vercel_ai', 'ApprovalRequired'): snapshot(
        [
            {'type': 'tool-input-start', 'toolCallId': 'call_0', 'toolName': 'escalate'},
            {'type': 'tool-input-delta', 'toolCallId': 'call_0', 'inputTextDelta': '{"order_id": "o_1"}'},
            {
                'type': 'tool-input-available',
                'toolCallId': 'call_0',
                'toolName': 'escalate',
                'input': {'order_id': 'o_1'},
            },
            {'type': 'tool-approval-request', 'toolCallId': 'call_0'},
        ]
    ),
}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_redacted_tool_drops_the_mcp_app_and_keeps_metadata_events(protocol: ProtocolName):
    """The MCP Apps record holds the full result, so it goes; events the tool put in its metadata are its own channel."""
    agent = Agent(calling(('forecast', '{"city": "Paris"}'), ('notify', '{}')))
    app: dict[str, Any] = {
        'resource_uri': 'ui://weather/forecast',
        'tool_input': {'city': 'Paris'},
        'result': {'content': [{'type': 'text', 'text': SECRET}]},
        'toolset_id': 'weather',
    }

    @agent.tool_plain(ui_visibility='redacted')
    def forecast(city: str) -> ToolReturn:
        return ToolReturn(return_value=SECRET, metadata={'mcp_app': app})

    event: CustomEvent | DataChunk = (
        CustomEvent(name='progress', value='notified')
        if protocol == 'ag_ui'
        else DataChunk(type='data-progress', data='notified')
    )

    @agent.tool_plain(ui_visibility='redacted')
    def notify() -> ToolReturn:
        return ToolReturn(return_value=SECRET, metadata=event)

    run = await run_adapter(protocol, agent)

    assert SECRET not in run.wire
    assert 'ui://weather/forecast' not in run.wire
    assert 'notified' in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert 'ui://weather/forecast' not in dump(protocol, run.messages)


@dataclass(kw_only=True)
class StockChecked(PydanticAICustomEvent, name='stock_checked'):
    """What a `'private'` tool tells the client while it runs."""

    in_stock: int


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_private_tool_sends_no_metadata_events_and_can_emit_instead(protocol: ProtocolName):
    """A `'private'` result isn't sent, so neither are the events in its `metadata`; `ctx.emit` still reaches the client."""
    agent = Agent(calling(('check_stock', '{}')))
    event: CustomEvent | DataChunk = (
        CustomEvent(name='progress', value='from-metadata')
        if protocol == 'ag_ui'
        else DataChunk(type='data-progress', data='from-metadata')
    )

    @agent.tool(ui_visibility='private')
    async def check_stock(ctx: RunContext[object]) -> ToolReturn:
        await ctx.emit(StockChecked(in_stock=7))
        return ToolReturn(return_value=SECRET, metadata=event)

    run = await run_adapter(protocol, agent)

    assert 'from-metadata' not in run.wire
    assert 'from-metadata' not in dump(protocol, run.messages)
    assert 'stock_checked' in run.wire
    assert 'call_0' not in run.wire


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('ui_visibility', ['redacted', 'private'])
async def test_tool_return_content_is_a_user_message_visibility_does_not_cover(
    protocol: ProtocolName, ui_visibility: ToolUIVisibility
):
    """`ui_visibility` covers the call and its return value; `ToolReturn.content` is a separate user message.

    The model gets it as one, and `dump_messages` sends it as one, so data the client must not see belongs in the
    return value. See https://github.com/pydantic/pydantic-ai/issues/6662#issuecomment-5443082639.
    """
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility=ui_visibility)
    def lookup_order(order_id: str) -> ToolReturn:
        return ToolReturn(return_value=SECRET, content='Order o_1 is on its way.')

    run = await run_adapter(protocol, agent)

    assert SECRET not in run.wire
    # The adapters' streams never send a result event's `content`, for any tool.
    assert 'Order o_1 is on its way.' not in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert 'Order o_1 is on its way.' in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_tool_parts_without_a_definition_follow_the_adapter_default(protocol: ProtocolName):
    """A call to an unknown tool, and a native tool's parts, have no definition to take a setting from."""

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[BuiltinToolCallsReturns | DeltaToolCalls | str]:
        last = messages[-1]
        if isinstance(last, ModelRequest) and not any(isinstance(part, ToolReturnPart) for part in last.parts):
            yield {
                0: NativeToolCallPart(
                    tool_name='web_search', args={'query': 'o_1'}, tool_call_id='native_1', provider_name='function'
                )
            }
            yield {
                1: NativeToolReturnPart(
                    tool_name='web_search', content=SECRET, tool_call_id='native_1', provider_name='function'
                )
            }
            yield {2: DeltaToolCall(name='made_up', json_args='{}', tool_call_id='call_0')}
        else:
            yield 'Done.'

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain(ui_visibility='private')
    def internal_lookup() -> str:
        return SECRET  # pragma: no cover

    run = await run_adapter(protocol, agent, default_ui_visibility='redacted')

    assert SECRET not in run.wire
    # The retry for the unknown tool lists the available tools, a private one among them.
    assert 'internal_lookup' not in run.wire
    assert stamps(run.messages) == [
        ('builtin-tool-call', 'web_search', 'redacted'),
        ('builtin-tool-return', 'web_search', 'redacted'),
        ('tool-call', 'made_up', 'redacted'),
        ('tool-return', 'made_up', 'redacted'),
    ]
    assert 'internal_lookup' not in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_client_held_history_warns_about_non_public_tools(protocol: ProtocolName):
    """A client that only saw the placeholder would send it back as the result."""
    with pytest.warns(UserWarning, match="`manage_message_history='server'`") as record:
        await run_adapter(
            protocol, secret_agent(None), manage_message_history='client', default_ui_visibility='redacted'
        )
    assert len(record) == 1

    with pytest.warns(UserWarning, match="`manage_message_history='server'`") as record:
        await run_adapter(protocol, secret_agent('private'), manage_message_history='client')
    assert len(record) == 1

    # Any warning fails the test (`filterwarnings = ['error']`).
    await run_adapter(protocol, secret_agent('private'))
    await run_adapter(protocol, secret_agent(None), manage_message_history='client')


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_client_held_history_warns_when_an_interrupted_redacted_call_is_closed(protocol: ProtocolName):
    """A redacted call is sent in full, so the first thing hidden can be the placeholder that closes it on an error."""
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility='redacted')
    def lookup_order(order_id: str) -> str:
        raise RuntimeError('The order service is down.')

    with pytest.warns(UserWarning, match="`manage_message_history='server'`") as record:
        run = await run_adapter(protocol, agent, manage_message_history='client')
    assert len(record) == 1
    assert '[REDACTED]' in run.wire


ENQUEUED_RECORDS: dict[ProtocolName, str] = {
    'ag_ui': 'pydantic_ai_enqueued_messages',
    'vercel_ai': 'data-enqueued-messages',
}
"""The event each protocol streams messages a tool enqueues in."""


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_enqueued_tool_exchange_follows_the_stream_default(protocol: ProtocolName):
    """A tool exchange a tool enqueues has no definition to be stamped from, so the stream's default decides it."""
    if protocol == 'ag_ui' and parse_ag_ui_version(DEFAULT_AG_UI_VERSION) < ACTIVITY_EVENTS_VERSION:
        pytest.skip('this ag-ui-protocol has no activity events')  # pragma: lax no cover

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, ToolReturnPart) and part.tool_name == 'escalate' for m in messages for part in m.parts):
            yield 'Done.'
        else:
            yield {0: DeltaToolCall(name='escalate', json_args='{}', tool_call_id='call_0')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool
    def escalate(ctx: RunContext[object]) -> str:
        ctx.enqueue(
            ModelResponse(parts=[ToolCallPart('lookup_notes', {}, tool_call_id='notes_0')]),
            ModelRequest(parts=[ToolReturnPart('lookup_notes', SECRET, tool_call_id='notes_0')]),
        )
        return 'Escalated.'

    run = await run_adapter(protocol, agent, default_ui_visibility='private')

    assert ENQUEUED_RECORDS[protocol] in run.wire
    assert 'Escalated.' not in run.wire
    assert SECRET not in run.wire


async def test_standalone_stream_closes_no_output_call_it_hid():
    """An output call a standalone stream's non-public default hid has nothing to close when the run fails."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name='final_result', json_args='{"answer": ', tool_call_id='call_0')}
        raise RuntimeError('The model went away.')

    agent = Agent(FunctionModel(stream_function=stream), output_type=dict[str, str])
    event_stream = VercelAIEventStream[None, Any](default_ui_visibility='private')
    async with agent.run_stream_events('Where is my order?') as events:
        chunks = [chunk async for chunk in event_stream.encode_stream(event_stream.transform_stream(events))]

    wire = ''.join(chunks)
    assert 'The model went away.' in wire
    assert 'final_result' not in wire


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('prefix_tool', ['public', 'default_redacted'])
async def test_tool_name_streamed_in_pieces_is_stamped_from_the_complete_name(
    protocol: ProtocolName, prefix_tool: Literal['public', 'default_redacted']
):
    """A name that first matches another tool, `lookup` for `lookup_secret`, takes the complete name's stamp.

    The call has already started under the shorter name by then, which the docs call out; its result never goes out.
    """

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(part, ToolReturnPart) for m in messages for part in m.parts):
            yield 'Done.'
        else:
            yield {0: DeltaToolCall(name='lookup', tool_call_id='call_0')}
            yield {0: DeltaToolCall(name='_secret')}
            yield {0: DeltaToolCall(json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain(ui_visibility='public' if prefix_tool == 'public' else None)
    def lookup() -> str:
        return 'Nothing to see.'  # pragma: no cover

    @agent.tool_plain(ui_visibility='private')
    def lookup_secret() -> str:
        return SECRET

    default: ToolUIVisibility = 'redacted' if prefix_tool == 'default_redacted' else 'public'
    run = await run_adapter(protocol, agent, default_ui_visibility=default)

    assert stamps(run.messages) == [
        ('tool-call', 'lookup_secret', 'private'),
        ('tool-return', 'lookup_secret', 'private'),
    ]
    assert SECRET not in run.wire
    assert 'lookup_secret' not in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert 'lookup_secret' not in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('ui_visibility', ['redacted', 'private'])
@pytest.mark.parametrize('failure', [ToolFailed, ModelRetry])
async def test_failure_a_tool_reports_follows_its_visibility(
    protocol: ProtocolName, ui_visibility: ToolUIVisibility, failure: type[ToolFailed | ModelRetry]
):
    """A `ToolFailed` or `ModelRetry` message is the tool's result, so it stays off the stream and out of the dump.

    A `'redacted'` tool's failure still goes out, with the placeholder as its error text.
    """
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}')))

    @agent.tool_plain(ui_visibility=ui_visibility)
    def lookup_order(order_id: str) -> str:
        raise failure(SECRET)

    run = await run_adapter(protocol, agent)

    assert 'Done.' in run.wire
    assert SECRET not in run.wire
    assert SECRET not in dump(protocol, run.messages)
    assert ('[REDACTED]' in run.wire) == (ui_visibility == 'redacted')


async def test_standalone_event_stream_default_applies_only_to_unstamped_parts():
    """Without an adapter, nothing writes the default into the stamps, so only unstamped parts take it.

    That includes calls to external and output tools, which is why the adapter, not the stream, applies the default
    to tool definitions: under `'private'`, an output tool's call isn't sent, and a frontend tool's call is sent only
    once the run defers it, since the client has to run it.
    """
    agent = Agent(calling(('lookup_order', '{"order_id": "o_1"}'), ('order_status', '{"order_id": "o_1"}')))

    @agent.tool_plain
    def lookup_order(order_id: str) -> str:
        return SECRET

    @agent.tool_plain(ui_visibility='public')
    def order_status(order_id: str) -> str:
        return 'shipped'

    stream = AGUIEventStream[None, Any](default_ui_visibility='private')
    async with agent.run_stream_events('Where is my order?') as events:
        wire = [stream.encode_event(event) async for event in stream.transform_stream(events)]

    assert 'lookup_order' not in ''.join(wire)
    assert SECRET not in ''.join(wire)
    assert 'shipped' in ''.join(wire)

    frontend = ExternalToolset(
        [ToolDefinition(name='show_map', parameters_json_schema={'type': 'object', 'properties': {}})]
    )
    external = Agent(calling(('show_map', '{}')), toolsets=[frontend], output_type=[str, DeferredToolRequests])
    output = Agent(calling(('final_result', '{"response": "It shipped."}')), output_type=ToolOutput(str))
    edge_wires: dict[str, str] = {}
    for name, edge_agent in (('external', external), ('output', output)):
        edge_stream = AGUIEventStream[None, Any](default_ui_visibility='private')
        async with edge_agent.run_stream_events('Where is my order?') as events:
            edge_wires[name] = ''.join(
                [edge_stream.encode_event(event) async for event in edge_stream.transform_stream(events)]
            )

    assert 'show_map' in edge_wires['external']
    assert 'final_result' not in edge_wires['output']
    assert 'It shipped.' not in edge_wires['output']


def test_ui_visibility_flows_from_each_tool_setter_into_the_tool_definition() -> None:
    """A unit test: `ToolDefinition.ui_visibility` never reaches the provider, so a recorded request can't observe it."""

    def plain() -> str:
        return 'never called'  # pragma: no cover

    def with_ctx(ctx: RunContext[None]) -> str:
        return 'never called'  # pragma: no cover

    toolset = FunctionToolset[None](ui_visibility='private')
    toolset.tool_plain(plain)
    toolset.tool(ui_visibility='public')(with_ctx)
    toolset.add_tool(Tool(plain, name='as_tool'))
    toolset.add_tool(Tool(plain, name='kept', ui_visibility='redacted'))
    toolset.add_function(plain, name='as_function', ui_visibility='redacted')
    toolset.add_function(plain, name='function_default')
    assert {name: tool.tool_def.ui_visibility for name, tool in toolset.tools.items()} == {
        'plain': 'private',
        'with_ctx': 'public',
        'as_tool': 'private',
        'kept': 'redacted',
        'as_function': 'redacted',
        'function_default': 'private',
    }
    assert FunctionToolset[None]([plain]).tools['plain'].tool_def.ui_visibility is None

    seen: dict[str, ToolUIVisibility | None] = {}

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.update({tool_def.name: tool_def.ui_visibility for tool_def in info.function_tools})
        return ModelResponse(parts=[TextPart('Done.')])

    agent = Agent(FunctionModel(answer), deps_type=type(None))
    agent.tool_plain(ui_visibility='private')(plain)
    agent.tool(ui_visibility='redacted')(with_ctx)
    capability = Capability[None]()
    capability.tool_plain(ui_visibility='private')(plain)
    capability.tool(ui_visibility='redacted')(with_ctx)
    for tooled in (agent, Agent(FunctionModel(answer), deps_type=type(None), capabilities=[capability])):
        seen.clear()
        tooled.run_sync('Hi')
        assert seen == {'plain': 'private', 'with_ctx': 'redacted'}


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_prepare_tools_sets_visibility_in_bulk(protocol: ProtocolName):
    """`PrepareTools` reaches every tool, including toolset and MCP tools the decorators don't."""

    def redact_lookups(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [replace(td, ui_visibility='redacted') if td.name.startswith('lookup_') else td for td in tool_defs]

    agent = secret_agent(None, capabilities=[PrepareTools(redact_lookups)])

    run = await run_adapter(protocol, agent)

    assert SECRET not in run.wire
    assert stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'redacted'),
        ('tool-return', 'lookup_order', 'redacted'),
    ]


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_dispatch_request_passes_the_default_to_the_adapter(protocol: ProtocolName):
    """`dispatch_request` hands `default_ui_visibility` to `from_request`, which sets it on the adapter it builds."""
    if protocol == 'ag_ui':
        adapter_type: type[AGUIAdapter[None, Any] | VercelAIAdapter[None, Any]] = AGUIAdapter
        body = RunAgentInput(
            thread_id='thread_1',
            run_id='run_1',
            messages=[UserMessage(id='msg_1', content='Where is my order?')],
            state={},
            tools=[],
            context=[],
            forwarded_props={},
        ).model_dump_json(by_alias=True)
    else:
        adapter_type = VercelAIAdapter
        body = SubmitMessage(
            trigger='submit-message',
            id='chat_1',
            messages=[UIMessage(id='msg_1', role='user', parts=[TextUIPart(text='Where is my order?')])],
        ).model_dump_json(by_alias=True)

    async def receive() -> dict[str, Any]:
        return {'type': 'http.request', 'body': body.encode()}

    request = Request(
        scope={'type': 'http', 'method': 'POST', 'headers': [(b'content-type', b'application/json')]},
        receive=receive,
    )
    response = await adapter_type.dispatch_request(
        request, agent=secret_agent(None), manage_message_history='server', default_ui_visibility='redacted'
    )

    assert isinstance(response, StreamingResponse)
    wire = ''.join(
        [chunk if isinstance(chunk, str) else bytes(chunk).decode() async for chunk in response.body_iterator]
    )
    assert 'lookup_order' in wire
    assert SECRET not in wire


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('ui_visibility', ['redacted', 'private'])
def test_dump_projects_a_legacy_retry_as_its_calls_result(protocol: ProtocolName, ui_visibility: ToolUIVisibility):
    """A deprecated `RetryPromptPart` bound to a call is that call's result, so it takes the call's stamp.

    A unit test: no run makes one any more, and a stored one is translated when it loads.
    """
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Where is my order?'),
        ModelResponse(parts=[ToolCallPart('lookup_order', {}, tool_call_id='call_0', ui_visibility=ui_visibility)]),
        ModelRequest(parts=[legacy_retry_prompt_part(SECRET, tool_name='lookup_order', tool_call_id='call_0')]),
    ]

    assert SECRET not in dump(protocol, messages)


def test_unknown_visibility_raises_rather_than_sends_the_result():
    """A value outside `ToolUIVisibility`, which only code that ignores its types can set, never reads as `'public'`."""
    call = ToolCallPart('lookup_order', {}, tool_call_id='call_0', ui_visibility='Private')  # pyright: ignore[reportArgumentType]

    with pytest.raises(AssertionError):
        dump('vercel_ai', [ModelResponse(parts=[call])])


def _lookup_call() -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('lookup_order', {'order_id': 'o_1'}, tool_call_id='call_0')])


@dataclass
class AnswersForTheModel(AbstractCapability[Any]):
    """Answers the first request with a call to `lookup_order` instead of the model, the way `path` says."""

    path: Literal['wrap', 'skip', 'recover']

    async def wrap_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        if self.path == 'recover' or any(
            isinstance(part, ToolReturnPart) for message in request_context.messages for part in message.parts
        ):
            return await handler(request_context)
        if self.path == 'skip':
            raise SkipModelRequest(_lookup_call())
        return _lookup_call()

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        return _lookup_call()


def answered_agent(path: Literal['wrap', 'skip', 'recover']) -> Agent[None, Any]:
    """A `'private'` `lookup_order` whose call comes from `AnswersForTheModel`; the model only fails, then answers."""

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            raise RuntimeError('The model is down.')
        return ModelResponse(parts=[TextPart('Done.')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        answer(messages, info)
        yield 'Done.'

    agent = Agent(FunctionModel(answer, stream_function=stream), capabilities=[AnswersForTheModel(path)])

    @agent.tool_plain(ui_visibility='private')
    def lookup_order(order_id: str) -> dict[str, str]:
        return {'order_id': order_id, 'margin': SECRET}

    return agent


ANSWER_PATHS: list[Literal['wrap', 'skip', 'recover']] = ['wrap', 'skip', 'recover']


@pytest.mark.parametrize('protocol', PROTOCOLS)
@pytest.mark.parametrize('path', ANSWER_PATHS)
async def test_response_a_capability_supplies_is_stamped(
    protocol: ProtocolName, path: Literal['wrap', 'skip', 'recover']
):
    """A call in a response a capability gives instead of the model's is stamped from its tool like the model's own."""
    run = await run_adapter(protocol, answered_agent(path))

    assert stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'private'),
        ('tool-return', 'lookup_order', 'private'),
    ]
    assert 'o_1' not in run.wire
    assert SECRET not in run.wire
    assert SECRET not in dump(protocol, run.messages)


@pytest.mark.parametrize('protocol', PROTOCOLS)
async def test_recovered_response_keeps_its_tools_setting_under_a_non_public_default(protocol: ProtocolName):
    """A recovered response is stamped before `after_model_request`, so the adapter's default can't write over it."""
    run = await run_adapter(protocol, answered_agent('recover'), default_ui_visibility='redacted')

    assert stamps(run.messages) == [
        ('tool-call', 'lookup_order', 'private'),
        ('tool-return', 'lookup_order', 'private'),
    ]
    assert 'o_1' not in run.wire


@pytest.mark.parametrize('path', ANSWER_PATHS)
async def test_response_a_capability_supplies_is_stamped_without_streaming(path: Literal['wrap', 'skip', 'recover']):
    """A run that doesn't stream stamps the supplied response too, so a later dump with no default leaves it out."""
    result = await answered_agent(path).run('Where is my order?')

    assert SECRET not in dump('vercel_ai', result.all_messages())


def test_dump_drops_a_response_left_empty_by_private_parts():
    """A response that only held a private call leaves no assistant message behind."""
    messages: list[ModelMessage] = [
        ModelRequest.user_text_prompt('Where is my order?'),
        ModelResponse(parts=[ToolCallPart('lookup_order', {'order_id': 'o_1'}, 'call_0', ui_visibility='private')]),
        ModelRequest(parts=[ToolReturnPart('lookup_order', SECRET, 'call_0', ui_visibility='private')]),
        ModelResponse(parts=[TextPart('Done.')]),
    ]
    for protocol in PROTOCOLS:
        dumped = json.loads(dump(protocol, messages))
        assert [message['role'] for message in dumped] == ['user', 'assistant']
        assert 'Done.' in json.dumps(dumped[-1])
        assert 'lookup_order' not in json.dumps(dumped)
