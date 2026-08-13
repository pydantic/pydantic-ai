"""Tests for the availability gate: a tool cannot be called unless it is available.

A deferred tool must have been revealed, and a capability-owned tool's capability loaded,
before a call is allowed. Split out of `test_capabilities.py`, which is at the 1 MB
`check-added-large-files` cap (#7304); availability is one coherent subject, so it earns
its own module rather than an arbitrary slice.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    Capability,
    ProcessHistory,
)
from pydantic_ai.exceptions import (
    ModelRetry,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    CompactionPart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._deferred_capability_loader import (
    LOAD_CAPABILITY_TOOL_NAME,
)

from ._inline_snapshot import snapshot
from .capability_models import (
    make_text_response,
)
from .conftest import iter_message_parts

_INVALID_WIRE_BOUNDARIES = [
    pytest.param(CompactionPart(content='foreign', provider_name='anthropic'), 'openai', id='foreign-provider'),
    pytest.param(CompactionPart(provider_name='openai'), 'openai', id='openai-without-encrypted-content'),
    pytest.param(CompactionPart(provider_name='anthropic'), 'anthropic', id='anthropic-without-content'),
]


def _provider_response(parts: list[Any], provider_name: str | None) -> ModelResponse:
    """Build a response whose explicit provenance drives the retrospective evidence window."""
    return ModelResponse(parts=parts, provider_name=provider_name)


@pytest.mark.parametrize(('boundary', 'provider_name'), _INVALID_WIRE_BOUNDARIES)
async def test_deferred_tool_call_uses_serving_providers_wire_window(boundary: CompactionPart, provider_name: str):
    """A boundary the serving provider skipped cannot erase deferred-tool callability evidence.

    The test explicitly stamps `ModelResponse.provider_name`; a bare `FunctionModel` would exercise
    the missing-provenance fallback instead.
    """
    calls = 0
    advertised: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        advertised.append([tool.name for tool in info.function_tools])
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'hidden' for msg in messages for part in msg.parts
        ):
            return _provider_response([make_text_response('done').parts[0]], provider_name)
        return _provider_response([ToolCallPart(tool_name='hidden', args={}, tool_call_id='h1')], provider_name)

    toolset = FunctionToolset[Any]()
    toolset.add_function(lambda: ToolReturn(return_value='ran', tools=['hidden']), name='hidden', defer_loading=True)
    agent = Agent(FunctionModel(model_fn), toolsets=[toolset])
    history = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden'])]),
        ModelResponse(parts=[boundary]),
    ]

    result = await agent.run('go', message_history=history)

    assert result.output == 'done'
    assert calls == 2
    # The retrospective supplement did not widen the shared prospective set: the tool was still
    # absent from the serving request, its explicit re-disclosure survived pruning, and only the
    # following request advertised it.
    assert 'hidden' not in advertised[0]
    assert 'hidden' in advertised[1]
    assert any(
        isinstance(part, ToolAvailabilityDeltaPart) and part.tools_added == ['hidden']
        for message in result.all_messages()
        for part in message.parts
    )


@pytest.mark.parametrize(('boundary', 'provider_name'), _INVALID_WIRE_BOUNDARIES)
async def test_capability_tool_call_uses_serving_providers_wire_window(boundary: CompactionPart, provider_name: str):
    """The provider-exact supplement covers capability load and tool discovery independently.

    The test explicitly stamps `ModelResponse.provider_name`; a bare `FunctionModel` would exercise
    the missing-provenance fallback instead.
    """

    def guarded_tool() -> str:
        return 'ran'

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'guarded_tool'
            for msg in messages
            for part in msg.parts
        ):
            return _provider_response([make_text_response('done').parts[0]], provider_name)
        return _provider_response([ToolCallPart(tool_name='guarded_tool', args={}, tool_call_id='g1')], provider_name)

    capability = Capability[Any](
        id='guarded', description='Guarded.', toolsets=[FunctionToolset([guarded_tool])], defer_loading=True
    )
    agent = Agent(FunctionModel(model_fn), capabilities=[capability])
    history = [
        ModelResponse(
            parts=[LoadCapabilityCallPart(args={'id': 'guarded'}, tool_call_id='load')], provider_name=provider_name
        ),
        ModelRequest(
            parts=[
                LoadCapabilityReturnPart(content={}, tool_call_id='load'),
                ToolAvailabilityDeltaPart(tools_added=['guarded_tool']),
            ]
        ),
        ModelResponse(parts=[boundary]),
    ]

    result = await agent.run('go', message_history=history)

    assert result.output == 'done'


async def test_compaction_inside_serving_response_does_not_reset_tool_evidence():
    """Anthropic may compact mid-response; explicit provenance exercises the strict-before rule."""
    toolset = FunctionToolset[Any]()
    toolset.add_function(lambda: 'ran', name='hidden', defer_loading=True)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(
            isinstance(part, ToolReturnPart) and part.tool_name == 'hidden' for msg in messages for part in msg.parts
        ):
            return _provider_response([make_text_response('done').parts[0]], 'anthropic')
        return _provider_response(
            [
                CompactionPart(content='summary', provider_name='anthropic'),
                ToolCallPart(tool_name='hidden', args={}, tool_call_id='h1'),
            ],
            'anthropic',
        )

    result = await Agent(FunctionModel(model_fn), toolsets=[toolset]).run(
        'go', message_history=[ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden'])])]
    )

    assert result.output == 'done'


async def test_missing_provider_name_uses_agnostic_window():
    """An unstamped `FunctionModel` response has `provider_name=None`, so the agnostic cut wins."""
    toolset = FunctionToolset[Any]()
    toolset.add_function(lambda: 'ran', name='hidden', defer_loading=True)
    history = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden'])]),
        ModelResponse(parts=[CompactionPart(content='summary', provider_name='anthropic')]),
    ]

    def call_hidden(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in iter_message_parts(messages, ModelRequest, RetryPromptPart):
            return make_text_response(str(part.content))
        return ModelResponse(parts=[ToolCallPart(tool_name='hidden', args={}, tool_call_id='h1')])

    result = await Agent(FunctionModel(call_hidden), toolsets=[toolset]).run('go', message_history=history)

    assert 'is not available yet' in result.output


async def test_boundary_the_serving_provider_honored_still_hides_evidence():
    """Anchoring makes the window exact, not permissive.

    The counterpart to the cases above: when the serving provider is the one that emitted the
    boundary and the payload it renders is there, the request really did start over, so the
    pre-boundary reveal is gone from the model's view and the call has to be refused.
    """
    toolset = FunctionToolset[Any]()
    toolset.add_function(lambda: 'ran', name='hidden', defer_loading=True)
    history = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden'])]),
        ModelResponse(parts=[CompactionPart(content='summary', provider_name='anthropic')], provider_name='anthropic'),
    ]

    def call_hidden(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in iter_message_parts(messages, ModelRequest, RetryPromptPart):
            return _provider_response([make_text_response(str(part.content)).parts[0]], 'anthropic')
        return _provider_response([ToolCallPart(tool_name='hidden', args={}, tool_call_id='h1')], 'anthropic')

    result = await Agent(FunctionModel(call_hidden), toolsets=[toolset]).run('go', message_history=history)

    assert 'is not available yet' in result.output


def secret_op() -> str:
    """A capability-owned tool an owner's `prepare_tools` filters out."""
    return 'SECRET EXECUTED'


def _report_secret_op_outcome(messages: list[ModelMessage]) -> ModelResponse | None:
    """Report the blocked call, once the retry prompt refusing it is in history.

    `secret_op` must not execute in any test that routes through here — that's the point — so a
    tool return for it would itself be the failure. The body is deliberately *not* marked
    unreachable: `test_loaded_capability_tool_survives_a_stripped_reveal_marker` runs it on
    purpose, since proving the tool is callable again is the whole assertion. These tests catch an
    unwanted execution by asserting on the refusal instead.
    """
    for part in iter_message_parts(messages, ModelRequest, RetryPromptPart):
        return make_text_response(f'BLOCKED: {part.content}')
    return None


def _call_secret_op(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Directly call `secret_op` without it ever being offered."""
    return _report_secret_op_outcome(messages) or ModelResponse(
        parts=[ToolCallPart(tool_name='secret_op', args={}, tool_call_id='s1')]
    )


def _load_compact_then_call_secret_op(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load `guarded`, emit a `CompactionPart` to reset that state, then direct-call its tool."""
    if (report := _report_secret_op_outcome(messages)) is not None:
        return report

    parts = [part for message in messages for part in message.parts]
    if not any(isinstance(p, ToolReturnPart) and p.tool_name == LOAD_CAPABILITY_TOOL_NAME for p in parts):
        return ModelResponse(
            parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'guarded'}, tool_call_id='l1')]
        )
    if not any(isinstance(p, CompactionPart) for p in parts):
        # Compaction lands in its own step, alongside a harmless call so the run continues.
        return ModelResponse(
            parts=[
                CompactionPart(content='Summary: guarded was loaded earlier.', provider_name='function'),
                ToolCallPart(tool_name='ping', args={}, tool_call_id='p1'),
            ]
        )
    return ModelResponse(parts=[ToolCallPart(tool_name='secret_op', args={}, tool_call_id='s1')])


class TestUnavailableCapabilityToolsAreNotCallable:
    """A deferred capability's tools cannot be called until the capability is available.

    Availability is the gate: the capability's instructions and hooks arrive as a bundle when it
    loads, so calling one of its tools earlier would run it without the context the model was meant
    to have read first. The refusal is a retry the model can act on, not an "unknown tool" dead end.
    """

    @staticmethod
    def _guarded_capability() -> Capability[Any]:
        return Capability[Any](
            id='guarded',
            description='Guarded tools.',
            toolsets=[FunctionToolset([secret_op])],
            defer_loading=True,
        )

    async def test_never_loaded_capability_tool_is_refused(self):
        """A direct call to an unloaded capability's tool is refused, and says how to fix it."""
        agent = Agent(FunctionModel(_call_secret_op), capabilities=[self._guarded_capability()])

        result = await agent.run('hello')

        assert result.output == snapshot(
            "BLOCKED: Tool 'secret_op' is not available yet: it belongs to capability 'guarded'. Call `load_capability` for it first, then call the tool again once you've read the capability's instructions."
        )

    async def test_capability_tool_is_refused_again_after_compaction(self):
        """A `CompactionPart` resets the load state, so the tool needs loading again."""
        agent = Agent(FunctionModel(_load_compact_then_call_secret_op), capabilities=[self._guarded_capability()])

        @agent.tool_plain
        def ping() -> str:
            return 'pong'

        result = await agent.run('hello')

        assert 'is not available yet' in result.output

    async def test_available_capability_does_not_excuse_an_undiscovered_tool(self):
        """An always-on capability's search-gated tool still has to be searched for.

        The capability being available makes its tools eligible to be shown, not evidence that any
        of them were — so availability must not short-circuit the discovery requirement.
        """
        toolset = FunctionToolset[Any]()
        toolset.add_function(secret_op, name='secret_op', defer_loading=True)
        eager = Capability[Any](id='eager', description='Always on.', toolsets=[toolset], defer_loading=False)

        agent = Agent(FunctionModel(_call_secret_op), capabilities=[eager])

        result = await agent.run('hello')

        assert result.output == snapshot(
            "BLOCKED: Tool 'secret_op' is not available yet: search for it first, then call it again once you've seen its schema."
        )

    async def test_an_availability_refusal_leaves_room_to_act_on_it(self):
        """The refusal grants one attempt beyond the tool's budget, and no more.

        It is not a mistake about the call's arguments — it says the run isn't in a state where the
        tool can be called, and names the step that fixes it. On the default budget of 1, charging
        it like a validation error would make a single act of disobedience fatal, so a message
        written to be acted on would never get the chance. The ceiling stays finite: a model that
        never takes the correction still ends the run.
        """

        def call_twice_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            refusals = list(iter_message_parts(messages, ModelRequest, RetryPromptPart))
            if len(refusals) >= 2:
                return make_text_response(f'refused {len(refusals)}x, still running')
            return ModelResponse(parts=[ToolCallPart(tool_name='secret_op', args={}, tool_call_id=f's{len(refusals)}')])

        agent = Agent(FunctionModel(call_twice_then_report), capabilities=[self._guarded_capability()])
        assert (await agent.run('hello')).output == snapshot('refused 2x, still running')

        def keep_calling(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            refusals = list(iter_message_parts(messages, ModelRequest, RetryPromptPart))
            return ModelResponse(parts=[ToolCallPart(tool_name='secret_op', args={}, tool_call_id=f's{len(refusals)}')])

        stubborn = Agent(FunctionModel(keep_calling), capabilities=[self._guarded_capability()])
        with pytest.raises(UnexpectedModelBehavior, match="Tool 'secret_op' exceeded max retries"):
            await stubborn.run('hello')

    async def test_an_availability_refusal_does_not_spend_the_budget_a_real_failure_needs(self):
        """A refusal must not leave the tool with nothing left when it is later called properly.

        The refusal says the run isn't in a state where the tool can be called. Charging it to
        `retries` would mean a tool refused once, then loaded and called correctly, aborts on its
        *first* genuine failure with no retry at all — the budget spent on a state problem rather
        than on the mistake it exists for.
        """
        kinds: list[str] = []

        def load_then_fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            parts = [part for message in messages for part in message.parts]
            loaded = any(
                isinstance(part, ToolReturnPart) and part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in parts
            )
            kinds[:] = [
                'availability' if 'is not available yet' in str(part.content) else 'real'
                for part in iter_message_parts(messages, ModelRequest, RetryPromptPart)
            ]
            if not loaded and not kinds:
                # Call before loading: refused for availability.
                return ModelResponse(parts=[ToolCallPart(tool_name='failing_op', args={}, tool_call_id='early')])
            if not loaded:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'guarded'}, tool_call_id='l')]
                )
            if 'real' in kinds:
                return make_text_response('the real retry survived')
            return ModelResponse(parts=[ToolCallPart(tool_name='failing_op', args={}, tool_call_id='real')])

        def failing_op() -> str:
            raise ModelRetry('give me better arguments')

        toolset = FunctionToolset[Any]([failing_op])
        capability = Capability[Any](id='guarded', description='Guarded.', toolsets=[toolset], defer_loading=True)

        agent = Agent(FunctionModel(load_then_fail), capabilities=[capability])
        result = await agent.run('hello')

        assert result.output == snapshot('the real retry survived')
        # The availability refusal came first and cost nothing; the genuine failure still got its retry.
        assert kinds == snapshot(['availability', 'real'])

    async def test_an_availability_refusal_is_charged_against_the_tools_own_budget(self):
        """A tool's configured `max_retries` governs its refusals, not the manager's default.

        The refusal is raised while resolving, before the caller has bound the resolved tool, so
        the budget could easily be taken from the manager default that an unresolvable name gets.
        A tool that asked for a larger budget must still get it.
        """
        refusals = 0

        def keep_calling(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal refusals
            refusals = len(list(iter_message_parts(messages, ModelRequest, RetryPromptPart)))
            return ModelResponse(parts=[ToolCallPart(tool_name='secret_op', args={}, tool_call_id=f's{refusals}')])

        toolset = FunctionToolset[Any]()
        toolset.add_function(secret_op, name='secret_op', retries=4)
        capability = Capability[Any](id='guarded', description='Guarded.', toolsets=[toolset], defer_loading=True)

        agent = Agent(FunctionModel(keep_calling), capabilities=[capability])
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 4'):
            await agent.run('hello')

        # The tool's own budget of 4, plus the one extra an availability refusal is granted.
        assert refusals == snapshot(5)

    async def test_unavailable_capability_tool_is_not_advertised(self):
        """The tool is withheld as well as uncallable — never visible-but-unusable."""
        advertised: list[list[str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            advertised.append(sorted(t.name for t in info.function_tools))
            return make_text_response('done')

        agent = Agent(FunctionModel(model_fn), capabilities=[self._guarded_capability()])

        @agent.tool_plain
        def untouched() -> str:
            return 'safe'  # pragma: no cover

        await agent.run('hello')

        assert advertised == snapshot([['load_capability', 'untouched']])


async def test_reveal_of_another_capabilitys_tool_is_rejected_even_while_loaded():
    """Being loaded exempts a capability's *own* tools, never another capability's.

    `load_capability` is allowed past the guard for the bundle it is activating, so the guard has to
    stay strict about everything else — otherwise one loaded capability would become a route to
    revealing any other capability's tools without activating them.
    """

    def other_op() -> str:
        return 'OTHER'  # pragma: no cover

    other = Capability[Any](
        id='other', description='Other tools.', toolsets=[FunctionToolset([other_op])], defer_loading=True
    )

    def smuggler() -> ToolReturn:
        return ToolReturn(return_value='ok', tools=['other_op'])

    smuggling = Capability[Any](id='smuggling', description='Smuggling tools.', toolsets=[FunctionToolset([smuggler])])

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='smuggler', args={}, tool_call_id='s1')])

    agent = Agent(FunctionModel(model_fn), capabilities=[other, smuggling])

    with pytest.raises(UserError, match="cannot reveal 'other_op'"):
        await agent.run('go')


async def test_loaded_capability_tool_survives_a_stripped_reveal_marker() -> None:
    """A loaded capability's tool stays callable when its reveal marker is gone from history.

    Every tool of a deferred capability is kept out of the search corpus, and reloading an
    already-active capability is refused, so the load exchange is the only thing that can ever
    disclose these tools. Requiring a separate reveal marker on top left the model with a refusal
    telling it to search — for a tool no search can return — and no way to regenerate the evidence,
    so the run burned its retry budget and aborted. History processing that drops availability
    deltas while keeping the load pair is enough to reach it.
    """
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        # Match by name: `LoadCapabilityReturnPart` subclasses `ToolReturnPart`, so the seeded load
        # return would otherwise read as "the tool already ran" and the call would never be made.
        returns = [p for p in iter_message_parts(messages, ModelRequest, ToolReturnPart) if p.tool_name == 'secret_op']
        if returns:
            return make_text_response('EXECUTED')
        return ModelResponse(parts=[ToolCallPart(tool_name='secret_op', args={}, tool_call_id=f'c{calls}')])

    def strip_deltas(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(message, parts=[p for p in message.parts if not isinstance(p, ToolAvailabilityDeltaPart)])
            if isinstance(message, ModelRequest)
            else message
            for message in messages
        ]

    toolset = FunctionToolset[Any]()
    toolset.add_function(secret_op, defer_loading=True)
    guarded = Capability[Any](id='guarded', description='Guarded tools.', toolsets=[toolset], defer_loading=True)
    agent = Agent(FunctionModel(model_fn), capabilities=[guarded, ProcessHistory(strip_deltas)])

    result = await agent.run(
        'use the tool',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='load it')]),
            ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'guarded'}, tool_call_id='l1')]),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='l1')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['secret_op'])]),
        ],
    )

    assert result.output == 'EXECUTED'
    refusals = [str(part.content) for part in iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart)]
    assert refusals == []


async def test_stripped_reveal_marker_survives_a_boundary_the_wire_skipped() -> None:
    """The load-is-the-reveal shortcut has to read the anchored window, not just the agnostic one.

    Where the two features meet, both halves of the evidence are hidden from the provider-agnostic
    window: the boundary moves the load out of `loaded_capability_ids`, and history processing
    removes the delta that would otherwise stand in for it. Only the anchored window still holds
    the load, and only the shortcut can turn it into an answer, since nothing else discloses a
    capability's own tools.
    """
    calls = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        returns = [p for p in iter_message_parts(messages, ModelRequest, ToolReturnPart) if p.tool_name == 'secret_op']
        if returns:
            return _provider_response([make_text_response('EXECUTED').parts[0]], 'openai')
        return _provider_response([ToolCallPart(tool_name='secret_op', args={}, tool_call_id=f'c{calls}')], 'openai')

    def strip_deltas(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(message, parts=[p for p in message.parts if not isinstance(p, ToolAvailabilityDeltaPart)])
            if isinstance(message, ModelRequest)
            else message
            for message in messages
        ]

    toolset = FunctionToolset[Any]()
    toolset.add_function(secret_op, defer_loading=True)
    guarded = Capability[Any](id='guarded', description='Guarded tools.', toolsets=[toolset], defer_loading=True)
    agent = Agent(FunctionModel(model_fn), capabilities=[guarded, ProcessHistory(strip_deltas)])

    result = await agent.run(
        'use the tool',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='load it')]),
            ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'guarded'}, tool_call_id='l1')]),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='l1')]),
            ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['secret_op'])]),
            # Stamped by another provider, so the OpenAI request that carried this call replayed
            # the whole history and the model saw the load exchange.
            ModelResponse(parts=[CompactionPart(content='foreign', provider_name='anthropic')]),
        ],
    )

    assert result.output == 'EXECUTED'
    refusals = [str(part.content) for part in iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart)]
    assert refusals == []
