"""Tests for the `ToolGuardrail` capability."""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Callable
from typing import Any, NoReturn

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer
from pydantic_ai import Agent, AgentRunResult, AgentSpec, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry, SkipToolExecution, ToolFailed, UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.guardrails import (
    GuardrailError,
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    ToolBlocked,
    ToolCallInfo,
    ToolGuardrail,
    ToolResultInfo,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def _run_ctx(
    *,
    trace_include_content: bool = False,
    tracer: Tracer | None = None,
    tool_call_approved: bool = False,
) -> RunContext[object]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        trace_include_content=trace_include_content,
        tracer=tracer if tracer is not None else NoOpTracer(),
        tool_call_approved=tool_call_approved,
    )


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, f'expected exactly one span, got {[s.name for s in spans]}'
    return spans[0]


def _call(tool_name: str = 'lookup', tool_call_id: str = 'call-1') -> ToolCallPart:
    return ToolCallPart(tool_name=tool_name, args='{}', tool_call_id=tool_call_id)


def _tool_def(name: str = 'lookup') -> ToolDefinition:
    return ToolDefinition(name=name)


async def _guard_args(
    guard: ToolGuardrail[object],
    args: dict[str, Any],
    *,
    tool_name: str = 'lookup',
    ctx: RunContext[object] | None = None,
) -> dict[str, Any]:
    return await guard.before_tool_execute(
        ctx if ctx is not None else _run_ctx(),
        call=_call(tool_name),
        tool_def=_tool_def(tool_name),
        args=args,
    )


async def _guard_result(
    guard: ToolGuardrail[object],
    result: Any,
    *,
    tool_name: str = 'lookup',
    ctx: RunContext[object] | None = None,
) -> Any:
    return await guard.after_tool_execute(
        ctx if ctx is not None else _run_ctx(),
        call=_call(tool_name),
        tool_def=_tool_def(tool_name),
        args={},
        result=result,
    )


def _scripted(*responses: ModelResponse) -> FunctionModel:
    """A model that replays `responses` in order, one per request."""
    remaining = iter(responses)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return next(remaining)

    return FunctionModel(respond)


def _calls_lookup(**args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args=args, tool_call_id='call-1')])


def _agent_with(guard: ToolGuardrail[None], *responses: ModelResponse) -> tuple[Agent[None, str], list[dict[str, Any]]]:
    """An agent whose single `lookup` tool records the arguments it was called with."""
    seen: list[dict[str, Any]] = []
    # `deps_type` is explicit because Pydantic AI cannot infer it from a capability
    # passed as an already-typed variable rather than constructed inline.
    agent = Agent(_scripted(*responses), deps_type=type(None), capabilities=[guard])

    @agent.tool_plain
    def lookup(query: str) -> str:
        seen.append({'query': query})
        return f'result for {query}'

    return agent, seen


class TestArgumentGuard:
    """The `guard` callable, evaluated before the tool runs."""

    async def test_allow_runs_the_tool(self):
        agent, seen = _agent_with(
            ToolGuardrail(guard=lambda call: True),
            _calls_lookup(query='weather'),
            ModelResponse(parts=[TextPart(content='done')]),
        )
        result = await agent.run('hi')

        assert result.output == 'done'
        assert seen == [{'query': 'weather'}]

    async def test_block_substitutes_a_refusal_and_skips_execution(self):
        agent, seen = _agent_with(
            ToolGuardrail(guard=lambda call: GuardrailResult.block('not allowed')),
            _calls_lookup(query='secrets'),
            ModelResponse(parts=[TextPart(content='understood')]),
        )
        result = await agent.run('hi')

        assert seen == []
        returns = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert [part.content for part in returns] == ['not allowed']

    async def test_bare_false_blocks_with_the_default_message(self):
        with pytest.raises(SkipToolExecution) as exc_info:
            await _guard_args(ToolGuardrail(guard=lambda call: False), {})

        assert exc_info.value.result == 'Tool call blocked by tool guardrail.'

    async def test_replace_substitutes_arguments(self):
        def redact(call: ToolCallInfo) -> GuardrailResult:
            return GuardrailResult.replace({**call.args, 'query': '[redacted]'})

        agent, seen = _agent_with(
            ToolGuardrail(guard=redact),
            _calls_lookup(query='sk-live-000'),
            ModelResponse(parts=[TextPart(content='done')]),
        )
        await agent.run('hi')

        assert seen == [{'query': '[redacted]'}]

    async def test_replace_with_non_string_keys_is_a_user_error(self):
        """They become keyword arguments, where the interpreter's TypeError names nothing useful."""
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.replace({1: 'x'}))

        with pytest.raises(UserError, match='must provide replacement arguments'):
            await _guard_args(guard, {'query': 'x'})

    async def test_the_arguments_a_guard_sees_are_read_only(self):
        """Mutating them would change the call while reporting `allow`, with no redaction span."""

        def mutate(call: ToolCallInfo) -> GuardrailResult:
            call.args['query'] = 'MUTATED'  # pyright: ignore[reportIndexIssue]
            return GuardrailResult.allow()  # pragma: no cover - the mutation above raises first

        with pytest.raises(TypeError):
            await _guard_args(ToolGuardrail(guard=mutate), {'query': 'original'})

    async def test_a_nested_mutation_does_not_reach_the_call(self):
        """The read-only view is shallow, so the guard sees a copy rather than the real dict."""

        def sneak(call: ToolCallInfo) -> GuardrailResult:
            options = call.args['options']
            assert isinstance(options, dict)
            options['path'] = '/etc/passwd'
            return GuardrailResult.allow()

        args = {'options': {'path': '/workspace/notes.txt'}}

        assert await _guard_args(ToolGuardrail(guard=sneak), args) == {'options': {'path': '/workspace/notes.txt'}}
        assert args == {'options': {'path': '/workspace/notes.txt'}}

    async def test_replace_with_a_non_mapping_is_a_user_error(self):
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.replace('not a mapping'))

        with pytest.raises(UserError, match='must provide replacement arguments'):
            await _guard_args(guard, {'query': 'x'})

    async def test_retry_asks_the_model_to_redo_the_call(self):
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.retry('narrow the query'))

        with pytest.raises(ModelRetry, match='narrow the query'):
            await _guard_args(guard, {})

    async def test_retry_without_a_message_uses_the_default(self):
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult(action='retry', message=''))

        with pytest.raises(ModelRetry, match='Tool call rejected by tool guardrail.'):
            await _guard_args(guard, {})

    async def test_approve_defers_the_call(self):
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.approve())

        with pytest.raises(ApprovalRequired):
            await _guard_args(guard, {})

    async def test_approve_surfaces_the_call_as_a_deferred_request(self):
        agent = Agent(
            _scripted(_calls_lookup(query='rm -rf')),
            capabilities=[ToolGuardrail(guard=lambda call: GuardrailResult.approve())],
            output_type=[str, DeferredToolRequests],
        )

        @agent.tool_plain
        def lookup(query: str) -> str:  # pragma: no cover - deferred before execution
            return query

        result = await agent.run('hi')

        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['lookup']


class TestHumanInTheLoop:
    """`approve` hands off to Pydantic AI's deferred-approval round trip."""

    def _deferring_agent(self) -> tuple[Agent[None, str | DeferredToolRequests], list[str]]:
        executed: list[str] = []
        agent = Agent(
            _scripted(_calls_lookup(query='prod'), ModelResponse(parts=[TextPart(content='done')])),
            deps_type=type(None),
            capabilities=[ToolGuardrail(guard=lambda call: GuardrailResult.approve())],
            output_type=[str, DeferredToolRequests],
        )

        @agent.tool_plain
        def lookup(query: str) -> str:
            executed.append(query)
            return f'result for {query}'

        return agent, executed

    async def test_approval_lets_the_call_through_on_resume(self):
        agent, executed = self._deferring_agent()
        deferred = await agent.run('hi')
        assert isinstance(deferred.output, DeferredToolRequests)

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': True}),
        )

        assert resumed.output == 'done'
        assert executed == ['prod']

    async def test_denial_reaches_the_model_without_running_the_tool(self):
        agent, executed = self._deferring_agent()
        deferred = await agent.run('hi')

        resumed = await agent.run(
            message_history=deferred.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': ToolDenied('not on a Friday')}),
        )

        assert resumed.output == 'done'
        assert executed == []
        returns = [
            part for message in resumed.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert [part.content for part in returns] == ['not on a Friday']

    async def test_a_guard_may_still_block_an_approved_call(self):
        """Approval clears `approve`, not the rest of the policy."""

        def policy(call: ToolCallInfo) -> GuardrailResult:
            return GuardrailResult.block('the window closed while you were deciding')

        guard = ToolGuardrail[object](guard=policy)
        ctx = _run_ctx(tool_call_approved=True)

        with pytest.raises(SkipToolExecution) as exc_info:
            await _guard_args(guard, {}, ctx=ctx)

        assert exc_info.value.result == 'the window closed while you were deciding'

    async def test_an_async_guard_approves_in_process_without_ending_the_run(self):
        """The blocking-prompt shape: the guard awaits a human and the run never unwinds."""
        asked: list[str] = []

        async def ask_the_operator(call: ToolCallInfo) -> GuardrailResult:
            asked.append(call.name)
            await asyncio.sleep(0)  # stands in for a prompt, a websocket, a queue
            return GuardrailResult.block('operator said no')

        agent, seen = _agent_with(
            ToolGuardrail(guard=ask_the_operator),
            _calls_lookup(query='prod'),
            ModelResponse(parts=[TextPart(content='understood')]),
        )
        result = await agent.run('hi')

        assert result.output == 'understood'
        assert asked == ['lookup']
        assert seen == []

    async def test_guard_receives_the_call_details(self):
        seen: list[ToolCallInfo] = []

        def record(call: ToolCallInfo) -> bool:
            seen.append(call)
            return True

        await _guard_args(ToolGuardrail(guard=record), {'query': 'x'})

        assert seen == [ToolCallInfo(name='lookup', args={'query': 'x'}, tool_call_id='call-1')]

    async def test_guard_may_take_run_context(self):
        async def policy(ctx: RunContext[object], call: ToolCallInfo) -> GuardrailResult:
            assert ctx.deps is None
            return GuardrailResult.block('denied')

        with pytest.raises(SkipToolExecution):
            await _guard_args(ToolGuardrail(guard=policy), {})

    async def test_no_guard_passes_arguments_through(self):
        assert await _guard_args(ToolGuardrail[object](), {'query': 'x'}) == {'query': 'x'}

    async def test_tools_restricts_which_calls_are_guarded(self):
        guard = ToolGuardrail[object](guard=lambda call: False, tools=['dangerous'])

        assert await _guard_args(guard, {'query': 'x'}, tool_name='safe') == {'query': 'x'}
        with pytest.raises(SkipToolExecution):
            await _guard_args(guard, {}, tool_name='dangerous')

    async def test_a_misspelled_tools_name_warns_after_the_unmatched_call(self):
        guarded: list[str] = []

        def block(call: ToolCallInfo) -> GuardrailResult:  # pragma: no cover - this test verifies it cannot run
            guarded.append(call.name)
            return GuardrailResult.block('blocked')

        agent, executed = _agent_with(
            ToolGuardrail(guard=block, tools=['lookpu']),
            _calls_lookup(query='weather'),
            ModelResponse(parts=[TextPart(content='done')]),
        )

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            result = await agent.run('hi')

        assert result.output == 'done'
        assert guarded == []
        assert executed == [{'query': 'weather'}]
        assert [str(warning.message) for warning in raised] == [
            "ToolGuardrail.tools names 'lookpu', which no tool offered during this run is called. "
            'Neither guard was applied to it.'
        ]


class TestResultGuard:
    """The `result_guard` callable, evaluated after the tool returns."""

    async def test_allow_returns_the_result_unchanged(self):
        assert await _guard_result(ToolGuardrail(result_guard=lambda info: True), 'clean') == 'clean'

    async def test_block_replaces_the_result_with_the_refusal(self):
        guard = ToolGuardrail[object](result_guard=lambda info: GuardrailResult.block('contained a secret'))

        assert await _guard_result(guard, 'sk-live-000') == 'contained a secret'

    async def test_block_without_a_message_uses_the_default(self):
        assert await _guard_result(ToolGuardrail(result_guard=lambda info: False), 'x') == (
            'Tool result blocked by tool guardrail.'
        )

    async def test_replace_substitutes_a_sanitized_result(self):
        def scrub(info: ToolResultInfo) -> GuardrailResult:
            return GuardrailResult.replace(str(info.result).replace('sk-live-000', '[redacted]'))

        agent = Agent(
            _scripted(_calls_lookup(query='key'), ModelResponse(parts=[TextPart(content='done')])),
            capabilities=[ToolGuardrail(result_guard=scrub)],
        )

        @agent.tool_plain
        def lookup(query: str) -> str:
            return 'the key is sk-live-000'

        result = await agent.run('hi')
        returns = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]

        assert [part.content for part in returns] == ['the key is [redacted]']

    async def test_retry_asks_the_model_to_redo_the_call(self):
        guard = ToolGuardrail[object](result_guard=lambda info: GuardrailResult.retry('try a narrower query'))

        with pytest.raises(ModelRetry, match='try a narrower query'):
            await _guard_result(guard, 'too much')

    async def test_approve_is_rejected_after_execution(self):
        guard = ToolGuardrail[object](result_guard=lambda info: GuardrailResult.approve())

        with pytest.raises(UserError, match='the tool has already run'):
            await _guard_result(guard, 'x')

    async def test_result_guard_receives_the_call_and_the_result(self):
        seen: list[ToolResultInfo] = []

        def record(info: ToolResultInfo) -> bool:
            seen.append(info)
            return True

        await _guard_result(ToolGuardrail(result_guard=record), {'rows': 2})

        assert seen == [ToolResultInfo(name='lookup', args={}, tool_call_id='call-1', result={'rows': 2})]

    async def test_replace_accepts_none_as_a_sanitized_result(self):
        """`None` is a real tool result, so it has to be a usable replacement."""
        guard = ToolGuardrail[object](result_guard=lambda info: GuardrailResult.replace(None))

        assert await _guard_result(guard, 'sensitive') is None

    async def test_no_result_guard_passes_the_result_through(self):
        assert await _guard_result(ToolGuardrail[object](), 'x') == 'x'

    async def test_tools_restricts_which_results_are_guarded(self):
        guard = ToolGuardrail[object](result_guard=lambda info: False, tools=['dangerous'])

        assert await _guard_result(guard, 'x', tool_name='safe') == 'x'
        assert await _guard_result(guard, 'x', tool_name='dangerous') == 'Tool result blocked by tool guardrail.'


class TestResultGuardOnAFailedTool:
    """A tool that fails sends text to the model too, and core routes it past `after_tool_execute`."""

    def _agent(
        self,
        verdict: Callable[[ToolResultInfo], GuardrailResult],
        error: Exception,
        *,
        tools: list[str] | None = None,
    ) -> tuple[Agent[None, str], list[object]]:
        seen: list[object] = []

        def result_guard(info: ToolResultInfo) -> GuardrailResult:
            seen.append(info.result)
            return verdict(info)

        agent = Agent(
            _scripted(_calls_lookup(query='key'), ModelResponse(parts=[TextPart(content='done')])),
            deps_type=type(None),
            capabilities=[ToolGuardrail[None](result_guard=result_guard, tools=tools)],
        )

        @agent.tool_plain
        def lookup(query: str) -> str:
            raise error

        return agent, seen

    @pytest.mark.parametrize('error', [ModelRetry('the key is sk-live-000'), ToolFailed('the key is sk-live-000')])
    async def test_a_failure_message_is_screened_before_the_model_sees_it(self, error: Exception):
        def scrub(info: ToolResultInfo) -> GuardrailResult:
            cleaned = str(info.result).replace('sk-live-000', '[redacted]')
            return GuardrailResult.replace(cleaned) if cleaned != str(info.result) else GuardrailResult.allow()

        agent, seen = self._agent(scrub, error)

        result = await agent.run('hi')

        assert seen == ['the key is sk-live-000'], 'the guard never saw the failed tool'
        rendered = str(result.all_messages())
        assert 'sk-live-000' not in rendered
        assert '[redacted]' in rendered

    @pytest.mark.parametrize('error', [ModelRetry('boom'), ToolFailed('boom')])
    async def test_a_failure_the_guard_allows_keeps_its_own_type_and_text(self, error: Exception):
        agent, seen = self._agent(lambda info: GuardrailResult.allow(), error)

        result = await agent.run('hi')

        assert seen == ['boom']
        assert 'boom' in str(result.all_messages())

    @pytest.mark.parametrize('error', [ModelRetry('boom'), ToolFailed('boom')])
    async def test_retry_on_a_failure_still_asks_the_model_to_redo_the_call(self, error: Exception):
        """Collapsing it into a replacement message hands the model a failed result instead."""
        agent, _ = self._agent(lambda info: GuardrailResult.retry('narrow the query'), error)

        result = await agent.run('hi')
        retries = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, RetryPromptPart)
        ]

        assert [part.content for part in retries] == ['narrow the query']

    async def test_a_failure_is_not_turned_into_a_success(self):
        """A screened failure is still a failure; a success would tell the model the call worked."""
        agent, _ = self._agent(lambda info: GuardrailResult.block('refused'), ToolFailed('the key is sk-live-000'))

        result = await agent.run('hi')
        returns = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]

        assert [(part.content, part.outcome) for part in returns] == [('refused', 'failed')]

    async def test_approve_is_rejected_on_a_failure_too(self):
        """The tool has already run, and failed; there is nothing left to approve."""
        agent, _ = self._agent(lambda info: GuardrailResult.approve(), ToolFailed('boom'))

        with pytest.raises(UserError, match='the tool has already run'):
            await agent.run('hi')

    async def test_a_tool_the_guard_does_not_cover_is_left_alone(self):
        agent, seen = self._agent(lambda info: GuardrailResult.block('refused'), ToolFailed('boom'), tools=[])

        result = await agent.run('hi')

        assert seen == []
        assert 'boom' in str(result.all_messages())


class TestConfigurationShape:
    """A bare string satisfies `Sequence[str]`, and both fields misbehave on one."""

    def test_a_bare_string_for_hidden_is_refused(self):
        """`set('danger')` holds six letters, so the tool it names would stay on the wire."""
        with pytest.raises(UserError, match='ToolGuardrail.hidden takes a collection'):
            ToolGuardrail[object](hidden='danger')  # pyright: ignore[reportArgumentType]

    def test_a_bare_string_for_tools_is_refused(self):
        """Substring membership would make it match any tool whose name it contains."""
        with pytest.raises(UserError, match='ToolGuardrail.tools takes a collection'):
            ToolGuardrail[object](tools='delete_all')  # pyright: ignore[reportArgumentType]


class TestHiddenTools:
    """`hidden` withholds tools from the model rather than refusing them."""

    async def test_hidden_tools_are_not_offered_to_the_model(self):
        offered: list[list[str]] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append(sorted(tool.name for tool in info.function_tools))
            return ModelResponse(parts=[TextPart(content='done')])

        agent = Agent(FunctionModel(respond), capabilities=[ToolGuardrail(hidden=['danger'])])

        @agent.tool_plain
        def danger() -> str:  # pragma: no cover - hidden, so never called
            return 'boom'

        @agent.tool_plain
        def safe() -> str:  # pragma: no cover - the model returns text instead
            return 'ok'

        await agent.run('hi')

        assert offered == [['safe']]

    async def test_a_hidden_tool_cannot_be_called(self):
        """Withholding the definition is only half of it; the name must not resolve either."""

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='gave up')])
            return ModelResponse(parts=[ToolCallPart(tool_name='danger', args={}, tool_call_id='c1')])

        agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[ToolGuardrail(hidden=['danger'])])

        @agent.tool_plain
        def danger() -> str:  # pragma: no cover - the point is that it is unreachable
            return 'boom'

        @agent.tool_plain
        def safe() -> str:  # pragma: no cover - never called by this script
            return 'ok'

        result = await agent.run('hi')
        retries = [p for m in result.all_messages() for p in m.parts if isinstance(p, RetryPromptPart)]

        assert result.output == 'gave up'
        assert "Unknown tool name: 'danger'" in str(retries[0].content)

    async def test_no_hidden_tools_leaves_definitions_untouched(self):
        tool_defs = [_tool_def('a'), _tool_def('b')]

        assert await ToolGuardrail[object]().prepare_tools(_run_ctx(), tool_defs) == tool_defs


class TestHiddenNameWarning:
    """`hidden` is the one setting whose typo fails open."""

    async def test_an_unstarted_agent_iteration_does_not_warn_about_hidden_names(self):
        agent = Agent(
            _scripted(ModelResponse(parts=[TextPart(content='done')])),
            deps_type=type(None),
            capabilities=[ToolGuardrail(hidden=['danger'])],
        )

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            async with agent.iter('hi'):
                pass

        assert raised == []

    async def test_an_unprepared_successful_run_does_not_warn_about_hidden_names(self):
        guard = ToolGuardrail[object](hidden=['danger'])

        async def succeed() -> AgentRunResult[object]:
            return AgentRunResult(output='done')

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            result = await guard.wrap_run(_run_ctx(), handler=succeed)

        assert result.output == 'done'
        assert raised == []

    async def test_a_run_without_function_tools_warns_about_hidden_names(self):
        agent = Agent(
            _scripted(ModelResponse(parts=[TextPart(content='done')])),
            deps_type=type(None),
            capabilities=[ToolGuardrail(hidden=['danger'])],
        )

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            result = await agent.run('hi')

        assert result.output == 'done'
        assert [str(warning.message) for warning in raised] == [
            "ToolGuardrail.hidden names 'danger', which no tool offered during this run is called. "
            'The name stays available to the model.'
        ]

    async def test_a_warning_does_not_mask_a_prepared_run_error(self):
        guard = ToolGuardrail[object](hidden=['danger'])
        await guard.prepare_tools(_run_ctx(), [_tool_def('safe')])

        async def fail() -> NoReturn:
            raise RuntimeError('boom')

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(RuntimeError, match='boom'):
                await guard.wrap_run(_run_ctx(), handler=fail)

    async def test_a_prepared_run_error_does_not_warn_about_hidden_names(self):
        def fail(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
            raise RuntimeError('boom')

        agent = Agent(
            FunctionModel(fail),
            deps_type=type(None),
            capabilities=[ToolGuardrail(hidden=['danger'])],
        )

        @agent.tool_plain
        def safe() -> str:  # pragma: no cover - the model fails before tool execution
            return 'safe'

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            with pytest.raises(RuntimeError, match='boom'):
                await agent.run('hi')

        assert raised == []

    async def test_a_name_no_tool_answers_to_is_warned_about(self):
        agent, _ = _agent_with(
            ToolGuardrail(hidden=['dangre']),
            ModelResponse(parts=[TextPart(content='done')]),
        )

        with pytest.warns(UserWarning, match="names 'dangre'"):
            result = await agent.run('hi')

        assert result.output == 'done'

    async def test_the_warning_fires_once_after_a_multi_step_run(self):
        agent, _ = _agent_with(
            ToolGuardrail(hidden=['dangre']),
            _calls_lookup(query='weather'),
            ModelResponse(parts=[TextPart(content='done')]),
        )

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            result = await agent.run('hi')

        assert result.output == 'done'
        assert [str(warning.message) for warning in raised] == [
            "ToolGuardrail.hidden names 'dangre', which no tool offered during this run is called. "
            'The name stays available to the model.'
        ]

    async def test_a_name_that_matches_is_not_warned_about(self):
        guard = ToolGuardrail[object](hidden=['danger'])

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            remaining = await guard.prepare_tools(_run_ctx(), [_tool_def('danger'), _tool_def('safe')])

        assert raised == []
        assert [tool_def.name for tool_def in remaining] == ['safe']

    async def test_a_tool_that_only_appears_later_is_not_warned_about(self):
        """A toolset may offer a tool on some steps and not others."""
        guard = ToolGuardrail[object](hidden=['danger'])

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            await guard.prepare_tools(_run_ctx(), [_tool_def('danger')])
            await guard.prepare_tools(_run_ctx(), [_tool_def('safe')])

        assert raised == []

    async def test_a_hidden_tool_that_appears_after_an_absent_step_is_not_warned_about(self):
        ready = False
        safe_tools = FunctionToolset[None]()
        danger_tools = FunctionToolset[None]()
        offered: list[list[str]] = []

        @safe_tools.tool_plain
        def safe() -> str:
            nonlocal ready
            ready = True
            return 'ready'

        @danger_tools.tool_plain
        def danger() -> str:  # pragma: no cover - hidden from the model
            return 'danger'

        def toolset(_: RunContext[None]) -> FunctionToolset[None]:
            return danger_tools if ready else safe_tools

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            offered.append(sorted(tool.name for tool in info.function_tools))
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='safe', args={}, tool_call_id='call-1')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[ToolGuardrail(hidden=['danger'])],
            toolsets=[toolset],
        )

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter('always')
            result = await agent.run('hi')

        assert result.output == 'done'
        assert offered == [['safe'], []]
        assert raised == []


class TestAgentSpec:
    def test_callable_guardrails_are_not_represented_in_agent_spec_schema(self):
        schema = AgentSpec.model_json_schema()

        schema_text = json.dumps(schema)
        assert 'InputGuardrail' not in schema_text
        assert 'OutputGuardrail' not in schema_text
        assert 'ToolGuardrail' not in schema_text

    @pytest.mark.parametrize('capability_type', [InputGuardrail, OutputGuardrail, ToolGuardrail])
    def test_agent_spec_rejects_callable_guardrails(self, capability_type: type[AbstractCapability[None]]):
        with pytest.raises(
            ValueError, match=rf'Custom capability class {capability_type.__name__} has opted out of serialization'
        ):
            Agent.from_spec(
                {'capabilities': [capability_type.__name__]},
                custom_capability_types=[capability_type],
                model=TestModel(),
            )


class TestHardFailure:
    """Raising from a guard propagates instead of substituting a result."""

    async def test_tool_blocked_propagates(self):
        def deny(call: ToolCallInfo) -> GuardrailResult:
            raise ToolBlocked(call.name, 'policy violation')

        with pytest.raises(ToolBlocked, match="Tool 'lookup' blocked: policy violation") as exc_info:
            await _guard_args(ToolGuardrail(guard=deny), {})

        assert exc_info.value.tool_name == 'lookup'
        assert exc_info.value.reason == 'policy violation'
        assert isinstance(exc_info.value, GuardrailError)

    async def test_tool_blocked_without_a_reason(self):
        error = ToolBlocked('lookup')

        assert str(error) == "Tool 'lookup' blocked"
        assert error.reason is None


class TestTracing:
    """Non-`allow` verdicts are recorded as spans, content only when opted in."""

    async def test_block_span_carries_the_message_when_content_is_included(self):
        tracer, exporter = _recording_tracer()
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.block('denied'))

        with pytest.raises(SkipToolExecution):
            await _guard_args(guard, {}, ctx=_run_ctx(trace_include_content=True, tracer=tracer))

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked tool args'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'tool args',
            'guardrail.action': 'block',
            'guardrail.tool': 'lookup',
            'guardrail.message': 'denied',
        }

    async def test_block_span_omits_the_message_by_default(self):
        tracer, exporter = _recording_tracer()
        guard = ToolGuardrail[object](result_guard=lambda info: GuardrailResult.block('denied'))

        await _guard_result(guard, 'x', ctx=_run_ctx(tracer=tracer))

        span = _only_span(exporter)
        assert span.name == 'guardrail blocked tool result'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'tool result',
            'guardrail.action': 'block',
            'guardrail.tool': 'lookup',
        }

    async def test_redaction_span_carries_values_when_content_is_included(self):
        tracer, exporter = _recording_tracer()
        guard = ToolGuardrail[object](result_guard=lambda info: GuardrailResult.replace('clean'))

        await _guard_result(guard, 'dirty', ctx=_run_ctx(trace_include_content=True, tracer=tracer))

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted tool result'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'tool result',
            'guardrail.action': 'replace',
            'guardrail.tool': 'lookup',
            'guardrail.original': 'dirty',
            'guardrail.replacement': 'clean',
        }

    async def test_redaction_span_omits_values_by_default(self):
        tracer, exporter = _recording_tracer()
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.replace({'query': 'clean'}))

        await _guard_args(guard, {'query': 'dirty'}, ctx=_run_ctx(tracer=tracer))

        span = _only_span(exporter)
        assert span.name == 'guardrail redacted tool args'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'tool args',
            'guardrail.action': 'replace',
            'guardrail.tool': 'lookup',
        }

    async def test_the_approval_span_is_the_only_record_of_the_call(self):
        """Deferring means the tool never runs, so no `execute_tool` span carries the arguments."""
        tracer, exporter = _recording_tracer()
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.approve())

        with pytest.raises(ApprovalRequired):
            await _guard_args(guard, {'q': 'x'}, ctx=_run_ctx(trace_include_content=True, tracer=tracer))

        span = _only_span(exporter)
        assert span.name == 'guardrail deferred tool args'
        assert dict(span.attributes or {}) == {
            'guardrail.direction': 'tool args',
            'guardrail.action': 'approve',
            'guardrail.tool': 'lookup',
            'guardrail.tool_call_id': 'call-1',
            'guardrail.arguments': "{'q': 'x'}",
        }

    async def test_the_approval_span_omits_the_arguments_by_default(self):
        tracer, exporter = _recording_tracer()
        guard = ToolGuardrail[object](guard=lambda call: GuardrailResult.approve())

        with pytest.raises(ApprovalRequired):
            await _guard_args(guard, {'q': 'x'}, ctx=_run_ctx(tracer=tracer))

        assert dict(_only_span(exporter).attributes or {}) == {
            'guardrail.direction': 'tool args',
            'guardrail.action': 'approve',
            'guardrail.tool': 'lookup',
            'guardrail.tool_call_id': 'call-1',
        }


class TestOrdering:
    """`ToolGuardrail` sits innermost so it sees final arguments and raw results."""

    def test_declares_innermost(self):
        assert ToolGuardrail[object]().get_ordering() == CapabilityOrdering(position='innermost')

    async def test_result_guard_runs_before_an_outer_capability_rewrites_the_result(self):
        """The guard is listed *first*, so only `position='innermost'` can put it second.

        A `ToolGuardrail` subclass would declare `innermost` too, leaving list order to decide and
        the assertion true whatever `get_ordering` returned. The truncating capability here is
        a plain one, the way `ToolOutputLimits` is.
        """
        order: list[str] = []

        class Truncating(AbstractCapability[object]):
            async def after_tool_execute(
                self,
                ctx: RunContext[object],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
            ) -> Any:
                order.append('outer')
                return str(result)[:4]

        seen: list[object] = []

        def record(info: ToolResultInfo) -> bool:
            order.append('guard')
            seen.append(info.result)
            return True

        agent = Agent(
            _scripted(_calls_lookup(query='k'), ModelResponse(parts=[TextPart(content='done')])),
            deps_type=type(None),
            capabilities=[ToolGuardrail[None](result_guard=record), Truncating()],
        )

        @agent.tool_plain
        def lookup(query: str) -> str:
            return 'the full untruncated result'

        await agent.run('hi')

        assert order == ['guard', 'outer']
        assert seen == ['the full untruncated result'], 'the guard saw the truncated result'


class TestSharedVerdicts:
    """`approve` joins the shared vocabulary, and the other guards reject it."""

    def test_an_unsupplied_replacement_reads_as_unset(self):
        assert 'replacement=<unset>' in repr(GuardrailResult.allow())

    def test_replace_still_requires_a_value(self):
        with pytest.raises(UserError, match="action='replace'"):
            GuardrailResult(action='replace')

    @pytest.mark.parametrize(
        ('action', 'kwargs'),
        [
            ('block', {'replacement': 'ignored'}),
            ('retry', {'message': 'm', 'replacement': 'ignored'}),
            ('replace', {'replacement': 'x', 'message': 'ignored'}),
        ],
    )
    def test_a_field_the_action_does_not_read_is_refused(self, action: str, kwargs: dict[str, object]):
        """Accepting it would discard a substitution or a message the guard believed it had set."""
        with pytest.raises(UserError, match=f"action='{action}'"):
            GuardrailResult(action=action, **kwargs)  # type: ignore[arg-type]

    def test_allow_rejects_an_explicit_none_replacement(self):
        with pytest.raises(UserError, match="action='allow'"):
            GuardrailResult(action='allow', replacement=None)

    def test_approve_rejects_a_message(self):
        with pytest.raises(UserError, match="action='approve'"):
            GuardrailResult(action='approve', message='why')

    def test_approve_rejects_a_replacement(self):
        with pytest.raises(UserError, match="action='approve'"):
            GuardrailResult(action='approve', replacement='x')

    async def test_input_guard_rejects_approve(self):
        from pydantic_ai_harness import InputGuardrail

        agent = Agent(TestModel(), capabilities=[InputGuardrail(guard=lambda prompt: GuardrailResult.approve())])

        with pytest.raises(UserError, match='approval applies to tool calls only'):
            await agent.run('hi')

    async def test_output_guard_rejects_approve(self):
        from pydantic_ai_harness import OutputGuardrail

        agent = Agent(TestModel(), capabilities=[OutputGuardrail(guard=lambda output: GuardrailResult.approve())])

        with pytest.raises(UserError, match='approval applies to tool calls only'):
            await agent.run('hi')
