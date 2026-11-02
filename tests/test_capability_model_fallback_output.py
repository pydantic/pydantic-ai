from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Literal, assert_type

import anyio
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, TypeAdapter
from pytest_mock import MockerFixture

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentSpec,
    AgentStreamEvent,
    ConcurrencyLimiter,
    ModelRetry,
    NativeOutput,
    PromptedOutput,
    RunContext,
    ToolDefinition,
    ToolOutput,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    CombinedCapability,
    Hooks,
    Instrumentation,
    ModelFallback,
    OutputFallbackEvent,
    OutputRetryContext,
    OutputRetrySource,
    ResolveModelId,
    SelectModel,
    WrapModelRequestHandler,
    WrapperCapability,
)
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelHTTPError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    CapabilityEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryFeedbackCause,
    RetryFeedbackPart,
    RetryInfo,
    TextPart,
    ToolCallPart,
    ToolRetryCause,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelSelectionContext
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, UsageLimits
from tests.conftest import try_import


class Answer(BaseModel):
    number: int


@dataclass
class RetryRequest(AbstractCapability[object]):
    after: bool

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if not self.after and ctx.retry == 0:
            raise ModelRetry('Retry the request.')
        return request_context

    async def after_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        if self.after and ctx.retry == 0:
            raise ModelRetry('Retry the request.')
        return response


class TestOutputFallback:
    async def test_last_candidate_keeps_its_wrapper(self) -> None:
        limiter = ConcurrencyLimiter(max_running=1)
        running: list[int] = []

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            running.append(limiter.running_count)
            return ModelResponse(parts=[TextPart('draft' if len(running) == 1 else 'ready')])

        agent = Agent(
            TestModel(custom_output_text='draft'),
            capabilities=[
                ModelFallback(ConcurrencyLimitedModel(FunctionModel(backup), limiter), output_retry_threshold=1)
            ],
            retries={'output': 2},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert running == [1, 1]

    async def test_thresholds_can_be_configured_per_model(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')])

        def middle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('middle')
            return ModelResponse(parts=[TextPart('draft')])

        first = FunctionModel(primary, model_name='primary')
        second = FunctionModel(middle, model_name='middle')
        thresholds: dict[str, int] = {first.model_id: 2, second.model_id: 1}
        fallback = ModelFallback(second, TestModel(custom_output_text='ready'), output_retry_threshold=thresholds)
        thresholds[first.model_id] = 1
        agent = Agent(first, capabilities=[fallback], retries={'output': 3})

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['primary', 'primary', 'middle']

    @pytest.mark.parametrize('thresholds', [{}, {'unused:model': 1}])
    async def test_mapping_leaves_unlisted_models_unchanged(self, thresholds: dict[str, int]) -> None:
        agent = Agent(
            TestModel(custom_output_text='draft'),
            capabilities=[ModelFallback(TestModel(custom_output_text='ready'), output_retry_threshold=thresholds)],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            assert value == 'draft'
            raise ModelRetry('Finish the draft.')

        with pytest.raises(UnexpectedModelBehavior, match='output retries'):
            await agent.run('Answer.')

    async def test_suspended_continuation_precedes_output_routing(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')  # pragma: no cover
            raise AssertionError('Continue the pinned response.')  # pragma: no cover

        def pinned_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('pinned')
            assert isinstance(messages[-1], ModelResponse) and messages[-1].state == 'suspended'
            return ModelResponse(parts=[TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            assert isinstance(messages[-1], ModelRequest)
            assert any(
                isinstance(part, UserPromptPart) and 'Finish the draft.' in part.content for part in messages[-1].parts
            )
            return ModelResponse(parts=[TextPart('ready')])

        pinned = FunctionModel(pinned_response, model_name='pinned')
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('Answer.')]),
            ModelResponse(
                parts=[TextPart('partial')],
                state='suspended',
                provider_response_id='job-1',
                model_id=pinned.model_id,
            ),
        ]
        agent = Agent(
            FunctionModel(primary),
            capabilities=[ModelFallback(pinned, FunctionModel(backup), output_retry_threshold=1)],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value.endswith('draft'):
                raise ModelRetry('Finish the draft.')
            return value

        result = await agent.run(message_history=history)
        assert result.output == 'ready'
        assert any(
            isinstance(part, RetryFeedbackPart)
            for message in result.all_messages()
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        assert calls == ['pinned', 'backup']

    async def test_producer_counts_survive_request_hook_retries(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            if calls.count('primary') == 2:
                raise ModelHTTPError(503, 'primary')
            return ModelResponse(parts=[TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            return ModelResponse(parts=[TextPart('request hook' if calls.count('backup') == 1 else 'ready')])

        def after_request(
            ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            if response.text == 'request hook':
                raise ModelRetry('Retry the request hook.')
            return response

        agent = Agent(
            FunctionModel(primary),
            capabilities=[
                ModelFallback(FunctionModel(backup), output_retry_threshold=2),
                Hooks(after_model_request=after_request),
            ],
            retries={'output': 3},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['primary', 'primary', 'backup', 'primary', 'backup']

    @pytest.mark.parametrize('enabled', [False, True])
    async def test_output_fallback_is_opt_in(self, enabled: bool) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            return ModelResponse(parts=[TextPart('ready')])

        candidate = FunctionModel(backup)
        fallback = ModelFallback(candidate, output_retry_threshold=1) if enabled else ModelFallback(candidate)
        agent = Agent(FunctionModel(primary), capabilities=[fallback])

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        if enabled:
            assert (await agent.run('Answer.')).output == 'ready'
            assert calls == ['primary', 'backup']
        else:
            with pytest.raises(UnexpectedModelBehavior, match='output retries'):
                await agent.run('Answer.')
            assert calls == ['primary', 'primary']

    async def test_request_fallback_does_not_return_to_a_retired_candidate(self) -> None:
        calls: list[str] = []
        tools: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')])

        def middle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('middle')
            if tools:
                raise ModelHTTPError(503, 'middle')
            return ModelResponse(parts=[ToolCallPart('lookup', {})])

        def final(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('final')
            return ModelResponse(parts=[TextPart('draft' if calls.count('final') == 1 else 'ready')])

        def lookup() -> str:
            tools.append('lookup')
            return 'evidence'

        agent = Agent(
            FunctionModel(primary),
            capabilities=[ModelFallback(FunctionModel(middle), FunctionModel(final), output_retry_threshold=1)],
            tools=[lookup],
            retries={'output': 2},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['primary', 'middle', 'middle', 'final', 'final']
        assert tools == ['lookup']

    @pytest.mark.parametrize('source', ['validator', 'empty', 'output_tool', 'function_tool', 'before', 'after'])
    async def test_retry_sources(
        self, source: Literal['validator', 'empty', 'output_tool', 'function_tool', 'before', 'after']
    ) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            if source == 'output_tool':
                return ModelResponse(parts=[ToolCallPart('final_result', {'number': 'bad'})])
            if source == 'function_tool' and len(calls) == 1:
                return ModelResponse(parts=[ToolCallPart('lookup', {})])
            return ModelResponse(parts=[] if source == 'empty' else [TextPart('draft')])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            if source == 'output_tool':
                return ModelResponse(parts=[ToolCallPart('final_result', {'number': 42})])
            return ModelResponse(parts=[TextPart('ready')])

        capability = ModelFallback(
            FunctionModel(primary, model_name='primary'),
            FunctionModel(backup, model_name='backup'),
            output_retry_threshold=1,
        )
        capabilities: list[AbstractCapability[object]] = [capability]
        if source in ('before', 'after'):
            capabilities.append(RetryRequest(after=source == 'after'))
        agent = Agent(
            capabilities=capabilities,
            output_type=Answer if source == 'output_tool' else str,
            retries={'output': 1},
        )
        if source == 'validator':

            @agent.output_validator
            def validate(value: Answer | str) -> Answer | str:
                if value == 'draft':
                    raise ModelRetry('Return a finished answer.')
                return value

        if source == 'function_tool':

            @agent.tool_plain
            def lookup() -> str:
                raise ModelRetry('Retry lookup.')

        await agent.run('Answer.')
        assert calls == (
            ['primary', 'backup']
            if source in ('validator', 'empty', 'output_tool')
            else ['primary']
            if source == 'before'
            else ['primary', 'primary']
        )

    @pytest.mark.parametrize('mode', ['native', 'prompted', 'tool'])
    async def test_schema_validation(self, mode: Literal['native', 'prompted', 'tool']) -> None:
        calls: list[str] = []

        def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('call')
            args = {'number': 'invalid' if len(calls) == 1 else 42}
            return ModelResponse(
                parts=[ToolCallPart('final_result', args) if mode == 'tool' else TextPart(json.dumps(args))]
            )

        profile = ModelProfile(supports_json_schema_output=True)
        first = FunctionModel(response, model_name='first', profile=profile)
        second = FunctionModel(response, model_name='second', profile=profile)
        fallback = ModelFallback(first, second, output_retry_threshold=1)
        output = NativeOutput(Answer) if mode == 'native' else PromptedOutput(Answer) if mode == 'prompted' else Answer
        result = await Agent(output_type=output, capabilities=[fallback]).run('Answer.')
        assert result.output == Answer(number=42)
        assert [m.model_name for m in result.all_messages() if isinstance(m, ModelResponse)] == ['first', 'second']

    async def test_thresholds_history_tools_and_usage(self) -> None:
        calls: list[str] = []
        tool_calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[TextPart('draft')], usage=RequestUsage(input_tokens=10, output_tokens=2))

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            if not tool_calls:
                return ModelResponse(parts=[ToolCallPart('lookup', {})])
            return ModelResponse(parts=[TextPart('ready')])

        def lookup() -> str:
            tool_calls.append('lookup')
            return 'evidence'

        fallback = ModelFallback(FunctionModel(primary), FunctionModel(backup), output_retry_threshold=2)
        agent = Agent(capabilities=[fallback], retries={'output': 3}, tools=[lookup])

        @agent.output_validator
        def validate(ctx: RunContext[object], value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            assert ctx.retry == 2
            return value

        result = await agent.run('Answer.')
        assert calls == ['primary', 'primary', 'backup', 'backup']
        assert tool_calls == ['lookup']
        assert result.usage.input_tokens >= 20
        feedback = [
            p
            for m in result.all_messages()
            if isinstance(m, ModelRequest)
            for p in m.parts
            if isinstance(p, RetryFeedbackPart)
        ]
        assert [p.retry.attempt for p in feedback] == [1, 2]
        assert all(p.content == 'Finish the draft.' for p in feedback)
        with pytest.raises(UsageLimitExceeded, match='request_limit'):
            await agent.run('Answer.', usage_limits=UsageLimits(request_limit=2))

    @pytest.mark.parametrize('tool_budget', [0, 1, 3])
    async def test_output_tool_budget(self, tool_budget: int) -> None:
        calls: list[str] = []

        def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('request')
            return ModelResponse(parts=[ToolCallPart('final_result', {'number': 'bad'})])

        model = FunctionModel(invalid)
        agent = Agent(
            output_type=ToolOutput(Answer, max_retries=tool_budget),
            retries={'output': 1},
            capabilities=[ModelFallback(model, model, output_retry_threshold=1)],
        )
        with pytest.raises(UnexpectedModelBehavior, match=f'output retries \\({tool_budget}\\)'):
            await agent.run('Answer.')
        assert len(calls) == tool_budget + 1

    @pytest.mark.parametrize('selection', ['ladder', 'last', 'run', 'override'])
    async def test_selection_precedence(self, selection: Literal['ladder', 'last', 'run', 'override']) -> None:
        first = TestModel(custom_output_text='ladder')
        other = TestModel(custom_output_text='other')
        fallback = ModelFallback(first, output_retry_threshold=1)
        caps: list[AbstractCapability[object]] = [SelectModel(lambda ctx: other), fallback]
        if selection == 'last':
            caps.reverse()
        agent = Agent(other, capabilities=caps)
        if selection == 'override':
            with agent.override(model=other):
                result = await agent.run('Answer.', model=first)
        elif selection == 'run':
            with pytest.warns(UserWarning, match='output_retry_threshold=1'):
                result = await agent.run('Answer.', model=other)
        else:
            result = await agent.run('Answer.')
        assert result.output == 'other'

    async def test_request_and_output_failures_share_chain(self) -> None:
        calls: list[str] = []

        def failed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('failed')
            raise ModelHTTPError(503, 'failed')

        def rejected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('rejected')
            return ModelResponse(parts=[])

        def valid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('valid')
            return ModelResponse(parts=[TextPart('ready')])

        agent = Agent(
            capabilities=[
                ModelFallback(
                    FunctionModel(failed), FunctionModel(rejected), FunctionModel(valid), output_retry_threshold=1
                ),
            ]
        )
        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['failed', 'rejected', 'valid']

    @pytest.mark.parametrize('threshold', [-1, 0, True])
    def test_invalid_threshold(self, threshold: int) -> None:
        with pytest.raises(UserError, match='positive'):
            ModelFallback('test', output_retry_threshold=threshold)

    async def test_tool_switch_does_not_reset_counters(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(parts=[ToolCallPart('first', {'number': 'bad'})])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('backup')
            return ModelResponse(parts=[ToolCallPart('second' if len(calls) == 2 else 'first', {'number': 'bad'})])

        agent = Agent(
            output_type=[
                ToolOutput(Answer, name='first', max_retries=1),
                ToolOutput(Answer, name='second', max_retries=3),
            ],
            retries={'output': 1},
            capabilities=[ModelFallback(FunctionModel(primary), FunctionModel(backup), output_retry_threshold=1)],
        )
        with pytest.raises(UnexpectedModelBehavior, match='output retries \\(1\\)'):
            await agent.run('Answer.')
        assert calls == ['primary', 'backup', 'backup']

    async def test_failed_siblings_do_not_skip_an_untried_candidate(self) -> None:
        calls: list[str] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('primary')
            return ModelResponse(
                parts=[
                    ToolCallPart('first', {'number': 'bad'}, 'one'),
                    ToolCallPart('second', {'number': 'bad'}, 'two'),
                ]
            )

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('final')
            return ModelResponse(parts=[ToolCallPart('first', {'number': 42})])

        def middle(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('middle')
            return ModelResponse(parts=[ToolCallPart('first', {'number': 'bad'})])

        agent = Agent(
            output_type=[ToolOutput(Answer, name='first'), ToolOutput(Answer, name='second')],
            retries={'output': 3},
            end_strategy='exhaustive',
            capabilities=[
                ModelFallback(
                    FunctionModel(primary), FunctionModel(middle), FunctionModel(backup), output_retry_threshold=1
                )
            ],
        )
        assert (await agent.run('Answer.')).output == Answer(number=42)
        assert calls == ['primary', 'middle', 'final']

    @pytest.mark.parametrize('mode', ['native', 'prompted', 'tool'])
    async def test_output_function_retry(self, mode: Literal['native', 'prompted', 'tool']) -> None:
        calls: list[str] = []

        def output(number: int) -> int:
            if number == 0:
                raise ModelRetry('Return a positive number.')
            return number

        def response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('call')
            args = {'number': 0 if len(calls) == 1 else 42}
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, args) if mode == 'tool' else TextPart(json.dumps(args))]
            )

        model = FunctionModel(response, profile=ModelProfile(supports_json_schema_output=True))
        agent = Agent(
            output_type=NativeOutput(output)
            if mode == 'native'
            else PromptedOutput(output)
            if mode == 'prompted'
            else ToolOutput(output),
            capabilities=[ModelFallback(model, model, output_retry_threshold=1)],
        )
        assert (await agent.run('Answer.')).output == 42
        assert calls == ['call', 'call']

    async def test_selected_model_prepares_its_own_output_mode_and_settings(self) -> None:
        seen: list[tuple[str, float | None]] = []

        def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append((info.model_request_parameters.output_mode, (info.model_settings or {}).get('temperature')))
            return ModelResponse(parts=[ToolCallPart('final_result', {'number': 'bad'})])

        def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append((info.model_request_parameters.output_mode, (info.model_settings or {}).get('temperature')))
            return ModelResponse(parts=[TextPart('{"number": 42}')])

        first = FunctionModel(primary, settings={'temperature': 0.1})
        second = FunctionModel(
            backup,
            settings={'temperature': 0.9},
            profile=ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native'),
        )
        agent = Agent(
            output_type=Answer,
            capabilities=[ModelFallback(first, second, output_retry_threshold=1)],
        )
        assert (await agent.run('Answer.')).output == Answer(number=42)
        assert seen == [('tool', 0.1), ('native', 0.9)]


@pytest.mark.parametrize('threshold', [1, {'test:test': 1}])
async def test_inferred_by_name_default_switches_after_output_retry(threshold: int | dict[str, int]) -> None:
    selected: list[str] = []

    def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        selected.append(ctx.model_id or ctx.model.model_id)
        return request_context

    agent = Agent(
        'test',
        deps_type=object,
        capabilities=[
            ModelFallback(TestModel(custom_output_text='ready', model_name='backup'), output_retry_threshold=threshold),
            Hooks(before_model_request=observe),
        ],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value != 'ready':
            raise ModelRetry('Use the backup.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert selected == ['test:test', 'test:backup']


with try_import() as otel_sdk_installed:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@dataclass
class RecordFallbacks(AbstractCapability[object]):
    events: list[OutputFallbackEvent] = field(default_factory=list[OutputFallbackEvent])

    async def on_event(self, ctx: RunContext[object], *, event: AgentStreamEvent) -> None:
        if isinstance(event, OutputFallbackEvent):
            self.events.append(event)


class TestOutputFallbackEvent:
    @pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')
    async def test_output_policy_yields_to_selected_model(self) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        model = TestModel(custom_output_text='draft')
        agent = Agent(
            capabilities=[
                ModelFallback(TestModel(custom_output_text='ready'), output_retry_threshold=1),
                SelectModel(lambda ctx: model),
                Instrumentation(settings=InstrumentationSettings(tracer_provider=provider)),
            ]
        )

        @agent.output_validator
        def validate(ctx: RunContext[object], value: str) -> str:
            if ctx.retry == 0:
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'draft'
        assert (
            sum(
                event.name == 'model_fallback.output_fallback'
                for span in exporter.get_finished_spans()
                for event in span.events
            )
            == 0
        )

    @pytest.mark.parametrize('include_content', [False, True])
    @pytest.mark.parametrize('producer', ['test', None])
    @pytest.mark.skipif(not otel_sdk_installed(), reason='opentelemetry-sdk not installed')
    async def test_retry_events_and_single_trace(self, include_content: bool, producer: str | None) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        recorder = RecordFallbacks()
        hooks = Hooks[object]()

        @hooks.on.after_model_request
        def set_producer(
            ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            response.model_name = producer
            return response

        fallback = ModelFallback(
            TestModel(custom_output_text='draft', model_name='primary'),
            TestModel(custom_output_text='ready', model_name='backup'),
            output_retry_threshold=1,
        )
        fallback.id = 'fallback'
        agent = Agent(
            capabilities=[
                fallback,
                recorder,
                hooks,
                Instrumentation(
                    settings=InstrumentationSettings(tracer_provider=provider, include_content=include_content)
                ),
            ],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        [event] = recorder.events
        assert_type(event.retry, RetryInfo[RetryFeedbackCause | ToolRetryCause])
        assert_type(event.source, OutputRetrySource)
        assert event.source == 'response_output'
        assert (event.source_model_id, event.target_model_id) == ('test:primary', 'test:backup')
        assert (event.retry.cause, event.retry.attempt, event.retry.max_retries) == ('model_retry', 1, 1)
        assert event.capability_id == 'fallback'
        adapter = TypeAdapter(CapabilityEvent)
        assert adapter.validate_json(adapter.dump_json(event)) == event
        spans = exporter.get_finished_spans()
        assert not any(span.name == 'model_fallback.output_fallback' for span in spans)
        [transition] = [e for s in spans for e in s.events if e.name == 'model_fallback.output_fallback']
        attributes: dict[str, object] = dict(transition.attributes or {})
        assert attributes == snapshot(
            {
                'model_fallback.source_model_id': 'test:primary',
                'model_fallback.target_model_id': 'test:backup',
                'model_fallback.source': 'response_output',
            }
        )
        assert len({s.context.trace_id for s in spans if s.context is not None}) == 1
        assert len([s for s in spans if s.parent is None]) == 1

    async def test_concurrent_runs_and_history_start_at_first_level(self) -> None:
        recorder = RecordFallbacks()
        agent = Agent(
            capabilities=[
                ModelFallback(
                    TestModel(custom_output_text='draft'),
                    TestModel(custom_output_text='ready'),
                    output_retry_threshold=1,
                ),
                recorder,
            ]
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        async def run() -> None:
            assert (await agent.run('Answer.')).output == 'ready'

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            group.start_soon(run)
        first = await agent.run('Answer.')
        assert (await agent.run('Answer again.', message_history=first.all_messages())).output == 'ready'
        assert len(recorder.events) == 4

    @pytest.mark.parametrize('threshold', [1, 2])
    async def test_applied_switch_uses_selection_aliases(self, threshold: int) -> None:
        recorder = RecordFallbacks()
        models: dict[str, TestModel] = {
            'primary': TestModel(custom_output_text='draft'),
            'backup': TestModel(custom_output_text='ready'),
        }
        agent = Agent(
            'primary',
            capabilities=[
                ModelFallback('backup', output_retry_threshold=threshold),
                ResolveModelId(lambda ctx, model_id: models.get(model_id)),
                recorder,
            ],
            retries={'output': threshold},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        [event] = recorder.events
        assert (event.source_model_id, event.target_model_id) == ('primary', 'backup')
        assert event.retry.attempt == threshold

    async def test_exhausted_budget_does_not_emit_an_unapplied_switch(self) -> None:
        recorder = RecordFallbacks()
        agent = Agent(
            TestModel(custom_output_text='draft'),
            capabilities=[ModelFallback(TestModel(), output_retry_threshold=2), recorder],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            raise ModelRetry('Finish the draft.')

        with pytest.raises(UnexpectedModelBehavior, match='output retries'):
            await agent.run('Answer.')
        assert recorder.events == []

    async def test_last_candidate_retries_without_another_switch(self) -> None:
        recorder = RecordFallbacks()
        calls: list[str] = []

        async def backup(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            calls.append('backup')
            yield 'draft' if len(calls) == 1 else 'ready'

        agent = Agent(
            TestModel(custom_output_text='draft'),
            capabilities=[ModelFallback(FunctionModel(stream_function=backup), output_retry_threshold=1), recorder],
            retries={'output': 2},
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert calls == ['backup', 'backup']
        assert len(recorder.events) == 1

    @pytest.mark.parametrize('method', ['run', 'handler', 'events'])
    async def test_applied_output_tool_switch_reaches_stream_consumers(
        self, method: Literal['run', 'handler', 'events']
    ) -> None:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            yield {0: DeltaToolCall(name='answer', json_args='{"number": "bad"}', tool_call_id='one')}

        recorder = RecordFallbacks()
        agent = Agent(
            output_type=ToolOutput(Answer, name='answer'),
            capabilities=[
                ModelFallback(
                    FunctionModel(stream_function=stream, model_name='primary'),
                    TestModel(custom_output_args={'number': 42}, model_name='backup'),
                    output_retry_threshold=1,
                ),
                recorder,
            ],
        )
        streamed_events: list[AgentStreamEvent] = []
        if method == 'handler':

            async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
                streamed_events.extend([event async for event in stream])

            assert (await agent.run('Answer.', event_stream_handler=collect)).output == Answer(number=42)
        elif method == 'events':
            async with agent.run_stream_events('Answer.') as event_stream:
                async for event in event_stream:
                    if isinstance(event, AgentRunResultEvent):
                        assert event.result.output == Answer(number=42)
                    else:
                        streamed_events.append(event)
        else:
            assert (await agent.run('Answer.')).output == Answer(number=42)
        [event] = recorder.events
        assert (event.source_model_id, event.target_model_id) == ('function:primary', 'test:backup')
        assert (event.source, event.tool_name, event.tool_call_id) == ('output_tool', 'answer', 'one')
        if method != 'run':
            assert [e for e in streamed_events if isinstance(e, OutputFallbackEvent)] == [event]

    @pytest.mark.parametrize('method', ['run', 'handler', 'events'])
    async def test_successful_sibling_does_not_emit_an_unapplied_switch(
        self, method: Literal['run', 'handler', 'events']
    ) -> None:
        calls: list[str] = []

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            calls.append('primary')
            yield {
                0: DeltaToolCall(name='first', json_args='{"number": "bad"}', tool_call_id='one'),
                1: DeltaToolCall(name='second', json_args='{"number": 42}', tool_call_id='two'),
            }

        recorder = RecordFallbacks()
        retries: list[OutputRetryContext] = []

        def record_retry(ctx: RunContext[object], *, retry_context: OutputRetryContext) -> None:
            retries.append(retry_context)

        agent = Agent(
            output_type=[ToolOutput(Answer, name='first'), ToolOutput(Answer, name='second')],
            end_strategy='exhaustive',
            capabilities=[
                ModelFallback(
                    FunctionModel(stream_function=stream),
                    TestModel(custom_output_args={'number': -1}),
                    output_retry_threshold=1,
                ),
                recorder,
                Hooks(output_retry=record_retry),
            ],
        )
        streamed_events: list[AgentStreamEvent] = []
        if method == 'handler':

            async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
                streamed_events.extend([event async for event in stream])

            assert (await agent.run('Answer.', event_stream_handler=collect)).output == Answer(number=42)
        elif method == 'events':
            async with agent.run_stream_events('Answer.') as event_stream:
                async for event in event_stream:
                    if isinstance(event, AgentRunResultEvent):
                        assert event.result.output == Answer(number=42)
                    else:
                        streamed_events.append(event)
        else:
            assert (await agent.run('Answer.')).output == Answer(number=42)
        [retry] = retries
        assert (retry.source, retry.tool_name, retry.tool_call_id) == ('output_tool', 'first', 'one')
        assert (retry.retry.attempt, retry.retry.max_retries) == (1, 1)
        assert recorder.events == []
        assert calls == ['primary']
        if method != 'run':
            assert [e for e in streamed_events if isinstance(e, OutputFallbackEvent)] == []


@pytest.mark.parametrize('alias', [False, True])
async def test_untried_model_instance_with_same_id_is_not_retired(alias: bool) -> None:
    primary = TestModel(custom_output_text='draft')
    backup = TestModel(custom_output_text='ready')
    assert primary.model_id == backup.model_id
    served: list[Model] = []

    def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        served.append(request_context.model)
        return request_context

    capabilities: list[AbstractCapability[object]] = [
        ModelFallback(backup, output_retry_threshold=1),
        Hooks(before_model_request=observe),
    ]
    if alias:
        capabilities.insert(0, ResolveModelId(lambda ctx, model_id: primary if model_id == primary.model_id else None))
    agent = Agent(primary.model_id if alias else primary, deps_type=object, capabilities=capabilities)

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert len(served) == 2
    assert served[0] is primary
    assert served[1] is backup


@dataclass
class ObserveStepModel(AbstractCapability[object]):
    phases: list[tuple[str, int, str]]
    attempts: list[tuple[int, str, int]]

    async def before_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.phases.append(('before', ctx.run_step, request_context.model.model_name))
        assert ctx.model is request_context.model
        return request_context

    async def wrap_model_request(
        self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        self.phases.append(('wrap', ctx.run_step, request_context.model.model_name))
        assert ctx.model is request_context.model
        return await handler(request_context)

    async def prepare_model_request(
        self, ctx: RunContext[object], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.attempts.append((ctx.run_step, request_context.model.model_name, request_context.attempt))
        return request_context

    async def prepare_tools(self, ctx: RunContext[object], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        self.phases.append(('tools', ctx.run_step, ctx.model.model_name))
        return tool_defs


@dataclass(frozen=True)
class SelectionCase:
    source: Literal['agent', 'fallback_default', 'run', 'override', 'selector_before', 'selector_after', 'wrapped']
    expected_models: tuple[str, ...] = ('primary', 'backup', 'backup')
    expected_output: str | None = 'ready'
    selected_steps: tuple[int, ...] = ()
    warning: str | None = None


SELECTION_CASES: tuple[SelectionCase, ...] = (
    SelectionCase('agent'),
    SelectionCase('fallback_default'),
    SelectionCase('run', ('primary', 'primary', 'primary'), None, warning='output_retry_threshold=1'),
    SelectionCase('override', ('primary', 'primary', 'primary'), None),
    SelectionCase('selector_before', ('primary', 'primary', 'primary'), None, (1, 1, 2, 3)),
    SelectionCase('selector_after', ('primary', 'primary', 'primary'), None, (1, 1, 2, 3)),
    SelectionCase('wrapped'),
)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.source) for case in SELECTION_CASES])
async def test_output_switch_selects_before_step_hooks(case: SelectionCase) -> None:
    source = case.source
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []
    selected_steps: list[int] = []
    backup_calls = 0

    def primary_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('draft')])

    def backup_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal backup_calls
        backup_calls += 1
        if backup_calls == 1:
            return ModelResponse(parts=[ToolCallPart('lookup', {})])
        return ModelResponse(parts=[TextPart('ready')])

    primary = FunctionModel(primary_response, model_name='primary')
    backup = FunctionModel(backup_response, model_name='backup')
    fallback = ModelFallback[object](
        *([primary, backup] if source == 'fallback_default' else [backup]), output_retry_threshold=1
    )
    policy = WrapperCapability(fallback) if source == 'wrapped' else fallback
    capabilities: list[AbstractCapability[object]] = [ObserveStepModel(phases, attempts), policy]

    def select(ctx: ModelSelectionContext[object]) -> Model:
        selected_steps.append(ctx.run_step)
        return primary

    if source == 'selector_before':
        capabilities.insert(0, SelectModel(select))
    elif source == 'selector_after':
        capabilities.append(SelectModel(select))

    def settings(ctx: RunContext[object]) -> ModelSettings:
        phases.append(('settings', ctx.run_step, ctx.model.model_name))
        return {'temperature': 0.2}

    def lookup(ctx: RunContext[object]) -> str:
        phases.append(('tool', ctx.run_step, ctx.model.model_name))
        return 'evidence'

    other = TestModel(custom_output_text='wrong')
    agent = Agent(
        primary if source in {'agent', 'wrapped'} else None if source == 'fallback_default' else other,
        deps_type=object,
        capabilities=capabilities,
        tools=[lookup],
        model_settings=settings,
        retries={'output': 2},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    output: str | None = None
    with (
        pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries')
        if case.expected_output is None
        else nullcontext()
    ):
        with pytest.warns(UserWarning, match=case.warning) if case.warning is not None else nullcontext():
            if source == 'override':
                with agent.override(model=primary):
                    output = (await agent.run('Answer.', model=other)).output
            elif source == 'run':
                output = (await agent.run('Answer.', model=primary)).output
            else:
                output = (await agent.run('Answer.')).output
    assert output == case.expected_output
    assert attempts == [(step, model, 1) for step, model in enumerate(case.expected_models, 1)]
    assert all(model == case.expected_models[step - 1] for _, step, model in phases)
    assert backup_calls == case.expected_models.count('backup')
    if case.expected_output is not None:
        assert ('tool', 2, 'backup') in phases
    for step in (1, 2, 3):
        assert {'before', 'wrap', 'settings', 'tools'} <= {phase for phase, number, _ in phases if number == step}
    # Resolving the run-bound fallback copy re-evaluates the bootstrap selector on step one.
    assert tuple(selected_steps) == case.selected_steps


@pytest.mark.parametrize('threshold', [None, 1])
async def test_wrapped_container_keeps_output_selection_a_default(threshold: int | None) -> None:
    primary = TestModel(custom_output_text='primary')
    selected = TestModel(custom_output_text='selected')
    agent = Agent(
        capabilities=[
            SelectModel(lambda ctx: selected),
            WrapperCapability(CombinedCapability([ModelFallback(primary, output_retry_threshold=threshold)])),
        ]
    )

    # Static container contributions retain their existing precedence. The opt-in run-bound
    # output selector stays a default even through a wrapped container.
    assert (await agent.run('Answer.')).output == ('primary' if threshold is None else 'selected')


async def test_output_counts_survive_serving_model_reconstruction() -> None:
    calls: list[str] = []
    reconstructed: list[FunctionModel] = []
    hooks = Hooks[object]()

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        return ModelResponse(parts=[TextPart('draft')])

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('backup')
        return ModelResponse(parts=[TextPart('ready')])

    @hooks.on.before_model_request
    def reconstruct(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        if request_context.model.model_name == 'primary':
            model = FunctionModel(primary, model_name='primary')
            reconstructed.append(model)
            return replace(request_context, model=model)
        return request_context

    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        capabilities=[hooks, ModelFallback(FunctionModel(backup), output_retry_threshold=2)],
        retries={'output': 3},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert calls == ['primary', 'primary', 'backup']
    assert reconstructed[0] is not reconstructed[1]


async def test_output_counts_use_configured_aliases() -> None:
    served: list[str | None] = []
    hooks = Hooks[object]()
    primary = TestModel(custom_output_text='draft')
    models: dict[str, Model] = {'first': primary, 'second': primary, 'backup': TestModel(custom_output_text='ready')}

    @hooks.on.before_model_request
    def record(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        served.append(request_context.model_id)
        return request_context

    agent = Agent(
        'first',
        capabilities=[
            hooks,
            ModelFallback('second', 'backup', output_retry_threshold={'first': 2, 'second': 2}),
            ResolveModelId(lambda ctx, model_id: models.get(model_id)),
        ],
        retries={'output': 4},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    assert (await agent.run('Answer.')).output == 'ready'
    assert served == ['first', 'first', 'second', 'second', 'backup']


@dataclass(frozen=True)
class SuspendedCase:
    id: str
    pin: Literal['known', 'outside', 'missing']
    mapped: bool
    expected_models: tuple[str, ...]


SUSPENDED_CASES: tuple[SuspendedCase, ...] = (
    SuspendedCase('known-uniform', 'known', False, ('pinned', 'backup')),
    SuspendedCase('known-mapped', 'known', True, ('pinned', 'backup')),
    SuspendedCase('outside-uniform', 'outside', False, ('primary', 'pinned', 'backup')),
    SuspendedCase('outside-mapped', 'outside', True, ('primary', 'pinned', 'backup')),
    SuspendedCase('missing-uniform', 'missing', False, ('primary', 'pinned', 'backup')),
    SuspendedCase('missing-mapped', 'missing', True, ('primary', 'pinned', 'backup')),
)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in SUSPENDED_CASES])
async def test_output_policy_preserves_suspended_selection(case: SuspendedCase) -> None:
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []

    def draft(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('draft')])

    primary = FunctionModel(draft, model_name='primary')
    pinned = FunctionModel(draft, model_name='pinned')
    backup = TestModel(custom_output_text='ready', model_name='backup')
    threshold: int | dict[str, int] = {primary.model_id: 1, pinned.model_id: 1} if case.mapped else 1
    fallback = ModelFallback[object](pinned, backup, output_retry_threshold=threshold)
    agent = Agent(
        primary,
        deps_type=object,
        capabilities=[ObserveStepModel(phases, attempts), fallback],
        retries={'output': 3},
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value.endswith('draft'):
            raise ModelRetry('Finish the draft.')
        return value

    pin_ids: dict[str, str | None] = {'known': pinned.model_id, 'outside': 'outside:model', 'missing': None}
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Answer.')]),
        ModelResponse(
            parts=[TextPart('partial')],
            state='suspended',
            provider_response_id='job',
            model_id=pin_ids[case.pin],
        ),
    ]
    result = await agent.run(message_history=history)

    assert result.output == 'ready'
    expected: list[tuple[int, str, int]] = [(step, model, 1) for step, model in enumerate(case.expected_models, 1)]
    assert attempts == expected
    for step, model, _ in expected:
        assert ('before', step, model) in phases
        assert ('wrap', step, model) in phases


@pytest.mark.parametrize('policy', ['disabled', 'empty', 'unlisted'])
async def test_request_only_suspended_hook_behavior_is_unchanged(
    policy: Literal['disabled', 'empty', 'unlisted'],
) -> None:
    phases: list[tuple[str, int, str]] = []
    attempts: list[tuple[int, str, int]] = []
    served: list[str] = []

    def pinned_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        served.append('pinned')
        return ModelResponse(parts=[TextPart('ready')])

    pinned = FunctionModel(pinned_response, model_name='pinned')
    threshold: dict[str, int] | None = (
        None if policy == 'disabled' else {} if policy == 'empty' else {'unused:model': 1}
    )
    agent = Agent(
        TestModel(model_name='primary'),
        deps_type=object,
        capabilities=[
            ObserveStepModel(phases, attempts),
            ModelFallback(pinned, output_retry_threshold=threshold),
        ],
    )
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Answer.')]),
        ModelResponse(
            parts=[TextPart('partial')], state='suspended', provider_response_id='job', model_id=pinned.model_id
        ),
    ]
    assert (await agent.run(message_history=history)).output.endswith('ready')
    assert served == ['pinned']
    assert ('before', 1, 'primary') in phases
    assert ('wrap', 1, 'primary') in phases
    assert attempts == [(1, 'primary', 1), (1, 'pinned', 2)]


@pytest.mark.parametrize(
    'threshold', [0, -1, True, 1.5, 'one', {'model': 0}, {'model': True}, {1: 2}, {'model': 'bad'}]
)
def test_invalid_output_retry_threshold_in_spec(threshold: object) -> None:
    with pytest.raises(ValueError, match='positive integer'):
        Agent.from_spec(
            {'capabilities': [{'ModelFallback': {'models': ['test'], 'output_retry_threshold': threshold}}]}
        )


@pytest.mark.parametrize('per_model', [False, True])
@pytest.mark.parametrize('initial_model', [False, True])
async def test_output_fallback_spec_roundtrip_and_custom_ids(per_model: bool, initial_model: bool) -> None:
    policy: int | dict[str, int] = {'app:first': 2} if per_model else 2
    specification = AgentSpec.model_validate(
        {
            'model': 'app:first' if initial_model else None,
            'retries': {'output': 2},
            'capabilities': [
                {
                    'ModelFallback': {
                        'models': ['app:next'] if initial_model else ['app:first', 'app:next'],
                        'output_retry_threshold': policy,
                    }
                },
            ],
        }
    )
    recovered = AgentSpec.model_validate_json(specification.model_dump_json())
    assert recovered == specification
    agent = Agent.from_spec(
        recovered,
        capabilities=[ResolveModelId(lambda ctx, model_id: TestModel(custom_output_text=model_id))],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'app:first':
            raise ModelRetry('Use the next candidate.')
        return value

    assert (await agent.run('Answer.')).output == 'app:next'
    assert 'output_retry_threshold' in str(AgentSpec.model_json_schema_with_capabilities())


async def test_output_retirement_restricts_sticky_fallback_and_resets_next_run() -> None:
    calls: list[str] = []
    steps: list[tuple[int, str]] = []
    backup_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        raise ModelHTTPError(status_code=500, model_name='primary')

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal backup_calls
        calls.append('backup')
        backup_calls += 1
        return ModelResponse(parts=[TextPart('draft' if backup_calls == 2 else 'ready')])

    def final(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('final')
        raise ModelHTTPError(status_code=500, model_name='final')

    def observe(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        steps.append((ctx.run_step, ctx.model.model_name))
        return request_context

    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        deps_type=object,
        capabilities=[
            ModelFallback(
                FunctionModel(backup, model_name='backup'),
                FunctionModel(final, model_name='final'),
                sticky=True,
                output_retry_threshold=1,
            ),
            Hooks(before_model_request=observe),
        ],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    first = await agent.run('Answer.')
    assert first.output == 'ready'
    assert calls == ['primary', 'backup']
    steps.clear()

    with pytest.raises(FallbackExceptionGroup):
        await agent.run('Continue.', message_history=first.all_messages())
    assert calls == ['primary', 'backup', 'backup', 'final']
    assert steps[-1] == (2, 'final')

    assert (await agent.run('Continue again.', message_history=first.all_messages())).output == 'ready'
    assert calls == ['primary', 'backup', 'backup', 'final', 'backup']


async def test_output_only_switch_does_not_start_sticky_history() -> None:
    calls: list[str] = []

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('primary')
        return ModelResponse(parts=[TextPart('draft')])

    def backup(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('backup')
        return ModelResponse(parts=[TextPart('ready')])

    agent = Agent(
        FunctionModel(primary, model_name='primary'),
        capabilities=[ModelFallback(FunctionModel(backup, model_name='backup'), sticky=True, output_retry_threshold=1)],
    )

    @agent.output_validator
    def validate(value: str) -> str:
        if value == 'draft':
            raise ModelRetry('Finish the draft.')
        return value

    first = await agent.run('Answer.')
    assert first.output == 'ready'
    assert (await agent.run('Continue.', message_history=first.all_messages())).output == 'ready'
    assert calls == ['primary', 'backup', 'primary', 'backup']


@pytest.fixture
def streaming_agent() -> tuple[Agent[object, str], list[str]]:
    calls: list[str] = []

    async def primary(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('primary')
        yield 'draft'

    async def backup(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls.append('backup')
        yield 'ready'

    agent = Agent(
        deps_type=object,
        capabilities=[
            ModelFallback(
                FunctionModel(stream_function=primary), FunctionModel(stream_function=backup), output_retry_threshold=1
            )
        ],
    )

    @agent.output_validator
    def validate(ctx: RunContext[object], value: str) -> str:
        if value == 'draft' and not ctx.partial_output:
            raise ModelRetry('Finish the draft.')
        return value

    return agent, calls


class TestStreaming:
    @pytest.mark.parametrize('method', ['handler', 'events', 'iter'])
    async def test_graph_streaming_can_advance(
        self, streaming_agent: tuple[Agent[object, str], list[str]], method: Literal['handler', 'events', 'iter']
    ) -> None:
        agent, calls = streaming_agent
        events: list[AgentStreamEvent] = []
        if method == 'handler':

            async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
                events.extend([event async for event in stream])

            result = await agent.run('Answer.', event_stream_handler=collect)
            assert result.output == 'ready'
        elif method == 'events':
            output: str | None = None
            async with agent.run_stream_events('Answer.') as stream:
                async for event in stream:
                    if isinstance(event, AgentRunResultEvent):
                        output = event.result.output
                    else:
                        events.append(event)
            assert output == 'ready'
        else:
            async with agent.iter('Answer.') as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            events.extend([event async for event in stream])
                assert run.result is not None
                assert run.result.output == 'ready'
        assert calls == ['primary', 'backup']
        if method != 'iter':
            retries = [event for event in events if isinstance(event, OutputFallbackEvent)]
            assert len(retries) == 1

    async def test_delivered_final_stream_is_not_replaced(
        self, streaming_agent: tuple[Agent[object, str], list[str]]
    ) -> None:
        agent, calls = streaming_agent
        delivered: list[str] = []
        with pytest.raises(UnexpectedModelBehavior):
            async with agent.run_stream('Answer.') as result:
                async for value in result.stream_output(debounce_by=None):
                    delivered.append(value)
        assert 'draft' in delivered
        assert calls == ['primary']


@pytest.mark.parametrize('source', ['agent', 'wrapped', 'group', 'runtime', 'factory', 'run_spec', 'spec_capability'])
@pytest.mark.parametrize('threshold', [1, {'test:chosen': 1, 'chosen': 2}])
async def test_explicit_run_model_warns_before_output_is_rejected(
    source: Literal['agent', 'wrapped', 'group', 'runtime', 'factory', 'run_spec', 'spec_capability'],
    threshold: int | dict[str, int],
) -> None:
    backup = TestModel(model_name='backup', custom_output_text='backup')
    fallback = ModelFallback[object](backup, output_retry_threshold=threshold)
    chosen = TestModel(model_name='chosen', custom_output_text='chosen')
    capabilities: list[AgentCapability[object]] = [fallback]
    run_capabilities: list[AgentCapability[object]] = []
    explicit_model: Model | None = chosen
    spec: dict[str, object] | None = None
    expected_model_id = chosen.model_id
    if source == 'wrapped':
        capabilities = [WrapperCapability(fallback)]
    elif source == 'group':
        children: list[AbstractCapability[object]] = [fallback]
        capabilities = [WrapperCapability(CombinedCapability(children))]
    elif source == 'runtime':
        capabilities = []
        run_capabilities = [fallback]
    elif source == 'factory':
        capabilities = []
        run_capabilities = [lambda ctx: fallback]
    elif source in {'run_spec', 'spec_capability'}:
        models: dict[str, Model] = {'chosen': chosen, 'backup': backup}
        capabilities = [ResolveModelId(lambda ctx, model_id: models.get(model_id))]
        spec = {'model': 'chosen'}
        if source == 'run_spec':
            capabilities.append(fallback)
        else:
            spec['capabilities'] = [{'ModelFallback': {'models': ['backup'], 'output_retry_threshold': threshold}}]
        explicit_model = None
        expected_model_id = 'chosen'
    else:
        assert source == 'agent'
    hooks = Hooks[object]()

    @hooks.on.before_model_request
    def check_warning(ctx: RunContext[object], request_context: ModelRequestContext) -> ModelRequestContext:
        assert len(emitted) == 1
        return request_context

    capabilities.append(hooks)
    agent = Agent(TestModel(custom_output_text='default'), deps_type=object, capabilities=capabilities)

    with pytest.warns(UserWarning, match='output switching.*explicit run') as emitted:
        result = await agent.run('Answer.', model=explicit_model, spec=spec, capabilities=run_capabilities)

    assert result.output == 'chosen'
    assert len(emitted) == 1
    assert emitted[0].filename == __file__
    message = str(emitted[0].message)
    assert f'output_retry_threshold={threshold!r}' in message
    assert f'`model` {expected_model_id!r}' in message


@pytest.mark.parametrize('threshold', [None, {}])
async def test_request_only_fallback_does_not_warn(threshold: dict[str, int] | None) -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=threshold)]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        assert (await agent.run('Answer.', model=TestModel(custom_output_text='chosen'))).output == 'chosen'

    assert emitted == []


@pytest.mark.parametrize('source', ['override', 'selector'])
async def test_override_and_capability_composition_do_not_warn(source: Literal['override', 'selector']) -> None:
    chosen = TestModel(custom_output_text='chosen')
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    if source == 'selector':
        capabilities.append(SelectModel(lambda ctx: chosen))
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        if source == 'override':
            with agent.override(model=chosen):
                result = await agent.run('Answer.', model=TestModel())
        else:
            result = await agent.run('Answer.')

    assert result.output == 'chosen'
    assert emitted == []


async def test_warning_does_not_query_model_contributions(mocker: MockerFixture) -> None:
    fallback = ModelFallback[object](TestModel(), output_retry_threshold=1)
    getter = mocker.patch.object(fallback, 'get_model', wraps=fallback.get_model)
    capabilities: list[AbstractCapability[object]] = [fallback]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)
    getter.reset_mock()

    with pytest.warns(UserWarning, match='output switching.*explicit run'):
        assert (await agent.run('Answer.', model=TestModel(custom_output_text='chosen'))).output == 'chosen'

    getter.assert_not_called()


@pytest.mark.parametrize('entrypoint', ['iter', 'run_stream'])
async def test_warning_points_to_async_caller(entrypoint: Literal['iter', 'run_stream']) -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)
    chosen = TestModel(custom_output_text='chosen')

    with pytest.warns(UserWarning, match='output switching.*explicit run') as emitted:
        if entrypoint == 'iter':
            async with agent.iter('Answer.', model=chosen):
                pass
        else:
            assert entrypoint == 'run_stream'
            async with agent.run_stream('Answer.', model=chosen) as result:
                assert await result.get_output() == 'chosen'

    assert len(emitted) == 1
    assert emitted[0].filename == __file__


def test_warning_points_to_sync_caller() -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    agent = Agent(TestModel(), deps_type=object, capabilities=capabilities)

    with pytest.warns(UserWarning, match='output switching.*explicit run') as emitted:
        assert agent.run_sync('Answer.', model=TestModel(custom_output_text='chosen')).output == 'chosen'

    assert len(emitted) == 1
    assert emitted[0].filename == __file__


async def test_warning_without_frame_inspection(mocker: MockerFixture) -> None:
    capabilities: list[AbstractCapability[object]] = [ModelFallback(TestModel(), output_retry_threshold=1)]
    agent = Agent(TestModel(), name='fallback_warning', deps_type=object, capabilities=capabilities)
    mocker.patch('pydantic_ai.agent.inspect.currentframe', return_value=None)

    with pytest.warns(UserWarning, match='output switching.*explicit run'):
        assert (await agent.run('Answer.', model=TestModel(custom_output_text='chosen'))).output == 'chosen'
