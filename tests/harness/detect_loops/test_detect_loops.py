"""Tests for `DetectLoops`, driven through `Agent(..., capabilities=[DetectLoops(...)])`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import count, pairwise

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, WrapModelRequestHandler
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai_harness import ClearToolResults, DetectLoops, SlidingWindowCompaction
from pydantic_ai_harness.detect_loops import DetectedLoop, LoopDetectedError
from tests.function_model_prefix import COMPACTION_REWRITES_HISTORY


def _scripted(*responses: Sequence[ModelResponsePart]) -> FunctionModel:
    """A model that sends `responses` in order, then answers `'done'`."""
    steps = iter(responses)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=list(next(steps, [TextPart('done')])))

    return FunctionModel(model)


def _call(tool_name: str = 'read_file', args: dict[str, object] | str | None = None) -> ToolCallPart:
    return ToolCallPart(tool_name, {'path': 'missing.txt'} if args is None else args)


def _agent(
    responses: Sequence[Sequence[ModelResponsePart]] | FunctionModel, detector: DetectLoops[object]
) -> Agent[object, str]:
    model = responses if isinstance(responses, FunctionModel) else _scripted(*responses)
    agent = Agent(model, capabilities=[detector], retries=10)
    progress = count()

    @agent.tool_plain
    def read_file(path: str) -> str:
        return 'Error: file not found'

    @agent.tool_plain
    def poll(job: str) -> str:
        return f'pending ({next(progress)})'

    @agent.tool_plain
    def list_files(path: str) -> dict[str, object]:
        return {'path': path, 'entries': [f'file-{next(progress)}']}

    @agent.tool_plain
    def describe(path: str) -> dict[str, object]:
        return {'path': path, 'size': 0}

    @agent.tool_plain
    def fetch(url: str) -> str:
        raise ModelRetry('Connection refused')

    return agent


def _collect() -> tuple[list[DetectedLoop], DetectLoops[object]]:
    loops: list[DetectedLoop] = []
    return loops, DetectLoops(action=lambda ctx, loop: loops.append(loop))


def _nudges(messages: Sequence[ModelMessage]) -> list[SystemPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]


def _responses_before_each_nudge(messages: Sequence[ModelMessage]) -> list[int]:
    """For each nudge, how many model responses the run had produced when it was added."""
    responses = 0
    counts: list[int] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            responses += 1
        counts.extend(responses for _ in _nudges([message]))
    return counts


class TestDetectLoops:
    async def test_repeated_result_nudges_the_model(self):
        seen: list[list[ModelMessage]] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            if 'Change your approach' in str(messages[-1]):
                return ModelResponse(parts=[TextPart('missing.txt does not exist.')])
            return ModelResponse(parts=[_call()])

        result = await _agent(FunctionModel(model), DetectLoops()).run('Summarize missing.txt')

        assert result.output == 'missing.txt does not exist.'
        assert _responses_before_each_nudge(result.all_messages()) == [3]
        [nudge] = _nudges(result.all_messages())
        assert (nudge.content, nudge.metadata, nudge.scope) == snapshot(
            (
                'Your last 3 calls to `read_file` used the same arguments and returned the same result, so calling it again will not change the outcome. Change your approach, or explain what is blocking you.',
                {
                    'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
                    'pattern': 'repeated_result',
                    'tool_names': ['read_file'],
                },
                'conversation',
            )
        )
        # The model sees the nudge on the request that follows the third result.
        assert len(seen) == 4

    async def test_repeated_call_with_changing_results(self):
        loops, detector = _collect()

        await _agent([[_call('poll', {'job': 'build'})]] * 6, detector).run('Wait for the build')

        assert loops == snapshot(
            [
                DetectedLoop(
                    pattern='repeated_call',
                    tool_names=('poll',),
                    count=5,
                    window=5,
                    fingerprints=('5c0143f8081a875f',),
                    message='You have called `poll` with the same arguments 5 times in your last 5 tool calls, which is not making progress. Change your approach, or explain what is blocking you.',
                )
            ]
        )

    async def test_alternating_calls(self):
        loops, detector = _collect()
        responses = [[_call('list_files', {'path': 'src'})], [_call('list_files', {'path': 'docs'})]] * 3

        await _agent(responses, detector).run('Find the config file')

        assert loops == snapshot(
            [
                DetectedLoop(
                    pattern='alternating_calls',
                    tool_names=('list_files', 'list_files'),
                    count=3,
                    window=6,
                    fingerprints=('fc8412cf202b493c', '9b5242dea5d36d1f'),
                    message='You have alternated between two calls to `list_files` with different arguments 3 times without making progress. Change your approach, or explain what is blocking you.',
                )
            ]
        )

    async def test_repeated_text(self):
        loops, detector = _collect()
        agent = _agent([[TextPart('I will  look into it.')], [TextPart('i will look into it.')]] * 2, detector)

        @agent.output_validator
        def needs_a_file_name(output: str) -> str:
            if output != 'done':
                raise ModelRetry('Name the file.')
            return output

        await agent.run('Which file is the config in?')

        assert loops == snapshot(
            [
                DetectedLoop(
                    pattern='repeated_text',
                    tool_names=(),
                    count=3,
                    window=3,
                    fingerprints=('a0ca180735f6ca6d',),
                    message='Your last 3 responses had the same text and no tool calls, so repeating it will not change the outcome. Change your approach, or explain what is blocking you.',
                )
            ]
        )

    async def test_tool_retries_count_as_the_same_result(self):
        with pytest.raises(LoopDetectedError) as exc_info:
            await _agent([[_call('fetch', {'url': 'http://localhost'})]] * 5, DetectLoops(action='error')).run('Fetch')

        assert exc_info.value.loop == snapshot(
            DetectedLoop(
                pattern='repeated_result',
                tool_names=('fetch',),
                count=3,
                window=3,
                fingerprints=('3d7c98f4772d06ec',),
                message='Your last 3 calls to `fetch` used the same arguments and returned the same result, so calling it again will not change the outcome. Change your approach, or explain what is blocking you.',
            )
        )
        assert str(exc_info.value) == exc_info.value.loop.message

    async def test_every_call_in_a_response_is_counted(self):
        loops, detector = _collect()
        responses = [[_call(), _call(), _call(), _call('describe', {'path': 'a'})]]

        await _agent(responses, detector).run('Read it')

        assert [(loop.pattern, loop.count, loop.window) for loop in loops] == [('repeated_result', 3, 4)]

    async def test_a_reported_loop_must_build_up_again(self):
        loops, detector = _collect()
        # Detected after the third call; the fourth and fifth alone are not a loop, the sixth is.
        responses = [[_call()]] * 5 + [[_call(), _call('describe', {'path': 'a'})]]

        await _agent(responses, detector).run('Read it')

        assert [loop.count for loop in loops] == [3, 3]

    async def test_a_nudged_loop_must_build_up_again(self):
        result = await _agent([[_call()]] * 7, DetectLoops()).run('Read it')

        assert _responses_before_each_nudge(result.all_messages()) == [3, 6]

    async def test_below_every_threshold_nothing_happens(self):
        loops, detector = _collect()
        responses = [
            [_call()],
            [_call()],
            [_call('poll', {'job': 'build'})],
            [_call('poll', {'job': 'build'})],
            [_call('list_files', {'path': 'src'})],
            [_call('describe', {'path': 'src'})],
            [_call('list_files', {'path': 'src'})],
            [_call('describe', {'path': 'docs'})],
            [TextPart('Still looking.')],
            [TextPart('Still looking.')],
        ]
        agent = _agent(responses, detector)

        @agent.output_validator
        def reject_until_done(output: str) -> str:
            if output != 'done':
                raise ModelRetry('Keep going.')
            return output

        result = await agent.run('Find the config file')

        assert result.output == 'done'
        assert loops == []

    async def test_a_repeated_call_is_not_an_alternation(self):
        loops, detector = _collect()

        # The responses also share one tool call id, as some models' do: each result is matched to the
        # call in the response before it, so the changing results are not mistaken for one result.
        await _agent([[_call('poll', {'job': 'build'})]] * 4, detector).run('Wait')

        assert loops == []

    async def test_different_arguments_break_the_streak(self):
        loops, detector = _collect()
        responses = [[_call()], [_call()], [_call('read_file', {'path': 'other.txt'})]]

        await _agent(responses, detector).run('Read it')

        assert loops == []

    async def test_a_different_result_breaks_the_streak(self):
        loops, detector = _collect()
        statuses = iter(['busy', 'busy', 'idle', 'busy'])
        agent = _agent([[_call('status', {'job': 'build'})]] * 4, detector)

        @agent.tool_plain
        def status(job: str) -> str:
            return next(statuses)

        await agent.run('Wait for the build')

        assert loops == []

    async def test_checks_can_be_turned_off(self):
        detector = DetectLoops[object](
            repeated_call_threshold=None,
            repeated_result_threshold=None,
            alternating_calls_threshold=None,
            repeated_text_threshold=None,
            action='error',
        )

        result = await _agent([[_call()]] * 8, detector).run('Read it')

        assert result.output == 'done'

    async def test_equivalent_arguments_and_results_match(self):
        loops, detector = _collect()
        responses = [
            [_call('describe', {'path': 'a'})],
            [_call('describe', '{"path": "a"}')],
            [_call('describe', '{ "path" : "a" }')],
        ]

        await _agent(responses, detector).run('Describe a')

        assert [loop.pattern for loop in loops] == ['repeated_result']

    async def test_different_arguments_do_not_match(self):
        loops, detector = _collect()
        responses = [[_call('describe', {'path': str(index)})] for index in range(10)]

        await _agent(responses, detector).run('Describe everything')

        assert loops == []

    async def test_async_handler_can_steer(self):
        async def steer(ctx: RunContext[object], loop: DetectedLoop) -> None:
            ctx.enqueue(SystemPromptPart(f'Stop calling {loop.tool_names[0]}.'))

        result = await _agent([[_call()]] * 3, DetectLoops(action=steer)).run('Read it')

        assert [nudge.content for nudge in _nudges(result.all_messages())] == ['Stop calling read_file.']

    async def test_handler_can_stop_the_run(self):
        def stop(ctx: RunContext[object], loop: DetectedLoop) -> None:
            raise LoopDetectedError(loop)

        with pytest.raises(LoopDetectedError):
            await _agent([[_call()]] * 3, DetectLoops(action=stop)).run('Read it')

    async def test_tool_names_in_the_nudge_are_identifiers(self):
        loops, detector = _collect()
        agent = Agent(_scripted(*[[ToolCallPart('ns.read-file', {})]] * 3), capabilities=[detector])

        @agent.tool_plain(name='ns.read-file')
        def read() -> str:
            return 'nothing'

        await agent.run('Read it')

        [loop] = loops
        assert loop.tool_names == ('ns.read-file',)
        assert '`ns_read_file`' in loop.message

    async def test_runs_do_not_share_counts(self):
        loops, detector = _collect()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if sum(isinstance(message, ModelResponse) for message in messages) % 3 == 2:
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[_call()])

        agent = _agent(FunctionModel(model), detector)
        first = await agent.run('Read it')
        # Two calls in each run: the earlier run's calls are not counted toward this run's loop.
        second = await agent.run('Read it again', message_history=first.all_messages())
        await agent.run('Read it once more')

        assert len(second.new_messages()) == 6
        assert loops == []

    async def test_records_a_span(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent([[_call()]] * 3, DetectLoops())
        agent.instrument = InstrumentationSettings(tracer_provider=provider)

        await agent.run('Read it')

        [span] = [span for span in exporter.get_finished_spans() if span.name == 'loop detected']
        assert dict(span.attributes or {}) == snapshot(
            {
                'detect_loops.pattern': 'repeated_result',
                'detect_loops.action': 'nudge',
                'detect_loops.count': 3,
                'detect_loops.window': 3,
                'detect_loops.tool_names': ('read_file',),
                'detect_loops.fingerprints': ('cad0390702834b89',),
            }
        )

    @pytest.mark.parametrize(
        ('create', 'message'),
        [
            (lambda: DetectLoops[object](window=1), '`window` must be at least 2, got 1.'),
            (
                lambda: DetectLoops[object](repeated_text_threshold=1),
                '`repeated_text_threshold` must be at least 2 or `None`, got 1.',
            ),
            (
                lambda: DetectLoops[object](repeated_call_threshold=11),
                '`repeated_call_threshold` needs 11 tool calls, more than the 10 that `window` keeps, so it could '
                'never fire.',
            ),
            (
                lambda: DetectLoops[object](alternating_calls_threshold=6),
                '`alternating_calls_threshold` needs 12 tool calls, more than the 10 that `window` keeps, so it '
                'could never fire.',
            ),
        ],
    )
    def test_rejects_thresholds_that_cannot_fire(self, create: Callable[[], DetectLoops[object]], message: str):
        with pytest.raises(ValueError) as exc_info:
            create()

        assert str(exc_info.value) == message

    async def test_span_leaves_out_fingerprints_without_content(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent([[_call()]] * 3, DetectLoops())
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=False)

        await agent.run('Read it')

        [span] = [span for span in exporter.get_finished_spans() if span.name == 'loop detected']
        assert 'detect_loops.fingerprints' not in (span.attributes or {})
        assert (span.attributes or {})['detect_loops.pattern'] == 'repeated_result'

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_cleared_tool_results_are_not_mistaken_for_a_loop(self):
        loops, detector = _collect()
        # Each call has its own arguments and result. Clearing old results and inputs makes them look alike
        # in history, but they were recorded as they arrived.
        agent = Agent(
            _scripted(*[[_call('describe', {'path': str(index)})] for index in range(8)]),
            capabilities=[ClearToolResults(max_messages=6, keep_pairs=1, clear_tool_inputs=True), detector],
        )

        @agent.tool_plain
        def describe(path: str) -> dict[str, object]:
            return {'path': path, 'size': len(path)}

        result = await agent.run('Describe everything')

        assert loops == []
        assert '[tool result cleared]' in str(result.all_messages())

    @pytest.mark.moves_cache_prefix(reason=COMPACTION_REWRITES_HISTORY)
    async def test_dropped_messages_do_not_shift_detection(self):
        responses_at_detection: list[int] = []
        responses = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal responses
            responses += 1
            return ModelResponse(parts=[_call() if responses <= 7 else TextPart('done')])

        def record(ctx: RunContext[object], loop: DetectedLoop) -> None:
            responses_at_detection.append(responses)

        agent = Agent(
            FunctionModel(model),
            capabilities=[SlidingWindowCompaction(max_messages=6, keep_messages=4), DetectLoops(action=record)],
        )

        @agent.tool_plain
        def read_file(path: str) -> str:
            return 'Error: file not found'

        result = await agent.run('Read it')

        assert responses_at_detection == [3, 6]
        assert len(result.all_messages()) < 16

    async def test_nudges_keep_the_request_prefix_stable(self):
        requests: list[list[ModelMessage]] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            requests.append(messages)
            if sum(isinstance(message, ModelResponse) for message in messages) % 5 == 4:
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[_call()])

        agent = _agent(FunctionModel(model), DetectLoops())
        first = await agent.run('Read it')
        second = await agent.run('Read it again', message_history=first.all_messages())

        assert len(_nudges(second.all_messages())) == 2
        for previous, current in pairwise(requests):
            assert ModelMessagesTypeAdapter.dump_json(current[: len(previous)]) == ModelMessagesTypeAdapter.dump_json(
                previous
            )

    async def test_results_are_read_before_compaction_clears_them(self):
        loops, detector = _collect()
        # Listed first, `ClearToolResults` would clear the newest results before they were read.
        agent = Agent(
            _scripted(*[[_call('poll', {'job': 'build'})]] * 4),
            capabilities=[ClearToolResults(max_messages=1, keep_pairs=0), detector],
        )
        progress = count()

        @agent.tool_plain
        def poll(job: str) -> str:
            return f'pending ({next(progress)})'

        result = await agent.run('Wait for the build')

        assert loops == []
        assert '[tool result cleared]' in str(result.all_messages())

    async def test_skipped_model_requests_do_not_overwrite_recorded_results(self):
        loops, detector = _collect()

        @dataclass
        class SkipSecondRequest(AbstractCapability[object]):
            requests: int = 0

            async def before_model_request(
                self, ctx: RunContext[object], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                self.requests += 1
                if self.requests == 2:
                    raise SkipModelRequest(ModelResponse(parts=[_call('describe', {'path': 'a'})]))
                return request_context

        agent = Agent(_scripted(*[[_call()]] * 3), capabilities=[detector, SkipSecondRequest()])

        @agent.tool_plain
        def read_file(path: str) -> str:
            return 'Error: file not found'

        @agent.tool_plain
        def describe(path: str) -> str:
            return 'Error: file not found'

        await agent.run('Read it')

        # The skipped response was never recorded, so its result is not mistaken for the first call's.
        assert [(loop.pattern, loop.count) for loop in loops] == [('repeated_result', 3)]

    async def test_results_follow_their_own_response_when_a_request_is_short_circuited(self):
        loops, detector = _collect()

        @dataclass
        class ShortCircuitSecondRequest(AbstractCapability[object]):
            requests: int = 0

            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

            async def wrap_model_request(
                self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
            ) -> ModelResponse:
                self.requests += 1
                if self.requests == 2:
                    return ModelResponse(parts=[_call('describe', {'path': 'a'})])
                return await handler(request_context)

        # Listed first, the short-circuit wraps `DetectLoops`, whose hooks then don't run for the second request.
        agent = Agent(_scripted(*[[_call()]] * 3), capabilities=[ShortCircuitSecondRequest(), detector])

        @agent.tool_plain
        def read_file(path: str) -> str:
            return 'Error: file not found'

        @agent.tool_plain
        def describe(path: str) -> str:
            return 'Error: file not found'

        await agent.run('Read it')

        assert [(loop.pattern, loop.count) for loop in loops] == [('repeated_result', 3)]

    async def test_results_are_read_when_a_wrapper_replaces_the_response(self):
        loops, detector = _collect()

        @dataclass
        class AnnotateResponses(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

            async def wrap_model_request(
                self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
            ) -> ModelResponse:
                # A new timestamp means no response in history is the one `DetectLoops` recorded.
                return replace(await handler(request_context), timestamp=datetime.now(UTC))

        agent = Agent(_scripted(*[[_call()]] * 3), capabilities=[AnnotateResponses(), detector])

        @agent.tool_plain
        def read_file(path: str) -> str:
            return 'Error: file not found'

        await agent.run('Read it')

        assert [(loop.pattern, loop.count) for loop in loops] == [('repeated_result', 3)]

    async def test_results_are_read_when_a_wrapper_rewrites_the_calls(self):
        loops, detector = _collect()

        @dataclass
        class TagArguments(AbstractCapability[object]):
            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

            async def wrap_model_request(
                self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
            ) -> ModelResponse:
                response = await handler(request_context)
                parts = [
                    replace(part, args={**part.args_as_dict(), 'tagged': True})
                    if isinstance(part, ToolCallPart)
                    else part
                    for part in response.parts
                ]
                return replace(response, parts=parts)

        agent = Agent(_scripted(*[[_call()]] * 3), capabilities=[TagArguments(), detector])

        @agent.tool_plain
        def read_file(path: str, tagged: bool = False) -> str:
            return 'Error: file not found'

        await agent.run('Read it')

        # The rewritten response keeps its timestamp and tool call ids, so it is still found.
        assert [(loop.pattern, loop.count) for loop in loops] == [('repeated_result', 3)]

    async def test_an_earlier_call_with_a_reused_id_is_not_mistaken_for_the_latest(self):
        loops, detector = _collect()
        statuses = iter(['busy', 'busy', 'idle'])

        @dataclass
        class TagThirdResponse(AbstractCapability[object]):
            requests: int = 0

            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

            async def wrap_model_request(
                self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
            ) -> ModelResponse:
                self.requests += 1
                response = await handler(request_context)
                if self.requests != 3:
                    return response
                parts = [
                    replace(part, args={**part.args_as_dict(), 'tagged': True})
                    if isinstance(part, ToolCallPart)
                    else part
                    for part in response.parts
                ]
                return replace(response, parts=parts)

        call = ToolCallPart('status', {'job': 'build'}, tool_call_id='call_0')
        agent = Agent(_scripted(*[[call]] * 3), capabilities=[TagThirdResponse(), detector])

        @agent.tool_plain
        def status(job: str, tagged: bool = False) -> str:
            return next(statuses)

        await agent.run('Wait for the build')

        # The second response matches the third's recorded calls exactly, but its `busy` is not the third's result.
        assert loops == []

    async def test_a_short_circuited_response_that_reuses_tool_call_ids_is_not_mistaken_for_the_recorded_one(self):
        loops, detector = _collect()
        outputs = iter(['first', 'second', 'second', 'second'])

        @dataclass
        class ShortCircuitSecondRequest(AbstractCapability[object]):
            requests: int = 0

            def get_ordering(self) -> CapabilityOrdering:
                return CapabilityOrdering(position='outermost')

            async def wrap_model_request(
                self, ctx: RunContext[object], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
            ) -> ModelResponse:
                self.requests += 1
                if self.requests == 2:
                    return ModelResponse(
                        parts=[ToolCallPart('read_file', {'path': 'missing.txt'}, tool_call_id='call_0')]
                    )
                return await handler(request_context)

        call = ToolCallPart('read_file', {'path': 'missing.txt'}, tool_call_id='call_0')
        agent = Agent(_scripted(*[[call]] * 3), capabilities=[ShortCircuitSecondRequest(), detector])

        @agent.tool_plain
        def read_file(path: str) -> str:
            return next(outputs)

        await agent.run('Read it')

        # The recorded calls returned `first`, `second`, `second`; the canned response's `second` is not the first's.
        assert loops == []
