"""Tests for `DetectLoops` checks within one model response: streamed thinking, text and tool call arguments."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal

import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    RetryFeedbackPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai_harness import DetectLoops
from pydantic_ai_harness.detect_loops import (
    DetectedLoop,
    LoopDetectedError,
    LoopDetector,
    LoopHandler,
    LoopJudge,
    Repetition,
    RepetitionDetector,
    StreamSample,
)
from tests.conftest import no_leaked_tasks
from tests.harness.conftest import agent_run_names

Chunk = str | DeltaToolCalls | DeltaThinkingCalls

LOOP = 'I should check the config file again before answering. '
"""A sentence a degenerate model keeps repeating."""

PROGRESS = ' '.join(f'Step {index}: inspect module number {index} and note what it exports.' for index in range(80))
"""Long text that never repeats, because every sentence names a different step."""


def _thinking(text: str) -> DeltaThinkingCalls:
    return {0: DeltaThinkingPart(content=text)}


def _args(text: str, *, index: int = 0, name: str | None = None) -> DeltaToolCalls:
    return {index: DeltaToolCall(name=name, json_args=text, tool_call_id=None if name is None else f'call_{index}')}


@dataclass
class _Script:
    """A streaming model that sends `responses` in order, then answers `'done'`, and records what it sent."""

    responses: Sequence[Sequence[Chunk]]
    sent: list[int]
    """How many chunks each response got out before the stream ended."""
    requests: list[list[ModelMessage]]

    def model(self) -> FunctionModel:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Chunk]:
            index = len(self.requests)
            self.requests.append(messages)
            chunks = self.responses[index] if index < len(self.responses) else ['done']
            self.sent.append(0)
            for chunk in chunks:
                yield chunk
                self.sent[index] += 1

        return FunctionModel(stream_function=stream)


def _script(*responses: Sequence[Chunk]) -> _Script:
    return _Script(responses=responses, sent=[], requests=[])


def _agent(script: _Script, detector: DetectLoops[object]) -> Agent[object, str]:
    agent = Agent(script.model(), capabilities=[detector], retries=5)

    @agent.tool_plain
    def read_file(path: str) -> str:
        return 'contents'

    return agent


async def _drain(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in events:
        pass


def _nudges(messages: Sequence[ModelMessage]) -> list[SystemPromptPart]:
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]


def _retry_feedback(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, RetryFeedbackPart)
    ]


def _collect() -> tuple[list[DetectedLoop], DetectLoops[object]]:
    loops: list[DetectedLoop] = []
    return loops, DetectLoops(action=lambda ctx, loop: loops.append(loop))


def _judge(verdicts: list[bool], prompts: list[str] | None = None, passage: str = LOOP.strip()) -> FunctionModel:
    """A judge model that answers each request with the next verdict, quoting `passage` when it finds a loop."""

    def judge(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if prompts is not None:
            prompts.append(str(messages[-1]))
        looping = verdicts.pop(0)
        args = {'looping': looping, 'repeated_passage': passage if looping else ''}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(judge)


class TestDetectLoops:
    async def test_looping_thinking_is_stopped_and_the_request_retried_with_a_nudge(self):
        script = _script([*(_thinking(LOOP) for _ in range(60)), 'answer'], ['The config is fine.'])

        result = await _agent(script, DetectLoops()).run('Is the config fine?', event_stream_handler=_drain)

        assert result.output == 'The config is fine.'
        # The stream was cut long before the model finished its 61 chunks.
        assert script.sent[0] < 45
        [nudge] = _nudges(result.all_messages())
        assert (nudge.content, nudge.metadata, nudge.scope) == snapshot(
            (
                'Your thinking repeated the same passage 37 times in a row, which is not making progress. Change your approach, or explain what is blocking you.',
                {
                    'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
                    'pattern': 'repeated_thinking',
                    'tool_names': [],
                },
                'conversation',
            )
        )
        assert _retry_feedback(result.all_messages()) == snapshot(
            ['Your previous response was rejected because it kept repeating itself.']
        )
        # The stopped response is not kept, so the looping thinking doesn't fill the context.
        assert not any(
            isinstance(part, ThinkingPart)
            for message in result.all_messages()
            if isinstance(message, ModelResponse)
            for part in message.parts
        )

    async def test_looping_text_is_stopped(self):
        loops: list[DetectedLoop] = []
        script = _script([LOOP] * 60)

        def record(ctx: RunContext[object], loop: DetectedLoop) -> None:
            loops.append(loop)
            raise ModelRetry('Stop repeating yourself.')

        result = await _agent(script, DetectLoops(action=record)).run('Explain', event_stream_handler=_drain)

        assert result.output == 'done'
        assert script.sent[0] < 45
        assert loops == snapshot(
            [
                DetectedLoop(
                    pattern='repeated_stream_text',
                    tool_names=(),
                    count=37,
                    window=2048,
                    fingerprints=('dc3adfd1bb4f501d',),
                    message='Your response repeated the same passage 37 times in a row, which is not making progress. Change your approach, or explain what is blocking you.',
                    repeated_text='k the config file again before answering. I should chec',
                )
            ]
        )
        assert _retry_feedback(result.all_messages()) == ['Stop repeating yourself.']

    async def test_looping_tool_arguments_are_stopped_before_the_tool_runs(self):
        body = 'value = transform(value, step)\\n' * 200
        chunks = [
            {0: DeltaToolCall(name='write_file', tool_call_id='call_0')},
            _args('{"path": "a.py", "content": "'),
            *(_args(body[start : start + 40]) for start in range(0, len(body), 40)),
            _args('"}'),
        ]
        script = _script(chunks)
        agent = Agent(script.model(), capabilities=[DetectLoops()], retries=5)

        @agent.tool_plain
        def write_file(path: str, content: str) -> str:
            raise AssertionError('a looping call must not run')  # pragma: no cover

        result = await agent.run('Write a.py', event_stream_handler=_drain)

        assert result.output == 'done'
        assert script.sent[0] < 80
        [nudge] = _nudges(result.all_messages())
        assert (nudge.content, nudge.metadata) == snapshot(
            (
                'The arguments of your call to `write_file` repeated the same passage 63 times in a row, which is not making progress. Change your approach, or explain what is blocking you.',
                {
                    'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
                    'pattern': 'repeated_tool_args',
                    'tool_names': ['write_file'],
                },
            )
        )

    async def test_repeated_parallel_calls_are_stopped_before_the_tools_run(self):
        calls = [_args('{"path": "a.txt"}', index=index, name='read_file') for index in range(8)]
        script = _script(calls)
        agent = Agent(script.model(), capabilities=[DetectLoops()], retries=5)

        @agent.tool_plain
        def read_file(path: str) -> str:
            raise AssertionError('repeated calls must not run')  # pragma: no cover

        result = await agent.run('Read a.txt', event_stream_handler=_drain)

        assert result.output == 'done'

        assert script.sent[0] < 8
        [nudge] = _nudges(result.all_messages())
        assert (nudge.content, nudge.metadata) == snapshot(
            (
                'You called `read_file` with the same arguments 5 times in one response, which is not making progress. Change your approach, or explain what is blocking you.',
                {
                    'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
                    'pattern': 'repeated_parallel_call',
                    'tool_names': ['read_file'],
                },
            )
        )

    async def test_run_stream_retries_a_response_stopped_while_thinking(self):
        script = _script([*(_thinking(LOOP) for _ in range(60)), 'answer'], ['The config is fine.'])

        async with _agent(script, DetectLoops()).run_stream('Is the config fine?') as result:
            output = await result.get_output()

        assert output == 'The config is fine.'
        assert [nudge.metadata for nudge in _nudges(result.all_messages())] == [
            {'origin': 'pydantic-ai-harness.detect_loops.nudge.v1', 'pattern': 'repeated_thinking', 'tool_names': []}
        ]

    async def test_text_that_makes_progress_is_not_a_loop(self):
        loops, detector = _collect()
        words = PROGRESS.split(' ')
        script = _script([*(_thinking(word + ' ') for word in words), *(word + ' ' for word in words)])

        result = await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert loops == []
        assert result.output.startswith('Step 0')

    async def test_short_repetition_is_not_a_loop(self):
        loops, detector = _collect()
        # A long run of a short unit, like a divider or a row of zeros, repeats a unit shorter than `min_period`.
        script = _script(['=' * 50] * 20 + ['0, '] * 300)

        await _agent(script, detector).run('Draw a table', event_stream_handler=_drain)

        assert loops == []

    def test_a_passage_around_a_run_of_a_short_unit(self):
        detector = RepetitionDetector[object]()
        unit = 'Checking the configuration again. ' + '0' * 400

        assert detector.find(unit * 8) == Repetition(text=unit, count=8)
        assert detector.find('=' * 1000 + '0, ' * 1000) is None

    async def test_repetitive_content_written_on_purpose_is_not_a_loop(self):
        loops, detector = _collect()
        contents = [
            '0 0 0 0 0 0 0 0 0 0\\n' * 100,
            '| --- | --- | --- | --- |\\n' * 100,
            "INSERT INTO t VALUES (1, 'x', 'y');\\n" * 50,
        ]
        calls = [
            _args(json.dumps({'path': f'{index}.txt', 'content': content}), index=index, name='write_file')
            for index, content in enumerate(contents)
        ]
        agent = _agent(_script(calls, ['| a | b |\n' + '| --- | --- |\n' * 100]), detector)

        @agent.tool_plain
        def write_file(path: str, content: str) -> str:
            return 'written'

        await agent.run('Write the fixtures', event_stream_handler=_drain)

        assert loops == []

    async def test_channels_can_be_turned_off(self):
        loops: list[DetectedLoop] = []
        detector = DetectLoops[object](stream_channels=('text',), action=lambda ctx, loop: loops.append(loop))
        script = _script([*(_thinking(LOOP) for _ in range(30)), 'answer'])

        result = await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert result.output == 'answer'
        assert loops == []

    async def test_in_response_checks_can_be_turned_off(self):
        detector = DetectLoops[object](
            stream_detector=None,
            repeated_parallel_call_threshold=None,
            repeated_call_threshold=None,
            repeated_result_threshold=None,
            action='error',
        )
        calls = [_args('{"path": "a.txt"}', index=index, name='read_file') for index in range(6)]
        script = _script(calls, [*(_thinking(LOOP) for _ in range(30)), *([LOOP] * 30)])

        result = await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert result.output == LOOP * 30

    async def test_error_stops_the_stream_and_the_run(self):
        script = _script([LOOP] * 60)

        with pytest.raises(LoopDetectedError) as exc_info:
            await _agent(script, DetectLoops(action='error')).run('Explain', event_stream_handler=_drain)

        assert exc_info.value.loop.pattern == 'repeated_stream_text'
        assert str(exc_info.value) == exc_info.value.loop.message
        assert script.sent[0] < 45

    async def test_a_handler_that_returns_lets_the_response_continue(self):
        loops, detector = _collect()
        script = _script([LOOP] * 80)

        result = await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert result.output == LOOP * 80
        assert script.sent == [80]
        # The loop is reported, then has to build up again before it's reported a second time.
        assert [(loop.pattern, loop.count) for loop in loops] == snapshot(
            [('repeated_stream_text', 37), ('repeated_stream_text', 37)]
        )

    async def test_a_response_that_was_not_streamed_is_checked_when_it_arrives(self):
        responses = iter([ModelResponse(parts=[ThinkingPart(LOOP * 40), TextPart('answer')])])
        seen: list[list[ModelMessage]] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(messages)
            return next(responses, ModelResponse(parts=[TextPart('The config is fine.')]))

        agent = Agent(FunctionModel(model), capabilities=[DetectLoops()], retries=5)

        result = await agent.run('Is the config fine?')

        assert result.output == 'The config is fine.'
        assert len(seen) == 2
        [nudge] = _nudges(result.all_messages())
        assert nudge.metadata == {
            'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
            'pattern': 'repeated_thinking',
            'tool_names': [],
        }
        # The response arrived whole, so it stays in history ahead of the retry, as for any rejected response.
        first_response = result.all_messages()[1]
        assert isinstance(first_response, ModelResponse)
        assert first_response.text == 'answer'

    async def test_a_response_that_was_not_streamed_is_checked_for_repeated_parallel_calls(self):
        loops: list[DetectedLoop] = []
        calls = [ToolCallPart('read_file', {'path': 'a.txt'}, tool_call_id=f'call_{index}') for index in range(6)]
        responses = iter([ModelResponse(parts=calls)])

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return next(responses, ModelResponse(parts=[TextPart('done')]))

        detector = DetectLoops[object](stream_detector=None, action=lambda ctx, loop: loops.append(loop))
        agent = Agent(FunctionModel(model), capabilities=[detector])

        @agent.tool_plain
        def read_file(path: str) -> str:
            return 'contents'

        await agent.run('Read a.txt')

        # Reported once: the response's calls are not counted again across responses.
        assert [(loop.pattern, loop.count, loop.window) for loop in loops] == [('repeated_parallel_call', 5, 5)]

    async def test_a_response_that_was_not_streamed_is_checked_on_the_chosen_channels(self):
        loops: list[DetectedLoop] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # A native tool call is the provider's own, so its arguments are never checked.
            search = NativeToolCallPart('web_search', {'query': LOOP * 40})
            return ModelResponse(parts=[search, TextPart(LOOP * 40)])

        detector = DetectLoops[object](
            stream_channels=('thinking', 'tool_args'), action=lambda ctx, loop: loops.append(loop)
        )

        result = await Agent(FunctionModel(model), capabilities=[detector]).run('Explain')

        assert result.output == LOOP * 40
        assert loops == []

    async def test_a_nudged_response_is_not_counted_again_across_responses(self):
        calls = [ToolCallPart('read_file', {'path': 'a.txt'}, tool_call_id=f'call_{index}') for index in range(6)]
        responses = iter([ModelResponse(parts=calls)])

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return next(responses, ModelResponse(parts=[TextPart('done')]))

        agent = Agent(FunctionModel(model), capabilities=[DetectLoops()])

        @agent.tool_plain
        def read_file(path: str) -> str:
            raise AssertionError('repeated calls must not run')  # pragma: no cover

        result = await agent.run('Read a.txt')

        assert [nudge.metadata for nudge in _nudges(result.all_messages())] == [
            {
                'origin': 'pydantic-ai-harness.detect_loops.nudge.v1',
                'pattern': 'repeated_parallel_call',
                'tool_names': ['read_file'],
            }
        ]

    async def test_a_streamed_loop_a_handler_let_through_is_not_counted_again(self):
        loops, detector = _collect()
        calls = [_args('{"path": "a.txt"}', index=index, name='read_file') for index in range(6)]

        await _agent(_script(calls), detector).run('Read a.txt', event_stream_handler=_drain)

        assert [(loop.pattern, loop.count) for loop in loops] == [('repeated_parallel_call', 5)]

    async def test_arguments_that_arrive_as_an_object_are_checked_as_json(self):
        loops, detector = _collect()
        call = ToolCallPart('write_file', {'path': 'a.py', 'content': 'value = transform(value, step)\n' * 100})
        responses = iter([ModelResponse(parts=[call])])

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return next(responses, ModelResponse(parts=[TextPart('done')]))

        agent = Agent(FunctionModel(model), capabilities=[detector])

        @agent.tool_plain
        def write_file(path: str, content: str) -> str:
            return 'written'

        await agent.run('Write a.py')

        assert [(loop.pattern, loop.tool_names) for loop in loops][:1] == [('repeated_tool_args', ('write_file',))]

    async def test_run_stream_cannot_retry_output_that_already_streamed(self):
        script = _script([LOOP] * 60)

        with pytest.raises(UnexpectedModelBehavior, match='retries are not supported in `run_stream\\(\\)`'):
            async with _agent(script, DetectLoops()).run_stream('Explain') as result:
                await result.get_output()

        assert script.sent[0] < 45

    async def test_no_task_outlives_a_stopped_or_finished_stream(self):
        def stop(ctx: RunContext[object], loop: DetectedLoop) -> None:
            raise ModelRetry('Stop repeating yourself.')

        def let_through(ctx: RunContext[object], loop: DetectedLoop) -> None:
            pass

        actions: list[Literal['nudge', 'error'] | LoopHandler[object]] = ['nudge', 'error', stop, let_through]
        for action in actions:
            with no_leaked_tasks(), contextlib.suppress(LoopDetectedError):
                await _agent(_script([LOOP] * 60), DetectLoops(action=action)).run(
                    'Explain', event_stream_handler=_drain
                )

    async def test_cancelling_the_run_while_a_loop_is_handled(self):
        handling = asyncio.Event()

        async def block(ctx: RunContext[object], loop: DetectedLoop) -> None:
            handling.set()
            await asyncio.Event().wait()

        agent = _agent(_script([LOOP] * 60), DetectLoops(action=block))
        with no_leaked_tasks():
            run = asyncio.create_task(agent.run('Explain', event_stream_handler=_drain))
            await handling.wait()
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run

    async def test_a_part_is_checked_when_it_ends(self):
        loops: list[DetectedLoop] = []
        detector = DetectLoops[object](stream_check_every=100_000, action=lambda ctx, loop: loops.append(loop))

        await _agent(_script([LOOP] * 40), detector).run('Explain', event_stream_handler=_drain)

        assert [(loop.pattern, loop.window) for loop in loops] == [('repeated_stream_text', len(LOOP) * 40)]

    async def test_streamed_and_whole_responses_reach_the_same_decision(self):
        streamed, streamed_detector = _collect()
        whole, whole_detector = _collect()
        chunks = [LOOP[:13], *([LOOP] * 50)]
        await _agent(_script(chunks), streamed_detector).run('Explain', event_stream_handler=_drain)

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(''.join(chunks))])

        await Agent(FunctionModel(model), capabilities=[whole_detector]).run('Explain')

        assert streamed == whole
        assert streamed != []

    async def test_a_custom_detector(self):
        @dataclass
        class Keyword(LoopDetector[object]):
            keyword: str

            async def detect(self, ctx: RunContext[object], sample: StreamSample) -> Repetition | None:
                count = sample.text.count(self.keyword)
                return Repetition(text=self.keyword, count=count) if count >= 3 else None

        loops: list[DetectedLoop] = []
        detector = DetectLoops[object](
            stream_detector=Keyword('Hmm.'), stream_check_every=16, action=lambda ctx, loop: loops.append(loop)
        )
        script = _script(['Hmm. Let me think. '] * 3 + ['Done.'])

        await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert [(loop.pattern, loop.count, loop.repeated_text) for loop in loops] == [
            ('repeated_stream_text', 3, 'Hmm.')
        ]

    async def test_detectors_get_the_channel_and_how_much_streamed(self):
        samples: list[StreamSample] = []

        @dataclass
        class Record(LoopDetector[object]):
            @property
            def sample_chars(self) -> int:
                return 10

            async def detect(self, ctx: RunContext[object], sample: StreamSample) -> Repetition | None:
                samples.append(sample)
                return None

        detector = DetectLoops[object](stream_detector=Record(), stream_check_every=8)
        script = _script([_thinking('abcdefghij'), 'klmnopqrst', _args('{"path": "abc"}', index=2, name='read_file')])

        await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert samples == snapshot(
            [
                StreamSample(channel='thinking', text='abcdefgh', total_chars=8, new_chars=8, tool_name=None),
                StreamSample(channel='thinking', text='abcdefghij', total_chars=10, new_chars=2, tool_name=None),
                StreamSample(channel='text', text='klmnopqr', total_chars=8, new_chars=8, tool_name=None),
                StreamSample(channel='text', text='klmnopqrst', total_chars=10, new_chars=2, tool_name=None),
                StreamSample(channel='tool_args', text='{"path":', total_chars=8, new_chars=8, tool_name='read_file'),
                StreamSample(
                    channel='tool_args', text='h": "abc"}', total_chars=15, new_chars=7, tool_name='read_file'
                ),
                StreamSample(channel='text', text='done', total_chars=4, new_chars=4, tool_name=None),
            ]
        )

    async def test_records_a_span(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent(_script([LOOP] * 60), DetectLoops())
        agent.instrument = InstrumentationSettings(tracer_provider=provider)

        await agent.run('Explain', event_stream_handler=_drain)

        [span] = [span for span in exporter.get_finished_spans() if span.name == 'loop detected']
        assert dict(span.attributes or {}) == snapshot(
            {
                'detect_loops.pattern': 'repeated_stream_text',
                'detect_loops.action': 'nudge',
                'detect_loops.count': 37,
                'detect_loops.window': 2048,
                'detect_loops.tool_names': (),
                'detect_loops.fingerprints': ('dc3adfd1bb4f501d',),
                'detect_loops.repeated_text': 'k the config file again before answering. I should chec',
            }
        )

    async def test_span_leaves_out_the_repeated_text_without_content(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent(_script([LOOP] * 60), DetectLoops())
        agent.instrument = InstrumentationSettings(tracer_provider=provider, include_content=False)

        await agent.run('Explain', event_stream_handler=_drain)

        [span] = [span for span in exporter.get_finished_spans() if span.name == 'loop detected']
        attributes = dict(span.attributes or {})
        assert 'detect_loops.repeated_text' not in attributes
        assert 'detect_loops.fingerprints' not in attributes
        assert attributes['detect_loops.pattern'] == 'repeated_stream_text'

    @pytest.mark.parametrize(
        ('create', 'message'),
        [
            (lambda: RepetitionDetector[object](min_period=0), '`min_period` must be at least 1, got 0.'),
            (
                lambda: RepetitionDetector[object](min_period=30, max_period=20),
                '`max_period` must be at least `min_period` (30), got 20.',
            ),
            (lambda: RepetitionDetector[object](min_repeats=1), '`min_repeats` must be at least 2, got 1.'),
            (
                lambda: RepetitionDetector[object](min_distinct_chars=0),
                '`min_distinct_chars` must be at least 1, got 0.',
            ),
            (lambda: DetectLoops[object](stream_check_every=0), '`stream_check_every` must be at least 1, got 0.'),
            (
                lambda: DetectLoops[object](repeated_parallel_call_threshold=1),
                '`repeated_parallel_call_threshold` must be at least 2 or `None`, got 1.',
            ),
            (lambda: LoopJudge[object](), 'Provide a judge `model` or a judge `agent`.'),
            (
                lambda: LoopJudge[object](model=_judge([]), agent=Agent(_judge([]))),
                'Provide either a judge `model` or a judge `agent`, not both.',
            ),
            (lambda: LoopJudge[object](model=_judge([]), every=0), '`every` must be at least 1 or `None`, got 0.'),
        ],
    )
    def test_rejects_invalid_settings(self, create: object, message: str):
        assert callable(create)
        with pytest.raises(ValueError) as exc_info:
            create()

        assert str(exc_info.value) == message


class TestLoopJudge:
    async def test_the_judge_confirms_a_suspected_loop(self):
        prompts: list[str] = []
        judge = LoopJudge[object](model=_judge([True], prompts))
        script = _script([LOOP] * 60)

        result = await _agent(script, DetectLoops(stream_detector=judge)).run('Explain', event_stream_handler=_drain)

        assert result.output == 'done'
        assert script.sent[0] < 45
        assert [nudge.metadata['pattern'] for nudge in _nudges(result.all_messages()) if nudge.metadata] == [
            'repeated_stream_text'
        ]
        [prompt] = prompts
        assert '<streamed_output channel="text">' in prompt

    async def test_the_judge_can_overrule_the_heuristic(self):
        verdicts = [False] * 20
        loops: list[DetectedLoop] = []
        detector = DetectLoops[object](
            stream_detector=LoopJudge(model=_judge(verdicts)), action=lambda ctx, loop: loops.append(loop)
        )
        script = _script([LOOP] * 40)

        result = await _agent(script, detector).run('Explain', event_stream_handler=_drain)

        assert result.output == LOOP * 40
        assert loops == []
        # Asked only when the heuristic suspected a loop, not on every check.
        assert 0 < 20 - len(verdicts) < 10

    async def test_the_judge_can_detect_what_the_heuristic_misses(self):
        prompts: list[str] = []
        passage = 'Let me check the config file again'
        judge = LoopJudge[object](model=_judge([False, True], prompts, passage), every=600)
        # Each attempt has a different number, so the text never repeats exactly.
        script = _script([f'{passage} (attempt {index}). ' for index in range(40)])

        with pytest.raises(LoopDetectedError) as exc_info:
            await _agent(script, DetectLoops(stream_detector=judge, action='error')).run(
                'Explain', event_stream_handler=_drain
            )

        assert len(prompts) == 2
        loop = exc_info.value.loop
        assert (loop.pattern, loop.repeated_text) == ('repeated_stream_text', passage)
        assert loop.count > 2

    @pytest.mark.usefixtures('instrument_all_agents')
    async def test_the_judge_is_not_traced_even_when_every_agent_is(self, capfire: CaptureLogfire):
        judge = LoopJudge[object](model=_judge([True]))

        await Agent(_script([LOOP] * 60).model(), name='outer', capabilities=[DetectLoops(stream_detector=judge)]).run(
            'Explain', event_stream_handler=_drain
        )

        # The judge is shown the model's own output, which the host agent's settings may keep out of traces.
        assert agent_run_names(capfire) == ['outer']

    async def test_the_judges_usage_is_added_to_the_run(self):
        verdicts = [True]
        judge = LoopJudge[object](model=_judge(verdicts))

        result = await _agent(_script([LOOP] * 60), DetectLoops(stream_detector=judge)).run(
            'Explain', event_stream_handler=_drain
        )

        assert verdicts == []
        # The retried request and the judge's: the stopped stream's request isn't counted.
        assert result.usage.requests == 2

    async def test_a_judge_that_quotes_no_repeating_passage(self):
        judge = LoopJudge[object](model=_judge([True], passage=''), every=600)
        script = _script([f'Let me check the config file again (attempt {index}). ' for index in range(40)])

        with pytest.raises(LoopDetectedError) as exc_info:
            await _agent(script, DetectLoops(stream_detector=judge, action='error')).run(
                'Explain', event_stream_handler=_drain
            )

        assert (exc_info.value.loop.count, exc_info.value.loop.message) == snapshot(
            (
                0,
                'Your response kept repeating itself without making progress. Change your approach, or explain what is blocking you.',
            )
        )

    async def test_a_failing_judge_falls_back_to_the_heuristic(self):
        def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('judge is down')

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        script = _script([LOOP] * 60)
        agent = _agent(script, DetectLoops(stream_detector=LoopJudge(model=FunctionModel(broken))))
        agent.instrument = InstrumentationSettings(tracer_provider=provider)

        result = await agent.run('Explain', event_stream_handler=_drain)

        assert result.output == 'done'
        assert [nudge.metadata['pattern'] for nudge in _nudges(result.all_messages()) if nudge.metadata] == [
            'repeated_stream_text'
        ]
        [span] = [span for span in exporter.get_finished_spans() if span.name == 'loop judge failed']
        assert dict(span.attributes or {}) == {'error.type': 'RuntimeError'}

    async def test_a_slow_judge_falls_back_to_the_heuristic(self):
        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            await asyncio.sleep(10)
            raise AssertionError('unreachable')  # pragma: no cover

        script = _script([LOOP] * 60)
        judge = LoopJudge[object](model=FunctionModel(slow), timeout=0.1)

        result = await _agent(script, DetectLoops(stream_detector=judge)).run('Explain', event_stream_handler=_drain)

        assert [nudge.metadata['pattern'] for nudge in _nudges(result.all_messages()) if nudge.metadata] == [
            'repeated_stream_text'
        ]

    async def test_a_judge_agent(self):
        verdicts = [True]
        script = _script([LOOP] * 60)
        judge = LoopJudge[object](agent=Agent(_judge(verdicts), instructions='Be strict.'))

        await _agent(script, DetectLoops(stream_detector=judge)).run('Explain', event_stream_handler=_drain)

        assert verdicts == []
