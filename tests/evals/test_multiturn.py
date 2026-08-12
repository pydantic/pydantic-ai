from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

import pytest
from pydantic import BaseModel, TypeAdapter
from tenacity import stop_after_attempt

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.retries import RetryConfig
from pydantic_evals import Case, Dataset, multiturn as multiturn_module
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, HasMatchingSpan
from pydantic_evals.multiturn import ConversationResult, ConversationTask, ConversationTurn

from ..conftest import try_import

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.anyio
needs_logfire = pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')


async def test_mixed_sync_and_async_callbacks_finish_naturally() -> None:
    async def first_message(inputs: str) -> str:
        return inputs

    def target_factory(inputs: str) -> Callable[[str], str]:
        del inputs

        def target(message: str) -> str:
            return message.upper()

        return target

    def simulator_factory(
        inputs: str,
    ) -> Callable[[Sequence[ConversationTurn[str, str]]], Awaitable[str | None]]:
        del inputs

        async def simulator(turns: Sequence[ConversationTurn[str, str]]) -> str | None:
            return 'second' if len(turns) == 1 else None

        return simulator

    task = ConversationTask[str, str, str](
        first_message=first_message,
        target_factory=target_factory,
        simulator_factory=simulator_factory,
        max_turns=3,
    )
    result = await task.run('first')

    assert result == ConversationResult(
        turns=(
            ConversationTurn(user_message='first', output='FIRST'),
            ConversationTurn(user_message='second', output='SECOND'),
        ),
        stop_reason='simulator_finished',
    )
    assert result.turn_count == 2
    assert result.final_output == 'SECOND'


async def test_simulator_is_called_after_last_allowed_target_turn() -> None:
    simulator_calls = 0

    def simulator_factory(inputs: None) -> Callable[[Sequence[ConversationTurn[str, str]]], str | None]:
        del inputs

        def simulator(turns: Sequence[ConversationTurn[str, str]]) -> str | None:
            nonlocal simulator_calls
            simulator_calls += 1
            return None if len(turns) == 2 else 'continue'

        return simulator

    task = ConversationTask[None, str, str](
        first_message=lambda inputs: 'start',
        target_factory=lambda inputs: lambda message: message,
        simulator_factory=simulator_factory,
        max_turns=2,
    )
    result = await task.run(None)

    assert simulator_calls == 2
    assert result.stop_reason == 'simulator_finished'


async def test_max_turns_stop_reason_when_simulator_wants_to_continue() -> None:
    task = ConversationTask[None, str, str](
        first_message=lambda inputs: 'start',
        target_factory=lambda inputs: lambda message: message,
        simulator_factory=lambda inputs: lambda turns: 'continue',
        max_turns=lambda inputs: 1,
    )
    result = await task.run(None)

    assert result.turn_count == 1
    assert result.stop_reason == 'max_turns'


async def test_invalid_max_turns_fails_before_creating_sessions() -> None:
    factories_called = False

    def target_factory(inputs: None) -> Callable[[str], str]:
        nonlocal factories_called
        factories_called = True
        return lambda message: message

    task = ConversationTask[None, str, str](
        first_message=lambda inputs: 'start',
        target_factory=target_factory,
        simulator_factory=lambda inputs: lambda turns: None,
        max_turns=0,
    )

    with pytest.raises(ValueError, match='max_turns must be >= 1, got 0'):
        await task.run(None)

    assert factories_called is False


async def test_factories_isolate_concurrent_runs() -> None:
    created_target_sessions = 0
    created_simulator_sessions = 0

    def target_factory(inputs: str) -> Callable[[str], Awaitable[str]]:
        nonlocal created_target_sessions
        created_target_sessions += 1
        call_count = 0

        async def target(message: str) -> str:
            nonlocal call_count
            await asyncio.sleep(0)
            call_count += 1
            return f'{inputs}:{call_count}:{message}'

        return target

    def simulator_factory(inputs: str) -> Callable[[Sequence[ConversationTurn[str, str]]], str | None]:
        nonlocal created_simulator_sessions
        created_simulator_sessions += 1

        def simulator(turns: Sequence[ConversationTurn[str, str]]) -> str | None:
            return 'again' if len(turns) == 1 else None

        return simulator

    task = ConversationTask[str, str, str](
        first_message=lambda inputs: 'start',
        target_factory=target_factory,
        simulator_factory=simulator_factory,
    )

    first, second = await asyncio.gather(task.run('same'), task.run('same'))

    assert [turn.output for turn in first.turns] == ['same:1:start', 'same:2:again']
    assert [turn.output for turn in second.turns] == ['same:1:start', 'same:2:again']
    assert created_target_sessions == 2
    assert created_simulator_sessions == 2


async def test_repeat_and_retry_create_fresh_sessions() -> None:
    repeat_sessions = 0

    def repeat_target_factory(inputs: str) -> Callable[[str], str]:
        nonlocal repeat_sessions
        repeat_sessions += 1
        return lambda message: message

    repeat_task = ConversationTask[str, str, str](
        first_message=lambda inputs: inputs,
        target_factory=repeat_target_factory,
        simulator_factory=lambda inputs: lambda turns: None,
    )
    repeat_report = await Dataset(name='repeated conversations', cases=[Case(name='case', inputs='hello')]).evaluate(
        repeat_task.run,
        repeat=3,
        progress=False,
    )

    assert len(repeat_report.cases) == 3
    assert repeat_sessions == 3

    retry_sessions = 0

    def retry_target_factory(inputs: str) -> Callable[[str], str]:
        nonlocal retry_sessions
        retry_sessions += 1
        session_number = retry_sessions

        def target(message: str) -> str:
            if session_number == 1:
                raise RuntimeError('retry this conversation')
            return message

        return target

    retry_task = ConversationTask[str, str, str](
        first_message=lambda inputs: inputs,
        target_factory=retry_target_factory,
        simulator_factory=lambda inputs: lambda turns: None,
    )
    retry_report = await Dataset(name='retried conversation', cases=[Case(inputs='hello')]).evaluate(
        retry_task.run,
        retry_task=RetryConfig(stop=stop_after_attempt(2)),
        progress=False,
    )

    assert len(retry_report.cases) == 1
    assert retry_sessions == 2


class Patch(BaseModel):
    species: str


async def test_result_with_structured_output_is_serializable() -> None:
    task = ConversationTask[None, str, Patch](
        first_message=lambda inputs: 'start',
        target_factory=lambda inputs: lambda message: Patch(species='cat'),
        simulator_factory=lambda inputs: lambda turns: None,
    )
    result = await task.run(None)

    assert TypeAdapter(ConversationResult[str, Patch]).dump_python(result, mode='json') == {
        'turns': [{'user_message': 'start', 'output': {'species': 'cat'}}],
        'stop_reason': 'simulator_finished',
    }


async def test_dataset_records_metrics_attributes_and_failures() -> None:
    @dataclass(repr=False)
    class ObservabilityEvaluator(Evaluator[str, ConversationResult[str, str], object]):
        def evaluate(self, ctx: EvaluatorContext[str, ConversationResult[str, str], object]) -> bool:
            assert ctx.metrics['conversation_turn_count'] == 1
            assert ctx.metrics['conversation_target_turn_duration_seconds_avg'] >= 0
            assert (
                ctx.metrics['conversation_target_turn_duration_seconds_max']
                == ctx.metrics['conversation_target_turn_duration_seconds_avg']
            )
            assert ctx.attributes == {'conversation_stop_reason': 'simulator_finished'}
            return True

    successful_task = ConversationTask[str, str, str](
        first_message=lambda inputs: inputs,
        target_factory=lambda inputs: lambda message: message.upper(),
        simulator_factory=lambda inputs: lambda turns: None,
    )
    successful_dataset = Dataset(
        name='successful conversation',
        cases=[Case(inputs='hello')],
        evaluators=[ObservabilityEvaluator()],
    )

    report = await successful_dataset.evaluate(successful_task.run, progress=False)

    assert report.cases[0].assertions['ObservabilityEvaluator'].value is True

    def failing_target(message: str) -> str:
        raise RuntimeError(f'cannot handle {message}')

    failing_task = ConversationTask[str, str, str](
        first_message=lambda inputs: inputs,
        target_factory=lambda inputs: failing_target,
        simulator_factory=lambda inputs: lambda turns: None,
    )
    failed_report = await Dataset(name='failed conversation', cases=[Case(inputs='bad')]).evaluate(
        failing_task.run, progress=False
    )

    assert failed_report.cases == []
    assert failed_report.failures[0].error_message == 'RuntimeError: cannot handle bad'


async def test_target_turn_duration_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([1.0, 1.25, 2.0, 2.75])
    monkeypatch.setattr(multiturn_module, 'perf_counter', lambda: next(timestamps))

    task = ConversationTask[None, str, str](
        first_message=lambda inputs: 'first',
        target_factory=lambda inputs: lambda message: message,
        simulator_factory=lambda inputs: lambda turns: 'second' if len(turns) == 1 else None,
    )
    report = await Dataset(name='conversation timing', cases=[Case(inputs=None)]).evaluate(task.run, progress=False)

    assert report.cases[0].metrics == {
        'conversation_turn_count': 2,
        'conversation_target_turn_duration_seconds_avg': 0.5,
        'conversation_target_turn_duration_seconds_max': 0.75,
    }


@needs_logfire
async def test_role_spans_are_available_to_standard_evaluators(capfire: CaptureLogfire) -> None:
    assert capfire
    task = ConversationTask[str, str, str](
        first_message=lambda inputs: inputs,
        target_factory=lambda inputs: lambda message: message.upper(),
        simulator_factory=lambda inputs: lambda turns: None,
    )
    dataset = Dataset(
        name='conversation spans',
        cases=[Case(inputs='hello')],
        evaluators=[
            HasMatchingSpan(
                query={
                    'name_equals': 'multiturn target',
                    'has_attributes': {'conversation_role': 'target', 'conversation_turn': 1},
                },
                evaluation_name='target_span',
            ),
            HasMatchingSpan(
                query={
                    'name_equals': 'multiturn simulator',
                    'has_attributes': {'conversation_role': 'simulator', 'conversation_turn': 1},
                },
                evaluation_name='simulator_span',
            ),
        ],
    )

    report = await dataset.evaluate(task.run, progress=False)

    assert report.cases[0].assertions['target_span'].value is True
    assert report.cases[0].assertions['simulator_span'].value is True


@dataclass(frozen=True)
class AgentScenario:
    name: str
    first_message: str


async def test_from_agents_preserves_histories_deps_and_prompt_hooks() -> None:
    target_requests: list[list[ModelMessage]] = []
    simulator_requests: list[list[ModelMessage]] = []
    simulator_deps: list[AgentScenario] = []
    target_prompts: list[tuple[str, int]] = []
    simulator_outputs: list[str] = []

    def target_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        target_requests.append(list(messages))
        return ModelResponse(parts=[TextPart(content=f'target response {len(target_requests)}')])

    def simulator_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        simulator_requests.append(list(messages))
        response = 'continue' if len(simulator_requests) == 1 else 'done'
        return ModelResponse(parts=[TextPart(content=response)])

    target_agent = Agent(FunctionModel(target_model))
    simulator_agent = Agent(FunctionModel(simulator_model), deps_type=AgentScenario)

    @simulator_agent.instructions
    def simulator_instructions(ctx: RunContext[AgentScenario]) -> str:
        simulator_deps.append(ctx.deps)
        return f'Simulate {ctx.deps.name}'

    async def target_prompt(inputs: AgentScenario, message: str, turn_index: int) -> str:
        target_prompts.append((message, turn_index))
        return f'{message}\nAttachment: {inputs.name}.txt'

    def simulator_prompt(
        inputs: AgentScenario,
        turn: ConversationTurn[str, str],
    ) -> str:
        simulator_outputs.append(turn.output)
        return f'{inputs.name}: {turn.output}'

    task = ConversationTask.from_agents(
        target_agent=target_agent,
        simulator_agent=simulator_agent,
        first_message=lambda inputs: inputs.first_message,
        next_message=lambda output: None if output == 'done' else output,
        target_prompt=target_prompt,
        simulator_prompt=simulator_prompt,
        max_turns=3,
    )

    scenario = AgentScenario(name='miso', first_message='My cat is unwell.')
    result = await task.run(scenario)

    assert [turn.user_message for turn in result.turns] == ['My cat is unwell.', 'continue']
    assert target_prompts == [('My cat is unwell.', 1), ('continue', 2)]
    assert simulator_outputs == ['target response 1', 'target response 2']
    assert len(target_requests[1]) > len(target_requests[0])
    assert len(simulator_requests[1]) > len(simulator_requests[0])
    assert simulator_deps == [scenario, scenario]
    assert result.stop_reason == 'simulator_finished'
