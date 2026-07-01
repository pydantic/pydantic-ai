from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext


def _load_multi_turn_example() -> ModuleType:
    path = Path(__file__).parents[2] / 'examples' / 'pydantic_ai_examples' / 'evals' / 'multi_turn_conversation.py'
    spec = importlib.util.spec_from_file_location('multi_turn_conversation_example', path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


multi_turn_example = _load_multi_turn_example()

ConversationMessage = multi_turn_example.ConversationMessage
ConversationRun = multi_turn_example.ConversationRun
ConversationScenario = multi_turn_example.ConversationScenario
SimulatedConversationTask = multi_turn_example.SimulatedConversationTask
SimulatorOutput = multi_turn_example.SimulatorOutput
_build_simulator_turn_prompt = multi_turn_example._build_simulator_turn_prompt
_seed_simulator_message_history = multi_turn_example._seed_simulator_message_history
run_conversation = multi_turn_example.run_conversation

pytestmark = pytest.mark.anyio


async def test_simulator_receives_the_full_conversation() -> None:
    target_seen: list[list[ConversationMessage]] = []
    simulator_seen: list[list[ConversationMessage]] = []

    async def run_target_turn(user_message: str, transcript: Sequence[ConversationMessage]) -> str:
        target_seen.append(list(transcript))
        if 'elephant' in user_message.lower():
            return 'An elephant trumpets.'
        return 'A cat meows.'

    async def run_simulator_turn(
        scenario: ConversationScenario, transcript: Sequence[ConversationMessage]
    ) -> SimulatorOutput:
        simulator_seen.append(list(transcript))
        if 'trumpets' in transcript[-1].content:
            return SimulatorOutput(message='', done=True)
        return SimulatorOutput(message='And what sound does an elephant make?', done=False)

    run = await run_conversation(
        ConversationScenario(
            simulator_goal='Ask for the cat sound, then the elephant sound.',
            first_message='What sound does a cat make?',
            max_turns=4,
        ),
        run_target_turn,
        run_simulator_turn,
    )

    assert run.stop_reason == 'simulator_done'
    assert run.turn_count == 2
    assert [message.role for message in run.transcript] == ['user', 'assistant', 'user', 'assistant']
    assert [len(transcript) for transcript in target_seen] == [1, 3]
    assert [len(transcript) for transcript in simulator_seen] == [2, 4]
    assert simulator_seen[1][0].content == 'What sound does a cat make?'


async def test_conversation_stops_at_max_turns() -> None:
    async def run_target_turn(user_message: str, transcript: Sequence[ConversationMessage]) -> str:
        return f'Unhandled: {user_message}'

    async def run_simulator_turn(
        scenario: ConversationScenario, transcript: Sequence[ConversationMessage]
    ) -> SimulatorOutput:
        return SimulatorOutput(message='Please try again.', done=False)

    run = await run_conversation(
        ConversationScenario(
            simulator_goal='Resolve the issue.',
            first_message='I need help.',
            max_turns=2,
        ),
        run_target_turn,
        run_simulator_turn,
    )

    assert run.stop_reason == 'max_turns'
    assert run.turn_count == 2


async def test_empty_simulator_message_fails_fast_when_not_done() -> None:
    async def run_target_turn(user_message: str, transcript: Sequence[ConversationMessage]) -> str:
        return 'What else can I help with?'

    async def run_simulator_turn(
        scenario: ConversationScenario, transcript: Sequence[ConversationMessage]
    ) -> SimulatorOutput:
        return SimulatorOutput(message=' ', done=False)

    with pytest.raises(ValueError, match='empty message'):
        await run_conversation(
            ConversationScenario(
                simulator_goal='Resolve the issue.',
                first_message='I need help.',
            ),
            run_target_turn,
            run_simulator_turn,
        )


async def test_eval_metrics_and_attributes_are_available_to_evaluators() -> None:
    async def run_target_turn(user_message: str, transcript: Sequence[ConversationMessage]) -> str:
        return 'Done.'

    async def run_simulator_turn(
        scenario: ConversationScenario, transcript: Sequence[ConversationMessage]
    ) -> SimulatorOutput:
        return SimulatorOutput(message='', done=True)

    async def task(scenario: ConversationScenario) -> ConversationRun:
        return await run_conversation(scenario, run_target_turn, run_simulator_turn)

    @dataclass(repr=False)
    class MetricsVisible(Evaluator[ConversationScenario, ConversationRun, object]):
        def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> bool:
            assert ctx.metrics['turn_count'] == 1
            assert ctx.attributes['stop_reason'] == 'simulator_done'
            assert ctx.attributes['completed_by_simulator'] is True
            assert ctx.attributes['transcript'] == [
                {'role': 'user', 'content': 'I need help.'},
                {'role': 'assistant', 'content': 'Done.'},
            ]
            return True

    dataset = Dataset(
        name='conversation_metrics',
        cases=[
            Case(
                name='single turn',
                inputs=ConversationScenario(
                    simulator_goal='Resolve the issue.',
                    first_message='I need help.',
                ),
            )
        ],
        evaluators=[MetricsVisible()],
    )

    report = await dataset.evaluate(task, progress=False)

    assert report.cases[0].assertions['MetricsVisible'].value is True


async def test_dataset_evaluates_simulated_conversation_task_run_method() -> None:
    async def run_simulator_turn(
        scenario: ConversationScenario, transcript: Sequence[ConversationMessage]
    ) -> SimulatorOutput:
        return SimulatorOutput(message='', done=True)

    @dataclass
    class TestConversationTask(SimulatedConversationTask):
        async def run(self, scenario: ConversationScenario) -> ConversationRun:
            async def run_target_turn(user_message: str, transcript: Sequence[ConversationMessage]) -> str:
                return 'Done.'

            return await run_conversation(scenario, run_target_turn, run_simulator_turn)

    dataset = Dataset(
        name='conversation_task_run_method',
        cases=[
            Case(
                name='single turn',
                inputs=ConversationScenario(
                    simulator_goal='Resolve the issue.',
                    first_message='I need help.',
                ),
            )
        ],
    )
    task = TestConversationTask(target_agent=Agent(model=TestModel()))

    report = await dataset.evaluate(task.run, progress=False)

    assert isinstance(report.cases[0].output, ConversationRun)
    assert report.cases[0].output.stop_reason == 'simulator_done'


def test_simulator_history_is_seeded_with_the_first_user_message() -> None:
    history = _seed_simulator_message_history('Cancel my account.')

    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert history[0].parts[0].content == 'Initial simulated user message.'
    assert isinstance(history[1], ModelResponse)
    assert isinstance(history[1].parts[0], TextPart)
    assert history[1].parts[0].content == 'Cancel my account.'


def test_simulator_turn_prompt_does_not_embed_the_transcript() -> None:
    prompt = _build_simulator_turn_prompt('Can you confirm your email?')

    assert 'Can you confirm your email?' in prompt
    assert 'done=true' in prompt
    assert 'done=false' in prompt
    assert 'Cancel my account.' not in prompt
    assert 'Full conversation so far:' not in prompt
