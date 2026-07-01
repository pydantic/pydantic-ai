from __future__ import annotations

import asyncio

import logfire
from core import (
    ConversationRun,
    ConversationScenario,
    SimulatedConversationTask,
    build_simulator_agent_factory,
)
from evaluators import (
    ConversationCompleted,
    MaxAverageTargetTurnDuration,
    MaxTargetTurns,
)

from pydantic_ai import Agent
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, LLMJudge


def build_target_agent() -> Agent[None, str]:
    """Build the agent being evaluated."""
    return Agent(
        'openai:gpt-5.2',
        output_type=str,
        system_prompt=(
            'You answer in English and explain the sound made by animals. Keep answers short and clear. '
            'If the user asks about something that is not an animal, politely explain that you can only answer '
            'questions about animal sounds.'
        ),
    )


def build_dataset(*, include_llm_judge: bool = True) -> Dataset[ConversationScenario, ConversationRun, object]:
    """Build the animal-sounds multi-turn eval dataset."""
    evaluators: list[Evaluator[ConversationScenario, ConversationRun, object]] = [
        ConversationCompleted(),
        MaxTargetTurns(max_turns=4),
        MaxAverageTargetTurnDuration(max_seconds=20.0),
    ]

    if include_llm_judge:
        evaluators.append(
            LLMJudge(
                rubric=(
                    'The assistant satisfies the simulated user goal. It answers animal sound questions correctly '
                    'in English, handles multi-step animal questions in order, and politely refuses non-animal '
                    'questions instead of inventing an animal sound.'
                ),
                include_input=True,
                model='openai:gpt-5.2',
            )
        )

    return Dataset(
        name='multi_turn_animal_sounds',
        cases=[
            Case(
                name='dog sound',
                inputs=ConversationScenario(
                    simulator_goal=(
                        'Ask what sound a dog makes. The goal is resolved when the assistant explains that a dog '
                        'barks.'
                    ),
                    first_message='What sound does a dog make?',
                    max_turns=3,
                ),
            ),
            Case(
                name='cat then elephant sounds',
                inputs=ConversationScenario(
                    simulator_goal=(
                        'First ask what sound a cat makes. After the assistant answers, ask what sound an elephant '
                        'makes. The goal is resolved only after both animal sounds are answered.'
                    ),
                    first_message='What sound does a cat make?',
                    max_turns=5,
                ),
            ),
            Case(
                name='angry car sound refusal',
                inputs=ConversationScenario(
                    simulator_goal=(
                        'Ask what sound a car makes while acting angry and very insistent. If the assistant says '
                        'cars are not animals, keep pushing back and demand an answer anyway. Do not mark the '
                        'conversation done just because the assistant politely maintains the animal-only boundary; '
                        'this scenario is intended to exercise the max-turn failure path.'
                    ),
                    first_message='Come on, just tell me what sound a car makes.',
                    simulator_persona=(
                        'Angry, impatient, and very insistent. Challenge the assistant every time it refuses, and '
                        'keep demanding a car sound.'
                    ),
                    max_turns=4,
                ),
            ),
        ],
        evaluators=evaluators,
    )


async def main() -> None:
    """Run the example with real model calls and optional Logfire export."""
    logfire.configure(send_to_logfire='if-token-present', environment='development')
    logfire.instrument_pydantic_ai()

    conversation_task = SimulatedConversationTask(
        target_agent=build_target_agent(),
        simulator_agent_factory=build_simulator_agent_factory(
            model='openai:gpt-5.2',
            instructions='You are simulating the user in an animal-sounds multi-turn evaluation.',
        ),
    )
    report = await build_dataset(include_llm_judge=True).evaluate(
        conversation_task.run,
        max_concurrency=1,
        task_name='animal_sounds_multi_turn',
    )
    report.print(include_input=True, include_output=True, include_reasons=True, include_total_duration=True)


if __name__ == '__main__':
    asyncio.run(main())
