"""Minimal multi-turn evaluation with a target agent and a simulated user.

This example shows the short user journey. See `example_06_multiturn_veterinary`
for application-specific composition built on the same primitives.
"""

from __future__ import annotations

import asyncio

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, models
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.multiturn import ConversationResult, ConversationTask


class ConversationScenario(BaseModel):
    """Goal and opening message for one simulated conversation."""

    goal: str
    first_message: str
    persona: str = 'Polite, concise, and realistic.'
    max_turns: int = Field(default=4, ge=1)


class SimulatorDecision(BaseModel):
    """The simulated user's decision after seeing a target response."""

    message: str = Field(
        description='The next user message, or an empty string when done is true.'
    )
    done: bool = Field(description='Whether the conversation goal has been satisfied.')


def build_target_agent(
    model: models.Model | models.KnownModelName | str = 'openai:gpt-5.2',
) -> Agent[None, str]:
    return Agent(
        model,
        instructions=(
            'Answer in English and explain the sound made by animals. Keep answers short and clear. '
            'If the user asks about something that is not an animal, politely explain that you only answer '
            'questions about animal sounds.'
        ),
    )


def build_simulator_agent(
    model: models.Model | models.KnownModelName | str = 'openai:gpt-5.2',
) -> Agent[ConversationScenario, SimulatorDecision]:
    def simulator_instructions(ctx: RunContext[ConversationScenario]) -> str:
        return (
            'Act as the user in an evaluation conversation. Return the next realistic user message unless the '
            'goal is fully satisfied. Set done=true only when it is satisfied, and do not ask for optional '
            f'follow-up.\n\nPersona: {ctx.deps.persona}\nGoal: {ctx.deps.goal}'
        )

    return Agent(
        model,
        deps_type=ConversationScenario,
        output_type=SimulatorDecision,
        instructions=simulator_instructions,
    )


def build_conversation_task(
    *,
    target_model: models.Model | models.KnownModelName | str = 'openai:gpt-5.2',
    simulator_model: models.Model | models.KnownModelName | str = 'openai:gpt-5.2',
) -> ConversationTask[ConversationScenario, str, str]:
    """Build the reusable task; message histories are managed by `from_agents`."""
    return ConversationTask.from_agents(
        target_agent=build_target_agent(target_model),
        simulator_agent=build_simulator_agent(simulator_model),
        first_message=lambda scenario: scenario.first_message,
        next_message=lambda decision: None if decision.done else decision.message,
        max_turns=lambda scenario: scenario.max_turns,
    )


def build_dataset(
    *, include_llm_judge: bool = True
) -> Dataset[ConversationScenario, ConversationResult[str, str], object]:
    evaluators = (
        [
            LLMJudge(
                rubric=(
                    'The assistant satisfies the simulated user goal, answers animal-sound questions correctly, '
                    'and politely refuses non-animal questions.'
                ),
                include_input=True,
                model='openai:gpt-5.2',
            )
        ]
        if include_llm_judge
        else []
    )
    return Dataset(
        name='multiturn_animal_sounds',
        cases=[
            Case(
                name='dog sound',
                inputs=ConversationScenario(
                    goal='Learn that a dog barks.',
                    first_message='What sound does a dog make?',
                ),
            ),
            Case(
                name='cat then elephant',
                inputs=ConversationScenario(
                    goal='First learn what sound a cat makes, then what sound an elephant makes.',
                    first_message='What sound does a cat make?',
                    max_turns=5,
                ),
            ),
        ],
        evaluators=evaluators,
    )


async def main() -> None:
    logfire.configure(send_to_logfire='if-token-present', environment='development')
    logfire.instrument_pydantic_ai()

    conversation = build_conversation_task()
    report = await build_dataset().evaluate(
        conversation.run,
        name='animal sounds - pydantic agent',
        max_concurrency=1,
    )
    report.print(include_input=True, include_output=True, include_reasons=True)


if __name__ == '__main__':
    asyncio.run(main())
