from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from textwrap import dedent
from typing import Literal

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import (
    EvaluationReason,
    Evaluator,
    EvaluatorContext,
    LLMJudge,
)

Role = Literal['user', 'assistant']
StopReason = Literal['simulator_done', 'max_turns']


class ConversationMessage(BaseModel):
    role: Role
    content: str


class ConversationScenario(BaseModel):
    """Inputs for a simulated multi-turn conversation eval case."""

    simulator_goal: str
    first_message: str
    simulator_persona: str = 'Polite, concise, and realistic.'
    max_turns: int = Field(default=6, ge=1)


class SimulatorOutput(BaseModel):
    """Normalized result of one simulator turn."""

    message: str = Field(description='The next user message. Empty only when done is true.')
    done: bool = Field(description='True when the simulator goal has been resolved.')


class ConversationTurn(BaseModel):
    turn_index: int
    user_message: str
    assistant_message: str
    simulator_message: str | None
    target_duration: float
    simulator_duration: float


class ConversationRun(BaseModel):
    transcript: list[ConversationMessage]
    turns: list[ConversationTurn]
    stop_reason: StopReason
    turn_count: int
    target_duration: float
    simulator_duration: float
    average_target_turn_duration: float


TargetTurn = Callable[[str, Sequence[ConversationMessage]], Awaitable[str]]
SimulatorTurn = Callable[[ConversationScenario, Sequence[ConversationMessage]], Awaitable[SimulatorOutput]]
SimulatorAgentFactory = Callable[[ConversationScenario], Agent[None, SimulatorOutput]]


def _seed_simulator_message_history(first_message: str) -> list[ModelMessage]:
    normalized_message = first_message.strip()
    if not normalized_message:
        return []
    return [
        ModelRequest(parts=[UserPromptPart(content='Initial simulated user message.')]),
        ModelResponse(parts=[TextPart(content=normalized_message)]),
    ]


def _build_simulator_agent(scenario: ConversationScenario) -> Agent[None, SimulatorOutput]:
    instructions = '\n'.join(
        [
            'You are simulating the user in an animal-sounds multi-turn evaluation.',
            'Return the next user message unless the objective is fully completed.',
            'Do not roleplay as the assistant.',
            'Set done=true only when the simulation objective is fully completed.',
            'When done=true, return an empty message.',
            'When done=false, return the next realistic user message.',
            'Never ask for optional refinements or extra follow-up after the assistant has satisfied the goal.',
            'Persona:',
            scenario.simulator_persona,
            'Simulation objective:',
            scenario.simulator_goal,
        ]
    )
    return Agent(
        'openai:gpt-5.2',
        output_type=SimulatorOutput,
        instructions=instructions,
    )


def _build_simulator_turn_prompt(assistant_message: str) -> str:
    return dedent(
        f"""
        {assistant_message}

        If the assistant has satisfied the simulation goal, set done=true and message="".
        Otherwise, set done=false and return the next user message.
        """
    ).strip()


async def run_conversation(
    scenario: ConversationScenario,
    run_target_turn: TargetTurn,
    run_simulator_turn: SimulatorTurn,
) -> ConversationRun:
    """Run a target agent against a simulator while recording a simplified transcript."""
    transcript: list[ConversationMessage] = []
    turns: list[ConversationTurn] = []
    user_message = scenario.first_message
    target_duration_total = 0.0
    simulator_duration_total = 0.0
    stop_reason: StopReason = 'max_turns'

    for turn_index in range(1, scenario.max_turns + 1):
        transcript.append(ConversationMessage(role='user', content=user_message))

        target_started = time.perf_counter()
        assistant_message = await run_target_turn(user_message, tuple(transcript))
        target_duration = time.perf_counter() - target_started
        target_duration_total += target_duration
        transcript.append(ConversationMessage(role='assistant', content=assistant_message))

        simulator_started = time.perf_counter()
        simulator_output = await run_simulator_turn(scenario, tuple(transcript))
        simulator_duration = time.perf_counter() - simulator_started
        simulator_duration_total += simulator_duration

        next_user_message = simulator_output.message.strip()
        turns.append(
            ConversationTurn(
                turn_index=turn_index,
                user_message=user_message,
                assistant_message=assistant_message,
                simulator_message=None if simulator_output.done else next_user_message,
                target_duration=target_duration,
                simulator_duration=simulator_duration,
            )
        )

        if simulator_output.done:
            stop_reason = 'simulator_done'
            break
        if not next_user_message:
            raise ValueError('Simulator returned an empty message with `done=False`.')

        user_message = next_user_message

    turn_count = len(turns)
    average_target_turn_duration = target_duration_total / turn_count if turn_count else 0.0
    run = ConversationRun(
        transcript=transcript,
        turns=turns,
        stop_reason=stop_reason,
        turn_count=turn_count,
        target_duration=target_duration_total,
        simulator_duration=simulator_duration_total,
        average_target_turn_duration=average_target_turn_duration,
    )
    _record_eval_observability(run)
    return run


def _record_eval_observability(run: ConversationRun) -> None:
    increment_eval_metric('turn_count', run.turn_count)
    increment_eval_metric('target_duration_seconds', run.target_duration)
    increment_eval_metric('simulator_duration_seconds', run.simulator_duration)
    increment_eval_metric('average_target_turn_duration_seconds', run.average_target_turn_duration)
    set_eval_attribute('stop_reason', run.stop_reason)
    set_eval_attribute('completed_by_simulator', run.stop_reason == 'simulator_done')
    set_eval_attribute('transcript', [message.model_dump(mode='json') for message in run.transcript])


@dataclass
class SimulatedConversationTask:
    target_agent: Agent[None, str]
    simulator_agent_factory: SimulatorAgentFactory = _build_simulator_agent

    async def run(self, scenario: ConversationScenario) -> ConversationRun:
        target_message_history: list[ModelMessage] = []
        simulator_message_history = _seed_simulator_message_history(scenario.first_message)
        simulator_agent = self.simulator_agent_factory(scenario)

        async def run_target_turn(user_message: str, transcript: Sequence[ConversationMessage]) -> str:
            del transcript
            result = await self.target_agent.run(user_message, message_history=target_message_history)
            target_message_history[:] = result.all_messages()
            return result.output

        async def run_simulator_turn(
            scenario: ConversationScenario, transcript: Sequence[ConversationMessage]
        ) -> SimulatorOutput:
            del scenario
            assistant_message = transcript[-1].content
            result = await simulator_agent.run(
                _build_simulator_turn_prompt(assistant_message),
                message_history=simulator_message_history,
            )
            simulator_message_history[:] = result.all_messages()
            message = result.output.message.strip()
            return SimulatorOutput(message='' if result.output.done else message, done=result.output.done)

        return await run_conversation(scenario, run_target_turn, run_simulator_turn)


@dataclass(repr=False)
class ConversationCompleted(Evaluator[ConversationScenario, ConversationRun, object]):
    def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> EvaluationReason:
        completed = ctx.output.stop_reason == 'simulator_done'
        reason = None if completed else f'Conversation stopped after {ctx.output.turn_count} turns.'
        return EvaluationReason(value=completed, reason=reason)


@dataclass(repr=False)
class MaxTargetTurns(Evaluator[ConversationScenario, ConversationRun, object]):
    max_turns: int

    def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> EvaluationReason:
        passed = ctx.output.turn_count <= self.max_turns
        reason = None if passed else f'Conversation took {ctx.output.turn_count} turns; expected at most {self.max_turns}.'
        return EvaluationReason(value=passed, reason=reason)


@dataclass(repr=False)
class MaxAverageTargetTurnDuration(Evaluator[ConversationScenario, ConversationRun, object]):
    max_seconds: float

    def evaluate(self, ctx: EvaluatorContext[ConversationScenario, ConversationRun, object]) -> EvaluationReason:
        passed = ctx.output.average_target_turn_duration <= self.max_seconds
        reason = (
            None
            if passed
            else (
                'Average target-agent turn duration was '
                f'{ctx.output.average_target_turn_duration:.2f}s; expected at most {self.max_seconds:.2f}s.'
            )
        )
        return EvaluationReason(value=passed, reason=reason)


def build_dataset(*, include_llm_judge: bool = True) -> Dataset[ConversationScenario, ConversationRun, object]:
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


def build_live_task() -> SimulatedConversationTask:
    target_agent: Agent[None, str] = Agent(
        'openai:gpt-5.2',
        output_type=str,
        system_prompt=(
            'You answer in English and explain the sound made by animals. Keep answers short and clear. '
            'If the user asks about something that is not an animal, politely explain that you can only answer '
            'questions about animal sounds.'
        ),
    )
    return SimulatedConversationTask(target_agent=target_agent)


async def run_live() -> None:
    logfire.configure(send_to_logfire='if-token-present', environment='development')
    logfire.instrument_pydantic_ai()

    report = await build_dataset(include_llm_judge=True).evaluate(
        build_live_task().run,
        max_concurrency=1,
        task_name='animal_sounds_multi_turn',
    )
    report.print(include_input=True, include_output=True, include_reasons=True, include_total_duration=True)


if __name__ == '__main__':
    asyncio.run(run_live())
