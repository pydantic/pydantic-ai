from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from textwrap import dedent
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_evals import increment_eval_metric, set_eval_attribute

Role = Literal['user', 'assistant']
StopReason = Literal['simulator_done', 'max_turns']


class ConversationMessage(BaseModel):
    """A simplified transcript message for reports, evaluators, and debugging."""

    role: Role
    content: str


class ConversationScenario(BaseModel):
    """Inputs for one simulated multi-turn conversation case."""

    simulator_goal: str
    first_message: str
    simulator_persona: str = 'Polite, concise, and realistic.'
    max_turns: int = Field(default=6, ge=1)


class SimulatorOutput(BaseModel):
    """Structured response returned by the simulator agent for one turn."""

    message: str = Field(description='The next user message. Empty only when done is true.')
    done: bool = Field(description='True when the simulator goal has been resolved.')


class ConversationTurn(BaseModel):
    """Trace data for one target-agent exchange."""

    turn_index: int
    user_message: str
    assistant_message: str
    simulator_message: str | None
    target_duration: float
    simulator_duration: float


class ConversationRun(BaseModel):
    """Complete output of a simulated conversation eval case."""

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


def build_simulator_agent_factory(
    *,
    model: str,
    instructions: str | Sequence[str] = (),
) -> SimulatorAgentFactory:
    """Build a scenario-aware simulator factory from a model and optional domain instructions.

    The returned factory creates one simulator `Agent` per case. Core simulation
    mechanics stay here, while client examples only provide domain-specific
    instructions such as "you are asking about animal sounds".
    """
    extra_instructions = [instructions] if isinstance(instructions, str) else list(instructions)

    def build_simulator_agent(scenario: ConversationScenario) -> Agent[None, SimulatorOutput]:
        return Agent(
            model,
            output_type=SimulatorOutput,
            instructions='\n'.join(
                [
                    *extra_instructions,
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
            ),
        )

    return build_simulator_agent


def seed_simulator_message_history(first_message: str) -> list[ModelMessage]:
    """Seed the simulator's native model history with the first simulated user message.

    The simulator has its own conversation, separate from the target agent. The first
    user message is therefore represented as a previous model response from the
    simulator, so future simulator turns can see it through `message_history`.
    """
    normalized_message = first_message.strip()
    if not normalized_message:
        return []
    return [
        ModelRequest(parts=[UserPromptPart(content='Initial simulated user message.')]),
        ModelResponse(parts=[TextPart(content=normalized_message)]),
    ]


def build_simulator_turn_prompt(assistant_message: str) -> str:
    """Build the simulator prompt for the latest target-agent message only."""
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
    """Run a target agent against a simulator and collect multi-turn eval metrics."""
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
    """Adapter that turns a target `Agent` and simulator factory into an eval task."""

    target_agent: Agent[None, str]
    simulator_agent_factory: SimulatorAgentFactory

    async def run(self, scenario: ConversationScenario) -> ConversationRun:
        """Run one scenario as the callable passed to `Dataset.evaluate`."""
        target_message_history: list[ModelMessage] = []
        simulator_message_history = seed_simulator_message_history(scenario.first_message)
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
                build_simulator_turn_prompt(assistant_message),
                message_history=simulator_message_history,
            )
            simulator_message_history[:] = result.all_messages()
            message = result.output.message.strip()
            return SimulatorOutput(message='' if result.output.done else message, done=result.output.done)

        return await run_conversation(scenario, run_target_turn, run_simulator_turn)
