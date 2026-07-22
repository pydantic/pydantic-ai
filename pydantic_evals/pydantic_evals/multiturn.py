"""Primitives for evaluating simulated multi-turn conversations."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass
from time import perf_counter
from typing import Any, Generic, Literal

from typing_extensions import TypeAliasType, TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ._utils import logfire_span
from .dataset import increment_eval_metric, set_eval_attribute

__all__ = ('ConversationResult', 'ConversationTask', 'ConversationTurn')


InputsT = TypeVar('InputsT', default=Any)
UserMessageT = TypeVar('UserMessageT', default=Any)
TargetOutputT = TypeVar('TargetOutputT', default=Any)
SimulatorOutputT = TypeVar('SimulatorOutputT')
_T = TypeVar('_T')


@dataclass(frozen=True)
class ConversationTurn(Generic[UserMessageT, TargetOutputT]):
    """One completed exchange between a simulated user and the target."""

    _: KW_ONLY
    user_message: UserMessageT
    """The logical message produced by the simulated user."""

    output: TargetOutputT
    """The target's output for `user_message`."""


_TargetSession = TypeAliasType(
    '_TargetSession',
    Callable[[UserMessageT], TargetOutputT | Awaitable[TargetOutputT]],
    type_params=(UserMessageT, TargetOutputT),
)
_SimulatorSession = TypeAliasType(
    '_SimulatorSession',
    Callable[
        [Sequence[ConversationTurn[UserMessageT, TargetOutputT]]],
        UserMessageT | None | Awaitable[UserMessageT | None],
    ],
    type_params=(UserMessageT, TargetOutputT),
)


@dataclass(frozen=True)
class ConversationResult(Generic[UserMessageT, TargetOutputT]):
    """The completed trajectory and stop reason for a simulated conversation."""

    _: KW_ONLY
    turns: tuple[ConversationTurn[UserMessageT, TargetOutputT], ...]
    """Completed exchanges in chronological order."""

    stop_reason: Literal['simulator_finished', 'max_turns']
    """Whether the simulator finished or the target turn limit was reached."""

    @property
    def turn_count(self) -> int:
        """The number of completed target turns."""
        return len(self.turns)

    @property
    def final_output(self) -> TargetOutputT:
        """The target output from the final turn."""
        return self.turns[-1].output


class ConversationTask(Generic[InputsT, UserMessageT, TargetOutputT]):
    """Run one isolated simulated conversation for each evaluation case.

    Pass the [`run`][pydantic_evals.multiturn.ConversationTask.run] method to
    [`Dataset.evaluate`][pydantic_evals.dataset.Dataset.evaluate]. The factories
    create fresh target and simulator sessions for every run, preventing state
    such as message history from leaking across concurrent cases, repeats, or
    retries.
    """

    def __init__(
        self,
        *,
        first_message: Callable[[InputsT], UserMessageT | Awaitable[UserMessageT]],
        target_factory: Callable[[InputsT], _TargetSession[UserMessageT, TargetOutputT]],
        simulator_factory: Callable[[InputsT], _SimulatorSession[UserMessageT, TargetOutputT]],
        max_turns: int | Callable[[InputsT], int] = 6,
    ) -> None:
        """Create a conversation task.

        Args:
            first_message: Produce the first logical user message for a case.
            target_factory: Create a fresh target session for a case.
            simulator_factory: Create a fresh simulator session for a case. The
                session receives all completed turns and returns the next logical
                user message, or `None` to finish.
            max_turns: Maximum target turns, either fixed or derived from the case
                inputs. Must resolve to at least one.
        """
        self.first_message = first_message
        self.target_factory = target_factory
        self.simulator_factory = simulator_factory
        self.max_turns = max_turns

    async def run(self, inputs: InputsT) -> ConversationResult[UserMessageT, TargetOutputT]:
        """Run a simulated conversation for `inputs`."""
        max_turns = self.max_turns(inputs) if callable(self.max_turns) else self.max_turns
        if max_turns < 1:
            raise ValueError(f'max_turns must be >= 1, got {max_turns}')

        target = self.target_factory(inputs)
        simulator = self.simulator_factory(inputs)
        user_message = await _resolve(self.first_message(inputs))
        turns: list[ConversationTurn[UserMessageT, TargetOutputT]] = []
        target_turn_durations: list[float] = []

        for turn_index in range(1, max_turns + 1):
            with logfire_span(
                'multiturn target',
                conversation_role='target',
                conversation_turn=turn_index,
            ):
                target_started = perf_counter()
                output = await _resolve(target(user_message))
                target_turn_durations.append(perf_counter() - target_started)

            turn = ConversationTurn(user_message=user_message, output=output)
            turns.append(turn)

            with logfire_span(
                'multiturn simulator',
                conversation_role='simulator',
                conversation_turn=turn_index,
            ):
                next_message = await _resolve(simulator(tuple(turns)))

            if next_message is None:
                return _build_result(turns, target_turn_durations, 'simulator_finished')

            if turn_index < max_turns:
                user_message = next_message

        return _build_result(turns, target_turn_durations, 'max_turns')

    @staticmethod
    def from_agents(
        *,
        target_agent: Agent[None, TargetOutputT],
        simulator_agent: Agent[InputsT, SimulatorOutputT],
        first_message: Callable[[InputsT], str | Awaitable[str]],
        next_message: Callable[[SimulatorOutputT], str | None | Awaitable[str | None]],
        max_turns: int | Callable[[InputsT], int] = 6,
        target_prompt: Callable[[InputsT, str, int], str | Awaitable[str]] | None = None,
        simulator_prompt: Callable[[InputsT, ConversationTurn[str, TargetOutputT]], str | Awaitable[str]] | None = None,
    ) -> ConversationTask[InputsT, str, TargetOutputT]:
        """Create a task that manages two Pydantic AI agents' message histories.

        The target agent has no dependencies. The simulator receives the case
        inputs as its dependencies, allowing its instructions and tools to use the
        simulated user's goal or persona.

        Args:
            target_agent: The agent being evaluated.
            simulator_agent: The agent acting as the simulated user.
            first_message: Produce the first logical user message for a case.
            next_message: Convert the simulator output into the next logical user
                message, or `None` to finish.
            max_turns: Maximum target turns, fixed or derived from case inputs.
            target_prompt: Optionally transform each logical message before it is
                sent to the target.
            simulator_prompt: Optionally render the latest completed turn for the
                simulator. The default includes the logical user message and target
                output using their string representations.

        Returns:
            A conversation task whose `run` method can be passed to
            `Dataset.evaluate`.
        """

        def build_target(inputs: InputsT) -> _TargetSession[str, TargetOutputT]:
            message_history: list[ModelMessage] = []
            turn_index = 0

            async def run_target(message: str) -> TargetOutputT:
                nonlocal turn_index
                turn_index += 1
                prompt = (
                    await _resolve(target_prompt(inputs, message, turn_index)) if target_prompt is not None else message
                )
                result = await target_agent.run(prompt, message_history=message_history)
                message_history[:] = result.all_messages()
                return result.output

            return run_target

        def build_simulator(inputs: InputsT) -> _SimulatorSession[str, TargetOutputT]:
            message_history: list[ModelMessage] = []

            async def run_simulator(
                turns: Sequence[ConversationTurn[str, TargetOutputT]],
            ) -> str | None:
                turn = turns[-1]
                prompt = (
                    await _resolve(simulator_prompt(inputs, turn))
                    if simulator_prompt is not None
                    else f'User message: {turn.user_message}\nTarget output: {turn.output}'
                )
                result = await simulator_agent.run(prompt, deps=inputs, message_history=message_history)
                message_history[:] = result.all_messages()
                return await _resolve(next_message(result.output))

            return run_simulator

        return ConversationTask(
            first_message=first_message,
            target_factory=build_target,
            simulator_factory=build_simulator,
            max_turns=max_turns,
        )


async def _resolve(value: _T | Awaitable[_T]) -> _T:
    if inspect.isawaitable(value):
        return await value
    return value


def _build_result(
    turns: list[ConversationTurn[UserMessageT, TargetOutputT]],
    target_turn_durations: list[float],
    stop_reason: Literal['simulator_finished', 'max_turns'],
) -> ConversationResult[UserMessageT, TargetOutputT]:
    result = ConversationResult(turns=tuple(turns), stop_reason=stop_reason)
    increment_eval_metric('conversation_turn_count', result.turn_count)
    increment_eval_metric(
        'conversation_target_turn_duration_seconds_avg',
        sum(target_turn_durations) / len(target_turn_durations),
    )
    increment_eval_metric(
        'conversation_target_turn_duration_seconds_max',
        max(target_turn_durations),
    )
    set_eval_attribute('conversation_stop_reason', result.stop_reason)
    return result
