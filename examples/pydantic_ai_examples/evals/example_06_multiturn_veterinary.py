"""Build a domain-specific evaluation on top of the generic multi-turn primitives.

This example is intentionally more involved than `example_05_multiturn.py`. The
veterinary domain is only a vehicle for demonstrating that applications can add
their own prompt enrichment, structured outputs, tool assertions, and target
implementations without requiring those concepts in `ConversationTask` itself.

In particular, the example:

* adds the contents of a temporary `.txt` record to the first target prompt;
* evaluates a typed "patch" independently of how the target produced it;
* evaluates tool calls when the target is a Pydantic AI Agent; and
* compares that Agent with a deterministic regex implementation on the same cases.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, models
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, HasMatchingSpan
from pydantic_evals.multiturn import (
    ConversationResult,
    ConversationTask,
    ConversationTurn,
)

Species = Literal['dog', 'cat', 'exotic']
Urgency = Literal['yellow', 'red']

_DOG_PATTERN = re.compile(r'\b(dog|puppy|canine|retriever)\b', re.IGNORECASE)
_CAT_PATTERN = re.compile(r'\b(cat|kitten|feline)\b', re.IGNORECASE)
_EXOTIC_PATTERN = re.compile(
    r'\b(bird|parrot|rabbit|snake|lizard|iguana|tortoise)\b', re.IGNORECASE
)
_RED_PATTERN = re.compile(
    r'\b(difficulty breathing|cannot breathe|seizure|unconscious|heavy bleeding|collapsed)\b',
    re.IGNORECASE,
)
_YELLOW_PATTERN = re.compile(
    r'\b(limping|vomiting|diarrhea|not eating|lethargic|rash)\b', re.IGNORECASE
)

_RECORD_CONTENTS = {
    'miso': 'Miso is a cat who is having difficulty breathing and has collapsed.',
    'rex': 'Rex is a golden retriever who started limping this morning.',
}


@contextmanager
def temporary_record_files() -> Generator[dict[str, Path]]:
    """Create the external records used by the example, then delete them.

    The files are deliberately runtime fixtures rather than repository assets. Their
    purpose is to show that an application can adapt a logical simulator message to
    its own external context before invoking the target. `ConversationTask` does not
    need to know anything about files.
    """
    with TemporaryDirectory(prefix='pydantic-ai-veterinary-eval-') as directory:
        root = Path(directory)
        records: dict[str, Path] = {}
        for name, content in _RECORD_CONTENTS.items():
            path = root / f'{name}.txt'
            path.write_text(content, encoding='utf-8')
            records[name] = path
        yield records
    # TemporaryDirectory removes both `.txt` files here, including on failure.


class TriagePatch(BaseModel):
    """The domain output evaluated for every target implementation."""

    species: Species | None = None
    urgency: Urgency | None = None


class TriageReply(BaseModel):
    """A user-facing response plus the machine-readable state accumulated so far."""

    message: str
    patch: TriagePatch


class TriageScenario(BaseModel):
    """Case-specific simulator goal and external context."""

    goal: str
    first_message: str
    record_path: Path
    max_turns: int = Field(default=4, ge=1)


class TriageExpected(BaseModel):
    """Expected final patch, stored as the evaluation case metadata."""

    species: Species
    urgency: Urgency


class SimulatorDecision(BaseModel):
    """Domain-specific Agent output adapted to the generic `str | None` protocol."""

    message: str = Field(
        description='The next pet-owner message, or an empty string when done is true.'
    )
    done: bool


def classify_species(description: str) -> Species:
    """Classify the patient as a dog, cat, or exotic animal."""
    species = _classify_species(description)
    if species is None:
        raise ValueError('The animal species is not present in the description.')
    return species


def classify_urgency(description: str) -> Urgency:
    """Classify the clinical problem as yellow or red urgency."""
    urgency = _classify_urgency(description)
    if urgency is None:
        raise ValueError('The clinical problem is not present in the description.')
    return urgency


def build_target_agent(
    model: models.Model | models.KnownModelName | str = 'openai:gpt-4.1-mini',
) -> Agent[None, TriageReply]:
    """Build the implementation for which tool usage is meaningful."""
    agent = Agent(
        model,
        output_type=TriageReply,
        instructions=(
            'You route patients for a veterinary clinic. Collect enough information to determine species and '
            'urgency. Use both classification tools when their inputs are available. Return the current patch and '
            'ask one concise follow-up question when a value is still missing.'
        ),
    )
    agent.tool_plain(classify_species)
    agent.tool_plain(classify_urgency)
    return agent


def build_simulator_agent(
    model: models.Model | models.KnownModelName | str = 'openai:gpt-4.1-mini',
) -> Agent[TriageScenario, SimulatorDecision]:
    def simulator_instructions(ctx: RunContext[TriageScenario]) -> str:
        return (
            'Act as the pet owner. Answer follow-up questions using the scenario and stop when both species and '
            f'urgency are present in the patch.\n\nGoal: {ctx.deps.goal}'
        )

    return Agent(
        model,
        deps_type=TriageScenario,
        output_type=SimulatorDecision,
        instructions=simulator_instructions,
    )


def target_prompt(scenario: TriageScenario, message: str, turn_index: int) -> str:
    """Adapt the logical message to the target application's input needs.

    This hook is the extension point being demonstrated: the stored conversation
    still contains only the owner's message, while the actual target prompt can use
    external context. Nothing file-specific leaks into the multi-turn primitive.
    """
    if turn_index == 1:
        record = scenario.record_path.read_text(encoding='utf-8')
        return f'{message}\n\nPatient record:\n{record}'
    return message


def simulator_prompt(
    scenario: TriageScenario, turn: ConversationTurn[str, TriageReply]
) -> str:
    """Render a domain output into the information the simulator should see.

    The generic turn remains typed, so applications decide how much of their output
    to expose to the simulator. Here it needs both the reply and current patch.
    """
    del scenario
    return (
        f'Clinic response: {turn.output.message}\n'
        f'Current triage patch: {turn.output.patch.model_dump_json()}'
    )


def build_agent_conversation_task(
    *,
    target_model: models.Model | models.KnownModelName | str = 'openai:gpt-4.1-mini',
    simulator_model: models.Model | models.KnownModelName | str = 'openai:gpt-4.1-mini',
) -> ConversationTask[TriageScenario, str, TriageReply]:
    """Use the convenience adapter when both participants are Pydantic AI Agents."""
    return ConversationTask.from_agents(
        target_agent=build_target_agent(target_model),
        simulator_agent=build_simulator_agent(simulator_model),
        first_message=lambda scenario: scenario.first_message,
        next_message=lambda decision: None if decision.done else decision.message,
        max_turns=lambda scenario: scenario.max_turns,
        target_prompt=target_prompt,
        simulator_prompt=simulator_prompt,
    )


def build_regex_conversation_task(
    *,
    simulator_model: models.Model | models.KnownModelName | str = 'openai:gpt-4.1-mini',
) -> ConversationTask[TriageScenario, str, TriageReply]:
    """Compose the generic primitive with a non-Agent target.

    `target_factory` gives every case private state, while the callback still has the
    same simple message-in/output-out shape used by the conversation loop. This is
    the main extensibility claim of the example: users can replace one participant
    without rebuilding turn limits, trajectories, isolation, metrics, or reporting.
    """
    simulator_agent = build_simulator_agent(simulator_model)

    def target_factory(scenario: TriageScenario) -> Callable[[str], TriageReply]:
        # This state belongs to one case because the factory is called for every run.
        evidence = [scenario.record_path.read_text(encoding='utf-8')]

        def run_target(message: str) -> TriageReply:
            evidence.append(message)
            text = '\n'.join(evidence)
            patch = TriagePatch(
                species=_classify_species(text), urgency=_classify_urgency(text)
            )
            if patch.species is None:
                reply = 'What kind of animal is the patient?'
            elif patch.urgency is None:
                reply = 'What symptoms is the patient experiencing?'
            else:
                reply = (
                    'The patient has been routed to the appropriate veterinary team.'
                )
            return TriageReply(message=reply, patch=patch)

        return run_target

    def simulator_factory(
        scenario: TriageScenario,
    ) -> Callable[
        [Sequence[ConversationTurn[str, TriageReply]]], Awaitable[str | None]
    ]:
        async def run_simulator(
            turns: Sequence[ConversationTurn[str, TriageReply]],
        ) -> str | None:
            # A custom adapter may render the whole typed trajectory rather than
            # relying on Pydantic AI message history.
            result = await simulator_agent.run(_render_transcript(turns), deps=scenario)
            return None if result.output.done else result.output.message

        return run_simulator

    return ConversationTask(
        first_message=lambda scenario: scenario.first_message,
        target_factory=target_factory,
        simulator_factory=simulator_factory,
        max_turns=lambda scenario: scenario.max_turns,
    )


@dataclass(repr=False)
class TriagePatchEvaluator(
    Evaluator[TriageScenario, ConversationResult[str, TriageReply], TriageExpected]
):
    """Evaluate domain state without depending on the target implementation."""

    def evaluate(
        self,
        ctx: EvaluatorContext[
            TriageScenario, ConversationResult[str, TriageReply], TriageExpected
        ],
    ) -> dict[str, bool]:
        assert ctx.metadata is not None
        patch = ctx.output.final_output.patch
        return {
            'species_patch': patch.species == ctx.metadata.species,
            'urgency_patch': patch.urgency == ctx.metadata.urgency,
        }


def build_dataset(
    *, records: Mapping[str, Path], include_tool_evaluators: bool
) -> Dataset[TriageScenario, ConversationResult[str, TriageReply], TriageExpected]:
    """Build shared cases, adding only evaluators supported by the implementation."""
    # Patch evaluation works for both the Agent and regex target because it only
    # depends on their shared typed output.
    evaluators: list[
        Evaluator[
            TriageScenario,
            ConversationResult[str, TriageReply],
            TriageExpected,
        ]
    ] = [TriagePatchEvaluator()]
    if include_tool_evaluators:
        # Tool calls are an implementation-specific capability. Standard span
        # evaluators compose with the target spans emitted by ConversationTask, so
        # the conversation primitive needs no tool-evaluation API of its own.
        evaluators.extend(
            [
                _target_used_tool('classify_species'),
                _target_used_tool('classify_urgency'),
            ]
        )

    return Dataset(
        name='multiturn_veterinary_triage',
        cases=[
            Case(
                name='cat respiratory emergency',
                inputs=TriageScenario(
                    goal='Route Miso using the attached record.',
                    first_message='Can you help me understand where Miso should be seen?',
                    record_path=records['miso'],
                ),
                metadata=TriageExpected(species='cat', urgency='red'),
            ),
            Case(
                name='dog mobility problem',
                inputs=TriageScenario(
                    goal='Route Rex using the attached record.',
                    first_message='Which team should see Rex?',
                    record_path=records['rex'],
                ),
                metadata=TriageExpected(species='dog', urgency='yellow'),
            ),
        ],
        evaluators=evaluators,
    )


def _target_used_tool(tool_name: str) -> HasMatchingSpan:
    # Restrict the search to a target-turn span. Simulator internals can therefore
    # use their own tools without accidentally satisfying this assertion.
    return HasMatchingSpan(
        query={
            'name_equals': 'multiturn target',
            'some_descendant_has': {
                'has_attributes': {
                    'gen_ai.operation.name': 'execute_tool',
                    'gen_ai.tool.name': tool_name,
                },
            },
        },
        evaluation_name=f'used_{tool_name}',
    )


def _classify_species(text: str) -> Species | None:
    if _DOG_PATTERN.search(text):
        return 'dog'
    if _CAT_PATTERN.search(text):
        return 'cat'
    if _EXOTIC_PATTERN.search(text):
        return 'exotic'
    return None


def _classify_urgency(text: str) -> Urgency | None:
    if _RED_PATTERN.search(text):
        return 'red'
    if _YELLOW_PATTERN.search(text):
        return 'yellow'
    return None


def _render_transcript(turns: Sequence[ConversationTurn[str, TriageReply]]) -> str:
    return '\n\n'.join(
        f'Owner: {turn.user_message}\nClinic: {turn.output.message}\nPatch: {turn.output.patch.model_dump_json()}'
        for turn in turns
    )


async def main() -> None:
    logfire.configure(send_to_logfire='if-token-present', environment='development')
    logfire.instrument_pydantic_ai()

    # The temporary records model application-owned external data. They exist for
    # both experiments and are removed automatically when this block exits.
    with temporary_record_files() as records:
        # The first experiment takes the short, convenient path: two Agents and
        # from_agents() for automatic message-history management.
        agent_conversation = build_agent_conversation_task()
        agent_report = await build_dataset(
            records=records, include_tool_evaluators=True
        ).evaluate(
            agent_conversation.run,
            name='veterinary triage - pydantic agent',
            metadata={'implementation': 'pydantic_agent'},
            max_concurrency=1,
        )
        agent_report.print(
            include_input=True, include_output=True, include_reasons=True
        )

        # The second experiment swaps in a deterministic target but keeps the same
        # dataset definition, simulator behavior, patch evaluator, turn loop,
        # metrics, and reporting. Only the tool assertions are omitted because
        # regex has no tools.
        regex_conversation = build_regex_conversation_task()
        regex_report = await build_dataset(
            records=records, include_tool_evaluators=False
        ).evaluate(
            regex_conversation.run,
            name='veterinary triage - regex',
            metadata={'implementation': 'regex'},
            max_concurrency=1,
        )
        regex_report.print(
            include_input=True, include_output=True, include_reasons=True
        )

    # At this point the `.txt` files no longer exist.


if __name__ == '__main__':
    asyncio.run(main())
