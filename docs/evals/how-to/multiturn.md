# Multi-Turn Evaluation

Use [`ConversationTask`][pydantic_evals.multiturn.ConversationTask] to evaluate a target over a conversation driven by a simulated user. It provides the conversation loop, isolated sessions, a hard turn limit, a typed trajectory, and role-specific spans while continuing to use the normal [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate] workflow.

This is different from evaluating an existing message history: a multi-turn task actively alternates between a target and a simulator until the simulator finishes or the turn budget is exhausted.

## Evaluate two Pydantic AI agents

For the common case, [`ConversationTask.from_agents()`][pydantic_evals.multiturn.ConversationTask.from_agents] manages both agents' message histories. Define only the case-specific scenario and the simulator's structured decision:

```python {test="skip" lint="skip"}
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_evals import Case, Dataset
from pydantic_evals.multiturn import ConversationTask


class Scenario(BaseModel):
    goal: str
    first_message: str
    max_turns: int = 4


class SimulatorDecision(BaseModel):
    message: str
    done: bool


target_agent = Agent('openai:gpt-5.2')
simulator_agent = Agent(
    'openai:gpt-5.2',
    deps_type=Scenario,
    output_type=SimulatorDecision,
    instructions=lambda ctx: f'Act as the user. Your goal is: {ctx.deps.goal}',
)

conversation = ConversationTask.from_agents(
    target_agent=target_agent,
    simulator_agent=simulator_agent,
    first_message=lambda scenario: scenario.first_message,
    next_message=lambda decision: None if decision.done else decision.message,
    max_turns=lambda scenario: scenario.max_turns,
)

dataset = Dataset(
    name='support_conversations',
    cases=[
        Case(
            name='reset password',
            inputs=Scenario(
                goal='Successfully learn how to reset a forgotten password.',
                first_message='I forgot my password.',
            ),
        )
    ],
)

report = await dataset.evaluate(conversation.run)
```

Each invocation of [`ConversationTask.run()`][pydantic_evals.multiturn.ConversationTask.run] creates fresh target and simulator sessions. Message histories therefore cannot leak between concurrent cases, repeated runs, or task retries.

## Keep the primitive small and extend it locally

`ConversationTask` deliberately owns only the mechanics shared by multi-turn evaluations: alternating participants, isolated sessions, a turn limit, a typed trajectory, metrics, and role spans. It does not need concepts for files, patches, tools, or any particular target implementation.

Applications can build those concerns on top of the primitive using existing extension points:

| Application requirement | Extension point | Core behavior reused |
| --- | --- | --- |
| Add an external `.txt` record to a target request | `target_prompt` | logical messages and Agent history |
| Return and evaluate a domain-specific patch | generic target output and a regular `Evaluator` | typed trajectory and reporting |
| Assert which Agent tools were used | target role spans and `HasMatchingSpan` | tracing and span-based evaluation |
| Compare an Agent with a regex implementation | `target_factory` | cases, isolation, turn loop, metrics, and reports |

The advanced veterinary example combines all four to demonstrate composability, not to prescribe a veterinary-specific abstraction. Its `.txt` records are created as temporary runtime fixtures and deleted after both experiments. The same dataset definition and patch evaluator are then used for an Agent target and a deterministic target; only the Agent run adds tool-use evaluators because only that implementation has tools.

## Inspect the trajectory

The task returns a [`ConversationResult`][pydantic_evals.multiturn.ConversationResult]. Its `turns` contain the logical simulated-user message and the target output for every completed exchange:

```python {test="skip" lint="skip"}
final_output = report.cases[0].output.final_output
turn_count = report.cases[0].output.turn_count
stop_reason = report.cases[0].output.stop_reason
```

`stop_reason` is either:

- `simulator_finished`, when the simulator returned `None`;
- `max_turns`, when the simulator wanted to continue after the final allowed target turn.

Simulator completion is a control signal, not a correctness judgment. Use normal Pydantic Evals evaluators to decide whether the resulting conversation succeeded.

Every completed conversation records these metrics automatically:

- `conversation_turn_count`;
- `conversation_target_turn_duration_seconds_avg`;
- `conversation_target_turn_duration_seconds_max`.

The standard `cost` metric still represents all instrumented model calls in the task. Target and simulator calls have separate role spans, but costs are not split into report metrics: generic callbacks do not have a common cost contract. Role-specific cost aggregation can be added later without changing the conversation protocol.

## Transform prompts without changing the trajectory

The logical message stored in [`ConversationTurn.user_message`][pydantic_evals.multiturn.ConversationTurn.user_message] can differ from the prompt sent to an Agent. For example, an application can append the contents of a temporary text record to the first target prompt while keeping the trajectory concise:

```python {test="skip" lint="skip"}
from pathlib import Path


class DocumentScenario(Scenario):
    record_path: Path


def target_prompt(scenario: DocumentScenario, message: str, turn_index: int) -> str:
    if turn_index == 1:
        record = scenario.record_path.read_text(encoding='utf-8')
        return f'{message}\n\nAttached record:\n{record}'
    return message
```

Pass this callback as `target_prompt=` to `from_agents()`. Both `target_prompt` and `simulator_prompt` may be synchronous or asynchronous. `simulator_prompt` receives the latest typed [`ConversationTurn`][pydantic_evals.multiturn.ConversationTurn], so it can render structured target outputs such as patches or actions for the simulated user.

## Evaluate structured outputs and tool use

Target outputs remain typed in every turn. A regular evaluator can therefore inspect a final patch without knowing whether the target was an Agent, a deterministic function, or a remote service:

```python
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.multiturn import ConversationResult


class TriagePatch(BaseModel):
    species: Literal['dog', 'cat', 'exotic'] | None = None
    urgency: Literal['yellow', 'red'] | None = None


@dataclass(repr=False)
class PatchIsComplete(Evaluator[object, ConversationResult[str, TriagePatch], object]):
    def evaluate(self, ctx: EvaluatorContext[object, ConversationResult[str, TriagePatch], object]) -> bool:
        patch = ctx.output.final_output
        return patch.species is not None and patch.urgency is not None
```

Every target and simulator call is wrapped in a distinct span. Combine this with [`HasMatchingSpan`][pydantic_evals.evaluators.HasMatchingSpan] to evaluate tool usage only inside target turns:

```python
from pydantic_evals.evaluators import HasMatchingSpan

used_species_tool = HasMatchingSpan(
    query={
        'name_equals': 'multiturn target',
        'some_descendant_has': {
            'has_attributes': {
                'gen_ai.operation.name': 'execute_tool',
                'gen_ai.tool.name': 'classify_species',
            },
        },
    },
    evaluation_name='used_species_classifier',
)
```

See [span-based evaluation](../evaluators/span-based.md) for the complete query syntax.

## Use a custom target or simulator

The generic constructor accepts factories that create one target session and one simulator session for each case. The returned callbacks can capture private state without requiring a public state model:

```python {test="skip" lint="skip"}
def regex_target_factory(scenario: Scenario):
    messages: list[str] = []

    def run_target(message: str) -> str:
        messages.append(message)
        return classify_with_regex('\n'.join(messages))

    return run_target


conversation = ConversationTask(
    first_message=lambda scenario: scenario.first_message,
    target_factory=regex_target_factory,
    simulator_factory=simulator_factory,
    max_turns=lambda scenario: scenario.max_turns,
)
```

This lets the same [`Dataset`][pydantic_evals.dataset.Dataset] and output evaluators compare different implementations. Evaluators that assert implementation-specific behavior, such as Agent tool calls, should only be attached to implementations that provide that capability.

Complete runnable examples are available in `pydantic_ai_examples.evals.example_05_multiturn` and `pydantic_ai_examples.evals.example_06_multiturn_veterinary`.
