"""Default guidance for autonomous software work."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import AbstractModel, Model
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import AgentDepsT, RunContext

Autonomy = Literal['low', 'medium', 'high', 'max']
"""How far `Coder` works on its own before it checks in with the user."""

_AUTONOMY: dict[Autonomy, str] = {
    'low': 'Propose a plan and wait for approval before editing.',
    'medium': 'Work until done; ask when a choice really matters.',
    'high': 'Work until done. Ask only for credentials, approvals, or real ambiguity.',
    'max': 'Work fully autonomously; nobody will answer questions. Choose sensible defaults and finish.',
}

_DELEGATED = 'You are running a delegated task. Choose sensible defaults and finish.'
"""Replaces the autonomy sentence in a delegate. The parent already chose the task, so a plan-and-wait level would
leave it idle. It does not say whether anyone can answer: `ask_parent`, when the run has it, says when to ask."""

_INTRO = "You are a software engineering agent. Match the surrounding code's style, structure, and imports."

_READING = (
    'Read files with `read_file`, not batched `cat`. '
    "Save effort on repeated work, not on understanding the task. Don't re-read files that haven't changed."
)

_REPO_FILES = "Instruction files ({names}) for the project, if any, are already in your context; don't search for them."
"""Only when the bundled `RepoContext` autoloads instruction files."""

_NESTED_REPO_FILES = (
    "A subdirectory's instruction files are added to your context "
    'the first time you read, list, write or edit something in it.'
)
"""Only when that `RepoContext` inlines nested files (`nested_traversal` and `nested_inject='contents'`)."""

_DELEGATION = (
    "Don't spawn sub-agents with `{tool_name}` unless the user or AGENTS.md asks, "
    'whatever other instructions say about delegating.'
)
"""Overrides the instruction a `DelegationTasks` owner adds, which tells the model to delegate when it
saves context. A delegate starts from a fresh context and re-explores the project, so unrequested
delegation multiplies tool calls and cost. Sent only to models whose profile sets `delegates_unprompted`: other
models don't delegate unasked, and keep the `DelegationTasks` guidance as it is."""

_TARGETS = 'Measure results against any numeric target the task states.'
"""Wherever the agent decides on its own when the work is good enough: `'high'`, `'max'`, and every delegate."""

_CLEANUP = 'Leave only the requested change; delete your scratch files.'
"""At `'max'` only: nobody reviews the workspace before the result is used."""

BENCHMARK_INSTRUCTIONS = (
    'Deliverables may be used outside this workspace, where only the files you were asked to produce are present. '
    'Make them self-contained, with no reliance on other workspace files, caches, or data you did not include, '
    'and check that they still work when run from a clean directory. '
    'Make sure your solution uses every field the provided data supplies unless the task says to ignore it.'
)
"""Added by `BenchmarkInstructions`: a verifier grades only the requested files, in a clean directory,
and benchmark data is built so that every field matters."""

_CONFIRM = (
    'Before finishing, go through each requirement and edge case the task states and confirm each one explicitly.'
)

_EXACT = "Write checks from the task's stated behavior, not your code."
_MISREAD = (
    'If inputs seem inconsistent or goals impossible, assume you misread. '
    'Re-read, and try other interpretations before settling.'
)
_MISREAD_MAX = (
    'If inputs seem inconsistent or a criterion impossible (no feasible plan, matching item, or reachable target), '
    'assume you misread. Re-read the task and inputs, and try interpretations meeting every stated criterion '
    'before settling for a partial answer. A result that leaves a stated criterion unmet means your '
    'interpretation is likely wrong, not that the answer is "none".'
)
"""Replaces `_MISREAD` at `'max'` only: nobody can confirm a reading, so a partial answer goes unchallenged."""

_MISREAD_CLASSIFY = (
    'If inputs look inconsistent, first check whether the task or its rules classify them '
    '(for example as malformed, conflicting, or rejected) and apply that rule. '
    'Only when no rule covers them, assume you misread: re-read, and try other interpretations before settling.'
)
"""Refines `_MISREAD_MAX` for models whose profile sets `reinterprets_malformed_inputs`, at `'max'` only: they get both."""

_STRUCTURED_HANDOFF = (
    'When asked for notes or a handoff, use headed sections: each decision and why, '
    'the exact verification commands with their observed results, the remaining risk, '
    'and what was out of scope or not done.'
)
"""For models whose profile sets `omits_handoff_evidence`, at `'max'` only."""

_RUNTIME_CONSTRAINTS = (
    'When a task states constraints on where or how the code will run '
    '(a restricted or sandboxed interpreter, no dependencies, a specific Python version, pure functions, no I/O), '
    "find out what's actually allowed before writing code. Then prefer the simplest constructs that fit, "
    'and if possible check the code under those constraints.'
)
"""For models whose profile sets `ignores_stated_runtime_constraints`, at `'max'` only."""
_HEREDOC = 'Check behavior with heredocs for one-offs, not new files.'

_FINISHING = (
    'Finish long-running work before replying; poll until done or blocked. '
    'Servers the task needs may keep running; stop the rest.'
)


def _checks(*, measure: bool, cleanup: bool, misread_max: bool) -> str:
    return ' '.join(
        [
            *([_TARGETS] if measure else []),
            _CONFIRM,
            _MISREAD_MAX if misread_max else _MISREAD,
            _EXACT,
            _HEREDOC,
            *([_CLEANUP] if cleanup else []),
        ]
    )


def _compose(lead: str, *, context: Sequence[str], checks: str, suffix: str) -> str:
    return '\n\n'.join([_INTRO, lead, *context, _READING, checks, _FINISHING]) + '\n' + suffix


def _profile_says(model: AbstractModel, flag: Callable[[ModelProfile], bool]) -> bool:
    try:
        return isinstance(model, Model) and flag(model.profile)
    except NotImplementedError:
        # `FallbackModel` picks its model per request, so no one profile speaks for it.
        return False


def _model_specific(
    name: str, flag: Callable[[ModelProfile], bool], *, flagged: str
) -> Callable[[RunContext[AgentDepsT]], InstructionPart]:
    """A named block whose text depends on the profile of the model handling the request.

    The block, `capability:coder_instructions:<name>`, is contributed from the first request and appends its changes.
    So a conversation that moves to a model with a different profile keeps the instructions it started with and is
    sent the block as a change, or told it was withdrawn.
    """

    def instruction(ctx: RunContext[AgentDepsT]) -> InstructionPart:
        return InstructionPart(flagged if _profile_says(ctx.model, flag) else '', name=name, on_change='append')

    return instruction


def model_specific_instructions(
    *, autonomy: Autonomy, delegate_tool: str | None
) -> list[Callable[[RunContext[AgentDepsT]], InstructionPart]]:
    """The blocks of `Coder`'s guidance that only some models get, each empty for the others.

    - `delegation`, with `SubAgents` only: the rule against unrequested delegation, for `delegates_unprompted`.
    - `misread`: at `'max'`, `_MISREAD_CLASSIFY`, for `reinterprets_malformed_inputs`. It refines the `_MISREAD_MAX`
      check the shared text carries for every model.
    - `handoff`: at `'max'`, `_STRUCTURED_HANDOFF`, for `omits_handoff_evidence`.
    - `runtime_constraints`: at `'max'`, `_RUNTIME_CONSTRAINTS`, for `ignores_stated_runtime_constraints`.
    """
    is_max = autonomy == 'max'
    return [
        *(
            [
                _model_specific(
                    'delegation',
                    lambda p: p.get('delegates_unprompted', False),
                    flagged=_DELEGATION.format(tool_name=delegate_tool),
                )
            ]
            if delegate_tool is not None
            else []
        ),
        _model_specific(
            'misread',
            lambda p: p.get('reinterprets_malformed_inputs', False),
            flagged=_MISREAD_CLASSIFY if is_max else '',
        ),
        _model_specific(
            'handoff',
            lambda p: p.get('omits_handoff_evidence', False),
            flagged=_STRUCTURED_HANDOFF if is_max else '',
        ),
        _model_specific(
            'runtime_constraints',
            lambda p: p.get('ignores_stated_runtime_constraints', False),
            flagged=_RUNTIME_CONSTRAINTS if is_max else '',
        ),
    ]


def coder_instructions(
    *,
    autonomy: Autonomy,
    extra: str | None = None,
    repo_files: Sequence[str] | None = None,
    nested_instructions: bool = False,
) -> tuple[str, str]:
    """The top-level guidance and the delegate guidance.

    They differ in the lead sentence. A delegate also measures numeric targets, because it decides on its
    own when the work is good enough. Non-empty `extra` is appended after a blank line.
    `repo_files` names instruction files already loaded (`None` when none are); `nested_instructions`
    says a subdirectory's files are inlined the first time the model reads, lists, writes or edits there.
    """
    is_max = autonomy == 'max'
    parent_checks = _checks(measure=autonomy in ('high', 'max'), cleanup=is_max, misread_max=is_max)
    delegate_checks = _checks(measure=True, cleanup=is_max, misread_max=is_max)
    context = [
        *([_REPO_FILES.format(names=', '.join(repo_files))] if repo_files is not None else []),
        *([_NESTED_REPO_FILES] if nested_instructions else []),
    ]
    suffix = '\n' + extra if extra else ''
    return (
        _compose(_AUTONOMY[autonomy], context=context, checks=parent_checks, suffix=suffix),
        _compose(_DELEGATED, context=context, checks=delegate_checks, suffix=suffix),
    )


def project_instructions(*, unrestricted: bool) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]:
    """Name the working directory, so the model neither hunts for the project nor works outside it.

    The workspace keeps its working directory for its lifetime, so the text is the same on every
    request and does not break prompt caching. `Coder` fails a run without a workspace at its
    start, so a workspace is always there to ask.
    """
    files = 'relative file paths resolve from it' if unrestricted else 'the file tools only accept paths inside it'

    async def instructions(ctx: RunContext[AgentDepsT]) -> str:
        return f'Your project is `{await ctx.workspace.working_dir()}`. Shell commands start there, and {files}.'

    return instructions
