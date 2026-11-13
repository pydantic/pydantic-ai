"""Default guidance for autonomous software work."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

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
delegation multiplies tool calls and cost."""

_TARGETS = 'Measure results against any numeric target the task states.'
"""Wherever the agent decides on its own when the work is good enough: `'high'`, `'max'`, and every delegate."""

_CLEANUP = 'Leave only the requested change; delete your scratch files.'
"""At `'max'` only: nobody reviews the workspace before the result is used."""

_DELIVER = (
    'Deliverables may be used outside this workspace, where only the files you were asked to produce are present. '
    'Make them self-contained, with no reliance on other workspace files, caches, or data you did not include, '
    'and check that they still work when run from a clean directory.'
)

_CONFIRM = (
    'Before finishing, go through each requirement and edge case the task states and confirm each one explicitly.'
)

_EXACT = "Write checks from the task's stated behavior, not your code."
_EVERY_FIELD = 'Make sure your solution uses every field the provided data supplies unless the task says to ignore it.'
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
_HEREDOC = 'Check behavior with heredocs for one-offs, not new files.'

_FINISHING = (
    'Finish long-running work before replying; poll until done or blocked. '
    'Servers the task needs may keep running; stop the rest.'
)


def _checks(*, measure: bool, cleanup: bool, misread_max: bool) -> str:
    return ' '.join(
        [
            _DELIVER,
            *([_TARGETS] if measure else []),
            _CONFIRM,
            _MISREAD_MAX if misread_max else _MISREAD,
            _EXACT,
            _EVERY_FIELD,
            _HEREDOC,
            *([_CLEANUP] if cleanup else []),
        ]
    )


def _compose(lead: str, *, context: Sequence[str], checks: str, delegation: Sequence[str], suffix: str) -> str:
    return '\n\n'.join([_INTRO, lead, *context, _READING, *delegation, checks, _FINISHING]) + '\n' + suffix


def coder_instructions(
    *,
    autonomy: Autonomy,
    delegate_tool: str | None,
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
    delegation = [_DELEGATION.format(tool_name=delegate_tool)] if delegate_tool is not None else []
    context = [
        *([_REPO_FILES.format(names=', '.join(repo_files))] if repo_files is not None else []),
        *([_NESTED_REPO_FILES] if nested_instructions else []),
    ]
    suffix = '\n' + extra if extra else ''
    return (
        _compose(_AUTONOMY[autonomy], context=context, checks=parent_checks, delegation=delegation, suffix=suffix),
        _compose(_DELEGATED, context=context, checks=delegate_checks, delegation=delegation, suffix=suffix),
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
