"""Default guidance for autonomous software work."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic_ai.tools import AgentDepsT, RunContext

Agency = Literal['low', 'medium', 'high', 'max']
"""How far `Coder` works on its own before it checks in with the user."""

_AGENCY: dict[Agency, str] = {
    'low': 'Propose a plan and wait for approval before editing.',
    'medium': 'Work until done; ask when a choice really matters.',
    'high': 'Work until done. Ask only for credentials, approvals, or real ambiguity.',
    'max': 'Work fully autonomously; nobody will answer questions. Choose sensible defaults and finish.',
}

_INTRO = "You are a software engineering agent. Match the surrounding code's style, structure, and imports."

_READING = (
    'Read files with `read_file`, not batched `cat`. '
    "Save effort on repeated work, not on understanding the task. Don't re-read files that haven't changed."
)

_DELEGATION = "Don't spawn sub-agents with `{tool_name}` unless the user or AGENTS.md asks."
"""The rule Codex gives its models for `spawn_agent`. A delegate starts from a fresh context and
re-explores the project, so unrequested delegation multiplies tool calls and cost."""

_TARGETS = 'Measure results against any numeric target the task states.'
"""From `'high'` up, where the agent decides on its own when the work is good enough."""

_CLEANUP = 'Leave only the requested change; delete your scratch files.'
"""At `'max'` only: nobody reviews the workspace before the result is used."""

_FINISHING = 'Finish long-running work before replying; poll until done or blocked. Stop servers you no longer need.'


def coder_instructions(*, agency: Agency, delegate_tool: str | None) -> str:
    """`Coder`'s default guidance for an agency level, with the delegation rule only when it can delegate."""
    autonomous = agency in ('high', 'max')
    checks = [
        'If deliverables will run elsewhere, make them self-contained.',
        *([_TARGETS] if autonomous else []),
        'Before finishing, confirm each stated requirement and edge case.',
        'If inputs seem inconsistent or goals impossible, assume you misread. '
        'Re-read, and try other interpretations before settling.',
        "Write checks from the task's stated behavior, not your code.",
        'Check behavior with heredocs for one-offs, not new files.',
        *([_CLEANUP] if agency == 'max' else []),
    ]
    sections = [
        _INTRO,
        _AGENCY[agency],
        _READING,
        *([_DELEGATION.format(tool_name=delegate_tool)] if delegate_tool is not None else []),
        ' '.join(checks),
        _FINISHING,
    ]
    return '\n\n'.join(sections) + '\n'


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
