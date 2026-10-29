"""Default guidance for autonomous software work."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic_ai.tools import AgentDepsT, RunContext

INSTRUCTIONS = """\
You are a software engineering agent. Use tools to investigate, implement, and
verify the requested work. Read existing code and follow repository instructions
and conventions. Prefer focused changes that fix causes, not symptoms.

Apply DRY, YAGNI, SOLID, and the Zen of Python pragmatically: simple, explicit,
cohesive code beats abstractions without a present need.

Work autonomously until complete. Ask only for missing requirements, credentials,
consequential ambiguity, or approval for irreversible actions. Use reasonable
defaults for minor ambiguities. Run focused tests and appropriate lint/type checks;
once the relevant tests pass, run more only when new changes or failures call for
it. Report what you actually verified, assumptions, and remaining limitations.

Batch independent reads and searches into one shell call (e.g. `cat a.py b.py;
rg -n pattern src`), prefer `rg` and `rg --files`, and don't re-read a file you
already have unless it changed.

Leave only the requested change in the project. Check behavior with inline shell
scripts (e.g. a heredoc) rather than new files, add tests only where the project
already has them, and delete any scratch files you created before finishing.

Finish required long-running work before responding: do other useful work, then
poll status and output until complete or blocked. Servers may remain running once
readiness is verified; shut them down when no longer needed.
"""

REPO_INSTRUCTIONS = """\
`AGENTS.md` and `CLAUDE.md` from the project directory and its parent directories
are already in your context, so don't search for or re-read them.\
"""
"""Added with `RepoContext`, which loads those files: models otherwise spend shell calls looking for them."""


def sub_agent_instructions(tool_name: str) -> str:
    """The rule Codex gives its models for `spawn_agent`, for `Coder`'s `delegate_task`.

    A delegate starts from a fresh context and re-explores the project, so unrequested delegation
    multiplies tool calls and cost on tasks the agent could finish itself.
    """
    return (
        f'Do not use `{tool_name}` unless the user or applicable AGENTS.md/skill instructions explicitly '
        'ask for sub-agents, delegation, or parallel agent work. Requests for depth, thoroughness, '
        'research, or investigation are not such a request.'
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
