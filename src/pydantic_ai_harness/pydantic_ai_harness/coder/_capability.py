"""Complete coding-agent harness assembled from regular capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, ValidatedToolArgs
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UnexpectedModelBehavior
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai_harness._warn import warn_argument_ignored
from pydantic_ai_harness._workspace import RequireWorkspace
from pydantic_ai_harness.coder._instructions import Autonomy, coder_instructions, project_instructions
from pydantic_ai_harness.compaction import ClearToolResults
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import MAX_FOREGROUND_WAIT, Shell
from pydantic_ai_harness.subagents import SubAgents, TaskControl
from pydantic_ai_harness.subagents._control import ASK_PARENT_TOOL_NAME
from pydantic_ai_harness.subagents._toolset import CONTROL_TOOL_NAMES, delegation_depth
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

FILE_TOOL_NAMES: tuple[str, ...] = ('read_file', 'write_file', 'edit_file', 'list_files', 'grep')
"""The `FileSystem` tools `Coder` registers; `shell` covers directory creation, file metadata, and the rest."""


class _BoundToolOutputs(ToolOutputLimits[AgentDepsT]):
    # Unsets only the class-level default id, which is what the combine policy reads: two instances
    # sharing an id raise instead of merging as one configuration stated twice, so the `Anonymous`
    # `COMBINE_POLICY` entry holds. Instances still default to `id='tool_output_limits'` through the
    # inherited `__init__`; `Coder` passes `id='coder_tool_output_limits'`, which keeps its bound
    # limits separate from a standalone `ToolOutputLimits`.
    id: str | None = None

    def get_toolset(self) -> None:
        """Coder uses bounded truncation, so no spill-retrieval tool is needed."""
        return None


@dataclass
class _DefaultCaching(AbstractCapability[AgentDepsT]):
    """`Caching()` that yields to a `cache` value set before it, by the model, the agent, or a capability.

    Without `Caching`'s fixed id, so a `Caching` bound next to `Coder` is not combined away.
    """

    def get_model_settings(self) -> Callable[[RunContext[AgentDepsT]], ModelSettings]:
        def unless_configured(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            return ModelSettings() if 'cache' in (ctx.model_settings or {}) else ModelSettings(cache=True)

        return unless_configured


@dataclass
class _SetToolStrict(AbstractCapability[AgentDepsT]):
    """Sets `strict` on the named tools that leave it unset.

    Tool names are unique within an agent, so the names select `Coder`'s own tools and no tool bound next to it.
    """

    strict: bool
    tool_names: frozenset[str]

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return [
            replace(tool_def, strict=self.strict)
            if tool_def.strict is None and tool_def.name in self.tool_names
            else tool_def
            for tool_def in tool_defs
        ]


@dataclass
class _ReportExhaustedRetries(AbstractCapability[AgentDepsT]):
    """Returns a tool's last error to the model once its retry budget is spent, instead of ending the run.

    A tool that keeps asking for a retry (a stale `edit_file` match, a denied path) would otherwise
    raise `UnexpectedModelBehavior` and discard the whole run. The model gets the error as a failed
    tool result and can change approach; run-level usage limits still bound a model that never does.
    """

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        cause = error.__cause__
        if not isinstance(error, UnexpectedModelBehavior) or not isinstance(cause, ModelRetry):
            raise error
        raise ToolFailed(
            f'{cause.message}\n\n`{call.tool_name}` has failed several times in a row. Do not repeat the same '
            'call: check the current state and try a different approach.'
        ) from error


MAX_READ_CHARS = 50_000
"""Characters of complete lines per `read_file`, below the output cap."""

MAX_OUTPUT_CHARS = 64000
"""Characters kept from any tool result."""

MAX_FILE_TOOL_RETRIES = 5
"""Consecutive retries each file tool allows before its execution errors come back as plain failures.

A denied path or a stale edit is corrected from the error message. With the agent's default
budget of one, a second consecutive mistake with the same tool ends a long autonomous run
and discards its work.
"""


@dataclass
class _CoderGuidance(AbstractCapability[AgentDepsT]):
    """Literal coding guidance, with the delegate text chosen when this run is a delegate.

    `Coder` is flattened into the agent's capability list, so this leaf is what `for_run` reaches.
    The texts are strings: a function would be dynamic, and the static prompt-cache prefix ends at
    the first dynamic part. It contributes no toolset. `id` is set on the instance, not as a class
    default, so two copies are rejected rather than merged. A delegate's text comes from
    `delegation_depth()`, which the delegating run sets before the child run is awaited, and which
    a recorded delegation sets again on the worker.
    """

    guidance: str
    delegated_guidance: str
    project: tuple[Callable[[RunContext[AgentDepsT]], Awaitable[str]], ...]
    delegate: bool = False

    def __post_init__(self) -> None:
        self.id = 'coder_instructions'

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return [self.delegated_guidance if self.delegate else self.guidance, *self.project]

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        if delegation_depth() <= 1:
            return self
        return replace(self, delegate=True)


def _file_system(*, unrestricted: bool) -> FileSystem[AgentDepsT]:
    file_system = FileSystem[AgentDepsT](
        content_hashes=False,
        max_read_chars=MAX_READ_CHARS,
        tools=FILE_TOOL_NAMES,
        max_retries=MAX_FILE_TOOL_RETRIES,
    )
    if unrestricted:
        # Workspace paths are POSIX, so the filesystem root is `/` whatever the host platform.
        return replace(file_system, root_dir='/', read_only_patterns=[])
    return file_system


@dataclass(init=False)
class Coder(CombinedCapability[AgentDepsT]):
    """Autonomous coding with six tools, delegation, and context management, in the run's workspace.

    Files and commands go through `ctx.workspace`, and the project directory is
    its working directory: attach `LocalWorkspace('.')` for a local checkout, or
    a sandbox provider's capability for untrusted work, alongside `Coder()`. A
    run without a workspace fails at its start. Commands are unrestricted and
    can outlive runs.

    Args:
        workspace: Deprecated and ignored; attach a workspace capability instead.
        instructions: Appended to the default instructions.
        autonomy: How far the agent works on its own. This is guidance to the model, not a
            limit on its tools: nothing stops it from editing if it ignores the sentence.
            To require approval before writes or commands, add `Permissions`. `'low'` proposes
            a plan and waits for approval before editing. `'medium'` works until done but asks when a
            choice really matters. `'high'` asks only for credentials, approvals, or
            real ambiguity, and measures results against any numeric target the task
            states. `'max'` never asks, chooses sensible defaults, also deletes its
            scratch files, and treats an unmet stated criterion as a misreading rather
            than settling for a partial answer, for runs nobody can answer. A delegate run is told to choose
            sensible defaults and finish, and measures numeric targets, whatever the level.
        unrestricted_filesystem: Let the file tools reach the whole workspace
            filesystem rather than only the working directory.
        repo_context: Include `RepoContext`. Pass `False` when the agent binds its
            own, so the instruction files are not loaded twice. Pass a
            `RepoContext` to use it in place of the bundled
            `RepoContext(walk_up='repo_root')`, keeping the project-directory note;
            e.g. `RepoContext()` loads only the working directory's instruction files.
        sub_agents: Add `delegate_task`, which hands a self-contained sub-task to a
            fresh run of the same agent `Coder` is bound to, including capabilities
            bound next to `Coder`. This requires binding `Coder` with
            `Agent(capabilities=[...])` rather than passing it to `run()`.
        agent_folders: Also load disk-defined delegates, from `SubAgents` folder
            names or explicit workspace paths. `None` skips disk discovery.
        task_control: Passed to `SubAgents(task_control=...)`, so the agent can run
            delegations in the background, steer, answer, and cancel them.
            Requires `sub_agents=True`.
        caching: Turn on prompt caching through the unified `cache` model setting,
            unless the model, the agent, or a capability listed before `Coder`
            already sets `cache`. A capability listed after it and the run's own
            `model_settings` take precedence as usual.
        strict: The `strict` value for the tools `Coder` contributes. The default
            `False` turns off [strict mode](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#strict-mode),
            which OpenAI and Gemini 2.5+ would otherwise turn on, and relies on the
            bundled `RepairToolArguments` instead. `None` leaves the choice to the
            provider. Tools bound next to `Coder` keep their own value.
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        instructions: str | None = None,
        autonomy: Autonomy = 'high',
        unrestricted_filesystem: bool = False,
        repo_context: bool | RepoContext[AgentDepsT] = True,
        sub_agents: bool = True,
        agent_folders: str | Sequence[str | Path] | None = None,
        task_control: TaskControl | None = None,
        caching: bool = True,
        strict: bool | None = False,
    ) -> None:
        if autonomy not in get_args(Autonomy):
            raise ValueError(f'`autonomy` must be one of {", ".join(map(repr, get_args(Autonomy)))}, not {autonomy!r}.')
        if task_control is not None and not sub_agents:
            raise ValueError('`task_control` requires `sub_agents=True`.')
        if workspace is not None:
            warn_argument_ignored(
                'Coder',
                'workspace',
                "the project directory is the workspace's working directory; attach "
                f'`LocalWorkspace({str(workspace)!r})` (or a sandbox) alongside `Coder()`.',
                stacklevel=3,
            )
        bundled_repo_context = (
            repo_context
            if isinstance(repo_context, RepoContext)
            else RepoContext[AgentDepsT](walk_up='repo_root')
            if repo_context
            else None
        )
        delegation = (
            SubAgents[AgentDepsT](include_self=True, agent_folders=agent_folders, task_control=task_control)
            if sub_agents
            else None
        )
        guidance, delegated_guidance = coder_instructions(
            autonomy=autonomy,
            delegate_tool=delegation.tool_name if delegation is not None else None,
            extra=instructions,
            repo_files=bundled_repo_context.filenames
            if bundled_repo_context is not None and bundled_repo_context.autoload_instructions
            else None,
            nested_instructions=bundled_repo_context is not None
            and bundled_repo_context.nested_traversal
            and bundled_repo_context.nested_inject == 'contents',
        )
        # Naming the directory creates a lazily created sandbox, so it comes only with
        # `RepoContext`, which reads the workspace at run start anyway.
        project = (
            [project_instructions(unrestricted=unrestricted_filesystem)] if bundled_repo_context is not None else []
        )
        capabilities: list[AbstractCapability[AgentDepsT]] = [
            RequireWorkspace[AgentDepsT]('Coder'),
            _CoderGuidance[AgentDepsT](guidance, delegated_guidance, tuple(project)),
            _file_system(unrestricted=unrestricted_filesystem),
            Shell[AgentDepsT](
                denied_commands=[],
                default_timeout=MAX_FOREGROUND_WAIT,
                allow_interactive=True,
                # No env patterns: the workspace, not the host process, supplies the command environment.
                tools=['shell'],
            ),
        ]
        if bundled_repo_context is not None:
            capabilities.append(bundled_repo_context)
        tool_names = {*FILE_TOOL_NAMES, 'shell'}
        if delegation is not None:
            capabilities.append(delegation)
            tool_names.add(delegation.tool_name)
            if task_control is not None:
                # The task-control tools, and the `ask_parent` tool each delegated run gets.
                tool_names |= {*CONTROL_TOOL_NAMES, ASK_PARENT_TOOL_NAME}
        capabilities += [
            ClearToolResults[AgentDepsT](max_fraction=0.7),
            _BoundToolOutputs[AgentDepsT](
                id='coder_tool_output_limits',
                bands=[Band(over=MAX_OUTPUT_CHARS, action=Truncate(max_chars=MAX_OUTPUT_CHARS))],
            ),
            RepairToolArguments[AgentDepsT](),
            _ReportExhaustedRetries[AgentDepsT](),
        ]
        if caching:
            capabilities.append(_DefaultCaching[AgentDepsT]())
        if strict is not None:
            capabilities.append(_SetToolStrict[AgentDepsT](strict=strict, tool_names=frozenset(tool_names)))
        super().__init__(capabilities)
