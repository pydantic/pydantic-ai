"""Complete coding-agent harness assembled from regular capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, ValidatedToolArgs, WrapperCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import InstructionPart, ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai_harness._warn import warn_argument_ignored
from pydantic_ai_harness._workspace import RequireWorkspace
from pydantic_ai_harness.coder._instructions import (
    BENCHMARK_INSTRUCTIONS,
    Autonomy,
    coder_instructions,
    model_specific_instructions,
)
from pydantic_ai_harness.compaction import ClearToolResults
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.shell import MAX_FOREGROUND_WAIT, Shell
from pydantic_ai_harness.sub_agents import SubAgents, TaskControl
from pydantic_ai_harness.sub_agents._tasks import ASK_PARENT_TOOL_NAME
from pydantic_ai_harness.sub_agents._toolset import CONTROL_TOOL_NAMES, delegation_depth
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

FILE_TOOL_NAMES: tuple[str, ...] = ('read_file', 'write_file', 'edit_file', 'list_files', 'grep')
"""The `FileSystem` tools `Coder` registers; `shell` covers directory creation, file metadata, and the rest."""


class _BoundToolOutputs(ToolOutputLimits[AgentDepsT]):
    # Unsets only the class-level default id, which is what the combine policy reads: two instances
    # sharing an id raise instead of merging as one configuration stated twice, so the `Anonymous`
    # `combine` of its test case holds. Instances still default to `id='tool_output_limits'` through the
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


class _CoderShell(Shell[AgentDepsT]):
    """`Coder`'s own `Shell`, which refuses another `Shell` on the same agent.

    A `Shell` added beside `Coder` registers `run_command` and its siblings next to `Coder`'s
    unrestricted `shell` tool rather than narrowing it, so its `allowed_commands` restricts nothing.
    The tool names differ, so nothing else catches it: this checks the run's capabilities by type before
    every request, which also covers a `Coder` loaded from a deferred group. A stopgap until a second `Shell` narrows `Coder`'s instead, tracked in
    https://github.com/pydantic/pydantic-ai/issues/14515.
    """

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Check before every request, so a `Coder` loaded later from a deferred group is checked too."""
        _reject_another_shell(ctx)
        return await super().prepare_tools(ctx, tool_defs)


def _reject_another_shell(ctx: RunContext[Any]) -> None:
    """Raise when the run has a `Shell` besides `Coder`'s own."""
    extra: list[AbstractCapability[Any]] = []

    def collect(candidate: AbstractCapability[Any]) -> None:
        while isinstance(candidate, WrapperCapability):
            candidate = candidate.wrapped
        if isinstance(candidate, Shell) and not isinstance(candidate, _CoderShell):
            extra.append(candidate)

    if ctx.root_capability is not None:  # pragma: no branch
        ctx.root_capability.apply(collect)
    if extra:
        raise UserError(
            '`Coder` already gives the agent an unrestricted `shell` tool, and a `Shell` beside it adds '
            "more command tools rather than narrowing `Coder`'s: its `allowed_commands` and "
            '`denied_commands` do not apply to `shell`. Remove the extra `Shell`, and adjust `Coder` through '
            'its own parameters (`instructions`, `autonomy`, `sub_agents`); add `Permissions` to require '
            'approval before commands, or a sandbox workspace such as `BubblewrappedWorkspace` to contain '
            'them. For a restricted shell, build the agent from `FileSystem` and `Shell` without `Coder`. '
            'Narrowing `Coder` by adding capabilities is planned in '
            'https://github.com/pydantic/pydantic-ai/issues/14515.'
        )


MAX_READ_CHARS = 50_000
"""Characters of complete lines per `read_file`, below the output cap."""

MAX_IMAGE_SIDE = 1024
"""Longest side, in pixels, of an image `read_file` returns; a larger one is downscaled to it.

Providers reject images over their own size limits, or downscale them themselves
after the full-size upload.
"""

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
    a recorded delegation sets again on the worker. The text only some models get, such as the rule against
    unrequested delegation for models whose profile sets `delegates_unprompted`, is in blocks of its own: see
    `model_specific_instructions`.
    """

    guidance: str
    delegated_guidance: str
    model_specific: tuple[Callable[[RunContext[AgentDepsT]], InstructionPart], ...] = ()
    """The blocks from `model_specific_instructions`, resolved against the model handling each request.

    When there are any, they can differ between models, so running on a `FallbackModel` gets a warning.
    """
    delegate: bool = False

    def __post_init__(self) -> None:
        self.id = 'coder_instructions'

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return [
            self.delegated_guidance if self.delegate else self.guidance,
            *self.model_specific,
        ]

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        if self.model_specific and isinstance(ctx.model, FallbackModel):
            # Python's default warning filter shows this once per process: it's raised from one line.
            warnings.warn(_FALLBACK_MODEL_WARNING, UserWarning, stacklevel=1)
        if delegation_depth() <= 1:
            return self
        return replace(self, delegate=True)


_FALLBACK_MODEL_WARNING = (
    "`Coder` can't read model-specific instructions from `FallbackModel`, so every model gets the default ones; "
    "use the `ModelFallback` capability so each attempt gets its model's instructions."
)


@dataclass
class BenchmarkInstructions(AbstractCapability[AgentDepsT]):
    """Benchmark-only guidance, bound next to `Coder` for runs a verifier grades.

    It says deliverables are checked in a clean directory with only the requested files, and that
    every field of the provided data matters. Neither holds for everyday work in a project, so
    `Coder` leaves both out. `delegate_task`'s `self` delegate re-runs the same agent, so it gets it too.
    """

    _: KW_ONLY
    id: str | None = 'benchmark_instructions'

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return BENCHMARK_INSTRUCTIONS


def _file_system(*, unrestricted: bool, max_image_side: int | None) -> FileSystem[AgentDepsT]:
    file_system = FileSystem[AgentDepsT](
        content_hashes=False,
        max_read_chars=MAX_READ_CHARS,
        tools=FILE_TOOL_NAMES,
        max_retries=MAX_FILE_TOOL_RETRIES,
        max_image_side=max_image_side,
    )
    if unrestricted:
        # `/` is the filesystem root, or on Windows the working directory's drive.
        return replace(file_system, root_dir='/', read_only_patterns=[])
    return file_system


@dataclass(init=False)
class Coder(CombinedCapability[AgentDepsT]):
    """Autonomous coding with six tools, delegation, and context management, in the run's workspace.

    Files and commands go through `ctx.workspace`, and the project directory is
    its working directory: attach `LocalWorkspace('.')` (or `WindowsWorkspace('.')`
    on Windows) for a local checkout, or a sandbox provider's capability for
    untrusted work, alongside `Coder()`. A run without a workspace fails at its
    start. Commands are unrestricted and can outlive runs.

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
            `RepoContext(walk_up='repo_root')`; e.g. `RepoContext()` loads only the working directory's instruction files.
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
        max_image_side: Longest side, in pixels, of an image `read_file` returns; a larger
            one is downscaled to it first. Needs Pillow, which the `coder` extra installs.
            `None` sends images at full size and drops the Pillow requirement.
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
        max_image_side: int | None = MAX_IMAGE_SIDE,
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
            extra=instructions,
            repo_files=bundled_repo_context.filenames
            if bundled_repo_context is not None and bundled_repo_context.autoload_instructions
            else None,
            nested_instructions=bundled_repo_context is not None
            and bundled_repo_context.nested_traversal
            and bundled_repo_context.nested_inject == 'contents',
        )
        capabilities: list[AbstractCapability[AgentDepsT]] = [
            RequireWorkspace[AgentDepsT]('Coder'),
            _CoderGuidance[AgentDepsT](
                guidance,
                delegated_guidance,
                tuple(
                    model_specific_instructions(
                        autonomy=autonomy, delegate_tool=delegation.tool_name if delegation is not None else None
                    )
                ),
            ),
            _file_system(unrestricted=unrestricted_filesystem, max_image_side=max_image_side),
            _CoderShell[AgentDepsT](
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
