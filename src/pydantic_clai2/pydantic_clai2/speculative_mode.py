"""Speculative CodeMode wiring, ported from Code Puppy's `enable_speculative_code_mode`.

Every tool the run has, plugin and MCP tools included, becomes a function inside harness
`CodeMode`'s `run_code` except `write_file` and `edit_file`, which stay native so their diffs
render as usual. `shell` folds in too, as Code Puppy's shell tool does: harness `CodeMode` keeps
tools marked with `code_arg_name` metadata native by default, so `SpeculativeExecution` clears
that marker on `shell` only (`FOLDED_CODE_TOOLS`). Without it, a coding turn is all native `shell` calls
and eager execution never starts a build or test while the model is still writing.

Only the read-only tools in `SPECULATIVE_TOOLS`, provided by the capability listed there, may
launch speculatively. That allowlist is the safety contract: an early launch may run for a branch the snippet never takes, so it is reserved for
calls that are harmless to re-run or discard. Everything else waits for eager or normal
execution.

The sandbox gets Monty's `OSAccess` (isolated environment, host clock, in-memory scratch files)
and, when the run's `FileSystem` allows it (`_mount_mode`) and the run's workspace is this machine,
a mount of the workspace's working directory at its real path (`_workspace_mount`). A sandbox
plugin's workspace is not mounted: the host directory is not the filesystem its tools act on.
There is no network in the sandbox; anything remote goes through a wrapped tool such as `shell`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, TypeGuard

from pydantic_monty import MountDir, OSAccess

from pydantic_ai import RunContext
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    DynamicCapability,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.messages import AgentStreamEvent, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModelSettings
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, WrapperToolset
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.code_mode import (
    CodeMode,
    CodeModeToolset,
    SpeculativeCallClaimedEvent,
    SpeculativeCallEvictedEvent,
    SpeculativeCallMissedEvent,
)
from pydantic_ai_harness.filesystem import FileSystem

from .customization import CustomizationGuide, read_clai_customization_guide
from .eager_timing import NESTED_CALL, EagerExecutionCompletedEvent, EagerTiming
from .sandbox_calls import SandboxCallFinishedEvent, SandboxCallStartedEvent
from .speculation import SpeculationCounters

SPECULATIVE_TOOLS: Mapping[str, type[object]] = {
    'list_files': FileSystem,
    'read_file': FileSystem,
    'grep': FileSystem,
    read_clai_customization_guide.__name__: CustomizationGuide,
}
"""Tools that are pure with respect to the workspace, so safe to start early, re-run, or discard,
keyed to the capability that must provide them.

A plugin or MCP tool can share one of these names when `Coder` is off, so the name alone does not
vouch for it. The customization guide reads a file shipped inside the package and takes no arguments.
"""

NATIVE_TOOLS = frozenset({'write_file', 'edit_file'})
"""CLAI's equivalents of Code Puppy's native `create_file` and `replace_in_file`."""

FOLDED_CODE_TOOLS = frozenset({'shell'})
"""Tools marked `code_arg_name` that fold in anyway, as Code Puppy's shell tool does.

Other code-running tools, such as a workflow or capability-authoring plugin, keep harness
`CodeMode`'s default and stay native, so the model never passes a program as a string literal.
"""

GUIDANCE = """\
Speculative execution is on. Use `write_file` and `edit_file` as native tools,
outside `run_code`; they are not available as functions inside the sandbox.
Every other tool, including `shell`, is a function inside `run_code`, a
persistent sandboxed Python REPL. Call `run_code` with a Python snippet to use them; do
not attempt to call those functions as native tools. `await` the functions that
`run_code` lists as `async def`; call the ones listed as plain `def` without `await`.

The sandbox also has direct capabilities, no function call needed:

{workspace}
- Environment variables (isolated) and in-memory scratch files work. For the
  real clock use `datetime.datetime.now()` or `datetime.date.today()`.
  `time.sleep` and `asyncio.sleep` really wait.
- There is NO network in the sandbox: anything remote goes through a
  function like `shell` (e.g. `curl`).

Use raw Python strings for regex patterns so backslashes are not invalid escapes.

How to work. The runtime watches your code AS YOU WRITE IT and starts
eligible calls before the snippet is finished, so the SHAPE of your code
determines how fast it runs:

1. Emit small, flat statements, one per line. A read call (`list_files`,
   `read_file`, `grep`) starts executing the moment its line is complete,
   when its call has all-literal keyword arguments:

       hits = await grep(pattern="SpeculationCounters")
       src = await read_file(path="src/pydantic_clai2/speculation.py")

   Each such line runs while you are still writing the lines below it.
   Computing arguments forfeits that speculative head start. Literal
   calls can also be detected inside expressions or across multiple lines.
2. Go BIG in one `run_code` call. Do not split work across many small
   snippets: every extra round trip to the model wastes the runway that
   makes early execution pay. 60-100 lines with ten, twenty, thirty tool
   calls in a single snippet is the fast path: every additional literal
   read line is another call already running while you write the rest.
3. Front-load the reads: open every snippet with the literal read lines,
   one per line, then process the results with plain Python below them.
   Read the files you MIGHT need, not just the one you are sure of; an
   unused result costs nothing you were not already spending on generation.
4. Never introduce a variable just to pass it: `q = "x"` followed by
   `grep(pattern=q)` runs cold; `grep(pattern="x")` runs early. Repeat the
   literal even if it feels less DRY; here, DRY loses to speed.
5. Writes and shell commands never speculate, but eager execution can
   execute them as soon as their statements close, before generation ends.
   Put a slow build or test command on its own early line so it runs while
   you write the rest of the snippet.
   Obtain required approval BEFORE emitting a side-effectful statement;
   later code cannot undo it. Run independent calls concurrently with
   `await asyncio.gather(...)` (positional awaitables only; no other
   task-creation APIs exist in the sandbox).
6. Keep mutable state small and local: assign results to short fresh
   names, keep processing blocks brief, and never rebind a name a pending
   call's line already used. `print(...)` what matters and make the
   snippet's final expression the value you want returned.
7. State persists between snippets within a run: variables and functions
   carry over. Do not re-fetch what you already hold. Prefer the read
   functions over `pathlib` for discovery (they start early); use
   `pathlib` for surgical follow-ups on paths you already hold.
8. Read before you write, and verify after you change: re-read the file
   or run the tests in a follow-up snippet.
"""


_MountMode = Literal['read-write', 'read-only']
"""How much of the working directory the run's `FileSystem` lets the sandbox mount."""

_WORKSPACE_GUIDANCE: Mapping[str | None, str] = {
    'read-write': """\
- The workspace is mounted read-write at its real absolute path: use
  `pathlib.Path` to read, write, glob, and stat project files directly.""",
    'read-only': """\
- The workspace is mounted read-only at its real absolute path: use
  `pathlib.Path` to read, glob, and stat project files directly; `pathlib`
  writes to it fail.""",
    None: """\
- No host directory is mounted: `pathlib.Path` only reaches in-memory scratch
  files. Use the file functions to read project files.""",
    'remote': """\
- The workspace is not on this machine, so it is not mounted: `pathlib.Path`
  only reaches in-memory scratch files. Use the file functions to read and
  write project files.""",
}
"""The `GUIDANCE` line for each `_MountMode`, `None` when the file system allows no mount, and
`'remote'` when it would but the workspace is a sandbox rather than this machine."""


def guidance(mount: _MountMode | Literal['remote'] | None) -> str:
    """Code Puppy's guidance, describing the workspace the sandbox actually has."""
    return GUIDANCE.format(workspace=_WORKSPACE_GUIDANCE[mount])


def _is_local(workspace: Workspace) -> bool:
    """Whether the run's workspace is a directory on this machine, which a host mount can reach."""
    ref = workspace.ref
    return ref is not None and ref.provider == 'local'


def _is_capability(capability: AgentCapability[AgentDepsT]) -> TypeGuard[AbstractCapability[AgentDepsT]]:
    return isinstance(capability, AbstractCapability)


def _mount_mode(granted: Sequence[AgentCapability[AgentDepsT]]) -> _MountMode | None:
    """Mount only what the run's `FileSystem` already lets its tools reach, or nothing.

    `pathlib` calls on a mount never pass through `FileSystem`'s checks, so an unconditional
    read-write mount of the working directory let sandboxed code read or overwrite files the
    caller had restricted (Veria, #1078). The mount is the workspace's working directory, which
    `root_dir` always contains, and only when the file system registers `read_file`, since
    `pathlib` reads any file's content. It is writable only when it may also write every file there: `write_file` registered, not `read_only`, and no
    `read_only_patterns`. A mount cannot express `allowed_patterns` or `denied_patterns`, so either one
    leaves the sandbox unmounted, as do zero or several file systems and any capability function
    or `DynamicCapability`, which may only resolve to a file system at run time.
    """
    leaves: list[AbstractCapability[AgentDepsT]] = []
    for capability in granted:
        if not _is_capability(capability):
            return None
        capability.apply(leaves.append)
    file_systems = [leaf for leaf in leaves if isinstance(leaf, FileSystem)]
    if len(file_systems) != 1 or any(isinstance(leaf, DynamicCapability) for leaf in leaves):
        return None
    [file_system] = file_systems
    tools = set(file_system.tools)
    if file_system.allowed_patterns or file_system.denied_patterns or 'read_file' not in tools:
        return None
    writable = 'write_file' in tools and not file_system.read_only and not file_system.read_only_patterns
    return 'read-write' if writable else 'read-only'


async def _workspace_mount(workspace: Workspace, mode: _MountMode | None) -> MountDir | None:
    """The run's working directory at its real path, when `mode` allows a mount and the workspace is local.

    A sandbox plugin's workspace lives elsewhere, so a host mount would hand `pathlib` a different
    filesystem than the one the file tools act on. A read-only workspace is mounted read-only.
    """
    if mode is None or not _is_local(workspace):
        return None
    directory = await workspace.working_dir()
    return MountDir(virtual_path=directory, host_path=directory, mode='read-only' if workspace.read_only else mode)


@dataclass
class _MountWorkspace(WrapperToolset[AgentDepsT]):
    """Give `CodeMode`'s toolset this run's `_workspace_mount`, once the run's workspace is known."""

    mode: _MountMode | None = None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        code_mode = self.wrapped
        assert isinstance(code_mode, CodeModeToolset)
        mount = await _workspace_mount(ctx.workspace, self.mode)
        return await replace(code_mode, mount=mount).for_run(ctx)


@dataclass
class WorkspaceCodeMode(CodeMode[AgentDepsT]):
    """`CodeMode` whose mount is the run's local working directory, limited to `mount_mode`.

    Core selects the run's workspace after capabilities' `for_run`, so the mount is resolved in the
    toolset's `for_run`, which runs after the selection.
    """

    mount_mode: _MountMode | None = None

    def get_wrapper_toolset(self, toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT] | None:
        """`CodeMode`'s toolset, mounted per run by `_MountWorkspace`."""
        code_mode = super().get_wrapper_toolset(toolset)
        assert code_mode is not None
        return _MountWorkspace(code_mode, mode=self.mount_mode)


def _sandboxed(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> bool:
    return tool_def.name not in NATIVE_TOOLS


def _declarations(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> dict[str, object]:
    """The tool's metadata as `CodeMode(speculate='declared')` should read it.

    Only a `SPECULATIVE_TOOLS` entry from its expected capability is declared read-only; every
    other tool's own `read_only` or MCP `readOnlyHint` claim is overridden, so the allowlist stays
    Code Puppy's rather than whatever a plugin or MCP server says about itself.
    """
    metadata: dict[str, object] = dict(tool_def.metadata or {})
    if tool_def.name in FOLDED_CODE_TOOLS:
        metadata.pop('code_arg_name', None)
    owner = SPECULATIVE_TOOLS.get(tool_def.name)
    trusted = owner is not None and isinstance(ctx.capabilities.get(tool_def.capability_id or ''), owner)
    metadata['read_only'] = trusted
    annotations: Mapping[str, object] | None = tool_def.metadata and tool_def.metadata.get('annotations')
    if annotations and not trusted:
        metadata['annotations'] = {**annotations, 'readOnlyHint': False}
    return metadata


@dataclass
class SpeculativeExecution(AbstractCapability[AgentDepsT]):
    """Fold code tools in, teach the snippet shape, stream tool arguments, and count outcomes."""

    counters: SpeculationCounters
    mount: _MountMode | None = None
    """The sandbox's `_mount_mode`, so the guidance describes it."""

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str]:
        """Code Puppy's guidance, with CLAI's tool and argument names and the run's actual mount."""

        def describe(ctx: RunContext[AgentDepsT]) -> str:
            local = self.mount is None or _is_local(ctx.workspace)
            return guidance(self.mount if local else 'remote')

        return describe

    def get_model_settings(self) -> AnthropicModelSettings:
        """Anthropic buffers a tool call's input by default, which leaves eager execution no runway.

        Other providers stream tool arguments already and ignore the setting.
        """
        return {'anthropic_eager_input_streaming': True}

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        """Fold `FOLDED_CODE_TOOLS` in and declare which tools may launch speculatively.

        Core runs this inside every capability's wrapper toolset, so `CodeMode` sees the result.
        """
        return [replace(tool_def, metadata=_declarations(ctx, tool_def)) for tool_def in tool_defs]

    async def on_event(self, ctx: RunContext[AgentDepsT], *, event: AgentStreamEvent) -> None:
        """Consume telemetry without retaining generated code or rendering anything."""
        counters = self.counters
        if isinstance(event, EagerExecutionCompletedEvent):
            counters.eager_ms += max(0.0, event.saved_ms)
        elif isinstance(event, SpeculativeCallClaimedEvent):
            counters.hits += 1
            if event.ready_at_claim:
                counters.speculative_ms += max(0.0, event.elapsed_ms)
        elif isinstance(event, SpeculativeCallMissedEvent):
            counters.misses += 1
        elif isinstance(event, SpeculativeCallEvictedEvent):
            counters.wasted += 1


@dataclass
class ShowSandboxCalls(AbstractCapability[AgentDepsT]):
    """Report tools called from inside `run_code` so they render like direct calls.

    A cold call reports its start and result as it runs. A speculative launch may never be used,
    so its call and result are held and reported under the claiming call's id only when the
    snippet claims it; an evicted launch is dropped unseen. The result is always held by then:
    harness `SpeculationCoordinator.adopt` awaits the launch, which runs this hook to completion,
    before it emits `SpeculativeCallClaimedEvent`, even when the launch was not ready at the claim.
    """

    _launched: dict[str, tuple[ToolCallPart, ToolReturnPart | RetryPromptPart]] = field(
        default_factory=dict[str, tuple[ToolCallPart, ToolReturnPart | RetryPromptPart]], init=False, repr=False
    )
    _evicted: set[str] = field(default_factory=set[str], init=False, repr=False)
    """Launches evicted while still running, whose late result must not be held."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> 'ShowSandboxCalls[AgentDepsT]':
        """Keep held launches per run."""
        return ShowSandboxCalls()

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        """Report cold sandbox calls as they run; hold speculative results for the claim."""
        nested = NESTED_CALL.fullmatch(call.tool_call_id)
        if nested is None:
            return await handler(args)
        speculative = nested['speculative'] is not None
        if not speculative:
            await ctx.emit(SandboxCallStartedEvent(tool_call_id=call.tool_call_id, call=call))
        try:
            value = await handler(args)
        except Exception as error:
            failure = RetryPromptPart(content=str(error), tool_name=call.tool_name, tool_call_id=call.tool_call_id)
            await self._finish(ctx, call, failure, speculative=speculative)
            raise
        returned = ToolReturnPart(tool_name=call.tool_name, content=value, tool_call_id=call.tool_call_id)
        await self._finish(ctx, call, returned, speculative=speculative)
        return value

    async def _finish(
        self,
        ctx: RunContext[AgentDepsT],
        call: ToolCallPart,
        result: ToolReturnPart | RetryPromptPart,
        *,
        speculative: bool,
    ) -> None:
        if speculative and call.tool_call_id in self._evicted:
            self._evicted.discard(call.tool_call_id)
        elif speculative:
            self._launched[call.tool_call_id] = (call, result)
        else:
            await ctx.emit(SandboxCallFinishedEvent(tool_call_id=call.tool_call_id, result=result))

    async def on_event(self, ctx: RunContext[AgentDepsT], *, event: AgentStreamEvent) -> None:
        """Report a claimed launch as the claiming call; forget an evicted one."""
        if isinstance(event, SpeculativeCallEvictedEvent):
            if event.state == 'pending':
                self._evicted.add(event.launch_id)
            else:
                self._launched.pop(event.launch_id, None)
        elif isinstance(event, SpeculativeCallClaimedEvent) and event.launch_id in self._launched:
            call, result = self._launched.pop(event.launch_id)
            call_id = event.nested_tool_call_id
            await ctx.emit(SandboxCallStartedEvent(tool_call_id=call_id, call=replace(call, tool_call_id=call_id)))
            await ctx.emit(SandboxCallFinishedEvent(tool_call_id=call_id, result=replace(result, tool_call_id=call_id)))


def speculative_capabilities(
    counters: SpeculationCounters, granted: Sequence[AgentCapability[AgentDepsT]]
) -> 'list[AbstractCapability[AgentDepsT]]':
    """Fold tools into `run_code` with eager execution and read-only speculation.

    `granted` is every other capability the run binds; the sandbox mount follows its `FileSystem`.
    """
    mode = _mount_mode(granted)
    return [
        WorkspaceCodeMode(
            tools=_sandboxed,
            # `SpeculativeExecution.prepare_tools` writes the declarations from `SPECULATIVE_TOOLS`.
            speculate='declared',
            # Eager runs each streamed statement as it closes; speculation launches the
            # read-only calls beyond that frontier, and the eager feed claims them.
            eager=True,
            mount_mode=mode,
            os_access=OSAccess(),
        ),
        EagerTiming(),
        SpeculativeExecution(counters, mode),
        ShowSandboxCalls(),
    ]
