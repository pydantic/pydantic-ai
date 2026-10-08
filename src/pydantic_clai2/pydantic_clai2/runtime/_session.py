"""Conversation state and run-scoped capability plugins."""

import logging
import os
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, Generator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast
from uuid import uuid4

from anyio import get_cancelled_exc_class, move_on_after

from pydantic_ai import Agent, AgentModelSettings, AgentRunResult, AgentStreamEvent, RunContext, capture_run_messages
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    Capability,
    CapabilityOrdering,
    CombinedCapability,
    DynamicCapability,
    LocalWorkspace,
    ResolveModelId,
    WrapperCapability,
)
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, UserContent, UserPromptPart
from pydantic_ai.models import Model, ModelResolutionContext, infer_model
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationConflict,
    ConversationSummary,
    SqliteConversationStore,
    ensure_inactive,
)
from pydantic_ai_harness.subagents import DelegationReports, DelegationTasks
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError, raised_here, setup_errors
from pydantic_clai2.ui import telemetry

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_CURRENT_SESSION_ID: ContextVar[str | None] = ContextVar('clai2_session_id', default=None)


def current_session_id() -> str | None:
    """The saved conversation running in this task, including a background fork."""
    return _CURRENT_SESSION_ID.get()


@contextmanager
def session_context(session_id: str) -> Generator[None]:
    """Keep a saved conversation current through its run and shell lifecycle events."""
    token = _CURRENT_SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _CURRENT_SESSION_ID.reset(token)


FamilyDefaults = Callable[[str], ModelSettings | None]
"""CLAI's family default settings for a model, given its name."""


def _supports_local_workspace() -> bool:
    return sys.platform != 'win32'


def _supplies_workspace(plugins: Sequence[AgentCapability[DepsT]], *, include_dynamic: bool = True) -> bool:
    """Whether a plugin, such as a sandbox, supplies the run's workspace, so clai adds no `LocalWorkspace`.

    A plugin counts when it has a leaf, loaded up front, that overrides `get_workspace`. With
    `include_dynamic`, a capability function counts too, as its capability is known only once the run starts.
    """
    leaves: list[AbstractCapability[DepsT]] = []
    for plugin in plugins:
        # A capability function's capability exists only once the run starts.
        if isinstance(plugin, AbstractCapability):
            capability = cast('AbstractCapability[DepsT]', plugin)
            # `apply` visits a group's members, not a group subclass that supplies the workspace itself.
            if (
                isinstance(capability, CombinedCapability)
                and type(capability).get_workspace is not CombinedCapability.get_workspace
            ):
                return True
            capability.apply(leaves.append)
        elif include_dynamic:
            return True
    return any(
        not leaf.defer_loading and _overrides_get_workspace(leaf, include_dynamic=include_dynamic) for leaf in leaves
    )


def _overrides_get_workspace(leaf: AbstractCapability[DepsT], *, include_dynamic: bool) -> bool:
    while isinstance(leaf, WrapperCapability):
        if type(leaf).get_workspace is not WrapperCapability.get_workspace:
            return True
        wrapped: list[AbstractCapability[DepsT]] = []
        leaf.wrapped.apply(wrapped.append)
        if len(wrapped) != 1 or wrapped[0] is not leaf.wrapped:
            # A wrapped tree's leaves are visited on their own; only a lone wrapped capability hides behind its wrapper.
            return False
        leaf = leaf.wrapped
    if isinstance(leaf, DynamicCapability):
        # A capability function's capability, and so its workspace, is known only once the run starts.
        return include_dynamic
    return type(leaf).get_workspace is not AbstractCapability.get_workspace


@dataclass
class _LocalFallback(LocalWorkspace[DepsT]):
    """The session directory, for a run whose capability functions turn out to supply no workspace.

    Returned from a capability function itself, so core asks it only after the run's capability functions
    have resolved, alongside whatever workspace they supplied, which it defers to.
    """

    _asking: bool = field(default=False, init=False, repr=False)

    def get_workspace(self, ctx: RunContext[DepsT], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
        if self._asking:
            return None
        assert ctx.root_capability is not None, 'core sets the root capability before selecting a workspace'
        # Ask the resolved tree itself, so a provider of any shape (a group, a wrapper) is found; this
        # instance declines while asking. Selection does no I/O, so asking twice is harmless.
        self._asking = True
        try:
            other = ctx.root_capability.get_workspace(ctx, ref=ref)
        finally:
            self._asking = False
        return None if other is not None else super().get_workspace(ctx, ref=ref)


def _requested_model(ctx: RunContext[DepsT]) -> str:
    """The name a request's model goes by: what it was selected as, or its own name for an instance."""
    return ctx.model_id or ctx.model.model_name


@dataclass
class ModelDefaults(AbstractCapability[DepsT]):
    """CLAI's default settings for the model each request uses, beneath other capabilities' settings.

    Merged first among capabilities, wrapping every other one, even another `outermost` one, so
    another capability's settings and the run's own take precedence. An agent's own `model_settings`
    merge before any capability's, so these defaults still override them, as they did when CLAI
    passed them to the run. Resolved per request, so the defaults follow a model a capability selects.
    """

    defaults: FamilyDefaults
    """The defaults for a model, given its name: the run's model name when the run selected it by name."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost', wraps=[AbstractCapability])

    def get_model_settings(self) -> Callable[[RunContext[DepsT]], ModelSettings]:
        return lambda ctx: self.defaults(_requested_model(ctx)) or ModelSettings()


@dataclass
class SessionModels(ResolveModelId[DepsT]):
    """Resolve model names through CLAI first, as when CLAI resolved the selected model before each run.

    Outermost, so it is tried before a resolver on the agent or a plugin, unless that one is outermost
    too. Names CLAI returns unchanged are left to those resolvers, then to `infer_model`.
    """

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')


ModelNameResolver = Callable[[str], Model | str | Awaitable[Model | str]]
"""Builds the model a name selects, or returns a name for core to infer."""


async def resolve_model_name(resolve: ModelNameResolver, model_id: str) -> Model | None:
    """What `resolve` makes of `model_id`, as a `SessionModels` resolver returns it.

    A name it maps to another name is inferred by core. `None` when it returns `model_id` unchanged,
    leaving the name to the run's other resolvers.
    """
    model = resolve(model_id)
    if isinstance(model, Awaitable):
        model = await model
    if isinstance(model, Model):
        return model
    return infer_model(model) if model != model_id else None


def _agent_capabilities(agent: AbstractAgent[DepsT, OutputT]) -> list[AgentCapability[DepsT]]:
    """The capabilities the agent was built with, so a sandbox configured on it counts too.

    A run-level `LocalWorkspace` would be asked before the agent's own sandbox and win.
    """
    try:
        return [agent.root_capability]
    except NotImplementedError:  # A custom `AbstractAgent` need not expose its capabilities.
        return []


def _command_env() -> dict[str, str]:
    """The environment clai's commands get: this process's, minus LLM API keys.

    clai is a local coding CLI, so the model's commands see the user's shell environment the way
    the user's own commands would; only provider credentials are held back.
    """
    return {
        name: value
        for name, value in os.environ.items()
        if not any(fnmatchcase(name, pattern) for pattern in LLM_API_KEY_ENV_PATTERNS)
    }


def local_workspace(
    configured: Sequence[AgentCapability[DepsT]], directory: str | Path
) -> AgentCapability[DepsT] | None:
    """The `LocalWorkspace` at `directory` that CLAI adds beside `configured`; `None` when it adds none.

    None is added on platforms without a local workspace, or when a capability loaded up front supplies
    the workspace, such as a sandbox. When only a capability function might, the local one defers to it.
    """
    if not _supports_local_workspace() or _supplies_workspace(configured, include_dynamic=False):
        return None
    if _supplies_workspace(configured):
        # No id: a function's `LocalWorkspace` shares the default id and would replace this whole.
        fallback = _LocalFallback[DepsT](directory, env=_command_env(), id=None)
        return DynamicCapability[DepsT](lambda ctx: fallback)
    return LocalWorkspace[DepsT](directory, env=_command_env())


def _interrupted(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """`messages` with the last one marked interrupted, so core closes unanswered calls without replaying them."""
    marked = list(messages)
    if marked:
        last = marked[-1]
        if not isinstance(last, ModelResponse) or last.state != 'suspended':
            marked[-1] = replace(last, state='interrupted')
    return marked


def _fork_summary(busy: ConversationSummary) -> ConversationSummary:
    """A new saved session continuing `busy`, which another process is still running."""
    return ConversationSummary(
        workspace=busy.workspace,
        title=f'{busy.title} (fork)',
        subtitle=busy.subtitle,
        tags=busy.tags,
        title_source=busy.title_source,
        model=busy.model,
    )


def _stale_local_workspace(messages: Sequence[ModelMessage], workspace: str) -> bool:
    """Whether the history's latest response names a local directory other than `workspace`.

    `LocalWorkspace` declines such a reference, so continuing from it would leave the run without a
    workspace. A sandbox plugin's reference is not stale here: it continues in that sandbox.
    """
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    ref = response.workspace_ref if response is not None else None
    return ref is not None and ref.provider == 'local' and ref != WorkspaceRef(provider='local', id=workspace)


class StockAgent(Agent[DepsT, OutputT]):
    """A CLAI-owned agent whose configuration can be rebuilt with active plugins."""

    def __init__(
        self,
        model: Model | str | None,
        *,
        deps_type: type[DepsT],
        output_type: OutputSpec[OutputT],
        capabilities: Sequence[AgentCapability[DepsT]],
        defer_model_check: bool = False,
    ) -> None:
        super().__init__(
            model,
            deps_type=deps_type,
            output_type=output_type,
            capabilities=capabilities,
            defer_model_check=defer_model_check,
        )
        self._stock_deps_type = deps_type

    def with_plugins(
        self, plugins: Sequence[AgentCapability[DepsT]], *, model: str | None = None
    ) -> 'StockAgent[DepsT, OutputT]':
        """Bind a snapshot without mutating the agent used by another conversation.

        `model` replaces the agent's own model with a name each run resolves, through the session.
        """
        return StockAgent(
            self.model if model is None else model,
            deps_type=self._stock_deps_type,
            output_type=self.output_type,
            capabilities=[self.root_capability, *plugins],
            defer_model_check=model is not None,
        )


class Session(Generic[DepsT, OutputT]):
    """Run prompts to completion, retaining successful and interrupted turns in memory.

    Stock agents are rebuilt when the plugin snapshot or CLAI's unchosen default model changes,
    so delegates carry the same capabilities. Supplied agents keep their existing run-level plugins.
    """

    def __init__(
        self,
        agent: AbstractAgent[DepsT, OutputT],
        *,
        deps: DepsT,
        plugins: Sequence[AgentCapability[DepsT]] = (),
        message_history: Sequence[ModelMessage] = (),
        usage_limits: UsageLimits | None = None,
        conversations: SqliteConversationStore | None = None,
        workspace: Path | None = None,
        on_stream_event: Callable[[AgentStreamEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.delegations: DelegationTasks | None = None
        self.conversations = conversations
        self.workspace = str((workspace or Path.cwd()).resolve())
        self.summary = ConversationSummary(workspace=self.workspace)
        self.step_store: StepStore | None = (
            SqliteStepStore(database=conversations.database, max_snapshots_per_run=8) if conversations else None
        )
        self.model: str | None = None
        self.model_chosen = True
        """Whether the user chose `model`. A stock agent takes CLAI's default as its own model instead,
        so a capability that selects a model takes precedence over it; a chosen model is passed to each run."""
        self.model_settings: AgentModelSettings[DepsT] | None = None
        """Settings the user chose for this session's model, passed to each run, where they take precedence
        over all others. A request on another model, which a capability selected, does not get them. A
        callable is resolved before every model request, so it can change mid-run."""
        self.model_defaults: FamilyDefaults | None = None
        """CLAI's default settings for a model name, beneath the agent's capabilities' settings."""
        self.tool_retries: int | None = None
        self.instructions = ''
        """The user's own instructions, sent after the agent's and its plugins' on each request; empty sends none.

        A stock agent rebuilt for its plugins binds them too, so delegated tasks get them; otherwise each run
        gets them. Either way they are read on each request, so changing them does not rebuild the agent."""
        self.resolve_model: ModelNameResolver = lambda name: name
        self.agent = agent
        self._base_agent = agent
        self._bound_plugins: tuple[AgentCapability[DepsT], ...] = ()
        self._bound_model: str | None = None
        self.deps = deps
        self.plugins: Sequence[AgentCapability[DepsT]] = tuple(plugins)
        self.usage_limits = usage_limits
        self.on_stream_event = on_stream_event
        self._messages: list[ModelMessage] = list(message_history)
        self._running = False
        self._accepting_steering = False
        self._run_context: RunContext[DepsT] | None = None
        self._pending_steering: list[Sequence[UserContent]] = []
        self.on_context_usage: Callable[[int], None] | None = None
        self.on_context_window: Callable[[int], None] | None = None
        """Told the context window of each streamed request's model, when its profile knows one."""
        self.on_setup_error: Callable[[CapabilitySetupError], None] | None = None
        """Told when a guarded plugin capability rejected its configuration, before the failed turn's error propagates."""

    @property
    def running(self) -> bool:
        """Whether a run, resume, or history replacement holds the conversation now."""
        return self._running

    @property
    def messages(self) -> list[ModelMessage]:
        """Return a snapshot of the conversation's message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Start a new conversation without replacing the agent or plugins."""
        cleared = len(self._messages)
        self.replace_messages(())
        telemetry.record('conversation cleared', messages=cleared)
        self.summary = ConversationSummary(workspace=self.workspace)

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history, as `/compact` does after summarising it."""
        if self._running:
            raise RuntimeError('Cannot replace the history of a running conversation')
        self._messages = list(messages)

    async def commit_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Persist a between-turn history replacement before publishing it."""
        if self._running:
            raise RuntimeError('Cannot replace the history of a running conversation')
        self._running = True
        try:
            if self.conversations is not None:
                self.summary = await self.conversations.save(
                    summary=replace(self.summary, outcome='ready', run_id=None, model=self.model), messages=messages
                )
            self._messages = list(messages)
        finally:
            self._running = False

    async def resume(self, conversation_id: str, *, allow_other_workspace: bool = False) -> str:
        """Restore a saved head without invoking the model or replaying tools.

        A session another live process is running is not taken over: its newest saved state is
        copied into a new saved session, which this one continues while the original keeps running.
        """
        if self._running:
            raise RuntimeError('Cannot resume during a running conversation')
        self._running = True
        try:
            if self.conversations is None:
                raise ValueError('Session persistence is not configured')
            saved = await self.conversations.get(conversation_id=conversation_id)
            if not allow_other_workspace:
                self.check_workspace(saved.summary.workspace)
            busy: ConversationConflict | None = None
            try:
                ensure_inactive(saved.summary)
            except ConversationConflict as conflict:
                busy = conflict
            messages = saved.messages
            interrupted = saved.summary.outcome in ('running', 'failed', 'cancelled')
            warning = ''
            if interrupted:
                warning = ' Interrupted session: inspect external effects before continuing. No tools were replayed.'
            if saved.summary.outcome == 'running' and saved.summary.run_id and self.step_store:
                snapshot = await self.step_store.latest_snapshot(run_id=saved.summary.run_id, include_interrupted=True)
                if snapshot is not None:
                    messages = snapshot.messages
            if interrupted:
                messages = _interrupted(messages)
            summary = saved.summary
            notice = f'Resumed {summary.title} ({summary.id}).{warning}'
            if busy is not None:
                summary = await self.conversations.save(summary=_fork_summary(saved.summary), messages=messages)
                notice = (
                    f'{busy} Resumed a fork of it instead: {summary.title} ({summary.id}). '
                    'The original keeps running there and may still change files. No tools were replayed.'
                )
            self._messages = list(messages)
            self.summary = summary
            telemetry.record(
                'conversation resumed',
                outcome=saved.summary.outcome,
                messages=len(messages),
                other_workspace=saved.summary.workspace != self.workspace,
                forked=busy is not None,
            )
            # Keep the caller's current model and approval configuration. Saved models are informational.
            return notice
        finally:
            self._running = False

    def check_workspace(self, workspace: str) -> None:
        """Refuse a conversation from another directory unless the `/resume` browser confirmed it."""
        if workspace != self.workspace:
            raise ValueError(f'Session belongs to {workspace}. Select it in /resume to confirm.')

    def _mark_interrupted(self) -> None:
        self._messages = _interrupted(self._messages)

    async def _save_turn(self, *, outcome: Literal['running', 'completed', 'failed', 'cancelled']) -> None:
        if self.conversations is None:
            return
        self.summary = await self.conversations.save(
            summary=replace(self.summary, outcome=outcome, model=self.model, owner_pid=None), messages=self._messages
        )

    async def resolved_model(self) -> Model | str | None:
        """The selected model after `resolve_model`, else the agent's own (CLAI's default).

        A capability that selects a model can replace that default per request, so a run may use another.
        """
        if self.model is None:
            return self.agent.model
        model = self.resolve_model(self.model)
        return await model if isinstance(model, Awaitable) else model

    def _run_settings(self) -> AgentModelSettings[DepsT] | None:
        """`model_settings` for requests on this session's model only."""
        settings = self.model_settings
        own = self.model if self.model is not None else self.agent.model
        if settings is None or own is None:
            return settings
        name = own if isinstance(own, str) else own.model_name

        def for_own_model(ctx: RunContext[DepsT]) -> ModelSettings:
            if _requested_model(ctx) != name:
                return ModelSettings()
            return settings(ctx) if callable(settings) else settings

        return for_own_model

    def _bind(self) -> tuple[str | None, list[AgentCapability[DepsT]]]:
        """The model to pass to the next run and its run-level capabilities, rebinding a stock agent first."""
        run_model = self.model
        capabilities = list(self.plugins)
        if isinstance(self._base_agent, StockAgent):
            # CLAI's default is the agent's own model, so a capability that selects one wins.
            agent_model = None if self.model_chosen else self.model
            if agent_model is not None:
                run_model = None
            if (
                agent_model != self._bound_model
                or len(self.plugins) != len(self._bound_plugins)
                or any(new is not old for new, old in zip(self.plugins, self._bound_plugins))
            ):
                instructions = Capability[DepsT](instructions=self._user_instructions)
                self.agent = self._base_agent.with_plugins([*self.plugins, instructions], model=agent_model)
                self._bound_plugins = tuple(self.plugins)
                self._bound_model = agent_model
            # Already bound to the stock agent, including delegation and guardrails.
            capabilities = []
        if self.model is not None:
            capabilities.append(SessionModels[DepsT](self._resolve_model_id))
        if self.model_defaults is not None:
            capabilities.append(ModelDefaults[DepsT](self.model_defaults))
        return run_model, capabilities

    def _user_instructions(self) -> str | None:
        return self.instructions or None

    async def _resolve_model_id(self, ctx: ModelResolutionContext[DepsT], model_id: str) -> Model | None:
        return await resolve_model_name(self.resolve_model, model_id)

    def steer(self, text: str, *, images: Sequence[BinaryContent] = ()) -> bool:
        """Deliver input to the active run, or decline when no run is accepting input."""
        if not self._accepting_steering:
            return False
        content: Sequence[UserContent] = [text, *images]
        if self._run_context is None:
            self._pending_steering.append(content)
        else:
            self._run_context.enqueue(*content, priority='asap')
        return True

    async def prompt(self, text: str | None, *, images: Sequence[BinaryContent] = ()) -> AgentRunResult[OutputT]:
        """Execute the complete native agent loop, including tool calls."""
        if self._running:
            raise RuntimeError('A conversation can only run one prompt at a time')
        content: str | Sequence[UserContent] | None = (
            [*([text] if text is not None else []), *images] if images else text
        )
        submitted = [ModelRequest(parts=[UserPromptPart(content)])] if content is not None else []
        self._running = True
        self._accepting_steering = True
        with session_context(self.summary.id):
            try:
                previous = self._messages
                run_id = str(uuid4())
                candidate = replace(self.summary, run_id=run_id, owner_pid=os.getpid(), model=self.model)
                if self.conversations is not None:
                    if self.summary.revision == 0:
                        printable = ''.join(c for c in (text or '') if c.isprintable() or c.isspace())
                        title = ' '.join(printable.split())[:64]
                        candidate = replace(candidate, title=title or 'New session')
                    accepted: list[ModelMessage] = [*previous, *submitted]
                    self.summary = await self.conversations.save(
                        summary=replace(candidate, outcome='running'), messages=accepted
                    )
                    self._messages = accepted
                with (
                    capture_run_messages() as messages,
                    self.delegations.bind() if self.delegations is not None else nullcontext(),
                ):
                    try:
                        run_model, capabilities = self._bind()
                        if self.delegations is not None:
                            capabilities.append(
                                DelegationReports(
                                    self.delegations,
                                    conversation_id=self.summary.id,
                                    priority='asap' if text is None else 'when_idle',
                                )
                            )
                        workspace: Literal['new'] | None = None
                        configured = [*_agent_capabilities(self.agent), *self.plugins]
                        if (local := local_workspace(configured, self.workspace)) is not None:
                            capabilities.append(local)
                            if _stale_local_workspace(previous, self.workspace):
                                # A conversation resumed from another directory: work in this session's.
                                workspace = 'new'
                        result = await self.agent.run(
                            content,
                            deps=self.deps,
                            model=run_model,
                            model_settings=self._run_settings(),
                            # Run-level, not a capability: composing one more would split a plugin group that
                            # supplies the workspace into its members. A rebuilt stock agent has them bound already.
                            instructions=self._user_instructions if self.agent is self._base_agent else None,
                            retries={'tools': self.tool_retries} if self.tool_retries is not None else None,
                            message_history=previous,
                            conversation_id=self.summary.id,
                            run_id=run_id,
                            capabilities=capabilities,
                            workspace=workspace,
                            usage_limits=self.usage_limits,
                            event_stream_handler=self._stream,
                        )
                        self._accepting_steering = False
                        self._messages = result.all_messages()
                        await self._save_turn(outcome='completed')
                        return result
                    except get_cancelled_exc_class() as cancelled:
                        self._accepting_steering = False
                        # Core captures partial responses and tool results during cleanup.
                        # If cancellation precedes graph startup, retain at least the prompt.
                        self._messages = messages or [*previous, *submitted]
                        self._mark_interrupted()
                        try:
                            with move_on_after(5, shield=True):
                                await self._save_turn(outcome='cancelled')
                        except Exception as exc:  # noqa: BLE001 -- persistence failure must not swallow cancellation.
                            cancelled.add_note(f'Could not save cancelled turn: {exc}')
                            logging.getLogger(__name__).error('Could not save cancelled turn: %s', exc)
                        raise
                    except Exception as exc:
                        self._accepting_steering = False
                        self._report_setup_errors(exc)
                        if self.conversations is not None:
                            self._messages = messages or self._messages
                            self._mark_interrupted()
                            await self._save_turn(outcome='failed')
                        raise
            finally:
                self._accepting_steering = False
                self._run_context = None
                self._pending_steering.clear()
                self._running = False

    def _report_setup_errors(self, error: BaseException) -> None:
        """Tell `on_setup_error` about each setup failure from one of this session's guards; the turn still fails."""
        if self.on_setup_error is None:
            return
        for setup_error in setup_errors(error) or ():
            if raised_here(self.plugins, setup_error):
                self.on_setup_error(setup_error)

    async def _stream(self, ctx: RunContext[DepsT], events: AsyncIterable[AgentStreamEvent]) -> None:
        self._accepting_steering = True
        self._run_context = ctx
        for content in self._pending_steering:
            ctx.enqueue(*content, priority='asap')
        self._pending_steering.clear()
        if self.on_context_window is not None and (window := ctx.model.context_window):
            self.on_context_window(window)

        async def observed() -> AsyncIterable[AgentStreamEvent]:
            async for event in events:
                if self.on_context_usage is not None:
                    for message in reversed(ctx.messages):
                        if isinstance(message, ModelResponse) and message.usage.input_tokens:
                            self.on_context_usage(message.usage.total_tokens)
                            break
                if self.on_stream_event is not None:
                    await self.on_stream_event(event)
                yield event
            self._accepting_steering = False
            self._run_context = None

        # Preserve a supplied agent's handler instead of replacing its observers.
        handler = self.agent.event_stream_handler
        try:
            if handler is not None:
                await handler(ctx, observed())
            else:
                async for _ in observed():
                    pass
        finally:
            self._accepting_steering = False
            self._run_context = None
