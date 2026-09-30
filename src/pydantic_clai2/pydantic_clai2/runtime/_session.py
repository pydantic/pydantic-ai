"""Conversation state and run-scoped capability plugins."""

import logging
import os
import sys
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import replace
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast
from uuid import uuid4

from anyio import get_cancelled_exc_class, move_on_after

from pydantic_ai import AgentRunResult, AgentStreamEvent, RunContext, capture_run_messages
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentCapability,
    CombinedCapability,
    DynamicCapability,
    LocalWorkspace,
    WrapperCapability,
)
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, UserContent, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationSummary,
    SqliteConversationStore,
    ensure_inactive,
)

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


def _supports_local_workspace() -> bool:
    return sys.platform != 'win32'


def _supplies_workspace(plugins: Sequence[AgentCapability[DepsT]]) -> bool:
    """Whether a plugin, such as a sandbox, supplies the run's workspace, so clai adds no `LocalWorkspace`.

    A plugin counts when it has a leaf, loaded up front, that overrides `get_workspace` or is a
    capability function, whose capability is known only once the run starts.
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
    return any(not leaf.defer_loading and _overrides_get_workspace(leaf) for leaf in leaves)


def _overrides_get_workspace(leaf: AbstractCapability[DepsT]) -> bool:
    while isinstance(leaf, WrapperCapability):
        if type(leaf).get_workspace is not WrapperCapability.get_workspace:
            return True
        wrapped: list[AbstractCapability[DepsT]] = []
        leaf.wrapped.apply(wrapped.append)
        if len(wrapped) != 1 or wrapped[0] is not leaf.wrapped:
            # A wrapped tree's leaves are visited on their own; only a lone wrapped capability hides behind its wrapper.
            return False
        leaf = leaf.wrapped
    # A capability function's capability, and so its workspace, is known only once the run starts.
    return isinstance(leaf, DynamicCapability) or type(leaf).get_workspace is not AbstractCapability.get_workspace


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


def _stale_local_workspace(messages: Sequence[ModelMessage], workspace: str) -> bool:
    """Whether the history's latest response names a local directory other than `workspace`.

    `LocalWorkspace` declines such a reference, so continuing from it would leave the run without a
    workspace. A sandbox plugin's reference is not stale here: it continues in that sandbox.
    """
    response = next((message for message in reversed(messages) if isinstance(message, ModelResponse)), None)
    ref = response.workspace_ref if response is not None else None
    return ref is not None and ref.provider == 'local' and ref != WorkspaceRef(provider='local', id=workspace)


class Session(Generic[DepsT, OutputT]):
    """Run prompts to completion, retaining successful and interrupted turns in memory.

    Plugins are capabilities (or capability functions) bound per run, not to
    the agent itself, so the set can change between prompts.
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
        self.conversations = conversations
        self.workspace = str((workspace or Path.cwd()).resolve())
        self.summary = ConversationSummary(workspace=self.workspace)
        self.step_store: StepStore | None = (
            SqliteStepStore(database=conversations.database, max_snapshots_per_run=8) if conversations else None
        )
        self.model: str | None = None
        self.model_settings: ModelSettings | None = None
        self.tool_retries: int | None = None
        self.resolve_model: Callable[[str], Model | str | Awaitable[Model | str]] = lambda name: name
        self.agent = agent
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

    @property
    def messages(self) -> list[ModelMessage]:
        """Return a snapshot of the conversation's message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Start a new conversation without replacing the agent or plugins."""
        self.replace_messages(())
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
        """Restore a saved head without invoking the model or replaying tools."""
        if self._running:
            raise RuntimeError('Cannot resume during a running conversation')
        self._running = True
        try:
            if self.conversations is None:
                raise ValueError('Session persistence is not configured')
            saved = await self.conversations.get(conversation_id=conversation_id)
            if saved.summary.workspace != self.workspace and not allow_other_workspace:
                raise ValueError(f'Session belongs to {saved.summary.workspace}. Select it in /resume to confirm.')
            ensure_inactive(saved.summary)
            messages = saved.messages
            warning = ''
            if saved.summary.outcome in ('running', 'failed', 'cancelled'):
                warning = ' Interrupted session: inspect external effects before continuing. No tools were replayed.'
            if saved.summary.outcome == 'running' and saved.summary.run_id and self.step_store:
                snapshot = await self.step_store.latest_snapshot(run_id=saved.summary.run_id, include_interrupted=True)
                if snapshot is not None:
                    messages = snapshot.messages
            self._messages = list(messages)
            if saved.summary.outcome in ('running', 'failed', 'cancelled'):
                self._mark_interrupted()
            self.summary = saved.summary
            # Keep the caller's current model and approval configuration. Saved models are informational.
            return f'Resumed {saved.summary.title} ({saved.summary.id}).{warning}'
        finally:
            self._running = False

    def _mark_interrupted(self) -> None:
        # Let core close unanswered calls without replaying them on the next prompt.
        if self._messages:
            last = self._messages[-1]
            if not isinstance(last, ModelResponse) or last.state != 'suspended':
                self._messages[-1] = replace(last, state='interrupted')

    async def _save_turn(self, *, outcome: Literal['running', 'completed', 'failed', 'cancelled']) -> None:
        if self.conversations is None:
            return
        self.summary = await self.conversations.save(
            summary=replace(self.summary, outcome=outcome, model=self.model, owner_pid=None), messages=self._messages
        )

    async def resolved_model(self) -> Model | str | None:
        """The model the next run uses: the session's choice after `resolve_model`, else the agent's own."""
        if self.model is None:
            return self.agent.model
        model = self.resolve_model(self.model)
        return await model if isinstance(model, Awaitable) else model

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

    async def prompt(self, text: str, *, images: Sequence[BinaryContent] = ()) -> AgentRunResult[OutputT]:
        """Execute the complete native agent loop, including tool calls."""
        if self._running:
            raise RuntimeError('A conversation can only run one prompt at a time')
        content: str | Sequence[UserContent] = [text, *images] if images else text
        self._running = True
        self._accepting_steering = True
        try:
            previous = self._messages
            run_id = str(uuid4())
            candidate = replace(self.summary, run_id=run_id, owner_pid=os.getpid(), model=self.model)
            if self.conversations is not None:
                if self.summary.revision == 0:
                    title = ' '.join(''.join(c for c in text if c.isprintable() or c.isspace()).split())[:64]
                    candidate = replace(candidate, title=title or 'New session')
                accepted: list[ModelMessage] = [*previous, ModelRequest(parts=[UserPromptPart(content)])]
                self.summary = await self.conversations.save(
                    summary=replace(candidate, outcome='running'), messages=accepted
                )
                self._messages = accepted
            with capture_run_messages() as messages:
                try:
                    model = await self.resolved_model()
                    capabilities = list(self.plugins)
                    workspace: Literal['new'] | None = None
                    if _supports_local_workspace() and not _supplies_workspace(
                        [*_agent_capabilities(self.agent), *self.plugins]
                    ):
                        capabilities.append(LocalWorkspace[DepsT](self.workspace, env=_command_env()))
                        if _stale_local_workspace(previous, self.workspace):
                            # A conversation resumed from another directory: work in this session's.
                            workspace = 'new'
                    result = await self.agent.run(
                        content,
                        deps=self.deps,
                        model=model,
                        model_settings=self.model_settings,
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
                    self._messages = messages or [*previous, ModelRequest(parts=[UserPromptPart(content)])]
                    self._mark_interrupted()
                    try:
                        with move_on_after(5, shield=True):
                            await self._save_turn(outcome='cancelled')
                    except Exception as exc:
                        if sys.version_info >= (3, 11):  # `add_note` is 3.11+; the log below covers 3.10.
                            cancelled.add_note(f'Could not save cancelled turn: {exc}')
                        logging.getLogger(__name__).error('Could not save cancelled turn: %s', exc)
                    raise
                except Exception:
                    self._accepting_steering = False
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

    async def _stream(self, ctx: RunContext[DepsT], events: AsyncIterable[AgentStreamEvent]) -> None:
        self._accepting_steering = True
        self._run_context = ctx
        for content in self._pending_steering:
            ctx.enqueue(*content, priority='asap')
        self._pending_steering.clear()

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
