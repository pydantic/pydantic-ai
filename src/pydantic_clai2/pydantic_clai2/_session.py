"""Conversation state and run-scoped capability plugins."""

import logging
import os
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from anyio import get_cancelled_exc_class, move_on_after
from pydantic_ai import AgentRunResult, AgentStreamEvent, RunContext, capture_run_messages
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepStore
from pydantic_ai_harness.step_persistence.conversations import (
    ConversationSummary,
    SqliteConversationStore,
    ensure_inactive,
)

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


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
            self.summary = saved.summary
            # Keep the caller's current model and approval configuration. Saved models are informational.
            return f'Resumed {saved.summary.title} ({saved.summary.id}).{warning}'
        finally:
            self._running = False

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

    async def prompt(self, text: str) -> AgentRunResult[OutputT]:
        """Execute the complete native agent loop, including tool calls."""
        if self._running:
            raise RuntimeError('A conversation can only run one prompt at a time')
        self._running = True
        try:
            previous = self._messages
            run_id = str(uuid4())
            candidate = replace(self.summary, run_id=run_id, owner_pid=os.getpid(), model=self.model)
            if self.conversations is not None:
                if self.summary.revision == 0:
                    title = ' '.join(''.join(c for c in text if c.isprintable() or c.isspace()).split())[:64]
                    candidate = replace(candidate, title=title or 'New session')
                accepted: list[ModelMessage] = [*previous, ModelRequest(parts=[UserPromptPart(text)])]
                self.summary = await self.conversations.save(
                    summary=replace(candidate, outcome='running'), messages=accepted
                )
                self._messages = accepted
            with capture_run_messages() as messages:
                try:
                    model = await self.resolved_model()
                    result = await self.agent.run(
                        text,
                        deps=self.deps,
                        model=model,
                        model_settings=self.model_settings,
                        retries={'tools': self.tool_retries} if self.tool_retries is not None else None,
                        message_history=previous,
                        conversation_id=self.summary.id,
                        run_id=run_id,
                        capabilities=self.plugins,
                        usage_limits=self.usage_limits,
                        event_stream_handler=self._stream,
                    )
                    self._messages = result.all_messages()
                    await self._save_turn(outcome='completed')
                    return result
                except get_cancelled_exc_class() as cancelled:
                    # Core captures partial responses and tool results during cleanup.
                    # If cancellation precedes graph startup, retain at least the prompt.
                    self._messages = messages or [*previous, ModelRequest(parts=[UserPromptPart(text)])]
                    try:
                        with move_on_after(5, shield=True):
                            await self._save_turn(outcome='cancelled')
                    except Exception as exc:  # noqa: BLE001 -- persistence failure must not swallow cancellation.
                        cancelled.add_note(f'Could not save cancelled turn: {exc}')
                        logging.getLogger(__name__).error('Could not save cancelled turn: %s', exc)
                    raise
                except Exception:
                    if self.conversations is not None:
                        self._messages = messages or self._messages
                        await self._save_turn(outcome='failed')
                    raise
        finally:
            self._running = False

    async def _stream(self, ctx: RunContext[DepsT], events: AsyncIterable[AgentStreamEvent]) -> None:
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

        # Preserve a supplied agent's handler instead of replacing its observers.
        handler = self.agent.event_stream_handler
        if handler is not None:
            await handler(ctx, observed())
        else:
            async for _ in observed():
                pass
