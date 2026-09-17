"""Conversation state and run-scoped capability plugins."""

from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from typing import Generic, TypeVar

from anyio import get_cancelled_exc_class
from pydantic_ai import AgentRunResult, AgentStreamEvent, RunContext, capture_run_messages
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

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
        on_stream_event: Callable[[AgentStreamEvent], Awaitable[None]] | None = None,
    ) -> None:
        self.model: str | None = None
        self.model_settings: ModelSettings | None = None
        self.resolve_model: Callable[[str], Model | str | Awaitable[Model | str]] = lambda name: name
        self.agent = agent
        self.deps = deps
        self.plugins: Sequence[AgentCapability[DepsT]] = tuple(plugins)
        self.usage_limits = usage_limits
        self.on_stream_event = on_stream_event
        self._messages = list(message_history)
        self._running = False
        self.on_context_usage: Callable[[int], None] | None = None

    @property
    def messages(self) -> list[ModelMessage]:
        """Return a snapshot of the conversation's message list."""
        return list(self._messages)

    def clear(self) -> None:
        """Start a new conversation without replacing the agent or plugins."""
        self.replace_messages(())

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history, as `/compact` does after summarising it."""
        if self._running:
            raise RuntimeError('Cannot replace the history of a running conversation')
        self._messages = list(messages)

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
            model = await self.resolved_model()
            with capture_run_messages() as messages:
                try:
                    result = await self.agent.run(
                        text,
                        deps=self.deps,
                        model=model,
                        model_settings=self.model_settings,
                        message_history=self._messages,
                        capabilities=self.plugins,
                        usage_limits=self.usage_limits,
                        event_stream_handler=self._stream,
                    )
                    self._messages = result.all_messages()
                    return result
                except get_cancelled_exc_class():
                    # Core captures partial responses and tool results during cleanup.
                    # If cancellation precedes graph startup, retain at least the prompt.
                    self._messages = messages or [*self._messages, ModelRequest(parts=[UserPromptPart(text)])]
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
