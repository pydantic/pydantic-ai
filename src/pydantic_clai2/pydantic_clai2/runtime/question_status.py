"""Report open ask_user questions through the live-session contract."""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler, on_event
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.ask_user import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_clai2.plugins import LiveSession


@dataclass
class QuestionStatus(AbstractCapability[AgentDepsT]):
    """End each user wait on answer or run teardown, including cancelled runs."""

    session: LiveSession
    waits: dict[str | None, dict[str, Callable[[], None]]] = field(
        default_factory=dict[str | None, dict[str, Callable[[], None]]]
    )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        try:
            return await handler()
        finally:
            for finished in self.waits.pop(ctx.run_id, {}).values():
                finished()

    @on_event(AskUserRequestedEvent)
    async def _asked(self, ctx: RunContext[AgentDepsT], event: AskUserRequestedEvent) -> None:
        self.waits.setdefault(ctx.run_id, {})[event.request.id] = self.session.wait_for_user('an ask_user question')

    @on_event(AskUserAnsweredEvent)
    async def _answered(self, ctx: RunContext[AgentDepsT], event: AskUserAnsweredEvent) -> None:
        finished = self.waits.get(ctx.run_id, {}).pop(event.request_id, None)
        if finished is not None:
            finished()
