"""Expose questions awaiting a user in the process's live session registry."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler, on_event
from pydantic_ai.run import AgentRunResult
from pydantic_ai_harness.ask_user import AskUserAnsweredEvent, AskUserRequestedEvent
from pydantic_clai2.plugins import NoSettings, Plugin, PluginHost
from pydantic_clai2.runtime.live_runtimes import LiveRuntimes


@dataclass
class _Questions(AbstractCapability[None]):
    """Record live requests by run so a cancelled turn cannot leave a stale waiting state."""

    host: PluginHost[None]
    requests: dict[str, set[str]] = field(default_factory=dict[str, set[str]])

    def _update(self, conversation_id: str, request_id: str, *, answered: bool = False) -> None:
        live = self.host.live
        if isinstance(live, LiveRuntimes):
            for session in live.listing():
                if session.conversation.conversation_id == conversation_id:
                    live.registry.question(session.id, request_id, answered=answered)
                    break

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[object]:
        if ctx.run_id is None or ctx.conversation_id is None:
            return await handler()
        try:
            return await handler()
        finally:
            for request_id in self.requests.pop(ctx.run_id, set()):
                self._update(ctx.conversation_id, request_id, answered=True)

    @on_event(AskUserRequestedEvent)
    async def _asked(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        if ctx.run_id is None or ctx.conversation_id is None:
            return
        self.requests.setdefault(ctx.run_id, set()).add(event.request.id)
        self._update(ctx.conversation_id, event.request.id)

    @on_event(AskUserAnsweredEvent)
    async def _answered(self, ctx: RunContext[None], event: AskUserAnsweredEvent) -> None:
        if ctx.run_id is None or ctx.conversation_id is None:
            return
        self.requests.get(ctx.run_id, set()).discard(event.request_id)
        self._update(ctx.conversation_id, event.request_id, answered=True)


class LiveStatusPlugin(Plugin):
    """Keep registry question status current while a session runs."""

    scope = 'process'

    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self.questions = _Questions(host)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self.questions,)
