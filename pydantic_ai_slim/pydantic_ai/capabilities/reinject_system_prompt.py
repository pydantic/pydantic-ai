from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, SystemPromptPart
from pydantic_ai.models import (
    Model,
    _standing_system_prompt_count,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.settings import ModelSettings


@dataclass
class ReinjectSystemPrompt(AbstractCapability[AgentDepsT]):
    """Capability that reinjects the agent's configured `system_prompt` when missing from history.

    Ensures the agent's configured `system_prompt` is present at the head of the first
    `ModelRequest` on every model request.

    Intended for callers that reconstruct a `message_history` from a source that doesn't
    round-trip system prompts — UI frontends, database persistence layers, conversation
    compaction pipelines. By default, if any `SystemPromptPart` is already present anywhere
    in the history (for example, preserved from a prior run or handed off from another
    agent), this capability leaves the messages untouched so that existing system prompts
    remain authoritative. Set `replace_existing=True` to instead strip the `SystemPromptPart`s
    the run started with before prepending the agent's configured prompt — useful when the
    history comes from an untrusted source (such as a UI frontend) and the server's prompt
    must win. `SystemPromptPart`s added during the run (by a capability, a history processor,
    or [`ctx.enqueue`][pydantic_ai.tools.RunContext.enqueue]) are kept.

    The UI adapters automatically add this capability in `manage_system_prompt='server'` mode
    with `replace_existing=True`. Add it explicitly with
    `Agent(..., capabilities=[ReinjectSystemPrompt()])` or per-run via the `capabilities=`
    argument on [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] to get the same behavior
    anywhere.
    """

    replace_existing: bool = False
    """If `True`, strip the `SystemPromptPart`s that were in the message history when the run
    started before prepending the agent's configured prompt, keeping those added during the run.
    If `False` (the default), the capability is a no-op when any `SystemPromptPart` is already
    present."""

    _: KW_ONLY

    id: str | None = 'reinject_system_prompt'
    """One-off: an agent reinjects the system prompt one way, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

    _history_parts: list[SystemPromptPart] = field(default_factory=list[SystemPromptPart], init=False, repr=False)
    """With `replace_existing=True`, the system prompt parts in the history this run started with."""

    _head: list[SystemPromptPart] | None = field(default=None, init=False, repr=False)
    """The parts this run last put at the head of the history, or `None` while the head is still the
    agent's own system prompt, opening the first request of a fresh conversation."""

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> ReinjectSystemPrompt[AgentDepsT]:
        if not self.replace_existing:
            return self
        # Only the history the run starts with can carry a system prompt the caller didn't author
        # (e.g. one a UI client sent), so that's what gets replaced. It can't tell a client's
        # prompt from one an earlier server-side run persisted, so it replaces both.
        run_capability = replace(self)
        run_capability._history_parts = [
            part
            for message in ctx.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        ]
        # With a history, the agent adds no system prompt, so the head only holds the parts
        # snapshotted above and any added since (e.g. a compaction summary), which are kept.
        run_capability._head = [] if ctx.messages else None
        return run_capability

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_messages = list(request_context.messages)
        persistent_messages = ctx.messages
        if (
            not self.replace_existing
            and _has_system_prompt(request_messages)
            and _has_system_prompt(persistent_messages)
        ):
            return request_context
        # `ctx.agent` is always set during an agent run.
        if ctx.agent is None:
            # `ctx.agent` is always set during an agent run.
            return request_context  # pragma: no cover
        # `ctx.model` is an `AbstractModel`, which is a regular `Model` on every path that makes
        # model requests; fall back to the agent's configured model otherwise. The `cast` only
        # pins `Model`'s client type parameter, which `isinstance` can't recover.
        ctx_model = ctx.model
        model = cast('Model[Any]', ctx_model) if isinstance(ctx_model, Model) else None
        replaceable: list[SystemPromptPart] = []
        if self.replace_existing:
            # Besides the parts the run started with, replace the system prompt at the head: the one
            # this capability put there on an earlier step or, on a fresh conversation, the agent's own.
            head = self._head if self._head is not None else _standing_system_prompt(persistent_messages)
            replaceable = [*self._history_parts, *head]
        sys_parts = await ctx.agent.system_prompt_parts(
            deps=ctx.deps,
            model=model,
            message_history=_without_parts(request_messages, replaceable)
            if self.replace_existing
            else request_messages,
            prompt=ctx.prompt,
            usage=ctx.usage,
            # This hook only runs in the classic request pipeline, where `ctx.model_settings`
            # never holds `RealtimeModelSettings`.
            model_settings=cast('ModelSettings | None', ctx.model_settings),
        )
        if self.replace_existing:
            # A head part matching the configured prompt is that prompt, e.g. as another
            # reinjector already prepended it this step, unlike a summary a capability put there.
            configured = {(part.content, part.dynamic_ref) for part in sys_parts}
            replaceable += [
                part
                for part in _standing_system_prompt(persistent_messages)
                if (part.content, part.dynamic_ref) in configured
            ]
            self._head = sys_parts
            request_messages = _without_parts(request_messages, replaceable)
            persistent_messages = _without_parts(persistent_messages, replaceable)
        request_context.messages = _reinject_system_prompt(
            request_messages, sys_parts, replace_existing=self.replace_existing
        )
        ctx.messages[:] = _reinject_system_prompt(
            persistent_messages, sys_parts, replace_existing=self.replace_existing
        )
        return request_context


def _has_system_prompt(messages: Sequence[ModelMessage]) -> bool:
    for msg in messages:
        if isinstance(msg, ModelRequest) and any(isinstance(p, SystemPromptPart) for p in msg.parts):
            return True
    return False


def _standing_system_prompt(messages: Sequence[ModelMessage]) -> list[SystemPromptPart]:
    # The history always holds the request about to be sent.
    first_request = next(m for m in messages if isinstance(m, ModelRequest))
    standing = first_request.parts[: _standing_system_prompt_count(first_request)]
    return [part for part in standing if isinstance(part, SystemPromptPart)]


def _without_parts(messages: Sequence[ModelMessage], parts: Sequence[SystemPromptPart]) -> list[ModelMessage]:
    """Drop `parts` from `messages`, and any `ModelRequest` left empty.

    A dynamic system prompt in the history is re-rendered as a new part before the first model
    request, so a part sharing one of `parts`' `dynamic_ref`s is dropped too.
    """
    dynamic_refs = {part.dynamic_ref for part in parts if part.dynamic_ref is not None}

    def drop(part: ModelRequestPart) -> bool:
        return isinstance(part, SystemPromptPart) and (part in parts or part.dynamic_ref in dynamic_refs)

    kept: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            filtered_parts = [p for p in msg.parts if not drop(p)]
            if not filtered_parts:
                continue
            if len(filtered_parts) != len(msg.parts):
                msg = replace(msg, parts=filtered_parts)
        kept.append(msg)
    return kept


def _reinject_system_prompt(
    messages: Sequence[ModelMessage],
    sys_parts: list[SystemPromptPart],
    *,
    replace_existing: bool,
) -> list[ModelMessage]:
    messages = list(messages)
    if not replace_existing and _has_system_prompt(messages):
        return messages
    if sys_parts:
        i, first_request = next((i, m) for i, m in enumerate(messages) if isinstance(m, ModelRequest))
        messages[i] = replace(first_request, parts=[*sys_parts, *first_request.parts])
    return messages
