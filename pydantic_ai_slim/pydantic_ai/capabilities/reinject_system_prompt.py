from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext


# TODO (v2): consider making this the default behavior by adding `ReinjectSystemPrompt()` to
# every `Agent`'s default capabilities. Issue #1646 has been open since 2025-05 with community
# users repeatedly asking for this. Deferred to v2 because it changes the documented contract
# of `agent.run(message_history=[...])` — callers who deliberately omit the agent's system
# prompt (compaction pipelines, replay harnesses, OpenAI Responses with `previous_response_id`
# server-side memory) would see new content added to their requests.


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
    remain authoritative. Set `replace_existing=True` to instead strip any existing
    `SystemPromptPart`s before prepending the agent's configured prompt — useful when the
    history comes from an untrusted source (such as a UI frontend) and the server's prompt
    must win.

    The UI adapters automatically add this capability in `manage_system_prompt='server'` mode
    with `replace_existing=True`. Add it explicitly with
    `Agent(..., capabilities=[ReinjectSystemPrompt()])` or per-run via the `capabilities=`
    argument on [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] to get the same behavior
    anywhere.
    """

    replace_existing: bool = False
    """If `True`, strip any existing `SystemPromptPart`s from the history before prepending
    the agent's configured prompt. If `False` (the default), the capability is a no-op when
    any `SystemPromptPart` is already present."""

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        messages = request_context.messages
        if self.replace_existing:
            _strip_system_prompts(messages)
        elif _has_system_prompt(messages):
            return request_context
        if ctx.agent is None:
            return request_context  # pragma: no cover — ctx.agent is always set during an agent run
        sys_parts = await ctx.agent.system_prompt_parts(
            deps=ctx.deps,
            model=ctx.model,
            message_history=messages,
            prompt=ctx.prompt,
            usage=ctx.usage,
            model_settings=ctx.model_settings,
        )
        if sys_parts:
            _prepend_to_first_request(messages, sys_parts)
        return request_context


def _has_system_prompt(messages: list[ModelMessage]) -> bool:
    for msg in messages:
        if isinstance(msg, ModelRequest) and any(isinstance(p, SystemPromptPart) for p in msg.parts):
            return True
    return False


def _strip_system_prompts(messages: list[ModelMessage]) -> None:
    kept: list[ModelMessage] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            filtered_parts = [p for p in msg.parts if not isinstance(p, SystemPromptPart)]
            if not filtered_parts:
                continue
            if len(filtered_parts) != len(msg.parts):
                msg = replace(msg, parts=filtered_parts)
        kept.append(msg)
    messages[:] = kept


def _prepend_to_first_request(messages: list[ModelMessage], sys_parts: list[SystemPromptPart]) -> None:
    i, first_request = next((i, m) for i, m in enumerate(messages) if isinstance(m, ModelRequest))
    messages[i] = replace(first_request, parts=[*sys_parts, *first_request.parts])
