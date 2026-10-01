from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypeVar

from pydantic_graph import GraphRunContext
from pydantic_graph.basenode import NodeRunEndT

from .. import _system_prompt, exceptions, messages as _messages
from .._utils import dataclasses_no_defaults_repr
from ..tools import DeferredToolResult, DeferredToolResults, RunContext
from .graph import AgentNode
from .history import _clean_message_history, _repair_dangling_tool_calls, get_captured_run_messages
from .state import GraphAgentDeps, GraphAgentState, build_run_context

if TYPE_CHECKING:
    from .model_request import ModelRequestNode as ModelRequestNode
    from .model_response import CallToolsNode as CallToolsNode


DepsT = TypeVar('DepsT')


@dataclasses.dataclass
class UserPromptNode(AgentNode[DepsT, NodeRunEndT]):
    """The node that handles the user prompt and instructions."""

    user_prompt: str | Sequence[_messages.UserContent] | None

    _: dataclasses.KW_ONLY

    deferred_tool_results: DeferredToolResults | None = None

    instructions: str | None = None
    instructions_functions: list[_system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=list[_system_prompt.SystemPromptRunner[DepsT]]
    )

    system_prompts: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    system_prompt_functions: list[_system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=list[_system_prompt.SystemPromptRunner[DepsT]]
    )
    system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[DepsT]] = dataclasses.field(
        default_factory=dict[str, _system_prompt.SystemPromptRunner[DepsT]]
    )

    async def run(  # noqa: C901
        self, ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]]
    ) -> ModelRequestNode[DepsT, NodeRunEndT] | CallToolsNode[DepsT, NodeRunEndT]:
        from .model_request import ModelRequestNode, _get_instructions
        from .model_response import CallToolsNode

        try:
            ctx_messages = get_captured_run_messages()
        except LookupError:
            messages: list[_messages.ModelMessage] = []
        else:
            if ctx_messages.used:
                messages = []
            else:
                messages = ctx_messages.messages
                ctx_messages.used = True

        # Replace the `capture_run_messages` list with the message history
        messages[:] = _clean_message_history(ctx.state.message_history)
        # Use the `capture_run_messages` list as the message history so that new messages are added to it
        ctx.state.message_history = messages
        ctx.deps.new_message_index = len(messages)

        if self.deferred_tool_results is not None:
            return await self._handle_deferred_tool_results(self.deferred_tool_results, messages, ctx)

        messages[:] = _repair_interrupted_tail(messages, has_new_prompt=self.user_prompt is not None)

        next_message: _messages.ModelRequest | None = None
        is_resuming_without_prompt = False

        run_context: RunContext[DepsT] | None = None

        if messages and (last_message := messages[-1]):
            if isinstance(last_message, _messages.ModelRequest) and self.user_prompt is None:
                # Drop last message from history and reuse its parts
                messages.pop()
                next_message = _resumed_request(last_message)
                is_resuming_without_prompt = True

                if (prompt := _request_prompt(last_message)) is not None:
                    ctx.deps.prompt = prompt
            elif isinstance(last_message, _messages.ModelResponse):
                if last_message.state == 'suspended' and self.user_prompt is None:
                    # The history ends in a turn a provider paused mid-flight (Anthropic
                    # `pause_turn`, OpenAI background mode) and persisted. Resume it in
                    # `ModelRequestNode`, which re-issues the suspended turn and stitches the
                    # continuation into a single response with hooks firing once around the chain.
                    # `request` is an empty placeholder to satisfy `ModelRequestNode`'s dataclass:
                    # it is intentionally NOT appended to history (the suspended response is the real
                    # tail that gets echoed back), and nothing is sent for it. `_resume_suspended`
                    # drives `_prepare_resume_request` instead of the normal `_prepare_request` path.
                    return ModelRequestNode[DepsT, NodeRunEndT](
                        request=_messages.ModelRequest(parts=[]), _resume_suspended=last_message
                    )
                if self.user_prompt is None:
                    # The response may later be stamped with the run's workspace ref; don't mutate the caller's copy.
                    if ctx.deps.workspace.attached or ctx.deps.workspace_ref is not None:
                        last_message = replace(last_message)
                        messages[-1] = last_message
                        ctx.deps.adopted_response = last_message
                    # Align with the upcoming request step so we don't resolve dynamic toolsets twice.
                    run_context = replace(
                        build_run_context(ctx),
                        run_step=ctx.state.run_step + 1,
                        retry=ctx.state.output_retries_used,
                        max_retries=ctx.deps.tool_manager.default_max_retries,
                    )
                    ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)
                    if last_message.tool_calls:
                        # Pending tool calls must be processed before any new ModelRequest, regardless
                        # of instructions.  Instructions will be applied by ModelRequestNode.run() on
                        # the subsequent request after tool results are collected.
                        return CallToolsNode[DepsT, NodeRunEndT](last_message)
                    instruction_parts = await _get_instructions(ctx, run_context)
                    if not instruction_parts:
                        # No pending tool calls and no instructions — nothing new to send to the model.
                        return CallToolsNode[DepsT, NodeRunEndT](last_message)
                elif last_message.state == 'suspended':
                    # A new prompt on top of a suspended turn would abandon it, leaking the provider's
                    # server-side job (e.g. an OpenAI background run). Resume it first (run with this
                    # history and no new prompt) before starting a new turn.
                    raise exceptions.UserError(
                        'Cannot provide a new user prompt when the message history ends in a suspended response. '
                        'Resume it by running the agent with this message history and no new prompt.'
                    )
                elif last_message.tool_calls:
                    # An interrupted response's calls were already closed out by `_repair_interrupted_tail`.
                    raise exceptions.UserError(
                        'Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
                    )

        if not run_context:
            run_context = build_run_context(ctx)

        if messages:
            await self._reevaluate_dynamic_prompts(messages, run_context)

        if next_message:
            await self._reevaluate_dynamic_prompts([next_message], run_context)
        else:
            parts: list[_messages.ModelRequestPart] = []
            if not messages:
                parts.extend(await self._sys_parts(run_context))

            if self.user_prompt is not None:
                parts.append(_messages.UserPromptPart(self.user_prompt))

            next_message = _messages.ModelRequest(parts=parts)

        return ModelRequestNode[DepsT, NodeRunEndT](
            request=next_message, is_resuming_without_prompt=is_resuming_without_prompt
        )

    async def _handle_deferred_tool_results(
        self,
        deferred_tool_results: DeferredToolResults,
        messages: list[_messages.ModelMessage],
        ctx: GraphRunContext[GraphAgentState, GraphAgentDeps[DepsT, NodeRunEndT]],
    ) -> CallToolsNode[DepsT, NodeRunEndT]:
        from .model_response import CallToolsNode

        if not messages:
            raise exceptions.UserError('Tool call results were provided, but the message history is empty.')

        last_model_request: _messages.ModelRequest | None = None
        last_model_response: _messages.ModelResponse | None = None
        response_index: int | None = None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if isinstance(message, _messages.ModelRequest):
                last_model_request = message
            elif isinstance(message, _messages.ModelResponse):  # pragma: no branch
                last_model_response = message
                response_index = index
                break

        if not last_model_response:
            raise exceptions.UserError(
                'Tool call results were provided, but the message history does not contain a `ModelResponse`.'
            )
        if not last_model_response.tool_calls:
            raise exceptions.UserError(
                'Tool call results were provided, but the message history does not contain any unprocessed tool calls.'
            )

        assert response_index is not None
        last_model_response = replace(last_model_response)
        messages[response_index] = last_model_response

        tool_call_results: dict[str, DeferredToolResult | Literal['skip']] = {}
        tool_call_results.update(deferred_tool_results.to_tool_call_results())

        if last_model_request:
            for part in last_model_request.parts:
                if isinstance(part, _messages.ToolReturnPart | _messages.RetryPromptPart):
                    if part.tool_call_id in tool_call_results:
                        raise exceptions.UserError(
                            f'Tool call {part.tool_call_id!r} was already executed and its result cannot be overridden.'
                        )
                    tool_call_results[part.tool_call_id] = 'skip'

        # Skip ModelRequestNode and go directly to CallToolsNode
        return CallToolsNode[DepsT, NodeRunEndT](
            last_model_response,
            tool_call_results=tool_call_results,
            tool_call_metadata=deferred_tool_results.metadata or None,
            user_prompt=self.user_prompt,
        )

    async def _reevaluate_dynamic_prompts(
        self, messages: list[_messages.ModelMessage], run_context: RunContext[DepsT]
    ) -> None:
        """Reevaluate any `SystemPromptPart` with dynamic_ref in the provided messages by running the associated runner function."""
        # Only proceed if there's at least one dynamic runner.
        if self.system_prompt_dynamic_functions:
            for msg in messages:
                if isinstance(msg, _messages.ModelRequest):
                    reevaluated_message_parts: list[_messages.ModelRequestPart] = []
                    for part in msg.parts:
                        if isinstance(part, _messages.SystemPromptPart) and part.dynamic_ref:
                            # Look up the runner by its ref
                            if runner := self.system_prompt_dynamic_functions.get(  # pragma: lax no cover
                                part.dynamic_ref
                            ):
                                # To enable dynamic system prompt refs in future runs, use a placeholder string
                                updated_part_content = await runner.run(run_context)
                                part = _messages.SystemPromptPart(
                                    updated_part_content or '', dynamic_ref=part.dynamic_ref
                                )

                        reevaluated_message_parts.append(part)

                    # Replace message parts with reevaluated ones to prevent mutating parts list
                    if reevaluated_message_parts != msg.parts:
                        msg.parts = reevaluated_message_parts

    async def _sys_parts(self, run_context: RunContext[DepsT]) -> list[_messages.SystemPromptPart]:
        """Build the initial system-prompt messages for the conversation."""
        return await _system_prompt.resolve_system_prompts(
            self.system_prompts, self.system_prompt_functions, run_context
        )

    __repr__ = dataclasses_no_defaults_repr


def _repair_interrupted_tail(
    messages: list[_messages.ModelMessage], *, has_new_prompt: bool
) -> list[_messages.ModelMessage]:
    """Close out the tool calls that an interrupted end of the history leaves unanswered for good.

    A trailing request interrupted during tool execution means the last response's still-unanswered
    calls will never be executed. A response that was itself cut off (e.g. a cancelled stream) and is
    followed by a new prompt won't have its calls executed either. Both get synthesized returns. A
    'complete' trailing request (e.g. from a run that ended in `DeferredToolRequests`) is left alone:
    its response's open calls may still receive `deferred_tool_results`.
    """
    if not messages:
        return messages
    last_message = messages[-1]
    if (isinstance(last_message, _messages.ModelRequest) and last_message.state == 'interrupted') or (
        has_new_prompt
        and isinstance(last_message, _messages.ModelResponse)
        and last_message.state == 'interrupted'
        and last_message.tool_calls
    ):
        return _repair_dangling_tool_calls(messages, repair_last_response=True)
    return messages


def _resumed_request(request: _messages.ModelRequest) -> _messages.ModelRequest:
    """The request a run resuming from `request` without a new prompt sends, before its instructions are added."""
    return _messages.ModelRequest(
        parts=request.parts,
        run_id=request.run_id,
        conversation_id=request.conversation_id,
        metadata=request.metadata,
    )


def _request_prompt(request: _messages.ModelRequest) -> str | Sequence[_messages.UserContent] | None:
    """The user prompt a request carries, as a run resuming from it without a new prompt reports it."""
    user_prompt_parts = [part for part in request.parts if isinstance(part, _messages.UserPromptPart)]
    if not user_prompt_parts:
        return None
    if len(user_prompt_parts) == 1:
        return user_prompt_parts[0].content
    combined_content: list[_messages.UserContent] = []
    for part in user_prompt_parts:
        if isinstance(part.content, str):
            combined_content.append(part.content)
        else:
            combined_content.extend(part.content)
    return combined_content


def first_step_selection_messages(
    message_history: Sequence[_messages.ModelMessage] | None,
    user_prompt: str | Sequence[_messages.UserContent] | None,
    *,
    has_deferred_tool_results: bool = False,
) -> tuple[list[_messages.ModelMessage], str | Sequence[_messages.UserContent] | None]:
    """The `messages` and `prompt` a run's first-step `ModelSelectionContext` gets.

    The model is selected before `UserPromptNode` builds the first request, because building it
    needs the selected model. This previews what `RunContext.messages` and `RunContext.prompt` will
    hold when that request is sent, minus what depends on the model: the request's system prompt
    parts on a fresh run and its instructions. It shares `UserPromptNode`'s history cleanup and
    prompt extraction so the two can't drift.
    """
    messages = _clean_message_history(list(message_history or []))
    if has_deferred_tool_results:
        # The first request holds the results of tools that run with the selected model.
        return messages, user_prompt
    messages = _repair_interrupted_tail(messages, has_new_prompt=user_prompt is not None)
    if user_prompt is not None:
        return [*messages, _messages.ModelRequest(parts=[_messages.UserPromptPart(user_prompt)])], user_prompt
    last_message = messages[-1] if messages else None
    if isinstance(last_message, _messages.ModelRequest):
        # Resuming without a new prompt: the trailing request is the one being sent.
        return [*messages[:-1], _resumed_request(last_message)], _request_prompt(last_message)
    if isinstance(last_message, _messages.ModelResponse) and (
        last_message.tool_calls or last_message.state == 'suspended'
    ):
        # The step's request holds the results of tools that run with the selected model, or there is
        # none: a suspended response is resumed rather than answered.
        return messages, None
    # Without a new prompt, the request carries only what the selected model adds to it.
    return [*messages, _messages.ModelRequest(parts=[])], None
