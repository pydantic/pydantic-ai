"""Pass a sub-agent's deferred tool calls to the delegating run's caller, and route the caller's answers back.

With `SubAgents(forward_deferred_calls=True)`, a child run that ends with `DeferredToolRequests` makes the delegate
call raise `ApprovalRequired`, carrying the child's requests and history in the call's metadata. The caller sees one
`DeferredToolRequests` for the whole tree through `delegated_requests()`, with each child call's ID prefixed by the
delegate call's (`'<delegate call>:<child call>'`), and answers it in the usual way. `delegated_results()` turns those
answers into an approval of the delegate call whose metadata holds the child's state and its own results, so the
delegate resumes the child from its history instead of running the task again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved

STATE_KEY = 'sub_agent_run'
"""The key of the delegate call's metadata holding the child run's state."""

AGENT_PATH_KEY = 'agent_path'
"""The key of a child call's metadata in `delegated_requests()` naming the delegates it was made under, outermost first."""

_REQUESTS: TypeAdapter[DeferredToolRequests] = TypeAdapter(DeferredToolRequests)
_RESULTS: TypeAdapter[DeferredToolResults] = TypeAdapter(DeferredToolResults)

_V = TypeVar('_V')


def deferral_metadata(
    agent_name: str, requests: DeferredToolRequests, messages: Sequence[ModelMessage]
) -> dict[str, Any]:
    """The metadata a delegate call raises `ApprovalRequired` with, in a form that survives JSON serialization."""
    return {
        STATE_KEY: {
            'agent_name': agent_name,
            'requests': _REQUESTS.dump_python(requests, mode='json'),
            'messages': ModelMessagesTypeAdapter.dump_python(list(messages), mode='json'),
        }
    }


def _state(metadata: object) -> dict[str, Any] | None:
    if is_str_dict(metadata) and is_str_dict(state := metadata.get(STATE_KEY)):
        return state
    return None


def _invalid(error: Exception) -> UserError:
    return UserError(
        'The sub-agent state passed back with `deferred_tool_results` is not valid. Build the results with '
        f'`delegated_results()` from the `DeferredToolRequests` the run ended with: {error}'
    )


def _child_requests(state: dict[str, Any]) -> tuple[str, DeferredToolRequests]:
    try:
        return str(state['agent_name']), _REQUESTS.validate_python(state['requests'])
    except (KeyError, ValidationError) as error:
        raise _invalid(error) from error


def _flatten(requests: DeferredToolRequests) -> tuple[DeferredToolRequests, dict[str, list[str]]]:
    """`requests` with each delegate call replaced by its child's calls, and the delegate path of each of those."""
    approvals: list[ToolCallPart] = []
    calls = list(requests.calls)
    metadata = dict(requests.metadata)
    validated_args = dict(requests.validated_args)
    paths: dict[str, list[str]] = {}
    for call in requests.approvals:
        state = _state(requests.metadata.get(call.tool_call_id))
        if state is None:
            approvals.append(call)
            continue
        agent_name, child = _child_requests(state)
        child, child_paths = _flatten(child)
        del metadata[call.tool_call_id]
        validated_args.pop(call.tool_call_id, None)
        prefix = f'{call.tool_call_id}:'
        approvals.extend(replace(part, tool_call_id=prefix + part.tool_call_id) for part in child.approvals)
        calls.extend(replace(part, tool_call_id=prefix + part.tool_call_id) for part in child.calls)
        metadata.update({prefix + key: value for key, value in child.metadata.items()})
        validated_args.update({prefix + key: value for key, value in child.validated_args.items()})
        paths.update(
            {
                prefix + part.tool_call_id: [agent_name, *child_paths.get(part.tool_call_id, [])]
                for part in (*child.approvals, *child.calls)
            }
        )
    flat = DeferredToolRequests(calls=calls, approvals=approvals, metadata=metadata, validated_args=validated_args)
    return flat, paths


def delegated_requests(requests: DeferredToolRequests) -> DeferredToolRequests:
    """The deferred tool calls of a run and of the sub-agents it delegated to, as one `DeferredToolRequests`.

    Each delegate call that a sub-agent's deferred calls stopped, with `SubAgents(forward_deferred_calls=True)`, is
    replaced by those calls, nested delegations included. A child call's ID is the delegate call's ID and its own,
    joined by `:`, and its metadata gets an `'agent_path'` entry listing the delegates it was made under, outermost
    first, so a UI can say which delegate asks. Other calls are kept as they are.

    Answer the calls with `build_results()` on the returned requests, then pass the answers through
    `delegated_results()` to resume the run.
    """
    flat, paths = _flatten(requests)
    for tool_call_id, path in paths.items():
        flat.metadata[tool_call_id] = {**flat.metadata.get(tool_call_id, {}), AGENT_PATH_KEY: path}
    return flat


def _own(mapping: Mapping[str, _V], prefix: str) -> dict[str, _V]:
    return {key.removeprefix(prefix): value for key, value in mapping.items() if key.startswith(prefix)}


def delegated_results(requests: DeferredToolRequests, results: DeferredToolResults) -> DeferredToolResults:
    """The `deferred_tool_results` that resume a run with the answers to the calls of `delegated_requests(requests)`.

    `requests` is the `DeferredToolRequests` the run ended with, and `results` answers the calls of
    `delegated_requests(requests)`, keyed by their IDs. Each delegate call is approved with the child's state and
    answers in its metadata, so the delegate resumes the child from its history. Approving a delegate call some
    other way runs it again, but approves none of its child's calls.

    Raises:
        ValueError: A child call has no answer. A run resumes only once every call it waits on is answered.
    """
    routed = DeferredToolResults(
        approvals=dict(results.approvals),
        calls=dict(results.calls),
        metadata=dict(results.metadata),
        answered_by=dict(results.answered_by),
    )
    for call in requests.approvals:
        state = _state(requests.metadata.get(call.tool_call_id))
        if state is None:
            continue
        _, child = _child_requests(state)
        prefix = f'{call.tool_call_id}:'
        child_results = delegated_results(
            child,
            DeferredToolResults(
                approvals=_own(results.approvals, prefix),
                calls=_own(results.calls, prefix),
                metadata=_own(results.metadata, prefix),
                answered_by=_own(results.answered_by, prefix),
            ),
        )
        for mapping in (routed.approvals, routed.calls, routed.metadata, routed.answered_by):
            for key in [key for key in mapping if key.startswith(prefix)]:
                del mapping[key]
        answered = {*child_results.approvals, *child_results.calls}
        if missing := [p.tool_call_id for p in (*child.approvals, *child.calls) if p.tool_call_id not in answered]:
            raise ValueError(
                f'`results` has no answer for the calls {[prefix + tool_call_id for tool_call_id in missing]}'
            )
        routed.approvals[call.tool_call_id] = ToolApproved()
        routed.metadata[call.tool_call_id] = {
            STATE_KEY: {**state, 'results': _RESULTS.dump_python(child_results, mode='json')}
        }
    return routed


def _first_prompt(messages: Sequence[ModelMessage]) -> object:
    return next(
        (
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ),
        None,
    )


def _check_answers(
    requests: DeferredToolRequests, messages: Sequence[ModelMessage], results: DeferredToolResults
) -> None:
    """Refuse an answer that isn't for a call the child is waiting on, with the tool and arguments it was asked about."""
    # A later call with a reused ID replaces the earlier one, and core refuses an answer for a call that already ran.
    made = {
        call.tool_call_id: call
        for message in messages
        if isinstance(message, ModelResponse)
        for call in message.tool_calls
    }
    for answered, asked in ((results.approvals, requests.approvals), (results.calls, requests.calls)):
        asked_by_id = {call.tool_call_id: call for call in asked}
        for tool_call_id in answered:
            call, pending = asked_by_id.get(tool_call_id), made.get(tool_call_id)
            if (
                call is None
                or pending is None
                or pending.tool_name != call.tool_name
                or pending.args_as_dict() != call.args_as_dict()
            ):
                raise UserError(
                    f'The answer for sub-agent tool call {tool_call_id!r} does not match the call it was asked about.'
                )


def resume_state(
    metadata: object, *, agent_name: str, task: str
) -> tuple[list[ModelMessage], DeferredToolResults] | None:
    """The history and answers to resume a child with, from the metadata of an approved delegate call.

    `None` when the call was approved without a child's state, such as by an approval gate on the delegate tool.
    The state comes from the caller, so it must belong to this delegation, and each answer to a call the child is
    waiting on, with the tool and arguments it asked about.
    """
    state = _state(metadata)
    if state is None:
        return None
    name, requests = _child_requests(state)
    try:
        messages = ModelMessagesTypeAdapter.validate_python(state['messages'])
        results = _RESULTS.validate_python(state['results'])
    except (KeyError, ValidationError) as error:
        raise _invalid(error) from error
    if name != agent_name or _first_prompt(messages) != task:
        raise UserError(
            'The sub-agent state passed back with `deferred_tool_results` was saved for another delegation.'
        )
    _check_answers(requests, messages, results)
    return messages, results
