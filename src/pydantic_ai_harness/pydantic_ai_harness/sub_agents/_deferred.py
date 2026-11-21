"""Pass a sub-agent's deferred tool calls to the delegating run's caller, and route the caller's answers back.

With `SubAgents(forward_deferred_calls=True)`, a child run that ends with `DeferredToolRequests` makes the delegate
call raise `ApprovalRequired`, carrying the child's requests and history in the call's metadata, signed with the
capability's key. The caller sees one `DeferredToolRequests` for the whole tree through `delegated_requests()`, with
each child call's ID prefixed by the delegate call's (`'<delegate call>:<child call>'`), and answers it in the usual
way. `delegated_results()` turns those answers into an approval of the delegate call whose metadata holds the child's
state and its own answers, so the delegate resumes the child from its history instead of running the task again.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ToolCallPart
from pydantic_ai.tools import DeferredToolKind, DeferredToolRequests, DeferredToolResults, ToolApproved

STATE_KEY = 'sub_agent_run'
"""The key of the delegate call's metadata holding the child run's state."""

AGENT_PATH_KEY = 'agent_path'
"""The key of a child call's metadata in `delegated_requests()` naming the delegates it was made under, outermost first."""

_SIGNED_FIELDS = ('agent_name', 'task', 'requests', 'messages')

_REQUESTS: TypeAdapter[DeferredToolRequests] = TypeAdapter(DeferredToolRequests)
_RESULTS: TypeAdapter[DeferredToolResults] = TypeAdapter(DeferredToolResults)

_V = TypeVar('_V')


def _signature(key: bytes, state: dict[str, Any]) -> str:
    signed = json.dumps([state.get(name) for name in _SIGNED_FIELDS], sort_keys=True, separators=(',', ':'))
    return hmac.new(key, signed.encode(), hashlib.sha256).hexdigest()


def deferral_metadata(
    key: bytes, *, agent_name: str, task: str, requests: DeferredToolRequests, messages: Sequence[ModelMessage]
) -> dict[str, Any]:
    """The metadata a delegate call raises `ApprovalRequired` with, signed with `key`, in a JSON-serializable form."""
    state: dict[str, Any] = {
        'agent_name': agent_name,
        'task': task,
        'requests': _REQUESTS.dump_python(requests, mode='json'),
        'messages': ModelMessagesTypeAdapter.dump_python(list(messages), mode='json'),
    }
    state['signature'] = _signature(key, state)
    return {STATE_KEY: state}


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


def _kinds(requests: DeferredToolRequests) -> dict[str, DeferredToolKind]:
    return {call.tool_call_id: requests.kind(call.tool_call_id) for call in requests.calls}


def _flat_ids(requests: DeferredToolRequests) -> list[str]:
    return [call.tool_call_id for call in (*requests.approvals, *requests.calls)]


def _flatten(requests: DeferredToolRequests) -> tuple[DeferredToolRequests, dict[str, list[str]]]:
    """`requests` with each delegate call replaced by its child's calls, and the delegate path of each of those.

    Raises:
        ValueError: Two calls end up with the same ID, such as delegate call `a` with child call `b:c` and delegate
            call `a:b` with child call `c`.
    """
    approvals: list[ToolCallPart] = []
    calls = list(requests.calls)
    metadata = dict(requests.metadata)
    validated_args = dict(requests.validated_args)
    kinds = _kinds(requests)
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
        approvals.extend(_prefixed_calls(prefix, child.approvals))
        calls.extend(_prefixed_calls(prefix, child.calls))
        metadata.update(_prefixed(prefix, child.metadata))
        validated_args.update(_prefixed(prefix, child.validated_args))
        kinds.update(_prefixed(prefix, _kinds(child)))
        paths.update(_prefixed(prefix, {key: [agent_name, *child_paths.get(key, [])] for key in _flat_ids(child)}))
    flat = DeferredToolRequests(calls=calls, approvals=approvals, metadata=metadata, validated_args=validated_args)
    _require_unique(_flat_ids(flat))
    questions = {key: kind for key, kind in kinds.items() if kind == 'question'}
    if not questions:
        return flat, paths
    # Questions keep their kind, so `kind()` reads it as on the requests they came from. `DeferredToolRequests` takes a
    # call's kind only through its serialized form, which stored requests rely on to keep their questions.
    return _REQUESTS.validate_python({**_REQUESTS.dump_python(flat), '_kinds': questions}), paths


def _prefixed(prefix: str, mapping: Mapping[str, _V]) -> dict[str, _V]:
    return {prefix + key: value for key, value in mapping.items()}


def _prefixed_calls(prefix: str, calls: Sequence[ToolCallPart]) -> list[ToolCallPart]:
    return [replace(call, tool_call_id=prefix + call.tool_call_id) for call in calls]


def _require_unique(ids: Sequence[str]) -> None:
    if len(set(ids)) != len(ids):
        raise ValueError(
            f'Delegated tool call IDs collide once prefixed with their delegate call ID: {sorted(ids)}. '
            'Give the tool calls IDs without `:`.'
        )


def delegated_requests(requests: DeferredToolRequests) -> DeferredToolRequests:
    """The deferred tool calls of a run and of the sub-agents it delegated to, as one `DeferredToolRequests`.

    Each delegate call that a sub-agent's deferred calls stopped, with `SubAgents(forward_deferred_calls=True)`, is
    replaced by those calls, nested delegations included. A child call's ID is the delegate call's ID and its own,
    joined by `:`, and its metadata gets an `'agent_path'` entry listing the delegates it was made under, outermost
    first, so a UI can say which delegate asks. Other calls are kept as they are.

    Answer the calls with `build_results()` on the returned requests, then pass the answers through
    `delegated_results()` to resume the run.

    Raises:
        ValueError: Two calls end up with the same ID.
    """
    flat, paths = _flatten(requests)
    for tool_call_id, path in paths.items():
        flat.metadata[tool_call_id] = {**flat.metadata.get(tool_call_id, {}), AGENT_PATH_KEY: path}
    return flat


def delegated_results(requests: DeferredToolRequests, results: DeferredToolResults) -> DeferredToolResults:
    """The `deferred_tool_results` that resume a run with the answers to the calls of `delegated_requests(requests)`.

    `requests` is the `DeferredToolRequests` the run ended with, and `results` answers the calls of
    `delegated_requests(requests)`, keyed by their IDs. Each delegate call is approved with the child's state and
    answers in its metadata, so the delegate resumes the child from its history. Approving a delegate call some
    other way runs it again, but approves none of its child's calls.

    Raises:
        ValueError: A child call has no answer, as a run resumes only once every call it waits on is answered, or
            two calls end up with the same ID.
    """
    _flatten(requests)
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
        child_ids = _flat_ids(_flatten(child)[0])
        if missing := [prefix + key for key in child_ids if prefix + key not in {*results.approvals, *results.calls}]:
            raise ValueError(f'`results` has no answer for the calls {missing}')
        own = [(key, prefix + key) for key in child_ids]
        child_results = delegated_results(
            child,
            DeferredToolResults(
                approvals={key: results.approvals[flat] for key, flat in own if flat in results.approvals},
                calls={key: results.calls[flat] for key, flat in own if flat in results.calls},
                metadata={key: results.metadata[flat] for key, flat in own if flat in results.metadata},
                answered_by={key: results.answered_by[flat] for key, flat in own if flat in results.answered_by},
            ),
        )
        for _, flat in own:
            for mapping in (routed.approvals, routed.calls, routed.metadata, routed.answered_by):
                mapping.pop(flat, None)
        routed.approvals[call.tool_call_id] = ToolApproved()
        routed.metadata[call.tool_call_id] = {
            STATE_KEY: {**state, 'results': _RESULTS.dump_python(child_results, mode='json')}
        }
    return routed


def resume_state(
    key: bytes, metadata: object, *, agent_name: str, task: str
) -> tuple[list[ModelMessage], DeferredToolResults] | None:
    """The history and answers to resume a child with, from the metadata of an approved delegate call.

    `None` when the call was approved without a child's state, such as by an approval gate on the delegate tool.
    The state comes back from the caller, so it must carry this capability's signature, belong to this delegation,
    and answer only the calls the child asked about, each the way it asked.
    """
    state = _state(metadata)
    if state is None:
        return None
    if not hmac.compare_digest(str(state.get('signature')), _signature(key, state)):
        raise UserError(
            'The sub-agent state passed back with `deferred_tool_results` was not created by this `SubAgents`: '
            'pass back the state from the `DeferredToolRequests` the run ended with, through `delegated_results()`.'
        )
    if state.get('agent_name') != agent_name or state.get('task') != task:
        raise UserError(
            'The sub-agent state passed back with `deferred_tool_results` was saved for another delegation.'
        )
    _, requests = _child_requests(state)
    try:
        messages = ModelMessagesTypeAdapter.validate_python(state['messages'])
        results = _RESULTS.validate_python(state['results'])
    except (KeyError, ValidationError) as error:
        raise _invalid(error) from error
    for answered, asked in ((results.approvals, requests.approvals), (results.calls, requests.calls)):
        if unknown := sorted(set(answered) - {call.tool_call_id for call in asked}):
            raise UserError(f'The sub-agent was not waiting for these answers: {unknown}')
    return messages, results
