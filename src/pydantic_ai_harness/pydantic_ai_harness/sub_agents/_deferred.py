"""Pass a sub-agent's deferred tool calls to the delegating run's caller, and route the caller's answers back.

With `SubAgents(forward_deferred_calls=True)`, a child run that ends with `DeferredToolRequests` makes the delegate
call raise `ApprovalRequired`. Its history waits in the capability's `DeferredStateStore`, and the call's metadata
carries only the store's ID for it, the sub-agent's name and its pending calls. The caller sees one
`DeferredToolRequests` for the whole tree through `delegated_requests()`, with each child call's ID prefixed by the
delegate call's (`'<delegate call>:<child call>'`), and answers it in the usual way. `delegated_results()` turns those
answers into an approval of the delegate call whose metadata holds the child's own answers, so the delegate resumes
the child from its stored history instead of running the task again.
"""

from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Any, Protocol, TypeVar

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.tools import DeferredToolKind, DeferredToolRequests, DeferredToolResults, ToolApproved

STATE_KEY = 'sub_agent_run'
"""The key of the delegate call's metadata holding the child run's state."""

AGENT_PATH_KEY = 'agent_path'
"""The key of a child call's metadata in `delegated_requests()` naming the delegates it was made under, outermost first."""

_REQUESTS: TypeAdapter[DeferredToolRequests] = TypeAdapter(DeferredToolRequests)
_RESULTS: TypeAdapter[DeferredToolResults] = TypeAdapter(DeferredToolResults)

_V = TypeVar('_V')


class DeferredStateStore(Protocol):
    """Holds the state of a sub-agent whose deferred tool calls `SubAgents(forward_deferred_calls=True)` passed on.

    The state is an opaque blob holding the sub-agent's history and pending calls. It is saved under a new random ID
    when the delegation stops, and taken when the caller's answers arrive, so they resume it at most once. It stays
    on the server: the caller only sees its ID. Answers that name a state saved for another delegation save it back.
    """

    async def save(self, state_id: str, state: bytes) -> None:
        """Keep `state` under `state_id`."""
        ...

    async def take(self, state_id: str) -> bytes | None:
        """Remove the state saved under `state_id` and return it, or `None` if there is none.

        Make this atomic in a store shared by several processes, so two submissions of the same answers can't both
        resume the sub-agent and run its approved calls twice.
        """
        ...


class InMemoryDeferredStateStore:
    """A `DeferredStateStore` in this process's memory, the default.

    A resume must then reach the same store in the same process. It keeps at most `max_states` states, each for at
    most `max_age`, so the history of a delegation whose calls are never answered doesn't stay forever: a resume
    after that raises an error instead.
    """

    def __init__(self, *, max_states: int = 1000, max_age: timedelta = timedelta(days=1)) -> None:
        if max_states < 1:
            raise ValueError(f'`max_states` must be at least 1; got {max_states}.')
        if max_age <= timedelta(0):
            raise ValueError(f'`max_age` must be positive; got {max_age}.')
        self._max_states = max_states
        self._max_age = max_age.total_seconds()
        self._states: OrderedDict[str, tuple[float, bytes]] = OrderedDict()

    async def save(self, state_id: str, state: bytes) -> None:
        """Keep `state` under `state_id`, dropping expired states and then the oldest ones over `max_states`."""
        now = time.monotonic()
        while self._states and now - next(iter(self._states.values()))[0] >= self._max_age:
            self._states.popitem(last=False)
        self._states[state_id] = (now, state)
        while len(self._states) > self._max_states:
            self._states.popitem(last=False)

    async def take(self, state_id: str) -> bytes | None:
        """Remove the state saved under `state_id` and return it, or `None` if there is none or it expired."""
        saved_at, state = self._states.pop(state_id, (0.0, None))
        return state if time.monotonic() - saved_at < self._max_age else None


@dataclass(frozen=True)
class _StoredState:
    """What a `DeferredStateStore` keeps for a stopped delegation, serialized with `_STORED`."""

    agent_name: str
    task: str
    tool_call_id: str | None
    requests: DeferredToolRequests
    messages: list[ModelMessage]


_STORED: TypeAdapter[_StoredState] = TypeAdapter(_StoredState)


@dataclass(frozen=True)
class ResumedState:
    """A stored sub-agent state taken for a resume, with the caller's answers to its calls."""

    state_id: str
    messages: list[ModelMessage]
    results: DeferredToolResults


async def deferral_metadata(
    store: DeferredStateStore,
    *,
    agent_name: str,
    task: str,
    tool_call_id: str | None,
    requests: DeferredToolRequests,
    messages: Sequence[ModelMessage],
) -> dict[str, Any]:
    """Save a stopped child's state in `store`, and return the metadata its delegate call raises `ApprovalRequired` with.

    The metadata goes to the caller, so it holds only what the caller is asked about: the sub-agent's pending calls.
    """
    state_id = secrets.token_urlsafe(16)
    stored = _StoredState(agent_name, task, tool_call_id, requests, list(messages))
    await store.save(state_id, _STORED.dump_json(stored))
    return {
        STATE_KEY: {
            'state_id': state_id,
            'agent_name': agent_name,
            'requests': _REQUESTS.dump_python(requests, mode='json'),
        }
    }


def _state(metadata: object) -> dict[str, Any] | None:
    if is_str_dict(metadata) and is_str_dict(state := metadata.get(STATE_KEY)):
        return state
    return None


def _delegation(
    requests: DeferredToolRequests, call: ToolCallPart, tool_names: Collection[str]
) -> dict[str, Any] | None:
    """The forwarded state of a delegate call, `None` for any other call.

    Only a call of the delegate tool counts, so another tool that puts what looks like a state in its own approval
    metadata can't have its call hidden or approved through `delegated_results()`.
    """
    return _state(requests.metadata.get(call.tool_call_id)) if call.tool_name in tool_names else None


def _invalid(error: Exception) -> UserError:
    return UserError(
        'The sub-agent state passed back with `deferred_tool_results` is not valid. Build the results with '
        f'`delegated_results()` from the `DeferredToolRequests` the run ended with: {error}'
    )


def _child_requests(state: dict[str, Any]) -> tuple[str, DeferredToolRequests]:
    try:
        return str(state['agent_name']), _REQUESTS.validate_python(state['requests'])
    except (KeyError, ValidationError) as error:
        raise UserError(
            'A delegate call in these `DeferredToolRequests` has a forwarded sub-agent state that is not valid. Pass '
            f'the `DeferredToolRequests` the run ended with: {error}'
        ) from error


def _kinds(requests: DeferredToolRequests) -> dict[str, DeferredToolKind]:
    return {call.tool_call_id: requests.kind(call.tool_call_id) for call in requests.calls}


def _flat_ids(requests: DeferredToolRequests) -> list[str]:
    return [call.tool_call_id for call in (*requests.approvals, *requests.calls)]


def _flatten(
    requests: DeferredToolRequests, tool_names: Collection[str]
) -> tuple[DeferredToolRequests, dict[str, list[str]]]:
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
    # Drop the delegate calls' own entries first, so one's child call can't be dropped as another delegate call.
    delegated = {call.tool_call_id for call in requests.approvals if _delegation(requests, call, tool_names)}
    metadata = {key: value for key, value in metadata.items() if key not in delegated}
    validated_args = {key: value for key, value in validated_args.items() if key not in delegated}
    for call in requests.approvals:
        state = _delegation(requests, call, tool_names)
        if state is None:
            approvals.append(call)
            continue
        agent_name, child = _child_requests(state)
        child, child_paths = _flatten(child, tool_names)
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
    if repeated := sorted({tool_call_id for tool_call_id in ids if ids.count(tool_call_id) > 1}):
        raise ValueError(
            f'Delegated tool call IDs collide once prefixed with their delegate call ID: {repeated}. '
            'Give the tool calls IDs without `:`.'
        )


def delegated_requests(
    requests: DeferredToolRequests, *, tool_names: Collection[str] = ('delegate_task',)
) -> DeferredToolRequests:
    """The deferred tool calls of a run and of the sub-agents it delegated to, as one `DeferredToolRequests`.

    Each delegate call that a sub-agent's deferred calls stopped, with `SubAgents(forward_deferred_calls=True)`, is
    replaced by those calls, nested delegations included. A child call's ID is the delegate call's ID and its own,
    joined by `:`, and its metadata gets an `'agent_path'` entry listing the delegates it was made under, outermost
    first, so a UI can say which delegate asks. Other calls are kept as they are.

    Answer the calls with `build_results()` on the returned requests, then pass the answers through
    `delegated_results()` to resume the run. Only calls of the delegate tools named in `tool_names` are replaced, at
    every level, so list every name `SubAgents(tool_name=...)` gives one in the delegation tree.

    Raises:
        ValueError: Two calls end up with the same ID.
    """
    flat, paths = _flatten(requests, tool_names)
    for tool_call_id, path in paths.items():
        flat.metadata[tool_call_id] = {**flat.metadata.get(tool_call_id, {}), AGENT_PATH_KEY: path}
    return flat


def delegated_results(
    requests: DeferredToolRequests,
    results: DeferredToolResults,
    *,
    tool_names: Collection[str] = ('delegate_task',),
) -> DeferredToolResults:
    """The `deferred_tool_results` that resume a run with the answers to the calls of `delegated_requests(requests)`.

    `requests` is the `DeferredToolRequests` the run ended with, and `results` answers the calls of
    `delegated_requests(requests)`, keyed by their IDs. Each delegate call is approved with the child's state and
    answers in its metadata, so the delegate resumes the child from its history. Approving a delegate call some
    other way runs it again, but approves none of its child's calls. Pass the same `tool_names` as to
    `delegated_requests()`.

    Raises:
        ValueError: A child call has no answer, as a run resumes only once every call it waits on is answered, or
            two calls end up with the same ID.
    """
    _flatten(requests, tool_names)
    routed = DeferredToolResults(
        approvals=dict(results.approvals),
        calls=dict(results.calls),
        metadata=dict(results.metadata),
        answered_by=dict(results.answered_by),
    )
    for call in requests.approvals:
        state = _delegation(requests, call, tool_names)
        if state is None:
            continue
        _, child = _child_requests(state)
        prefix = f'{call.tool_call_id}:'
        child_ids = _flat_ids(_flatten(child, tool_names)[0])
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
            tool_names=tool_names,
        )
        for _, flat in own:
            for mapping in (routed.approvals, routed.calls, routed.metadata, routed.answered_by):
                mapping.pop(flat, None)
        routed.approvals[call.tool_call_id] = ToolApproved()
        routed.metadata[call.tool_call_id] = {
            STATE_KEY: {**state, 'results': _RESULTS.dump_python(child_results, mode='json')}
        }
    return routed


async def resume_state(
    store: DeferredStateStore, metadata: object, *, agent_name: str, task: str, tool_call_id: str | None
) -> ResumedState | None:
    """Take the stored state an approved delegate call names, with the caller's answers, to resume its child.

    `None` when the call was approved without a child's state, such as by an approval gate on the delegate tool.
    The answers come from the caller, so the state they name must be in `store`, saved by this same delegate call,
    and they may answer only the calls the child is waiting on. A state that doesn't match is saved back.
    """
    state = _state(metadata)
    if state is None:
        return None
    state_id = state.get('state_id')
    if not isinstance(state_id, str) or (stored := await store.take(state_id)) is None:
        raise UserError(
            'The sub-agent state named in `deferred_tool_results` is not in this `SubAgents` store: it was already '
            'resumed or expired, or was saved in another process or by another `SubAgents`. To resume in another '
            'process, give both the same `deferred_state_store`.'
        )
    try:
        return _resumed(state_id, stored, state, agent_name=agent_name, task=task, tool_call_id=tool_call_id)
    except UserError:
        await store.save(state_id, stored)
        raise


def _resumed(
    state_id: str,
    blob: bytes,
    state: dict[str, Any],
    *,
    agent_name: str,
    task: str,
    tool_call_id: str | None,
) -> ResumedState:
    stored = _STORED.validate_json(blob)
    if (stored.agent_name, stored.task, stored.tool_call_id) != (agent_name, task, tool_call_id):
        raise UserError('The sub-agent state named in `deferred_tool_results` was saved for another delegation.')
    try:
        results = _RESULTS.validate_python(state['results'])
    except (KeyError, ValidationError) as error:
        raise _invalid(error) from error
    requests = stored.requests
    for answered, asked in ((results.approvals, requests.approvals), (results.calls, requests.calls)):
        if unknown := sorted(set(answered) - {call.tool_call_id for call in asked}):
            raise UserError(f'The sub-agent was not waiting for these answers: {unknown}')
    return ResumedState(state_id, stored.messages, results)
