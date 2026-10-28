"""Prompt-cache prefix checks for conversations driven by `FunctionModel` and `TestModel`.

The cassette check in `cassette_utils` only sees requests that went over the wire, so the many tests that drive an
agent with `FunctionModel` or `TestModel` -- most of the Harness suite, and much of core -- are invisible to it. This
module records each request those models receive, renders it offline into the body `AnthropicModel` would send,
and runs the same prefix classification on consecutive requests of one conversation.
"""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pytest

from pydantic_ai._unsupported_settings import suppress_warnings
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters, override_allow_model_requests
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings, merge_model_settings

from .cassette_utils import (
    CassettePrefixViolation,
    PrefixBlock,
    anthropic_deferred_tool_blocks,
    canonical_prefix_blocks,
    classify_prefix_pair,
    moves_cache_prefix,
)

_ANTHROPIC_MESSAGES_URL = 'https://api.anthropic.com/v1/messages'

# Reasons for `@pytest.mark.moves_cache_prefix` shared by the tests whose conversations move the prefix on purpose.
# A test that moves it for a reason of its own gives that reason inline.

TOOLS_CHANGE_MID_RUN = (
    'the test changes the offered tools between steps (a `prepare`, filter, selector or dynamic toolset)'
)
INSTRUCTIONS_CHANGE_MID_RUN = 'the test changes the instructions between steps'
FALLBACK_REWINDS = 'a failed fallback continuation rewinds the history to before the response it abandoned'
COMPACTION_REWRITES_HISTORY = 'compaction or tool-result clearing rewrites earlier history by design'
REQUEST_SCOPED_PROMPT_DROPPED = (
    'a request-scoped prompt is sent with its own request only, so the next request drops it by design'
)
HOOK_REWRITES_REQUEST = (
    'a hook or history processor removes or rewrites parts of an earlier request, which the test exercises'
)


@dataclass
class RecordedRequest:
    """One request a `FunctionModel` or `TestModel` received, copied before the model or the run could change it."""

    model: Model
    """The model that received it, held so that two models in one test can't share an `id()`."""
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


@dataclass
class RequestRecorder:
    requests: list[RecordedRequest] = field(default_factory=list[RecordedRequest])

    def record(
        self,
        model: Model,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> None:
        try:
            # Anthropic renders the output mode the test model resolved for its own profile, the one the
            # conversation was actually run with (a `final_result` tool, say), rather than the one it would pick.
            # The rest of the preparation is left to `AnthropicModel`: preparing twice would render the instructions'
            # `<context>` blocks twice. The test model prepares the request again when it runs it, and warns about
            # ignored settings then.
            with suppress_warnings():
                _, prepared = model.prepare_request(model_settings, model_request_parameters)
            recorded = RecordedRequest(
                model,
                copy.deepcopy(messages),
                copy.deepcopy(merge_model_settings(model.settings, model_settings)),
                copy.deepcopy(replace(model_request_parameters, output_mode=prepared.output_mode)),
            )
        except UserError:
            # A request the test model rejects (a native tool it lacks, say) fails the run, and isn't checked.
            return
        self.requests.append(recorded)


def patch_models(monkeypatch: pytest.MonkeyPatch, recorder: RequestRecorder) -> None:
    """Record every request `FunctionModel` and `TestModel` receive, before they run it."""
    for model_class in (FunctionModel, TestModel):
        original_request = model_class.request
        original_request_stream = model_class.request_stream

        async def request(
            self: Any,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            _original: Any = original_request,
        ) -> Any:
            recorder.record(self, messages, model_settings, model_request_parameters)
            return await _original(self, messages, model_settings, model_request_parameters)

        def request_stream(
            self: Any,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: Any = None,
            _original: Any = original_request_stream,
        ) -> Any:
            recorder.record(self, messages, model_settings, model_request_parameters)
            return _original(self, messages, model_settings, model_request_parameters, run_context)

        monkeypatch.setattr(model_class, 'request', request)
        monkeypatch.setattr(model_class, 'request_stream', request_stream)


class _Captured(Exception):
    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body


class _CapturingAnthropicClient:
    """Stands in for `AsyncAnthropic`: hands the request body back instead of sending it."""

    base_url = 'https://api.anthropic.com'

    @property
    def timeout(self) -> Any:
        from anthropic import DEFAULT_TIMEOUT

        return DEFAULT_TIMEOUT

    @property
    def beta(self) -> Any:
        return self

    @property
    def messages(self) -> Any:
        return self

    async def create(self, *_args: Any, **kwargs: Any) -> Any:
        from anthropic import NOT_GIVEN, Omit

        raise _Captured(
            {key: value for key, value in kwargs.items() if value is not NOT_GIVEN and not isinstance(value, Omit)}
        )


def _render_anthropic_bodies(requests: list[RecordedRequest]) -> list[dict[str, Any] | None]:
    """The body `AnthropicModel` would send for each request, without `cache_control`, or `None` if it can't."""
    try:
        from anthropic import AsyncAnthropic

        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError:  # the slim test jobs don't install `anthropic`
        return [None] * len(requests)

    model = AnthropicModel(
        'claude-opus-5-5',
        provider=AnthropicProvider(anthropic_client=cast(AsyncAnthropic, _CapturingAnthropicClient())),
    )
    bodies: list[dict[str, Any] | None] = []
    for recorded in requests:
        with warnings.catch_warnings(), override_allow_model_requests(True):
            warnings.simplefilter('ignore')
            bodies.append(_render_anthropic_body(model, recorded))
    return bodies


def _render_anthropic_body(model: Model, recorded: RecordedRequest) -> dict[str, Any] | None:
    """Build the request body by stepping `model.request` until the stand-in client raises it.

    The coroutine is driven by hand rather than on an event loop: building the body awaits nothing, and a loop of
    our own would shut down whatever default executor the test's code gave it (DBOS gives it its own pool).
    """
    coroutine = model.request(recorded.messages, recorded.model_settings, recorded.model_request_parameters)
    try:
        coroutine.send(None)
    except _Captured as captured:
        return _without_cache_control(json.loads(json.dumps(captured.body, default=str)))
    except (UserError, NotImplementedError, RuntimeError, ValueError):
        # What a request built for a test model can hold that Anthropic doesn't take: a setting, output mode or
        # message part it rejects, a file type it can't read or infer, or a tool return value it can't serialize.
        return None
    # Building the body never waits on anything.
    coroutine.close()  # pragma: no cover
    raise AssertionError('`AnthropicModel.request` awaited something before sending the request')  # pragma: no cover


_DATA_KEYS = frozenset({'input', 'input_schema'})
"""Keys whose values are a tool call's arguments or a tool's schema: data that may hold a `cache_control` key."""


def _without_cache_control(value: Any) -> Any:
    """`value` without `cache_control` breakpoints: where one sits doesn't change what the cached prefix holds.

    Tool arguments and schemas are kept as they are, so a `cache_control` key in them still counts.
    """
    if isinstance(value, dict):
        return {
            key: item if key in _DATA_KEYS else _without_cache_control(item)
            for key, item in cast(dict[str, Any], value).items()
            if key != 'cache_control'
        }
    if isinstance(value, list):
        return [_without_cache_control(item) for item in cast(list[Any], value)]
    return value


def _continues(earlier: RecordedRequest, later: RecordedRequest) -> bool:
    """Whether `later` belongs to the conversation `earlier` was a request of, as far as its first message tells."""
    return (
        earlier.model is later.model
        and bool(earlier.messages)
        and bool(later.messages)
        and earlier.messages[0] == later.messages[0]
    )


def _run_id(recorded: RecordedRequest) -> str | None:
    """The run that sent `recorded`, from its latest message that names one (a paused response may not)."""
    return next((message.run_id for message in reversed(recorded.messages) if message.run_id is not None), None)


def _previous_request(requests: list[RecordedRequest], later_index: int) -> int | None:
    """The index of the request `requests[later_index]` follows in its conversation, if any.

    Within a run, that's the run's latest earlier request, so runs interleaved on one model (concurrent runs of the
    same prompt, say) are each compared with themselves, however they were scheduled. A run's first request continues
    the history of an earlier run: it follows the latest earlier request of its conversation that has fewer messages,
    as any request whose history it extends has. One with as many or more is another branch of the conversation
    (a run from a shorter `message_history`) or a concurrent run's later step.
    """
    later = requests[later_index]
    candidates = [index for index in range(later_index - 1, -1, -1) if _continues(requests[index], later)]
    if (run_id := _run_id(later)) is not None:
        same_run = next((index for index in candidates if _run_id(requests[index]) == run_id), None)
        if same_run is not None:
            return same_run
    return next((index for index in candidates if len(requests[index].messages) < len(later.messages)), None)


def _content_blocks(body: dict[str, Any]) -> list[PrefixBlock]:
    """An Anthropic request body's cache-ordered blocks, one per content block, the granularity Anthropic caches at.

    A request that adds content to the last message, like a prompt that joins the user message holding the
    previous step's tool results, or a continuation that adds parts to a paused response, extends the prefix.
    """
    canonical = canonical_prefix_blocks(body, _ANTHROPIC_MESSAGES_URL)
    assert canonical is not None  # the Messages API always has a known shape
    split: list[PrefixBlock] = []
    for level, block in canonical[1]:
        message: Any = json.loads(block)
        content = message.get('content') if level == 'messages' and is_str_dict(message) else None
        if isinstance(content, list):
            role = message.get('role')
            split.extend((level, json.dumps({'role': role, 'content': [item]})) for item in cast(list[Any], content))
        else:
            split.append((level, block))
    return split


def function_model_prefix_violations(requests: list[RecordedRequest]) -> list[CassettePrefixViolation]:
    """The prefix violations between each request and the request it follows in its conversation."""
    bodies = _render_anthropic_bodies(requests)
    violations: list[CassettePrefixViolation] = []
    for later_index, later_body in enumerate(bodies):
        earlier_index = _previous_request(requests, later_index)
        if later_body is None or earlier_index is None or (earlier_body := bodies[earlier_index]) is None:
            continue
        earlier, later = _content_blocks(earlier_body), _content_blocks(later_body)
        classification, block_index = classify_prefix_pair(earlier, later)
        earlier_deferred = anthropic_deferred_tool_blocks(earlier_body)
        later_deferred = anthropic_deferred_tool_blocks(later_body)
        if classification in ('identical', 'extension') and earlier_deferred != later_deferred[: len(earlier_deferred)]:
            violations.append(
                CassettePrefixViolation(
                    shape='anthropic',
                    pair_index=later_index,
                    level='deferred-tools',
                    block_index=0,
                    earlier_block=str(earlier_deferred)[:200],
                    later_block=str(later_deferred)[:200],
                )
            )
        if classification != 'shrunk' and not classification.endswith('-divergent'):
            continue
        violations.append(
            CassettePrefixViolation(
                shape='anthropic',
                pair_index=later_index,
                level=classification.removesuffix('-divergent'),
                block_index=block_index,
                earlier_block=(earlier[block_index][1] if block_index < len(earlier) else '<missing>')[:200],
                later_block=(later[block_index][1] if block_index < len(later) else '<missing>')[:200],
            )
        )
    return violations


def check_function_model_prefix_stability(node: pytest.Item, requests: list[RecordedRequest]) -> None:
    """Fail when a `FunctionModel` or `TestModel` conversation moves its prompt-cache prefix without an exemption."""
    if len(requests) < 2 or moves_cache_prefix(node):
        return
    fail_on_prefix_violations(function_model_prefix_violations(requests))


def fail_on_prefix_violations(violations: list[CassettePrefixViolation]) -> None:
    """Fail the test with a description of each violation, if there are any."""
    if not violations:
        return
    details = '\n'.join(
        f'request {violation.pair_index}, {violation.level} block {violation.block_index}:\n'
        f'  earlier: {violation.earlier_block}\n  later:   {violation.later_block}'
        for violation in violations
    )
    pytest.fail(
        f'{details}\nRendered as an Anthropic request, this test model conversation moves the prompt-cache prefix, '
        "which busts the provider's cache on every such turn; if the test's behavior is deliberately "
        'prefix-moving (compaction, dynamic tool disclosure, history rewriting), add '
        '@pytest.mark.moves_cache_prefix(reason=...) to the test'
    )
