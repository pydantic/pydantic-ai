"""Prompt-cache prefix checks for conversations driven by `FunctionModel` and `TestModel`.

The cassette check in `cassette_utils` only sees requests that went over the wire, so the many tests that drive an
agent with `FunctionModel` or `TestModel` -- most of the Harness suite, and much of core -- are invisible to it. This
module records each request those models receive, renders it offline into the body `AnthropicModel` would send,
and runs the same prefix classification on consecutive requests of one conversation.
"""

from __future__ import annotations

import asyncio
import copy
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestParameters, override_allow_model_requests
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from .cassette_utils import (
    CassettePrefixViolation,
    anthropic_deferred_tool_blocks,
    canonical_prefix_blocks,
    classify_prefix_pair,
)

_ANTHROPIC_MESSAGES_URL = 'https://api.anthropic.com/v1/messages'


@dataclass
class RecordedRequest:
    """One request a `FunctionModel` or `TestModel` received, copied before the model or the run could change it."""

    model_id: int
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


@dataclass
class RequestRecorder:
    requests: list[RecordedRequest] = field(default_factory=list[RecordedRequest])

    def record(
        self,
        model: object,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> None:
        try:
            # Resolved for the test model's own profile, so Anthropic renders the output mode the conversation
            # was actually run with (a `final_result` tool, say) rather than the one it would have picked.
            model_settings, model_request_parameters = cast(Model, model).prepare_request(
                model_settings, model_request_parameters
            )
            recorded = RecordedRequest(
                id(model),
                copy.deepcopy(messages),
                copy.deepcopy(model_settings),
                copy.deepcopy(model_request_parameters),
            )
        except Exception:  # pragma: no cover - a part holding an uncopyable value just isn't checked
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


async def _render_anthropic_bodies(requests: list[RecordedRequest]) -> list[dict[str, Any] | None]:
    """The body `AnthropicModel` would send for each request, or `None` where it can't render it."""
    try:
        from anthropic import AsyncAnthropic

        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError:  # pragma: no cover - the slim test jobs don't install `anthropic`
        return [None] * len(requests)

    model = AnthropicModel(
        'claude-opus-5-5',
        provider=AnthropicProvider(anthropic_client=cast(AsyncAnthropic, _CapturingAnthropicClient())),
    )
    bodies: list[dict[str, Any] | None] = []
    for recorded in requests:
        body: dict[str, Any] | None = None
        with warnings.catch_warnings(), override_allow_model_requests(True):
            warnings.simplefilter('ignore')
            try:
                await model.request(recorded.messages, recorded.model_settings, recorded.model_request_parameters)
            except _Captured as captured:
                body = json.loads(json.dumps(captured.body, default=str))
            except Exception:
                # A request built for a test model's profile that Anthropic can't take (a setting it rejects,
                # an output mode it lacks) has no Anthropic body to compare.
                body = None
        bodies.append(body)
    return bodies


def _continues(earlier: RecordedRequest, later: RecordedRequest) -> bool:
    """Whether `later` belongs to the conversation `earlier` was a request of, as far as its first message tells."""
    return (
        earlier.model_id == later.model_id
        and bool(earlier.messages)
        and bool(later.messages)
        and earlier.messages[0] == later.messages[0]
    )


async def function_model_prefix_violations(requests: list[RecordedRequest]) -> list[CassettePrefixViolation]:
    """Yield prefix violations between each request and the latest earlier request of its conversation."""
    bodies = await _render_anthropic_bodies(requests)
    blocks = [None if body is None else canonical_prefix_blocks(body, _ANTHROPIC_MESSAGES_URL) for body in bodies]
    violations: list[CassettePrefixViolation] = []
    for later_index, later in enumerate(requests):
        later_blocks = blocks[later_index]
        if later_blocks is None:
            continue
        earlier_index = next(
            (index for index in range(later_index - 1, -1, -1) if _continues(requests[index], later)), None
        )
        if earlier_index is None or (earlier_blocks := blocks[earlier_index]) is None:
            continue
        earlier_body, later_body = bodies[earlier_index], bodies[later_index]
        assert earlier_body is not None and later_body is not None
        classification, block_index = classify_prefix_pair(earlier_blocks[1], later_blocks[1])
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
        earlier, later_ = earlier_blocks[1], later_blocks[1]
        violations.append(
            CassettePrefixViolation(
                shape='anthropic',
                pair_index=later_index,
                level=classification.removesuffix('-divergent'),
                block_index=block_index,
                earlier_block=(earlier[block_index][1] if block_index < len(earlier) else '<missing>')[:200],
                later_block=(later_[block_index][1] if block_index < len(later_) else '<missing>')[:200],
            )
        )
    return violations


def check_function_model_prefix_stability(node: pytest.Item, requests: list[RecordedRequest]) -> None:
    """Fail when a `FunctionModel` or `TestModel` conversation moves its prompt-cache prefix without an exemption."""
    if len(requests) < 2 or node.get_closest_marker('moves_cache_prefix') is not None:
        # `check_cache_prefix_stability` already insists the marker gives a reason.
        return
    violations = asyncio.run(function_model_prefix_violations(requests))
    if violations:
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
