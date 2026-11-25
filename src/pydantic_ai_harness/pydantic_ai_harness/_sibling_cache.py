"""Start sibling sub-agent runs one at a time, so the ones after the first read the prompt prefix it caches.

Some providers, such as Anthropic, only serve a cached prefix to a request that arrives after another one has started
writing it, so sub-agents a fan-out starts at the same moment each write the whole shared prefix (tool definitions and
instructions) again. `SiblingCacheWarming` holds every sibling's first request until the first sibling's first request
has started streaming, which is when the provider has cached the prefix.

The first sibling never waits. Its first request is released for the others when it streams its first event, and also
when it fails or is cancelled first. A held sibling goes ahead on its own after `_RELEASE_TIMEOUT` seconds, so a slow
first sibling delays the others by at most that much.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Hashable
from contextlib import asynccontextmanager
from typing import Any, ClassVar, Literal, TypeAlias
from weakref import WeakValueDictionary

import anyio

from pydantic_ai._run_context import in_durable_unit
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import DurableRunContext
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

WarmSiblingCache: TypeAlias = bool | Literal['auto']
"""Whether a fan-out starts sibling sub-agents one at a time so they share the prompt cache.

- `'auto'`: when the model's profile says it pays off
  ([`prompt_cache_warms_siblings`][pydantic_ai.profiles.ModelProfile.prompt_cache_warms_siblings], True on Anthropic and
  on Claude via Bedrock) and the request has prompt caching configured.
- `True`: whenever several siblings start at once, whatever the model.
- `False`: never.
"""

_RELEASE_TIMEOUT = 10.0
"""How long, in seconds, a held sibling waits for the first sibling's first event before it goes ahead."""


class _Gate:
    """The siblings of one fan-out that share a prompt prefix: the first one through holds the others."""

    def __init__(self) -> None:
        self.released = anyio.Event()
        self.led = False


# A gate lives while a sibling run holds it, so a fan-out's gate goes away with its runs.
_gates: WeakValueDictionary[Hashable, _Gate] = WeakValueDictionary()


class SiblingCacheWarming(AbstractCapability[Any]):
    """Holds a sub-agent run's first model request until a sibling's first request has started streaming.

    A fan-out adds one to each sibling run it starts, with the same `scope` for the siblings that share a prompt
    prefix, such as the delegations to one sub-agent from one step of the parent run. Siblings on different models
    don't share a prefix, so they're held separately. Only the run's first request is held: later ones carry the
    sibling's own conversation, which no other sibling shares.
    """

    # It only times requests, so it can be added to a sub-agent run started at run time.
    _safe_at_runtime: ClassVar[bool] = True

    def __init__(self, scope: Hashable, mode: WarmSiblingCache) -> None:
        self.scope = scope
        self.mode: WarmSiblingCache = mode
        self.gate: _Gate | None = None
        self.started = False

    def get_wrapper_model(self, model: Model) -> Model | None:
        return None if self.started else _HeldModel(model, self)

    def applies(self, model: Model, model_settings: ModelSettings | None) -> bool:
        """Whether holding siblings can pay off for a request to `model` with these settings."""
        if self.mode != 'auto':
            return self.mode
        return (
            model.profile.get('prompt_cache_warms_siblings', False)
            and model.resolve_cache_retention(model_settings) is not None
        )

    def join(self, model: Model, model_settings: ModelSettings | None) -> tuple[_Gate, bool] | None:
        """Join the siblings with the run's first request: their gate and whether this run leads, or `None` to go ahead.

        The sibling that leads goes ahead now and must release the gate. The others wait on it.
        """
        if self.started:
            return None
        self.started = True
        if not self.applies(model, model_settings):
            return None
        key = (self.scope, model.system, model.model_name)
        gate = _gates.get(key)
        if gate is None:
            gate = _gates[key] = _Gate()
        # Held here too, so the gate outlives its first sibling while this one runs.
        self.gate = gate
        leads = not gate.led
        gate.led = True
        return gate, leads


class _HeldModel(WrapperModel):
    """Holds a sibling's first request behind the first sibling's, and releases the others from the first one's."""

    def __init__(self, wrapped: Model, warming: SiblingCacheWarming) -> None:
        super().__init__(wrapped)
        self.warming = warming

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        joined = self.warming.join(self.wrapped, model_settings)
        if joined is None or not joined[1]:
            if joined is not None:
                await _wait(joined[0])
            return await super().request(messages, model_settings, model_request_parameters)
        # The first sibling streams its request, so the others go ahead on its first event, not when it completes.
        released = joined[0].released
        try:
            async with self.wrapped.request_stream(messages, model_settings, model_request_parameters) as stream:
                stream._call_on_first_event(released.set)  # pyright: ignore[reportPrivateUsage]
                async for _ in stream:
                    pass
                return stream.get()
        finally:
            released.set()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        joined = self.warming.join(self.wrapped, model_settings)
        if joined is None or not joined[1]:
            if joined is not None:
                await _wait(joined[0])
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
            return
        released = joined[0].released
        try:
            async with self.wrapped.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                stream._call_on_first_event(released.set)  # pyright: ignore[reportPrivateUsage]
                yield stream
        finally:
            released.set()


async def _wait(gate: _Gate) -> None:
    with anyio.move_on_after(_RELEASE_TIMEOUT):
        await gate.released.wait()


def sibling_cache_warming(ctx: RunContext[Any], agent_name: str, mode: WarmSiblingCache) -> SiblingCacheWarming | None:
    """What to add to a run of `agent_name` that the parent run `ctx` starts, or `None` when siblings start together.

    The siblings are the runs of `agent_name` started from one step of the parent run, which share their first
    request's prefix. They start together under durable execution: the engine records each model request as a unit
    and numbers units in the order they start, so holding them behind an in-memory event that a replay doesn't see
    could change that order. A unit also runs in a worker that may not share the process, and so the gate, with its
    siblings.
    """
    # A `DurableRunContext` is the one a durable unit, like a Temporal activity, rebuilds from the parent's.
    if mode is False or isinstance(ctx, DurableRunContext) or ctx.in_durable_context or in_durable_unit():
        return None
    return SiblingCacheWarming((ctx.run_id, ctx.run_step, agent_name), mode)
