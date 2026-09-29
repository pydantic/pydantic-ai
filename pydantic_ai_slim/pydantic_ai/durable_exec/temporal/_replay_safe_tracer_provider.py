from __future__ import annotations

from typing import Any

from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.trace import Tracer
from temporalio.contrib.opentelemetry import create_tracer_provider


class ReplaySafeSDKTracerProvider(SDKTracerProvider):
    """An SDK tracer provider that hands out Temporal's replay-safe tracers.

    Temporal's `ReplaySafeTracerProvider` suppresses spans while workflow history is replaying, but it
    isn't an `opentelemetry.sdk.trace.TracerProvider`. Logfire's tracer proxy only forwards
    `force_flush()`, `shutdown()`, `add_span_processor()` and `resource` to an SDK provider, so installing
    Temporal's provider there directly would make `logfire.force_flush()` silently skip span export.

    This provider is an SDK provider sharing Logfire's span processor, so those calls keep working, and
    only `get_tracer()` is delegated to a replay-safe provider built on the same processor.
    """

    def __init__(self, provider: SDKTracerProvider):
        # OpenTelemetry does not expose a span processor accessor. Replace this private access if Logfire
        # adds a public way to share its configured processor with another tracer provider.
        active_span_processor = provider._active_span_processor
        self._replay_safe_provider = create_tracer_provider(
            resource=provider.resource,
            sampler=provider.sampler,
            active_span_processor=active_span_processor,
            shutdown_on_exit=False,
        )
        super().__init__(
            resource=provider.resource,
            sampler=provider.sampler,
            active_span_processor=active_span_processor,
            shutdown_on_exit=False,
        )

    def get_tracer(self, *args: Any, **kwargs: Any) -> Tracer:
        return self._replay_safe_provider.get_tracer(*args, **kwargs)
