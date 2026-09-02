from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING

from temporalio.plugin import SimplePlugin
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient

if TYPE_CHECKING:
    from logfire import Logfire


def _default_setup_logfire() -> Logfire:
    import logfire

    instance = logfire.DEFAULT_LOGFIRE_INSTANCE
    # `logfire.configure()` is a reset, not an additive call: it re-derives every unspecified argument
    # from the environment and shuts down the existing tracer provider, so calling it unconditionally on
    # every `Client.connect()` would silently discard the host's own configuration (scrubbing patterns,
    # console settings, additional span processors, service name, sampling). Only configure if the host
    # hasn't already. Logfire exposes no public way to ask whether it's been configured; replace this
    # with a public accessor (e.g. `is_configured()`) if one is added.
    if not instance.config._initialized:  # pyright: ignore[reportPrivateUsage]
        instance = logfire.configure()
    from pydantic_ai import Agent

    # `instrument_pydantic_ai()` is likewise a replace, not a merge: with no arguments it builds a
    # default `InstrumentationSettings` and assigns it to the process-wide `Agent._instrument_default`.
    # Calling it unconditionally would turn a host's deliberate `include_content=False` back on, putting
    # prompts, completions and tool call results on exported spans. Only instrument if the host hasn't.
    # `False` is the "never instrumented" sentinel, so a host that explicitly called
    # `Agent.instrument_all(False)` is indistinguishable from one that never called it and is still
    # instrumented here; telling those apart would need a separate sentinel.
    if Agent._instrument_default is False:  # pyright: ignore[reportPrivateUsage]
        instance.instrument_pydantic_ai()
    return instance


class LogfirePlugin(SimplePlugin):
    """Temporal client plugin for Logfire.

    Args:
        setup_logfire: Function that configures and returns a Logfire instance.
        metrics: Whether to send Temporal metrics to Logfire.
        metric_periodicity: How often to export Temporal metrics. Defaults to 60 seconds.
    """

    def __init__(
        self,
        setup_logfire: Callable[[], Logfire] = _default_setup_logfire,
        *,
        metrics: bool = True,
        metric_periodicity: timedelta = timedelta(seconds=60),
    ) -> None:
        try:
            import logfire  # noqa: F401 # pyright: ignore[reportUnusedImport]
            from opentelemetry.trace import get_tracer
            from temporalio.contrib.opentelemetry import TracingInterceptor
        except ImportError as _import_error:
            raise ImportError(
                'Please install the `logfire` package to use the Logfire plugin, '
                'you can use the `logfire` optional group — `pip install "pydantic-ai-slim[logfire]"`'
            ) from _import_error

        self.setup_logfire = setup_logfire
        self.metrics = metrics
        self.metric_periodicity = metric_periodicity

        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='LogfirePlugin',
            interceptors=[TracingInterceptor(get_tracer('temporalio'))],
        )

    async def connect_service_client(
        self, config: ConnectConfig, next: Callable[[ConnectConfig], Awaitable[ServiceClient]]
    ) -> ServiceClient:
        logfire = self.setup_logfire()

        if self.metrics:
            logfire_config = logfire.config
            token = logfire_config.token
            if logfire_config.send_to_logfire and isinstance(token, str) and logfire_config.metrics is not False:
                base_url = logfire_config.advanced.generate_base_url(token)
                metrics_url = base_url + '/v1/metrics'
                headers = {'Authorization': f'Bearer {token}'}

                config.runtime = Runtime(
                    telemetry=TelemetryConfig(
                        metrics=OpenTelemetryConfig(
                            url=metrics_url,
                            headers=headers,
                            metric_periodicity=self.metric_periodicity,
                        )
                    )
                )

        return await next(config)
