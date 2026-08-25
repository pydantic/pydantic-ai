from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from temporalio.plugin import SimplePlugin
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient

if TYPE_CHECKING:
    from logfire import Logfire
    from opentelemetry.trace import TracerProvider
    from temporalio.client import ClientConfig
    from temporalio.worker import ReplayerConfig, WorkerConfig


def _get_logfire() -> Logfire:
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
    return instance


def _setup_replay_safe_logfire() -> tuple[Logfire, TracerProvider]:
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
    from temporalio.contrib.opentelemetry import create_tracer_provider

    instance = _get_logfire()
    logfire_tracer_provider = instance.config.get_tracer_provider().provider
    assert isinstance(logfire_tracer_provider, SDKTracerProvider)
    # OpenTelemetry does not expose a span processor accessor. Replace this private access if Logfire
    # adds a public way to share its configured processor with another tracer provider.
    tracer_provider = create_tracer_provider(
        resource=logfire_tracer_provider.resource,
        sampler=logfire_tracer_provider.sampler,
        active_span_processor=logfire_tracer_provider._active_span_processor,  # pyright: ignore[reportPrivateUsage]
        shutdown_on_exit=False,
    )
    from pydantic_ai import Agent, __version__
    from pydantic_ai.models.instrumented import InstrumentationSettings

    host_settings = Agent._instrument_default  # pyright: ignore[reportPrivateUsage]
    if isinstance(host_settings, InstrumentationSettings):
        # `instrument_pydantic_ai()` replaces rather than merges the process-wide settings. Copy the host's
        # settings so its privacy and version choices survive while only the tracer becomes replay-safe.
        # `dataclasses.replace()` cannot do this because `InstrumentationSettings.__init__()` collapses its
        # `tracer_provider` argument into `tracer`, which is not itself an init parameter.
        settings = copy.copy(host_settings)
        settings.tracer = tracer_provider.get_tracer('pydantic-ai', __version__)
        Agent.instrument_all(settings)
    else:
        instance.instrument_pydantic_ai(tracer_provider=tracer_provider)
    return instance, tracer_provider


class LogfirePlugin(SimplePlugin):
    """Temporal client plugin for Logfire."""

    def __init__(
        self,
        setup_logfire: Callable[[], Logfire] | None = None,
        *,
        metrics: bool = True,
    ):
        """Initialize a Logfire plugin.

        Args:
            setup_logfire: Set up Logfire and Pydantic AI instrumentation. The default uses replay-safe
                instrumentation; providing a callback opts out and uses the global tracer provider.
            metrics: Whether to send Temporal metrics to Logfire.
        """
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
        self._replay_safe = setup_logfire is None
        self._logfire: Logfire | None = None

        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='LogfirePlugin',
            interceptors=[] if self._replay_safe else [TracingInterceptor(get_tracer('temporalio'))],
        )

    def _setup_replay_safe_instrumentation(self) -> Logfire:
        if self._logfire is None:
            from temporalio.contrib.opentelemetry import TracingInterceptor

            self._logfire, tracer_provider = _setup_replay_safe_logfire()
            # `SimplePlugin` reads this attribute in each `configure_*` hook rather than capturing it at init.
            self.interceptors = [TracingInterceptor(tracer_provider.get_tracer('temporalio'))]
        return self._logfire

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        if self._replay_safe:
            self._setup_replay_safe_instrumentation()
        return super().configure_client(config)

    def configure_replayer(self, config: ReplayerConfig) -> ReplayerConfig:
        if self._replay_safe:
            self._setup_replay_safe_instrumentation()
        return super().configure_replayer(config)

    def configure_worker(self, config: WorkerConfig) -> WorkerConfig:
        if self._replay_safe:
            self._setup_replay_safe_instrumentation()
        return super().configure_worker(config)

    async def connect_service_client(
        self, config: ConnectConfig, next: Callable[[ConnectConfig], Awaitable[ServiceClient]]
    ) -> ServiceClient:
        if self.setup_logfire is None:
            logfire = self._setup_replay_safe_instrumentation()
        else:
            logfire = self.setup_logfire()

        if self.metrics:
            logfire_config = logfire.config
            token = logfire_config.token
            if logfire_config.send_to_logfire and isinstance(token, str) and logfire_config.metrics is not False:
                base_url = logfire_config.advanced.generate_base_url(token)
                metrics_url = base_url + '/v1/metrics'
                headers = {'Authorization': f'Bearer {token}'}

                config.runtime = Runtime(
                    telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url=metrics_url, headers=headers))
                )

        return await next(config)
