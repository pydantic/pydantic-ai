from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING

from temporalio.plugin import SimplePlugin
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient

if TYPE_CHECKING:
    from logfire import Logfire
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


def _setup_replay_safe_logfire() -> Logfire:
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

    from pydantic_ai import Agent

    from ._replay_safe_tracer_provider import ReplaySafeSDKTracerProvider

    instance = _get_logfire()
    # Install the replay-safe provider *inside* Logfire's tracer proxy rather than beside it, so every tracer
    # Logfire hands out (including ones obtained before this ran) follows it, and scopes the host suppressed
    # with `logfire.suppress_scopes()` stay suppressed. `logfire.configure()` swaps a fresh provider into the
    # proxy, so this runs again on every client, worker and replayer to restore replay-safety.
    proxy = instance.config.get_tracer_provider()
    provider = proxy.provider
    if not isinstance(provider, ReplaySafeSDKTracerProvider):
        assert isinstance(provider, SDKTracerProvider)
        proxy.set_provider(ReplaySafeSDKTracerProvider(provider))

    # `instrument_pydantic_ai()` is a replace, not a merge: with no arguments it builds a default
    # `InstrumentationSettings` and assigns it to the process-wide `Agent._instrument_default`, which would turn
    # a host's deliberate `include_content=False` back on. A host's existing instrumentation already gets its
    # tracer through Logfire's proxy, so it becomes replay-safe without being replaced.
    if Agent._instrument_default is False:  # pyright: ignore[reportPrivateUsage]
        instance.instrument_pydantic_ai()
    return instance


class LogfirePlugin(SimplePlugin):
    """Temporal client plugin for Logfire.

    Args:
        setup_logfire: Function that configures Logfire and Pydantic AI instrumentation and returns the
            Logfire instance. By default, the plugin uses replay-safe instrumentation; providing a
            callback opts out and uses the global tracer provider.
        metrics: Whether to send Temporal metrics to Logfire.
        metric_periodicity: How often to export Temporal metrics. Defaults to 60 seconds.
    """

    def __init__(
        self,
        setup_logfire: Callable[[], Logfire] | None = None,
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
        self._replay_safe = setup_logfire is None

        super().__init__(  # type: ignore[reportUnknownMemberType]
            name='LogfirePlugin',
            interceptors=[] if self._replay_safe else [TracingInterceptor(get_tracer('temporalio'))],
        )

    def _setup_replay_safe_instrumentation(self) -> Logfire:
        instance = _setup_replay_safe_logfire()
        if not self.interceptors:
            from temporalio.contrib.opentelemetry import TracingInterceptor

            # Logfire's proxy re-points this tracer whenever its provider changes, so one interceptor suffices.
            # `SimplePlugin` reads this attribute in each `configure_*` hook rather than capturing it at init.
            self.interceptors = [TracingInterceptor(instance.config.get_tracer_provider().get_tracer('temporalio'))]
        return instance

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
                    telemetry=TelemetryConfig(
                        metrics=OpenTelemetryConfig(
                            url=metrics_url,
                            headers=headers,
                            metric_periodicity=self.metric_periodicity,
                        )
                    )
                )

        return await next(config)
