from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from temporalio.client import ClientConfig, Plugin as ClientPlugin
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient

if TYPE_CHECKING:
    from logfire import Logfire


def _default_setup_logfire() -> Logfire:
    import logfire

    instance = logfire.configure()
    logfire.instrument_pydantic_ai()
    return instance


class LogfirePlugin(ClientPlugin):
    """Temporal client plugin for Logfire."""

    def __init__(self, setup_logfire: Callable[[], Logfire] = _default_setup_logfire, *, metrics: bool = True):
        try:
            import logfire  # noqa: F401 # pyright: ignore[reportUnusedImport]
        except ImportError as _import_error:
            raise ImportError(
                'Please install the `logfire` package to use the Logfire plugin, '
                'you can use the `logfire` optional group — `pip install "pydantic-ai-slim[logfire]"`'
            ) from _import_error

        self.setup_logfire = setup_logfire
        self.metrics = metrics

    def init_client_plugin(self, next: ClientPlugin) -> None:
        self.next_client_plugin = next

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        from opentelemetry.trace import get_tracer
        from temporalio.contrib.opentelemetry import TracingInterceptor

        interceptors = config.get('interceptors', [])
        config['interceptors'] = [*interceptors, TracingInterceptor(get_tracer('temporalio'))]
        return self.next_client_plugin.configure_client(config)

    async def connect_service_client(self, config: ConnectConfig) -> ServiceClient:
        logfire = self.setup_logfire()

        if self.metrics:
            logfire_config = logfire.config
            token = logfire_config.token
            if logfire_config.send_to_logfire and token is not None and logfire_config.metrics is not False:
                base_url = logfire_config.advanced.generate_base_url(token)
                metrics_url = base_url + '/v1/metrics'
                headers = {'Authorization': f'Bearer {token}'}

                config.runtime = Runtime(
                    telemetry=TelemetryConfig(metrics=OpenTelemetryConfig(url=metrics_url, headers=headers))
                )

        return await self.next_client_plugin.connect_service_client(config)
