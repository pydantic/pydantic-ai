from __future__ import annotations

from typing import Callable

from logfire import Logfire
from opentelemetry.trace import get_tracer
from temporalio.client import ClientConfig, Plugin as ClientPlugin
from temporalio.contrib.opentelemetry import TracingInterceptor
from temporalio.runtime import OpenTelemetryConfig, Runtime, TelemetryConfig
from temporalio.service import ConnectConfig, ServiceClient


def _default_setup_logfire() -> Logfire:
    import logfire

    instance = logfire.configure()
    logfire.instrument_pydantic_ai()
    return instance


class LogfirePlugin(ClientPlugin):
    """Temporal client plugin for Logfire."""

    def __init__(self, setup_logfire: Callable[[], Logfire] = _default_setup_logfire, *, metrics: bool = True):
        self.setup_logfire = setup_logfire
        self.metrics = metrics

    def configure_client(self, config: ClientConfig) -> ClientConfig:
        interceptors = config.get('interceptors', [])
        config['interceptors'] = [*interceptors, TracingInterceptor(get_tracer('temporalio'))]
        return super().configure_client(config)

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

        return await super().connect_service_client(config)
