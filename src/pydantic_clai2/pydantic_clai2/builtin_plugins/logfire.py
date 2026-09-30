"""Default-enabled Logfire instrumentation, owned by the plugin rather than the process."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import logfire
from anyio import CancelScope, to_thread
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai.capabilities import AgentCapability, Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_clai2.plugins import Plugin, PluginHost, SessionEnd
from pydantic_clai2.ui.rendering import theme


class LogfireSettings(BaseModel):
    """Non-secret telemetry options; credentials stay in Logfire's environment or credential file."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    service_name: str = Field(default='pydantic-clai2', min_length=1)
    send_to_logfire: Literal[False, 'if-token-present'] = 'if-token-present'
    include_content: bool = True
    include_binary_content: bool = True


class LogfirePlugin(Plugin[LogfireSettings]):
    """Core instrumentation, without changing the supplied agent or global OTel providers."""

    def __init__(self, host: PluginHost[None], settings: LogfireSettings) -> None:
        super().__init__(host, settings)
        config_home = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
        if not config_home.is_absolute():
            config_home = Path.home() / '.config'
        private_dir = config_home / 'pydantic-clai2' / 'logfire'
        propagator = get_global_textmap()
        try:
            self.instance = logfire.configure(
                local=True,
                send_to_logfire=settings.send_to_logfire,
                service_name=settings.service_name,
                console=False,
                config_dir=private_dir,
                data_dir=private_dir,
            )
        finally:
            # Even local SDK configuration replaces the process-wide propagator.
            set_global_textmap(propagator)
        try:
            self.instrumentation = Instrumentation(
                settings=InstrumentationSettings(
                    tracer_provider=self.instance.config.get_tracer_provider(),
                    meter_provider=self.instance.config.get_meter_provider(),
                    include_content=settings.include_content,
                    include_binary_content=settings.include_binary_content,
                )
            )
        except BaseException:
            _shutdown(self.instance)
            raise

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self.instrumentation,)

    async def on_session_end(self, event: SessionEnd) -> None:
        with CancelScope(shield=True):
            finished = await to_thread.run_sync(_shutdown, self.instance)
            if not finished:
                self.host.console.print(
                    'Logfire shutdown timed out; some telemetry may not have been sent.',
                    style=theme.color(theme.WARNING),
                )


def _shutdown(instance: logfire.Logfire) -> bool:
    # SDK shutdown with flush=True can return on a flush timeout before stopping providers.
    try:
        flushed = instance.force_flush(timeout_millis=3000)
    finally:
        stopped = instance.shutdown(timeout_millis=3000, flush=False)
    return flushed and stopped
