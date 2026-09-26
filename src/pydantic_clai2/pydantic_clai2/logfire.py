"""Default-enabled Logfire instrumentation, owned by the plugin rather than the process."""

import os
from pathlib import Path
from typing import Literal

import logfire
from anyio import CancelScope, to_thread
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings

from . import theme
from .plugins import PluginHost, SessionEnd


class LogfireSettings(BaseModel):
    """Non-secret telemetry options; credentials stay in Logfire's environment or credential file."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    service_name: str = Field(default='pydantic-clai2', min_length=1)
    send_to_logfire: Literal[False, 'if-token-present'] = 'if-token-present'
    include_content: bool = True
    include_binary_content: bool = True


def activate(host: PluginHost[None]) -> None:
    """Add core instrumentation without changing the supplied agent or global OTel providers."""
    config = host.settings(LogfireSettings)
    config_home = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
    if not config_home.is_absolute():
        config_home = Path.home() / '.config'
    private_dir = config_home / 'pydantic-clai2' / 'logfire'
    propagator = get_global_textmap()
    try:
        instance = logfire.configure(
            local=True,
            send_to_logfire=config.send_to_logfire,
            service_name=config.service_name,
            console=False,
            config_dir=private_dir,
            data_dir=private_dir,
        )
    finally:
        # Even local SDK configuration replaces the process-wide propagator.
        set_global_textmap(propagator)
    try:
        host.add(
            Instrumentation(
                settings=InstrumentationSettings(
                    tracer_provider=instance.config.get_tracer_provider(),
                    meter_provider=instance.config.get_meter_provider(),
                    include_content=config.include_content,
                    include_binary_content=config.include_binary_content,
                )
            )
        )
    except BaseException:
        _shutdown(instance)
        raise

    @host.on('session_end')
    async def shutdown(event: SessionEnd) -> None:
        with CancelScope(shield=True):
            finished = await to_thread.run_sync(_shutdown, instance)
            if not finished:
                host.console.print(
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
