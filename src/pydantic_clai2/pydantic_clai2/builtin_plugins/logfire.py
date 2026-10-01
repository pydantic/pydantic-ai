"""The default-enabled `observability` plugin: Logfire instrumentation owned by the plugin, not the process.

With `ui_events` on, the same instance also records CLAI's UI interactions (see `pydantic_clai2.ui.telemetry`).
With `token` naming a `/keys` entry, everything goes to that key's Logfire project, such as one a team shares.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

import logfire
from anyio import CancelScope, to_thread
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_clai2.builtin_plugins.logfire_setup import Setup, https_origin, run_setup
from pydantic_clai2.config.api_keys import KeyReference, load_keys
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart, TurnEnd
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text


class LogfireSettings(BaseModel):
    """Non-secret telemetry options; a token stays in `LOGFIRE_TOKEN`, Logfire's credential file, or `/keys`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    service_name: str = Field(default='pydantic-clai2', min_length=1)
    send_to_logfire: Literal[False, 'if-token-present'] = 'if-token-present'
    include_content: bool = True
    include_binary_content: bool = True
    token: KeyReference | None = Field(
        default=None,
        description='A /keys entry holding the Logfire write token to send with, instead of LOGFIRE_TOKEN or the '
        'credentials file; its project receives the telemetry.',
    )
    base_url: Annotated[str, AfterValidator(https_origin)] | None = Field(
        default=None,
        description='The Logfire to send to, as the setup menu saves it. Unset, the SDK uses LOGFIRE_BASE_URL, '
        'else the region the token names.',
    )
    ui_events: bool = Field(
        default=False,
        description='Also record UI interactions: menus, commands, settings, plugins, keys, and prompt actions.',
    )


def activate(host: PluginHost[None]) -> None:
    """Add core instrumentation without changing the supplied agent or global OTel providers."""
    config = host.settings(LogfireSettings)
    token, send_to_logfire = _destination(config, host)
    config_home = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
    if not config_home.is_absolute():
        config_home = Path.home() / '.config'
    private_dir = config_home / 'pydantic-clai2' / 'logfire'
    propagator = get_global_textmap()
    try:
        instance = logfire.configure(
            local=True,
            send_to_logfire=send_to_logfire,
            token=token,
            service_name=config.service_name,
            console=False,
            config_dir=private_dir,
            data_dir=private_dir,
            # UI events name settings and keys, such as `sessions.naming` or `OPENAI_API_KEY`, that look like secrets.
            scrubbing=logfire.ScrubbingOptions(callback=telemetry.keep_names) if config.ui_events else None,
            advanced=logfire.AdvancedOptions(base_url=config.base_url) if config.base_url else None,
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
    if config.ui_events:
        _record_ui_events(host, instance)

    @host.configure
    async def configure() -> str:
        return await _configure(host, SETUP(host))

    @host.on('session_end')
    async def shutdown(event: SessionEnd) -> None:
        with CancelScope(shield=True):
            finished = await to_thread.run_sync(_shutdown, instance)
            if not finished:
                host.console.print(
                    'Logfire shutdown timed out; some telemetry may not have been sent.',
                    style=theme.color(theme.WARNING),
                )


def _announce(host: PluginHost[None]) -> Setup:
    def announce(line: str) -> None:
        # Links can come from a self-hosted server, so terminal controls in them are made inert.
        host.console.print(terminal_text(line), markup=False, highlight=False)

    return Setup(announce=announce)


SETUP: Callable[[PluginHost[None]], Setup] = _announce
"""How setup talks to the terminal, the browser, and Logfire; tests swap in scripted ones."""


async def _configure(host: PluginHost[None], setup: Setup) -> str:
    """The setup menu; saving new settings makes the loader load the plugin again, now sending to the project."""
    config = host.settings(LogfireSettings)
    chosen = await run_setup(setup, current=config.base_url, owned=config.token)
    if chosen is None:
        return 'Logfire setup cancelled; settings unchanged.'
    # Setting up a project means sending to it, even if sending had been turned off.
    update = {'token': chosen.token, 'base_url': chosen.base_url, 'send_to_logfire': 'if-token-present'}
    host.save_settings(config.model_copy(update=update))
    return (
        f'Logfire traces now go to {chosen.project.label}. Its write token is saved in /keys as '
        f'{chosen.token.name}; plugin settings keep only that name.'
    )


def _destination(
    config: LogfireSettings, host: PluginHost[None]
) -> tuple[str | None, Literal[False, 'if-token-present']]:
    """The token to send with, and whether to send: a chosen key missing from `/keys` keeps telemetry local.

    Falling back to `LOGFIRE_TOKEN` instead would send to a project the user did not choose.
    """
    if config.token is None or config.send_to_logfire is False:
        return None, config.send_to_logfire
    keys = load_keys()
    if config.token.name not in keys:
        host.console.print(
            f'Logfire is not sending telemetry: {config.token.name} is not in /keys. Save it there, then run '
            '/plugins reload observability.',
            style=theme.color(theme.WARNING),
            markup=False,
        )
        return None, False
    return keys[config.token.name].get_secret_value(), config.send_to_logfire


def _record_ui_events(host: PluginHost[None], instance: logfire.Logfire) -> None:
    """Subscribe `instance` to UI telemetry until the plugin unloads, and add the session and turn lifecycle.

    Registered before the shutdown handler, so the instance stops receiving UI events before it shuts down.
    """
    unsubscribe = telemetry.subscribe(instance)

    @host.on('session_end')
    async def stop(event: SessionEnd) -> None:
        unsubscribe()

    # Only to this plugin's own instance: every enabled copy of the plugin hears these events.
    @host.on('session_start')
    async def started(event: SessionStart) -> None:
        model = event.settings.model or 'agent default'
        instance.log('info', 'session started', attributes={'model': model}, tags=[telemetry.TAG])

    @host.on('turn_end')
    async def ended(event: TurnEnd) -> None:
        instance.log('info', 'turn {outcome}', attributes={'outcome': event.outcome}, tags=[telemetry.TAG])


def _shutdown(instance: logfire.Logfire) -> bool:
    # SDK shutdown with flush=True can return on a flush timeout before stopping providers.
    try:
        flushed = instance.force_flush(timeout_millis=3000)
    finally:
        stopped = instance.shutdown(timeout_millis=3000, flush=False)
    return flushed and stopped
