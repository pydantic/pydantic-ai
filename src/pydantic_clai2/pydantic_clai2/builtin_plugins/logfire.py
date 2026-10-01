"""The default-enabled `observability` plugin: Logfire instrumentation owned by the plugin, not the process.

With `ui_events` on, the same instance also records CLAI's UI interactions (see `pydantic_clai2.ui.telemetry`).
With `token` naming a `/keys` entry, everything goes to that key's Logfire project, such as one a team shares.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

import logfire
from anyio import CancelScope, to_thread
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from pydantic_ai.capabilities import AgentCapability, Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_clai2.builtin_plugins.logfire_setup import Setup, https_origin, run_setup
from pydantic_clai2.config.api_keys import KeyReference, load_keys
from pydantic_clai2.plugins import Plugin, PluginHost, SessionEnd, SessionStart, TurnEnd
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


class LogfirePlugin(Plugin[LogfireSettings]):
    """Core instrumentation, without changing the supplied agent or global OTel providers."""

    def __init__(self, host: PluginHost[None], settings: LogfireSettings) -> None:
        super().__init__(host, settings)
        token, send_to_logfire = _destination(settings, host)
        config_home = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
        if not config_home.is_absolute():
            config_home = Path.home() / '.config'
        private_dir = config_home / 'pydantic-clai2' / 'logfire'
        propagator = get_global_textmap()
        try:
            self.instance = logfire.configure(
                local=True,
                send_to_logfire=send_to_logfire,
                token=token,
                service_name=settings.service_name,
                console=False,
                config_dir=private_dir,
                data_dir=private_dir,
                # UI events name settings and keys, such as `sessions.naming` or `OPENAI_API_KEY`, that look like secrets.
                scrubbing=logfire.ScrubbingOptions(callback=telemetry.keep_names) if settings.ui_events else None,
                advanced=logfire.AdvancedOptions(base_url=settings.base_url) if settings.base_url else None,
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
        # Subscribed last, so a failed construction leaves nothing to unsubscribe.
        self._unsubscribe = telemetry.subscribe(self.instance) if settings.ui_events else None

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self.instrumentation,)

    async def configure(self) -> str:
        return await _configure(self.host, SETUP(self.host))

    # The UI lifecycle goes only to this plugin's own instance: every enabled copy of the plugin hears these events.
    async def on_session_start(self, event: SessionStart) -> None:
        if self.settings.ui_events:
            model = event.settings.model or 'agent default'
            self.instance.log('info', 'session started', attributes={'model': model}, tags=[telemetry.TAG])

    async def on_turn_end(self, event: TurnEnd) -> None:
        if self.settings.ui_events:
            self.instance.log('info', 'turn {outcome}', attributes={'outcome': event.outcome}, tags=[telemetry.TAG])

    async def on_session_end(self, event: SessionEnd) -> None:
        # Stop receiving UI events before the instance shuts down.
        if self._unsubscribe is not None:
            self._unsubscribe()
        with CancelScope(shield=True):
            finished = await to_thread.run_sync(_shutdown, self.instance)
            if not finished:
                self.host.console.print(
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


def _shutdown(instance: logfire.Logfire) -> bool:
    # SDK shutdown with flush=True can return on a flush timeout before stopping providers.
    try:
        flushed = instance.force_flush(timeout_millis=3000)
    finally:
        stopped = instance.shutdown(timeout_millis=3000, flush=False)
    return flushed and stopped
