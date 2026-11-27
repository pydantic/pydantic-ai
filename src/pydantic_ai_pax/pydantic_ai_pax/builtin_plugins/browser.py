"""Browse the web in a headless Chromium on this machine, which refuses private and loopback addresses.

The built-in `browser` plugin, off until enabled: harness `PlaywrightBrowser`, headless, with
`block_private_addresses` on and the `allowed_domains` setting as its egress allowlist. It needs the `playwright`
extra (`uv tool install 'pydantic-ai-pax[playwright]'`) and Playwright's Chromium (`playwright install chromium`);
enabling it without the extra fails the load with that command, and a missing Chromium is reported when it loads.

Chromium runs where Pax runs, also for a conversation in a `/sandbox`. Like the hosted integrations it is deferred,
one line in the capability catalog until the model loads it.
"""

import sys
from collections.abc import Sequence
from importlib.util import find_spec

import anyio
from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_pax.plugins import DepsT, Plugin, SessionStart
from pydantic_ai_pax.ui.rendering import theme

DESCRIPTION = 'A real Chromium browser for JavaScript-heavy pages and multi-step flows: navigate, click, type, read.'
"""The catalog line the model reads before it loads the tools."""

CHECK_TIMEOUT = 10
"""Seconds the load-time Chromium check may take; past that it is skipped, and the tools report a miss themselves."""


def install_chromium_command() -> str:
    """Playwright's installer run by Pax's own interpreter: a `playwright` command on `PATH` may be another one."""
    return f'{sys.executable} -m playwright install chromium'


class BrowserSettings(BaseModel):
    """The JSON a `browser` declaration may carry."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    allowed_domains: list[str] | None = Field(
        default=None,
        description=(
            'The only sites the browser may open or send data to, each with its subdomains; '
            'unset, any public site. Private and loopback addresses are refused either way.'
        ),
    )


class BrowserPlugin(Plugin[BrowserSettings, DepsT]):
    """Harness `PlaywrightBrowser`, headless and refusing private addresses, deferred until the model loads it."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        if find_spec('playwright') is None:
            raise ImportError(
                'The browser needs Playwright, which this Pax was installed without. Install Pax with it: '
                f"uv tool install 'pydantic-ai-pax[playwright]', then run {install_chromium_command()}."
            )
        # Imported here: the `playwright` extra is optional, and the check above names the command that installs it.
        from pydantic_ai_harness.playwright import PlaywrightBrowser

        return (
            PlaywrightBrowser[DepsT](
                headless=True,
                block_private_addresses=True,
                allowed_domains=self.settings.allowed_domains,
                defer_loading=True,
                description=DESCRIPTION,
            ),
        )

    async def on_session_start(self, event: SessionStart) -> None:
        """Say how to install Chromium when it is missing, rather than leaving the model's first call to find out."""
        from playwright.async_api import Error as PlaywrightError, async_playwright

        try:
            with anyio.fail_after(CHECK_TIMEOUT):
                async with async_playwright() as playwright:
                    executable = playwright.chromium.executable_path
                installed = await anyio.Path(executable).exists()
        except (PlaywrightError, OSError):
            # A driver that fails to start, or that `fail_after` cut off (its `TimeoutError` is an `OSError`): the
            # browser tools report a browser that cannot start themselves.
            return
        if not installed:
            self.host.console.print(
                f'The browser plugin needs Chromium. Run {install_chromium_command()}',
                style=theme.color(theme.WARNING),
                markup=False,
            )
