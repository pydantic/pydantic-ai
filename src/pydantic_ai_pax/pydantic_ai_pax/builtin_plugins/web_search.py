"""Search the web and read pages through Keenable, a hosted service that receives every query and page address.

The built-in `web_search` plugin, off until enabled: harness `KeenableSearch`, which adds a `web_search` tool and a
`get_page` tool. It needs no key and no extra install. Each search query, and each URL the model reads, is sent to
Keenable's API at `api.keenable.ai`, and `get_page` returns Keenable's indexed copy of the page, so no page is
fetched from this machine. `KEENABLE_API_KEY` in the environment raises Keenable's rate limits.

Every model gets these tools, including one whose provider offers native web search: Pax turns on no provider
search, so where queries go does not change with the model. Like the hosted integrations it is deferred, one line in
the capability catalog until the model loads it.
"""

from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.keenable import KeenableSearch
from pydantic_ai_pax.plugins import DepsT, NoSettings, Plugin

DESCRIPTION = 'Search the web and read pages through Keenable, which receives each query and page address.'
"""The catalog line the model reads before it loads the tools."""


class WebSearchPlugin(Plugin[NoSettings, DepsT]):
    """Harness `KeenableSearch` with its defaults, deferred until the model loads it."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (KeenableSearch[DepsT](defer_loading=True, description=DESCRIPTION),)
