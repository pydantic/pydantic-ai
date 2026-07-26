"""Nimble common tools for Pydantic AI agents.

Provides web search, page extract, site map, crawl job control, and Agent API V2
run lifecycle tools via the official [`nimble_python`](https://pypi.org/project/nimble_python/)
SDK.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from functools import partial
from inspect import signature
from typing import Literal, overload

from pydantic import TypeAdapter
from typing_extensions import Any, TypedDict

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import Tool

try:
    from nimble_python import AsyncNimble
except ImportError as _import_error:
    raise ImportError(
        'Please install `nimble_python` to use the Nimble tools, '
        'you can use the `nimble` optional group — `pip install "pydantic-ai-slim[nimble]"`'
    ) from _import_error

__all__ = (
    'NimbleToolset',
    'nimble_agent_run_result_tool',
    'nimble_agent_run_start_tool',
    'nimble_agent_run_status_tool',
    'nimble_agents_list_tool',
    'nimble_agent_templates_list_tool',
    'nimble_crawl_start_tool',
    'nimble_crawl_status_tool',
    'nimble_extract_tool',
    'nimble_map_tool',
    'nimble_search_tool',
)

_CLIENT_SOURCE = 'pydantic-ai'
"""Stable attribution slug sent as `X-Client-Source` on every Nimble request."""

_UNSET: Any = object()
"""Sentinel to distinguish "not provided" from None in factory kwargs."""


def _resolve_client(api_key: str | None, client: AsyncNimble | None) -> AsyncNimble:
    if client is not None:
        return client
    if api_key is None:
        raise ValueError('Either api_key or client must be provided')
    return AsyncNimble(api_key=api_key, client_source=_CLIENT_SOURCE)


def _bind_tool_params(func: Callable[..., Any], kwargs: dict[str, Any]) -> Callable[..., Any]:
    if not kwargs:
        return func
    original = func
    bound = partial(func, **kwargs)
    bound.__name__ = original.__name__  # type: ignore[attr-defined]
    bound.__qualname__ = original.__qualname__
    # partial with keyword args only updates defaults, not removes params.
    # Set __signature__ explicitly to exclude bound params from the tool schema.
    orig_sig = signature(original)
    bound.__signature__ = orig_sig.replace(  # type: ignore[attr-defined]
        parameters=[p for name, p in orig_sig.parameters.items() if name not in kwargs]
    )
    return bound


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class NimbleSearchResult(TypedDict):
    """A Nimble search result.

    See [Nimble Search API documentation](https://docs.nimbleway.com/)
    for more information.
    """

    title: str
    """The title of the search result."""
    url: str
    """The URL of the search result."""
    content: str
    """The result content, or the description when content is empty (e.g. `lite` depth)."""


nimble_search_ta = TypeAdapter(list[NimbleSearchResult])


@dataclass
class NimbleSearchTool:
    """The Nimble search tool."""

    client: AsyncNimble
    """The Nimble async client."""

    _: KW_ONLY

    max_results: int | None = None
    """The maximum number of results. If None, the Nimble default is used."""

    async def __call__(
        self,
        query: str,
        search_depth: Literal['lite', 'fast', 'deep'] = 'lite',
        time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> list[NimbleSearchResult]:
        """Searches Nimble for the given query and returns the results.

        Args:
            query: The search query to execute with Nimble.
            search_depth: Controls content richness and latency of search results.
            time_range: The time range back from the current date to filter results.
            include_domains: List of domains to specifically include in the search results.
            exclude_domains: List of domains to specifically exclude from the search results.

        Returns:
            A list of search results from Nimble.
        """
        search_kwargs: dict[str, Any] = {
            'query': query,
            'search_depth': search_depth,
            'time_range': time_range,
            'include_domains': include_domains,
            'exclude_domains': exclude_domains,
        }
        if self.max_results is not None:
            search_kwargs['max_results'] = self.max_results
        response = await self.client.search(**search_kwargs)
        projected = [
            {
                'title': result.title,
                'url': result.url,
                'content': result.content or result.description,
            }
            for result in response.results
        ]
        return nimble_search_ta.validate_python(projected)


@overload
def nimble_search_tool(
    api_key: str,
    *,
    max_results: int | None = None,
    search_depth: Literal['lite', 'fast', 'deep'] = _UNSET,
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]: ...


@overload
def nimble_search_tool(
    *,
    client: AsyncNimble,
    max_results: int | None = None,
    search_depth: Literal['lite', 'fast', 'deep'] = _UNSET,
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]: ...


def nimble_search_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
    max_results: int | None = None,
    search_depth: Literal['lite', 'fast', 'deep'] = _UNSET,
    time_range: Literal['hour', 'day', 'week', 'month', 'year'] | None = _UNSET,
    include_domains: list[str] | None = _UNSET,
    exclude_domains: list[str] | None = _UNSET,
) -> Tool[Any]:
    """Creates a Nimble search tool.

    `max_results` is always developer-controlled and does not appear in the LLM tool schema.
    Other parameters, when provided, are fixed for all searches and hidden from the LLM's
    tool schema. Parameters left unset remain available for the LLM to set per-call.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.

            You can get one from [Nimble's dashboard](https://online.nimbleway.com/account-settings/api-keys).
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
            This is useful for sharing a client across multiple tool instances.
        max_results: The maximum number of results. If None, the Nimble default is used.
        search_depth: Controls content richness and latency of search results.
        time_range: The time range back from the current date to filter results.
        include_domains: List of domains to specifically include in the search results.
        exclude_domains: List of domains to specifically exclude from the search results.
    """
    resolved = _resolve_client(api_key, client)
    func = NimbleSearchTool(client=resolved, max_results=max_results).__call__
    kwargs: dict[str, Any] = {}
    if search_depth is not _UNSET:
        kwargs['search_depth'] = search_depth
    if time_range is not _UNSET:
        kwargs['time_range'] = time_range
    if include_domains is not _UNSET:
        kwargs['include_domains'] = include_domains
    if exclude_domains is not _UNSET:
        kwargs['exclude_domains'] = exclude_domains
    return Tool[Any](
        _bind_tool_params(func, kwargs),
        name='nimble_search',
        description='Searches Nimble for the given query and returns the results.',
    )


# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------


@dataclass
class NimbleExtractTool:
    """The Nimble extract tool."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(self, url: str) -> str:
        """Extracts page content as markdown from a URL.

        Args:
            url: The URL to extract content from.

        Returns:
            The page content as markdown, or an empty string when unavailable.
        """
        response = await self.client.extract.run(url=url, formats=['markdown'])
        if response.data and response.data.markdown:
            return response.data.markdown
        return ''


@overload
def nimble_extract_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_extract_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_extract_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a Nimble extract tool that returns page content as markdown.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleExtractTool(client=resolved).__call__,
        name='nimble_extract',
        description='Extract page content as markdown from a URL. Use after search to read a specific page.',
    )


# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------


class NimbleMapLink(TypedDict):
    """A link discovered by Nimble Map."""

    url: str
    """The discovered URL."""
    title: str | None
    """The link title, if available."""
    description: str | None
    """The link description, if available."""


nimble_map_ta = TypeAdapter(list[NimbleMapLink])


@dataclass
class NimbleMapTool:
    """The Nimble map tool."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(
        self,
        url: str,
        limit: int | None = None,
        domain_filter: Literal['domain', 'subdomain', 'all'] | None = None,
        sitemap: Literal['skip', 'include', 'only'] | None = None,
    ) -> list[NimbleMapLink]:
        """Discovers links on a website.

        Args:
            url: The website URL to map.
            limit: Maximum number of links to return.
            domain_filter: Scope of domains to include (`domain`, `subdomain`, or `all`).
            sitemap: Sitemap handling strategy (`skip`, `include`, or `only`).

        Returns:
            Discovered links with optional titles and descriptions.
        """
        map_kwargs: dict[str, Any] = {'url': url}
        if limit is not None:
            map_kwargs['limit'] = limit
        if domain_filter is not None:
            map_kwargs['domain_filter'] = domain_filter
        if sitemap is not None:
            map_kwargs['sitemap'] = sitemap
        response = await self.client.map(**map_kwargs)
        projected = [
            {
                'url': link.url,
                'title': link.title,
                'description': link.description,
            }
            for link in response.links
        ]
        return nimble_map_ta.validate_python(projected)


@overload
def nimble_map_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_map_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_map_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a Nimble map tool for discovering links on a website.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleMapTool(client=resolved).__call__,
        name='nimble_map',
        description='Discover links on a website. Use to understand site structure before crawling or extracting pages.',
    )


# ---------------------------------------------------------------------------
# Crawl (resumable: start + status, no long polling)
# ---------------------------------------------------------------------------


class NimbleCrawlJob(TypedDict):
    """A Nimble crawl job snapshot."""

    crawl_id: str
    """The crawl job id."""
    status: str
    """The crawl job status."""
    url: str
    """The start URL for the crawl."""
    completed: float | None
    """Number of completed page tasks."""
    failed: float | None
    """Number of failed page tasks."""
    pending: float | None
    """Number of pending page tasks."""
    total: float | None
    """Total number of page tasks."""


def _crawl_job_from_response(response: Any) -> NimbleCrawlJob:
    return NimbleCrawlJob(
        crawl_id=response.crawl_id,
        status=response.status,
        url=response.url,
        completed=response.completed,
        failed=response.failed,
        pending=response.pending,
        total=response.total,
    )


@dataclass
class NimbleCrawlStartTool:
    """Starts a Nimble crawl job without waiting for completion."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(
        self,
        url: str,
        limit: int | None = None,
        max_discovery_depth: int | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        sitemap: Literal['skip', 'include', 'only'] | None = None,
        name: str | None = None,
    ) -> NimbleCrawlJob:
        """Starts a crawl job and returns its id/status.

        Args:
            url: The URL to start crawling from.
            limit: Maximum number of pages to crawl.
            max_discovery_depth: Maximum link-following depth from the start URL.
            include_paths: URL path patterns to include.
            exclude_paths: URL path patterns to exclude.
            sitemap: Sitemap handling strategy.
            name: Optional name for the crawl job.

        Returns:
            A crawl job snapshot including `crawl_id` for later status checks.
        """
        crawl_kwargs: dict[str, Any] = {'url': url}
        if limit is not None:
            crawl_kwargs['limit'] = limit
        if max_discovery_depth is not None:
            crawl_kwargs['max_discovery_depth'] = max_discovery_depth
        if include_paths is not None:
            crawl_kwargs['include_paths'] = include_paths
        if exclude_paths is not None:
            crawl_kwargs['exclude_paths'] = exclude_paths
        if sitemap is not None:
            crawl_kwargs['sitemap'] = sitemap
        if name is not None:
            crawl_kwargs['name'] = name
        response = await self.client.crawl.run(**crawl_kwargs)
        return _crawl_job_from_response(response)


@dataclass
class NimbleCrawlStatusTool:
    """Fetches status for a Nimble crawl job."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(self, crawl_id: str) -> NimbleCrawlJob:
        """Gets the current status of a crawl job.

        Args:
            crawl_id: The crawl job id returned by `nimble_crawl_start`.

        Returns:
            The latest crawl job snapshot.
        """
        response = await self.client.crawl.status(crawl_id)
        return _crawl_job_from_response(response)


@overload
def nimble_crawl_start_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_crawl_start_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_crawl_start_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that starts a Nimble crawl job (does not poll for completion).

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleCrawlStartTool(client=resolved).__call__,
        name='nimble_crawl_start',
        description=(
            'Start a Nimble crawl job and return its crawl_id immediately. '
            'Use nimble_crawl_status to poll progress across agent turns.'
        ),
    )


@overload
def nimble_crawl_status_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_crawl_status_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_crawl_status_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that fetches Nimble crawl job status.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleCrawlStatusTool(client=resolved).__call__,
        name='nimble_crawl_status',
        description='Get the status of a Nimble crawl job by crawl_id.',
    )


# ---------------------------------------------------------------------------
# Agent API V2 (list + start / status / result)
# ---------------------------------------------------------------------------


@dataclass
class NimbleAgentsListTool:
    """Lists Nimble agents."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(self, limit: int | None = None, offset: int | None = None) -> list[dict[str, Any]]:
        """Lists available Nimble agents.

        Args:
            limit: Maximum number of agents to return.
            offset: Pagination offset.

        Returns:
            Agent records as dictionaries.
        """
        list_kwargs: dict[str, Any] = {}
        if limit is not None:
            list_kwargs['limit'] = limit
        if offset is not None:
            list_kwargs['offset'] = offset
        response = await self.client.agents.list(**list_kwargs)
        return [item.model_dump(mode='json') for item in response.items]


@dataclass
class NimbleAgentTemplatesListTool:
    """Lists Nimble agent templates."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(self, limit: int | None = None, offset: int | None = None) -> list[dict[str, Any]]:
        """Lists available Nimble agent templates.

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            Template records as dictionaries.
        """
        list_kwargs: dict[str, Any] = {}
        if limit is not None:
            list_kwargs['limit'] = limit
        if offset is not None:
            list_kwargs['offset'] = offset
        response = await self.client.agents.templates.list(**list_kwargs)
        return [item.model_dump(mode='json') for item in response.items]


@dataclass
class NimbleAgentRunStartTool:
    """Starts an Agent API V2 run."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(
        self,
        agent_id: str,
        input: str,
        effort: Literal['low', 'medium', 'high', 'x-high', 'max'] | None = None,
    ) -> dict[str, Any]:
        """Starts an agent run and returns run metadata.

        Args:
            agent_id: The Nimble agent id to run.
            input: The prompt / input for the agent run.
            effort: Optional effort level for the run.

        Returns:
            Run metadata including `id` and `status` for later status/result calls.
        """
        run_kwargs: dict[str, Any] = {'agent_id': agent_id, 'input': input}
        if effort is not None:
            run_kwargs['effort'] = effort
        response = await self.client.agents.runs.create(**run_kwargs)
        return response.model_dump(mode='json')


@dataclass
class NimbleAgentRunStatusTool:
    """Fetches Agent API V2 run status."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Gets the status of an agent run.

        Args:
            agent_id: The Nimble agent id.
            run_id: The run id returned by `nimble_agent_run_start`.

        Returns:
            Run status metadata.
        """
        response = await self.client.agents.runs.get(run_id, agent_id=agent_id)
        return response.model_dump(mode='json')


@dataclass
class NimbleAgentRunResultTool:
    """Fetches Agent API V2 run result."""

    client: AsyncNimble
    """The Nimble async client."""

    async def __call__(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Gets the result of a completed agent run.

        Args:
            agent_id: The Nimble agent id.
            run_id: The run id returned by `nimble_agent_run_start`.

        Returns:
            The run result payload (text/JSON output and trust metadata when present).
        """
        response = await self.client.agents.runs.result(run_id, agent_id=agent_id)
        if hasattr(response, 'model_dump'):
            return response.model_dump(mode='json')
        return {'result': response}  # pragma: no cover


@overload
def nimble_agents_list_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_agents_list_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_agents_list_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that lists Nimble agents.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleAgentsListTool(client=resolved).__call__,
        name='nimble_agents_list',
        description='List available Nimble agents. Use an agent id with nimble_agent_run_start.',
    )


@overload
def nimble_agent_templates_list_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_agent_templates_list_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_agent_templates_list_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that lists Nimble agent templates.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleAgentTemplatesListTool(client=resolved).__call__,
        name='nimble_agent_templates_list',
        description='List available Nimble agent templates (research / enrichment starting points).',
    )


@overload
def nimble_agent_run_start_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_agent_run_start_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_agent_run_start_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that starts an Agent API V2 run (does not wait for completion).

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleAgentRunStartTool(client=resolved).__call__,
        name='nimble_agent_run_start',
        description=(
            'Start a Nimble agent run and return run id/status immediately. '
            'Use nimble_agent_run_status and nimble_agent_run_result across turns.'
        ),
    )


@overload
def nimble_agent_run_status_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_agent_run_status_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_agent_run_status_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that fetches Agent API V2 run status.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleAgentRunStatusTool(client=resolved).__call__,
        name='nimble_agent_run_status',
        description='Get the status of a Nimble agent run by agent_id and run_id.',
    )


@overload
def nimble_agent_run_result_tool(api_key: str) -> Tool[Any]: ...


@overload
def nimble_agent_run_result_tool(*, client: AsyncNimble) -> Tool[Any]: ...


def nimble_agent_run_result_tool(
    api_key: str | None = None,
    *,
    client: AsyncNimble | None = None,
) -> Tool[Any]:
    """Creates a tool that fetches Agent API V2 run results.

    Args:
        api_key: The Nimble API key. Required if `client` is not provided.
        client: An existing `AsyncNimble` client. If provided, `api_key` is ignored.
    """
    resolved = _resolve_client(api_key, client)
    return Tool[Any](
        NimbleAgentRunResultTool(client=resolved).__call__,
        name='nimble_agent_run_result',
        description='Get the result of a completed Nimble agent run by agent_id and run_id.',
    )


# ---------------------------------------------------------------------------
# Toolset
# ---------------------------------------------------------------------------


class NimbleToolset(FunctionToolset[Any]):
    """A toolset that provides Nimble tools with a shared client.

    By default includes search and extract. Map, crawl, and Agent API tools are opt-in.
    """

    def __init__(
        self,
        api_key: str,
        *,
        max_results: int | None = None,
        include_search: bool = True,
        include_extract: bool = True,
        include_map: bool = False,
        include_crawl: bool = False,
        include_agents: bool = False,
        id: str | None = None,
    ):
        """Creates a Nimble toolset with a shared client.

        Args:
            api_key: The Nimble API key.
            max_results: Developer-controlled max results for search.
            include_search: Whether to include `nimble_search`.
            include_extract: Whether to include `nimble_extract`.
            include_map: Whether to include `nimble_map`.
            include_crawl: Whether to include crawl start/status tools.
            include_agents: Whether to include Agent API V2 tools.
            id: Optional ID for the toolset, used for durable execution environments.
        """
        client = AsyncNimble(api_key=api_key, client_source=_CLIENT_SOURCE)
        tools: list[Tool[Any]] = []

        if include_search:
            tools.append(nimble_search_tool(client=client, max_results=max_results))
        if include_extract:
            tools.append(nimble_extract_tool(client=client))
        if include_map:
            tools.append(nimble_map_tool(client=client))
        if include_crawl:
            tools.append(nimble_crawl_start_tool(client=client))
            tools.append(nimble_crawl_status_tool(client=client))
        if include_agents:
            tools.append(nimble_agents_list_tool(client=client))
            tools.append(nimble_agent_templates_list_tool(client=client))
            tools.append(nimble_agent_run_start_tool(client=client))
            tools.append(nimble_agent_run_status_tool(client=client))
            tools.append(nimble_agent_run_result_tool(client=client))

        super().__init__(tools, id=id)
