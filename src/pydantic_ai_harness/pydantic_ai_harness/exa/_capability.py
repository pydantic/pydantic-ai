"""Exa search capability that gives an agent web research tools."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.exa._toolset import (
    EXA_MAX_NUM_RESULTS,
    EXA_MAX_PAGE_TEXT_CHARS,
    ExaClient,
    ExaSearchToolset,
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = (
    'You have web research tools backed by the Exa search API. Start broad: use `web_search` '
    'to survey several sources on a topic, then use `get_page` to read the most promising '
    'URLs in full before drawing conclusions. Prefer primary sources, and cite the URLs of '
    'the pages you relied on in your answer.'
)

_DEEP_INSTRUCTIONS_SUFFIX = (
    ' For questions that need synthesis across many sources, escalate to `deep_search`: it runs '
    'a full research pass in a single call, so save it for the questions that deserve that depth.'
)


@dataclass
class ExaSearch(AbstractCapability[AgentDepsT]):
    """Web research for agents, backed by the [Exa](https://exa.ai) search API.

    Adds two tools: `web_search`, which returns search results with their most
    relevant excerpts, and `get_page`, which retrieves the full text of a
    specific URL. Set `text_summary` to have `web_search` also return a
    synthesized text summary with the results. Set `include_deep_search=True`
    to also expose `deep_search`, which runs Exa's multi-step deep search and
    returns a synthesized, cited answer in one tool call.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.exa import ExaSearch

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])
    ```

    Authentication comes from the `EXA_API_KEY` environment variable by
    default; pass `client` to configure it explicitly.
    """

    num_results: int = 5
    """Number of results `web_search` returns per query (1 to 100, the Exa API range)."""

    max_text_chars: int = 10_000
    """Maximum characters of page text `get_page` returns (1 to 10,000, the Exa API range).

    One character of headroom above the cap is requested from Exa so local
    truncation can detect a longer page and append a truncation marker. At the
    API ceiling of 10,000 no headroom exists, so the marker cannot fire there.
    """

    text_summary: bool | str = False
    """Have `web_search` also return a synthesized text summary above the results. Off by default.

    When enabled, each `web_search` call requests Exa's plain-text output
    schema, so the response carries a short summary synthesized from the
    results in addition to the result list. Pass a string to describe the
    desired summary format (it is sent as the schema's `description`), or
    `True` for an unconstrained summary. The tool's return shape is unchanged:
    the summary is prepended as a `Summary:` line when Exa returns one.
    """

    include_deep_search: bool = False
    """Also expose the `deep_search` tool. Off by default.

    Deep search (Exa search `type='deep'`) runs a multi-step agentic search
    and synthesizes a cited answer in one call. Each call invests more time
    and search depth than `web_search` (Exa's research-grade mode), and the
    model decides when to invoke tools, so that investment is opt-in rather
    than the default.
    """

    include_domains: Sequence[str] = field(default_factory=list[str])
    """If non-empty, search results only come from these domains (allowlist).

    Applies to `web_search` and `deep_search`. Mutually exclusive with
    `exclude_domains`.
    """

    exclude_domains: Sequence[str] = field(default_factory=list[str])
    """Search results never come from these domains (denylist).

    Applies to `web_search` and `deep_search`. Mutually exclusive with
    `include_domains`.
    """

    guidance: str | None = None
    """Custom research guidance for the system prompt.

    Leave as `None` for the default guidance (which adapts to
    `include_deep_search`), or set `''` to contribute no instructions at all.
    """

    client: ExaClient | None = None
    """Exa client to use; when `None`, an `exa_py.AsyncExa` is built from `EXA_API_KEY`.

    Any object satisfying the `ExaClient` protocol works: use it to pass an API
    key explicitly, point at a different base URL, or substitute a fake in tests.
    """

    def __post_init__(self) -> None:
        """Validate configuration against the Exa API's documented bounds."""
        if not 1 <= self.num_results <= EXA_MAX_NUM_RESULTS:
            raise ValueError(f'num_results must be between 1 and {EXA_MAX_NUM_RESULTS}, got {self.num_results}')
        if not 1 <= self.max_text_chars <= EXA_MAX_PAGE_TEXT_CHARS:
            raise ValueError(
                f'max_text_chars must be between 1 and {EXA_MAX_PAGE_TEXT_CHARS}, got {self.max_text_chars}'
            )
        if self.include_domains and self.exclude_domains:
            raise ValueError('Specify include_domains or exclude_domains, not both.')

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static research guidance: search wide, read the promising pages in full, cite URLs.

        When `include_deep_search` is set, the default guidance also covers
        when to escalate to `deep_search`. A non-`None` `guidance` replaces the
        default; `''` disables instructions entirely.
        """
        if self.guidance is not None:
            return self.guidance or None
        instructions = _INSTRUCTIONS
        if self.include_deep_search:
            instructions += _DEEP_INSTRUCTIONS_SUFFIX
        return instructions

    def get_toolset(self) -> ExaSearchToolset[AgentDepsT]:
        """Build the toolset providing `web_search`, `get_page`, and the optional `deep_search` tool."""
        return ExaSearchToolset[AgentDepsT](
            client=self.client,
            num_results=self.num_results,
            max_text_chars=self.max_text_chars,
            include_deep_search=self.include_deep_search,
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            text_summary=self.text_summary,
        )

    @classmethod
    def from_spec(
        cls,
        *,
        num_results: int = 5,
        max_text_chars: int = 10_000,
        text_summary: bool | str = False,
        include_deep_search: bool = False,
        include_domains: Sequence[str] = (),
        exclude_domains: Sequence[str] = (),
        guidance: str | None = None,
    ) -> ExaSearch[AgentDepsT]:
        """Construct the capability from serializable spec options.

        The `client` field is not spec-serializable, so spec-loaded instances
        always build the default `exa_py.AsyncExa` from `EXA_API_KEY`.
        """
        return cls(
            num_results=num_results,
            max_text_chars=max_text_chars,
            text_summary=text_summary,
            include_deep_search=include_deep_search,
            include_domains=list(include_domains),
            exclude_domains=list(exclude_domains),
            guidance=guidance,
        )
