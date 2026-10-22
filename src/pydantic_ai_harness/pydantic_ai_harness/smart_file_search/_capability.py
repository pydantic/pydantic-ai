"""The `SmartFileSearch` capability: plain-English code search judged by a pluggable model."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.smart_file_search._toolset import SmartFileSearchToolset


@dataclass
class SmartFileSearch(AbstractCapability[AgentDepsT]):
    """Plain-English code search: find code by what it does, described in plain English.

    Adds a `smart_file_search` tool. A search lists the files under a directory in the
    run's workspace with `rg`, cuts them into snippets (overlapping line windows, or the
    functions, methods and long blocks of a Python file), shortlists the snippets with
    local BM25, and has a judge model score each shortlisted snippet's relevance.
    Matches come back ranked, with file and line ranges and a focused excerpt,
    plus coverage figures so an empty result is never mistaken for absence.

    The judge is any Pydantic AI model. A decision model is the recommended kind,
    since it answers each relevance question with a probability: any one served
    over the System One API (`'system-one:<name>'`), or TypeSafe's Jev
    (`'typesafe:jev-latest'`), which is the default when the `typesafe` SDK is
    installed and `TYPESAFE_API_KEY` is set. Otherwise the judge defaults to the
    run's own model; pass a small, fast `model` instead, as one search makes up
    to `candidates` (default 128) judge requests.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace

    from pydantic_ai_harness import SmartFileSearch

    agent = Agent(
        'openai:gpt-6-luna',
        capabilities=[LocalWorkspace('.'), SmartFileSearch(model='system-one:clm-latest')],
    )
    ```

    Files are listed and read through `ctx.workspace`, so attach a workspace
    that can run commands and has `rg` on its `PATH` (the `coder` extra
    installs it for a local workspace); the tool is not offered on one that
    cannot run commands, and a run without a workspace fails at its start.
    """

    model: Model | KnownModelName | str | None = None
    """The model that judges each snippet's relevance: any Pydantic AI model or model name.

    A decision model served over the System One API, such as
    `'system-one:clm-latest'`, or a `SystemOneModel`, answers each judgment with a
    probability. `None` (the default) picks TypeSafe's Jev (`'typesafe:jev-latest'`)
    when the `typesafe` SDK is installed and `TYPESAFE_API_KEY` is set, and the
    run's own model otherwise. Pin a model version (e.g. `'typesafe:jev-1.13.0'`)
    once you have tuned `threshold` against it.
    """

    threshold: float = 0.5
    """Relevance a snippet needs to be returned, from 0 to 1.

    A snippet's relevance is the product of two judgments: that it acts on what
    the query names, and that it performs the action the query describes.
    """

    concurrency: int = 8
    """How many snippets are judged at once."""

    cache_index: bool = False
    """Keep each searched directory's index in memory between searches, re-indexing only changed files.

    Off, every search reads, chunks and indexes its directory from scratch. On, later searches of the same
    directory still read every file to detect changes, but only re-chunk the ones that changed, and
    parallel searches of it build the index once. Up to four directories (with their `glob`) are kept for
    the life of the agent, so this holds memory in proportion to the source searched.
    """

    guidance: str | None = None
    """Custom discovery guidance for the system prompt, added whenever the tool is offered.

    Leave as `None` for the default, which tells the model to reach for
    `smart_file_search` before exploratory text search, or set `''` to contribute no
    instructions at all.
    """

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError(f'threshold must be between 0 and 1, got {self.threshold}')
        # Negated so NaN, which an agent spec can pass through unvalidated, fails too instead of hanging the search.
        if not self.concurrency >= 1:
            raise ValueError(f'concurrency must be a positive integer, got {self.concurrency}')

    def get_toolset(self) -> SmartFileSearchToolset[AgentDepsT]:
        """Build the toolset providing `smart_file_search`."""
        return SmartFileSearchToolset[AgentDepsT](
            model=self.model,
            threshold=self.threshold,
            concurrency=self.concurrency,
            cache_index=self.cache_index,
            guidance=self.guidance,
            id=self.id or 'smart_file_search',
        )
