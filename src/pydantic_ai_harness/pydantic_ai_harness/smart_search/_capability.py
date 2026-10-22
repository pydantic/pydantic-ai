"""The `SmartSearch` capability: plain-English code search judged by a pluggable model."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.smart_search._toolset import (
    DEFAULT_CONCURRENCY,
    DEFAULT_THRESHOLD,
    SmartSearchToolset,
    validate_settings,
)


@dataclass
class SmartSearch(AbstractCapability[AgentDepsT]):
    """Plain-English code search: find code by what it does, described in plain English.

    Adds a `smart_search` tool. A search lists the files under a directory in the
    run's workspace with `rg`, cuts them into snippets (overlapping line windows, or the
    functions, methods and long blocks of a Python file), shortlists the snippets with
    local BM25, and has a judge model score each shortlisted snippet's relevance.
    Matches come back ranked, with file and line ranges and a focused excerpt,
    plus coverage figures so the model can tell an empty result from a complete
    search that found nothing.

    The judge is any Pydantic AI model, and is required, since it decides which
    provider receives the searched source. A decision model is the recommended
    kind, since it answers each relevance question with a probability: any one
    served over the System One API (`'system-one:<name>'`), or TypeSafe's Jev
    (`'typesafe:jev-latest'`). One search makes up to `candidates` (default 128)
    judge requests, each counted as one request in the run's usage and checked
    against its usage limits.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace

    from pydantic_ai_harness import SmartSearch

    agent = Agent(
        'openai:gpt-6-astra',
        capabilities=[LocalWorkspace('.'), SmartSearch('system-one:clm-latest')],
    )
    ```

    Files are listed and read through `ctx.workspace`, so attach a workspace
    that can run commands and has `rg` on its `PATH` (the `coder` extra
    installs it for a local workspace); the tool is not offered on one that
    cannot run commands, and a run without a workspace fails at its start.
    """

    model: Model | KnownModelName | str
    """The model that judges each snippet's relevance: any Pydantic AI model or model name.

    A decision model served over the System One API, such as
    `'system-one:clm-latest'`, or a `SystemOneModel`, answers each judgment with a
    probability, as does TypeSafe's Jev (`'typesafe:jev-latest'`). Pin a model
    version (e.g. `'typesafe:jev-1.13.0'`) once you have tuned `threshold` against it.
    """

    _: KW_ONLY

    threshold: float = DEFAULT_THRESHOLD
    """Relevance a snippet needs to be returned, from 0 to 1.

    A snippet's relevance is the product of two judgments: that it acts on what
    the query names, and that it performs the action the query describes.
    """

    concurrency: int = DEFAULT_CONCURRENCY
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
    `smart_search` before exploratory text search, or set `''` to contribute no
    instructions at all.
    """

    _toolset: SmartSearchToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_settings(self.threshold, self.concurrency)

    def get_toolset(self) -> SmartSearchToolset[AgentDepsT]:
        """Provide the stable `smart_search` toolset, shared with every per-run copy.

        A run whose capabilities differ from the agent's (an instrumented one, or one given extra
        `capabilities`) asks for the toolset again; a new one would start with an empty `cache_index`.
        """
        if self._toolset is None:
            self._toolset = SmartSearchToolset[AgentDepsT](
                model=self.model,
                threshold=self.threshold,
                concurrency=self.concurrency,
                cache_index=self.cache_index,
                guidance=self.guidance,
                id=self.id or 'smart_search',
            )
        return self._toolset
