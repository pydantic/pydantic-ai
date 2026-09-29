"""The `SmartGrep` capability: plain-English code search judged by a pluggable model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness.smart_grep._toolset import SmartGrepToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_INSTRUCTIONS = """\
## Code discovery: smart_grep first
Use `smart_grep` as your first tool for gathering code context about an unfamiliar implementation.
Describe the behavior you need to find in plain English, scoped to the relevant directory. Do this
instead of guessing symbol names, running exploratory grep chains, or listing and reading many files
just to locate the implementation.

Use the returned excerpts and line ranges to decide which files need a targeted read. Do not
automatically read every match or repeat searches when the excerpts already answer the question.
Read the relevant code before editing; `smart_grep` is discovery, not a substitute for
understanding it.

Use regular text search for exact symbols, regexes, and exhaustive references, and read a known file
directly when its location is already established.

The lexical shortlist is not exhaustive, and no matches do not prove absence. If results are weak,
rephrase or narrow the query, or fall back to exact search. If `smart_grep` fails, fall back to
regular search and targeted reads rather than retrying. `smart_grep` sends selected source, paths and
the query to the judge model's provider, so honor requests to keep code local by using local search
instead."""


@dataclass
class SmartGrep(AbstractCapability[AgentDepsT]):
    """Plain-English code search: find code by what it does, described in plain English.

    Adds a `smart_grep` tool. A search lists the files under a directory in the
    run's workspace with `rg`, cuts them into syntax-aware snippets (functions,
    methods, and the blocks of long ones), shortlists the snippets with local
    BM25, and has a judge model score each shortlisted snippet's relevance.
    Matches come back ranked, with file and line ranges and a focused excerpt,
    plus coverage figures so an empty result is never mistaken for absence.

    The judge is any Pydantic AI model. TypeSafe's Jev decision model
    (`'typesafe:jev-latest'`) is the recommended judge, since it answers each
    relevance question with a calibrated probability, and is the default when
    the `typesafe` SDK is installed and `TYPESAFE_API_KEY` is set. Otherwise the
    judge defaults to the run's own model; pass a small, fast `model` instead,
    as one search makes up to `candidates` (default 128) judge requests.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace

    from pydantic_ai_harness import SmartGrep

    agent = Agent(
        'openai:gpt-6-luna',
        capabilities=[LocalWorkspace('.'), SmartGrep(model='typesafe:jev-latest')],
    )
    ```

    Files are listed and read through `ctx.workspace`, so attach a workspace
    that can run commands and has `rg` on its `PATH` (the `coder` extra
    installs it for a local workspace); the tool is not offered on one that
    cannot run commands, and a run without a workspace fails at its start.
    Install the `smart-grep` extra for syntax-aware chunking of fifteen
    languages beyond Python; without it, those files are cut into overlapping
    line windows.
    """

    model: Model | KnownModelName | str | None = None
    """The model that judges each snippet's relevance: any Pydantic AI model or model name.

    `None` (the default) picks TypeSafe's Jev (`'typesafe:jev-latest'`) when the
    `typesafe` SDK is installed and `TYPESAFE_API_KEY` is set, and the run's own
    model otherwise. Pin a Jev version (e.g. `'typesafe:jev-1.13.0'`) once you
    have tuned `threshold` against it.
    """

    threshold: float = 0.5
    """Relevance a snippet needs to be returned, from 0 to 1.

    A snippet's relevance is the product of two judgments: that it acts on what
    the query names, and that it performs the action the query describes.
    """

    concurrency: int = 8
    """How many snippets are judged at once."""

    guidance: str | None = None
    """Custom discovery guidance for the system prompt.

    Leave as `None` for the default, which tells the model to reach for
    `smart_grep` before exploratory text search, or set `''` to contribute no
    instructions at all.
    """

    def __post_init__(self) -> None:
        if not 0 <= self.threshold <= 1:
            raise ValueError(f'threshold must be between 0 and 1, got {self.threshold}')
        # Negated so NaN, which an agent spec can pass through unvalidated, fails too instead of hanging the search.
        if not self.concurrency >= 1:
            raise ValueError(f'concurrency must be a positive integer, got {self.concurrency}')

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """The discovery policy: smart_grep first for unfamiliar behaviour, exact search for exact symbols."""
        if self.guidance is not None:
            return self.guidance or None
        return _INSTRUCTIONS

    def get_toolset(self) -> SmartGrepToolset[AgentDepsT]:
        """Build the toolset providing `smart_grep`."""
        return SmartGrepToolset[AgentDepsT](model=self.model, threshold=self.threshold, concurrency=self.concurrency)
