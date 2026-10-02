"""Rank candidate context with a fast SystemOne decision model: TypeSafe's Jev, or Nimble on Ollama.

The built-in `system_one` plugin. While speculative execution is on, it adds `rank_relevance`, a
read-only tool, and guidance to gather context with it. The `run_code` sandbox may start it
speculatively only while its model runs on this machine: a launch for a branch the snippet never
takes must not send text to a remote API or spend its quota. Jev answers when a key named `JEV_API_KEY` is saved in `/keys` (or set in the environment);
otherwise Nimble on a local Ollama does (`ollama pull nimble`). Both speak the `/v1/systemone` API
through core `SystemOneModel`.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Annotated
from urllib.parse import urlsplit

import anyio
from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field

from pydantic_ai import ModelRetry, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_clai2.config.api_keys import load_keys
from pydantic_clai2.plugins import DepsT, Plugin
from pydantic_clai2.runtime.speculation import speculating

JEV_KEY = 'JEV_API_KEY'
"""The `/keys` name (or environment variable) holding a TypeSafe API key for Jev."""

MAX_CANDIDATES = 64
MAX_CHARS = 12_000
"""Per-candidate text cap, well under Ollama's 64 KiB request and Jev's 32k-token state."""
_CONCURRENCY = 4
_LOOPBACK = frozenset({'localhost', '127.0.0.1', '::1'})

GUIDANCE = """\
SystemOne decision models (TypeSafe's Jev, or Nimble on a local Ollama) are available for
context gathering through `rank_relevance`, a function inside `run_code`. They answer a yes/no
question about a text with a probability, much faster and cheaper than reading the text yourself.
Prefer them whenever you have more candidates than you want to read:

1. Collect candidates cheaply: `list_files`, `grep` hits, or the first lines of files.
2. Rank them: `await rank_relevance(question="Does this code parse the settings file?",
   candidates={"path/a.py": text_a, "path/b.py": text_b})`. Write the question as a yes/no about
   ONE candidate. Candidate keys are your own labels, such as paths or symbols.
3. Read only the top-ranked candidates in full.

A decision model judges; it cannot write, summarize, or follow multi-step reasoning. Treat its
scores as a ranking hint, not proof. If `rank_relevance` reports that no model is available, tell
the user once how to enable it and carry on with the other tools.
"""

Probability = Annotated[float, Field(ge=0, le=1)]
"""A bounded float output: the decision model's probability of yes, unrounded."""


class SystemOneSettings(BaseModel):
    """Where the `system_one` plugin finds its decision models."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    jev_model: str = Field(default='jev-latest', min_length=1, description='Jev model used when JEV_API_KEY is set.')
    jev_url: str = Field(
        default='https://api.typesafe.ai', min_length=1, description='TypeSafe SystemOne API base URL.'
    )
    ollama_model: str = Field(
        default='nimble', min_length=1, description='Local Ollama decision model used without a Jev key.'
    )
    ollama_url: str = Field(default='http://localhost:11434', min_length=1, description='Ollama base URL.')


@dataclass(frozen=True, kw_only=True)
class Backend:
    """The decision model one `rank_relevance` call uses."""

    url: str
    model: str
    api_key: str | None = None

    @property
    def local(self) -> bool:
        """Whether requests stay on this machine, so a discarded speculative launch discloses nothing."""
        return urlsplit(self.url).hostname in _LOOPBACK


def backend(settings: SystemOneSettings) -> Backend:
    """Jev when a key is saved in `/keys` or the environment, otherwise the local Ollama model."""
    saved = load_keys().get(JEV_KEY)
    key = saved.get_secret_value() if saved is not None else os.environ.get(JEV_KEY)
    if key:
        return Backend(url=settings.jev_url, model=settings.jev_model, api_key=key)
    return Backend(url=settings.ollama_url, model=settings.ollama_model)


def unavailable(chosen: Backend, settings: SystemOneSettings, error: Exception) -> str:
    """What the agent should tell the user when no decision model answered."""
    if chosen.api_key is not None:
        return (
            f'Jev ({chosen.model} at {chosen.url}) failed: {error}. Ask the user to check the {JEV_KEY} '
            'key saved in /keys. Meanwhile, gather context with the other tools.'
        )
    return (
        f'No SystemOne decision model is available: {settings.ollama_model} on Ollama at {chosen.url} failed '
        f'({error}). Tell the user that `rank_relevance` needs either a TypeSafe Jev API key saved as {JEV_KEY} '
        f'in /keys, or Ollama (v0.35 or later) running with `ollama pull {settings.ollama_model}`. Meanwhile, '
        'gather context with the other tools.'
    )


async def score(chosen: Backend, question: str, candidates: dict[str, str]) -> dict[str, float] | Exception:
    """Ask the decision model `question` about each candidate, concurrently; the first failure wins."""
    import httpx2

    from pydantic_ai import Agent
    from pydantic_ai.models.system_one import SystemOneModel
    from pydantic_ai.providers.system_one import SystemOneProvider

    scores: dict[str, float] = {}
    failures: list[Exception] = []
    limiter = anyio.CapacityLimiter(_CONCURRENCY)
    async with httpx2.AsyncClient(timeout=60) as client:
        provider = SystemOneProvider(base_url=chosen.url, api_key=chosen.api_key, http_client=client)
        agent = Agent(SystemOneModel(chosen.model, provider=provider), output_type=Probability, instructions=question)

        async def judge(name: str, text: str) -> None:
            async with limiter:
                if failures:
                    return
                try:
                    scores[name] = round((await agent.run(text[:MAX_CHARS])).output, 3)
                except (ModelAPIError, UnexpectedModelBehavior) as exc:
                    failures.append(exc)

        async with anyio.create_task_group() as group:
            for name, text in candidates.items():
                if text.strip():
                    group.start_soon(judge, name, text)
                else:
                    scores[name] = 0.0
    return failures[0] if failures else scores


@dataclass
class SystemOneContext(AbstractCapability[AgentDepsT]):
    """`rank_relevance` and its guidance, offered only on runs with speculative execution on."""

    settings: SystemOneSettings = field(default_factory=SystemOneSettings)

    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        """The one read-only tool, hidden on runs without the speculative sandbox."""
        return FunctionToolset[AgentDepsT]([Tool(self.rank_relevance, prepare=self._prepare)])

    async def _prepare(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
        """Offer the tool only beside the sandbox, and let it speculate only with a local model."""
        if not speculating(ctx):
            return None
        chosen = await to_thread.run_sync(partial(backend, self.settings))
        return replace(tool_def, metadata={**(tool_def.metadata or {}), 'read_only': chosen.local})

    def get_instructions(self) -> Callable[[RunContext[AgentDepsT]], str]:
        """Encourage ranking candidates with a decision model, only when the tool is offered."""

        def describe(ctx: RunContext[AgentDepsT]) -> str:
            return GUIDANCE if speculating(ctx) else ''

        return describe

    async def rank_relevance(self, question: str, candidates: dict[str, str]) -> dict[str, float] | str:
        """Score how relevant each candidate text is to a yes/no question, with a fast SystemOne decision model.

        Use it to gather context: rank files, grep hits, or snippets, then read only the best ones.
        Returns each candidate's probability of yes, most relevant first, or a message saying how
        to enable a decision model when none is available.

        Args:
            question: A yes/no question about one candidate, e.g. 'Does this code retry failed requests?'.
            candidates: Your label for each candidate (a path, a symbol) mapped to its text.
        """
        if len(candidates) > MAX_CANDIDATES:
            raise ModelRetry(f'Pass at most {MAX_CANDIDATES} candidates; narrow them with grep first.')
        if not candidates:
            return {}
        chosen = await to_thread.run_sync(partial(backend, self.settings))
        result = await score(chosen, question, candidates)
        if isinstance(result, Exception):
            return unavailable(chosen, self.settings, result)
        return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))


class SystemOnePlugin(Plugin[SystemOneSettings, DepsT]):
    """Rank candidate context with a SystemOne decision model while speculative execution is on."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (SystemOneContext[DepsT](settings=self.settings),)
