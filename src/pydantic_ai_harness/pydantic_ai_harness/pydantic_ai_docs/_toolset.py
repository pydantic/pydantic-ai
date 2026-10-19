"""Docs toolset: a single `read_pyai_docs` tool that locates one pyai doc on demand."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path

import httpx
from pydantic import TypeAdapter

from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._workspace import workspace_path

_REMOTE_BASE = 'https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs'
"""Raw-markdown base for the live fallback. Tracks `pydantic/pydantic-ai:main`,
and each topic maps to `{base}/{page}` -- byte-identical to a local checkout."""

_FETCH_TIMEOUT = 30.0
"""Per-request timeout (seconds) for the remote markdown fetch."""

_MODEL_ALIAS_RE = re.compile(r'\{\{ ([a-z0-9-]+:[a-z]+)(\.name)? \}\}')
"""A model alias in a docs page, as the docs site expands it.

`{{ openai:medium }}` stands for the full model string, like `openai:<model>`, and
`{{ openai:medium.name }}` for the bare model name, like `<model>`."""

_MODEL_ALIASES_FILE = 'model_aliases/aliases.json'
"""The file, relative to the docs directory, that maps each model alias to the model it stands for."""

_MODEL_ALIASES_ADAPTER = TypeAdapter(dict[str, str])


# Preserve string formatting of existing public enum members.
class PydanticAIDocsTopic(str, Enum):  # noqa: UP042
    """A Pydantic AI documentation page that `read_pyai_docs` can return.

    The value is the topic name the model passes. `_PAGES` maps it to the page's
    path relative to the pydantic-ai `docs/` directory, used for both a local
    checkout file and a raw-markdown URL.
    """

    capabilities = 'capabilities'
    hooks = 'hooks'
    tools = 'tools'
    tools_advanced = 'tools-advanced'
    toolsets = 'toolsets'
    agent = 'agent'


_PAGES: dict[PydanticAIDocsTopic, str] = {
    PydanticAIDocsTopic.capabilities: 'capabilities/overview.md',
    PydanticAIDocsTopic.hooks: 'hooks.md',
    PydanticAIDocsTopic.tools: 'tools.md',
    PydanticAIDocsTopic.tools_advanced: 'tools-advanced.md',
    PydanticAIDocsTopic.toolsets: 'toolsets.md',
    PydanticAIDocsTopic.agent: 'agent.md',
}
"""Each topic's page relative to pydantic-ai's `docs/` directory."""


class PydanticAIDocsToolset(FunctionToolset[AgentDepsT]):
    """Exposes one `read_pyai_docs` tool that locates and returns a pyai doc.

    Resolution per call: a configured local checkout first (when the topic's
    page exists there), otherwise a raw-markdown fetch from `main`. The full doc
    is returned with its model aliases, like `{{ openai:medium }}`, replaced by the
    models they stand for, read from the docs' alias file through the same
    resolution. Results are memoized in the run-scoped `cache` dict when caching
    is enabled.
    """

    def __init__(
        self,
        *,
        local_docs_path: Path | None,
        cache: dict[PydanticAIDocsTopic, str] | None,
    ) -> None:
        super().__init__()
        self._local_docs_path = local_docs_path
        # Shared with the run-scoped capability; `None` disables caching entirely.
        self._cache = cache
        self.add_function(self.read_pyai_docs, name='read_pyai_docs')

    async def read_pyai_docs(self, ctx: RunContext[AgentDepsT], topic: PydanticAIDocsTopic) -> str:  # noqa: D417
        """Return the Pydantic AI documentation for `topic` as markdown.

        Reads from a configured local docs checkout when available, otherwise
        fetches the page from the live docs source. Returns the full page; call
        it once per topic you need.

        Args:
            topic: Which documentation page to return. One of `capabilities`,
                `hooks`, `tools`, `tools-advanced`, `toolsets`, `agent`.
        """
        if self._cache is not None and topic in self._cache:
            return self._cache[topic]

        markdown = await self._read(ctx, _PAGES[topic])
        # Only pages that use aliases need the alias file, so checkouts from before it existed keep working.
        if _MODEL_ALIAS_RE.search(markdown):
            try:
                aliases = _MODEL_ALIASES_ADAPTER.validate_json(await self._read(ctx, _MODEL_ALIASES_FILE))
            except RuntimeError:
                # The page is still worth returning with its aliases as written.
                aliases = {}

            def resolve(match: re.Match[str]) -> str:
                alias, bare = match.groups()
                if (model := aliases.get(alias)) is None:
                    return match.group()
                return model.partition(':')[2] if bare else model

            markdown = _MODEL_ALIAS_RE.sub(resolve, markdown)

        if self._cache is not None:
            self._cache[topic] = markdown
        return markdown

    async def _read(self, ctx: RunContext[AgentDepsT], file: str) -> str:
        """Return `file`, relative to the docs directory, from the local checkout, else from the live source."""
        text = await self._read_local(ctx, file)
        if text is None:
            text = await self._fetch_remote(file)
        return text

    async def _read_local(self, ctx: RunContext[AgentDepsT], file: str) -> str | None:
        """Return the local checkout's `file`, or `None` to fall back to remote."""
        if self._local_docs_path is None:
            return None
        local_path = self._local_docs_path / file
        path = await ctx.workspace.resolve(workspace_path(local_path))
        try:
            return await ctx.workspace.read_text(path)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def _fetch_remote(self, file: str) -> str:
        """Fetch `file` from the live source, or raise a descriptive error."""
        url = f'{_REMOTE_BASE}/{file}'
        try:
            async with httpx.AsyncClient(timeout=_FETCH_TIMEOUT) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            local = 'no local checkout configured' if self._local_docs_path is None else str(self._local_docs_path)
            raise RuntimeError(
                f'Could not locate {file!r} in the Pydantic AI docs. Local source: {local}. '
                f'Remote fetch from {url} failed: {exc}'
            ) from exc
        return response.text
