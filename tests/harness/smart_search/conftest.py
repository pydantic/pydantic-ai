"""Shared helpers for the `SmartSearch` tests: judges, workspaces, and running the tool through an `Agent`.

Every search goes through the public surface: an `Agent` whose `FunctionModel` calls `smart_search`, with a
judge whose requests show which snippets (`path:line-end (symbol)`) it was asked about.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import pydantic_ai_harness.smart_search._index as smart_search_index
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.decision import (
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceTimeoutError
from pydantic_ai_harness.smart_search import SmartSearch, SmartSearchResult

PY_SOURCE = textwrap.dedent(
    '''\
    import os
    import sys


    def reject_expired(session):
        if session.expires_at < now():
            raise PermissionError("expired")
        return session


    class Cache:
        """A cache."""

        def get(self, key):
            return self._data.get(key)
    '''
)


def write_repo(root: Path) -> Path:
    """A small repository: the code under search, a test of it, and unrelated code."""
    (root / 'auth.py').write_text(PY_SOURCE)
    (root / 'tests').mkdir()
    (root / 'tests' / 'test_auth.py').write_text(
        'def test_expired_session_is_rejected():\n    assert reject_expired(old) is None\n'
    )
    (root / 'noise.py').write_text('def unrelated():\n    return 42\n')
    return root


@dataclass(init=False)
class FakeJev(DecisionModel[None]):
    """Answers P(yes)=`hit` when `needle` is in the state, else `miss`."""

    needle: str
    hit: float
    miss: float
    requests: list[DecisionRequest] = field(default_factory=list[DecisionRequest])

    def __init__(self, needle: str, hit: float = 0.9, miss: float = 0.1) -> None:
        self.needle, self.hit, self.miss = needle, hit, miss
        self.requests = []
        super().__init__()

    @property
    def model_name(self) -> str:
        return 'jev-fake'

    @property
    def system(self) -> str:
        return 'fake'

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        self.requests.append(request)
        p = self.hit if self.needle in str(request.state) else self.miss
        answers: dict[str, DecisionAnswer] = {name: NoulAnswer(noul=p) for name in request.questions}
        return DecisionResponse(answers=answers, model_name=self.model_name)


def language_judge(needle: str) -> FunctionModel:
    """A plain language model judge: fills `Relevance` as structured output."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = next(part.content for part in messages[-1].parts if isinstance(part, UserPromptPart))
        score = 0.9 if needle in str(prompt) else 0.1
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'entity': score, 'operation': score})])

    return FunctionModel(respond)


@dataclass(frozen=True)
class Judged:
    """A snippet the judge was asked about, read back from the state it was sent."""

    path: str
    line: int
    end_line: int
    symbol: str | None
    text: str

    @property
    def span(self) -> tuple[int, int, str | None]:
        return self.line, self.end_line, self.symbol


_HEADER = re.compile(r'(?P<path>.+):(?P<line>\d+)-(?P<end>\d+)(?: \((?P<symbol>.+)\))?')


def judged(model: FakeJev) -> list[Judged]:
    """Every snippet `model` judged, in file and line order."""
    out: list[Judged] = []
    for request in model.requests:
        header, _, text = str(request.state).partition('\n\n')
        match = _HEADER.fullmatch(header)
        assert match is not None, header
        out.append(Judged(match['path'], int(match['line']), int(match['end']), match['symbol'], text))
    return sorted(out, key=lambda s: (s.path, s.line, s.end_line))


def calls_smart_search(*calls: Mapping[str, object]) -> FunctionModel:
    """A run model that calls `smart_search` once per `calls` in one response (once by default), then stops."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('smart_search', {'query': 'reject expired sessions', **args})
                    for args in calls or ({},)
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def search_agent(
    root: Path | None,
    model: Model | str,
    *,
    threshold: float = 0.5,
    concurrency: int = 8,
    cache_index: bool = False,
) -> Agent[None, str]:
    """An agent searching `root` (or the run's `workspace`) with a `SmartSearch` judged by `model`."""
    capability = SmartSearch(model=model, threshold=threshold, concurrency=concurrency, cache_index=cache_index)
    run_model = calls_smart_search()  # each run passes its own calls
    if root is None:
        return Agent(run_model, capabilities=[capability])
    return Agent(run_model, capabilities=[LocalWorkspace(root), capability])


async def search_messages(
    agent: Agent[None, str], *calls: Mapping[str, object], workspace: Workspace | None = None
) -> list[ModelMessage]:
    # Unlimited requests: each judgment counts as one, and the request limit is tested on its own.
    result = await agent.run(
        'find it',
        model=calls_smart_search(*calls),
        workspace=workspace,
        usage_limits=UsageLimits(request_limit=None),
    )
    return result.all_messages()


def tool_returns(messages: Sequence[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


async def search_all(
    agent: Agent[None, str], *calls: Mapping[str, object], workspace: Workspace | None = None
) -> list[SmartSearchResult]:
    """The results of `smart_search` calls made in parallel in one model response."""
    results: list[SmartSearchResult] = []
    for part in tool_returns(await search_messages(agent, *calls, workspace=workspace)):
        assert isinstance(part.content, SmartSearchResult), part.content
        results.append(part.content)
    return results


async def search(agent: Agent[None, str], *, workspace: Workspace | None = None, **args: object) -> SmartSearchResult:
    """The result of one `smart_search` call with `args`."""
    [result] = await search_all(agent, args, workspace=workspace)
    return result


async def search_error(agent: Agent[None, str], *, workspace: Workspace | None = None, **args: object) -> str:
    """What the model was told when one `smart_search` call with `args` asked for a retry or failed."""
    [part] = tool_returns(await search_messages(agent, args, workspace=workspace))
    assert not isinstance(part.content, SmartSearchResult)
    return str(part.content)


class VanishingBackend(LocalWorkspaceBackend):
    """Lists files normally, then fails to read the ones named `gone*`, `slow*` or `odd*`."""

    async def read_bytes(self, path: str) -> bytes:
        name = Path(path).name
        if name.startswith('gone'):
            raise FileNotFoundError(2, 'No such file or directory', path)
        if name.startswith('slow'):
            raise WorkspaceTimeoutError('read timed out')
        if name.startswith('odd'):
            raise OSError('odd failure')
        return await super().read_bytes(path)


@pytest.fixture
def chunked(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The paths the index chunks, in order, to tell a rebuilt index from a reused one.

    Nothing public tells them apart: every search re-reads every file to detect changes, and both return the same
    result. So this counts calls to the index's chunker.
    """
    paths: list[str] = []
    source_chunks = smart_search_index.source_chunks

    def counting(text: str, path: str):
        paths.append(path)
        return source_chunks(text, path)

    monkeypatch.setattr(smart_search_index, 'source_chunks', counting)
    return paths
