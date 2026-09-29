"""Tests for `SmartFileSearch`: chunking, the pluggable judge, search, and the tool (the index is in `test_index.py`).

Ported from Code Puppy's `code_puppy_core_plugins/jev_grep` tests. The judge is exercised two ways, to pin
that it is pluggable: a fake `DecisionModel` answering the real Decisions protocol (the path TypeSafe's Jev
takes), and a plain `FunctionModel` filling the same fields as structured output.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from pydantic_ai import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import ModelHTTPError, ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.decision import (
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend, ReadOnlyWorkspace, Workspace, WorkspaceTimeoutError
from pydantic_ai_harness import SmartFileSearch
from pydantic_ai_harness.smart_file_search import (
    SmartFileSearchResult,
    SmartFileSearchToolset,
    _index,
    _judge,
)
from pydantic_ai_harness.smart_file_search._chunks import Chunk, LineTooLong, source_chunks, windows
from pydantic_ai_harness.smart_file_search._index import SnippetIndexes
from pydantic_ai_harness.smart_file_search._judge import TYPESAFE_MODEL, judge, resolve_judge_model
from pydantic_ai_harness.smart_file_search._retrieve import terms
from pydantic_ai_harness.smart_file_search._search import search_code

from ..conftest import agent_run_names
from .conftest import CountingChunker

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.usefixtures('allow_model_requests')  # `FakeJev` is a real `DecisionModel`


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


def _workspace(root: Path) -> Workspace:
    return Workspace(LocalWorkspaceBackend(root))


def _repo(root: Path) -> Path:
    (root / 'auth.py').write_text(PY_SOURCE)
    (root / 'tests').mkdir()
    (root / 'tests' / 'test_auth.py').write_text(
        'def test_expired_session_is_rejected():\n    assert reject_expired(old) is None\n'
    )
    (root / 'noise.py').write_text('def unrelated():\n    return 42\n')
    return root


async def _search(
    root: Path, model: Model, query: str = 'reject expired sessions', *, limit: int = 5, candidates: int | None = None
) -> SmartFileSearchResult:
    workspace = _workspace(root)
    indexes = SnippetIndexes(0)
    if candidates is None:  # the tool's own default
        return await search_code(
            workspace, model, query, '.', indexes=indexes, limit=limit, threshold=0.5, concurrency=8
        )
    return await search_code(
        workspace, model, query, '.', indexes=indexes, limit=limit, candidates=candidates, threshold=0.5, concurrency=8
    )


class VanishingBackend(LocalWorkspaceBackend):
    """Lists files normally, then fails to read the ones named `gone*` or `slow*`."""

    async def read_bytes(self, path: str) -> bytes:
        name = Path(path).name
        if name.startswith('gone'):
            raise FileNotFoundError(2, 'No such file or directory', path)
        if name.startswith('slow'):
            raise WorkspaceTimeoutError('read timed out')
        if name.startswith('odd'):
            raise OSError('odd failure')
        return await super().read_bytes(path)


# ---------------------------------------------------------------- chunks


def test_python_chunks_follow_declarations() -> None:
    chunks, parser = source_chunks(PY_SOURCE, 'a.py')
    assert parser == 'python'
    by_symbol = {c.symbol: c for c in chunks}
    imports = by_symbol[None]  # adjacent one-liners merge into one target
    assert (imports.line, imports.end_line) == (1, 2)
    fn = by_symbol['reject_expired']
    assert (fn.line, fn.end_line) == (3, 8)  # preceding blank gap is absorbed
    assert 'raise PermissionError' in fn.text
    assert by_symbol['Cache.get'].text.strip().startswith('def get')


def test_decorated_python_declaration_starts_at_its_decorator() -> None:
    chunks, _ = source_chunks('import os\n\n\n@cache\ndef load():\n    return os.environ\n', 'a.py')
    assert [(c.line, c.end_line, c.symbol) for c in chunks] == [(1, 1, None), (2, 6, 'load')]


def test_long_function_is_split_into_blocks_covering_every_line() -> None:
    body = '\n'.join(
        f'    if x == {i}:\n        a = {i}\n        b = {i}\n        c = {i}\n        d = {i}\n        e = {i}'
        for i in range(6)
    )
    src = f'def router(x):\n{body}\n'
    chunks, _ = source_chunks(src, 'r.py')
    assert len(chunks) > 1
    assert all(c.symbol == 'router' for c in chunks)
    covered = {n for c in chunks for n in range(c.line, c.end_line + 1)}
    assert covered == set(range(1, len(src.splitlines()) + 1))


def test_deeply_nested_single_statement_bodies_stop_splitting() -> None:
    src = 'def f():\n  with a:\n    with b:\n      with c:\n        with d:\n' + ''.join(
        f'          x{i} = {i}\n' for i in range(30)
    )
    chunks, _ = source_chunks(src, 'nested.py')
    assert [(c.line, c.end_line) for c in chunks] == [(1, 35)]


def test_non_python_and_broken_python_fall_back_to_windows() -> None:
    text = '\n'.join(f'line {i}' for i in range(130))
    for path in ('x.txt', 'broken.py'):
        src = text if path == 'x.txt' else 'def (:\n' + text
        chunks, parser = source_chunks(src, path)
        assert parser == 'overlapping-lines'
        assert chunks[0].line == 1 and chunks[0].end_line == 60
        assert chunks[1].line == 51  # 10-line overlap


def test_long_whitespace_lines_are_skipped_not_fatal() -> None:
    chunks = windows(['x = 1', ' ' * 13_000, 'y = 2'], 'w.py', size=1, overlap=0)
    assert [c.text for c in chunks] == ['x = 1', 'y = 2']


def test_giant_line_raises_and_giant_window_halves() -> None:
    with pytest.raises(LineTooLong):
        windows(['x' * 13_000], 'min.js')
    halves = windows(['y' * 5_000] * 4, 'big.txt', size=4, overlap=0)
    assert [(c.line, c.end_line) for c in halves] == [(1, 2), (3, 4)]


# ---------------------------------------------------------------- retrieve


def test_terms_split_identifiers_and_drop_stopwords() -> None:
    assert terms('where is refreshToken_value for HTTPServer') == ['refresh', 'token', 'value', 'httpserver']


# ---------------------------------------------------------------- judge


async def test_decision_judge_asks_two_probability_questions_and_multiplies() -> None:
    model = FakeJev(needle='expired', hit=0.8, miss=0.1)
    hit = Chunk(path='s.py', line=3, end_line=4, text='if expired: deny()', symbol='check')
    miss = Chunk(path='t.py', line=1, end_line=1, text="print('hi')")
    scores = await judge(model, 'reject expired sessions', [hit, miss], concurrency=2)
    assert scores == pytest.approx([0.64, 0.01])

    request = model.requests[0]
    assert set(request.questions) == {'entity', 'operation'}
    assert all(isinstance(q, NoulQuestion) for q in request.questions.values())
    states = {str(r.state) for r in model.requests}
    assert any('s.py:3-4 (check)' in s for s in states)  # path + symbol in state


async def test_any_language_model_can_judge() -> None:
    hit = Chunk(path='s.py', line=3, end_line=4, text='if expired: deny()')
    miss = Chunk(path='t.py', line=1, end_line=1, text="print('hi')")
    scores = await judge(language_judge('expired'), 'reject expired sessions', [hit, miss], concurrency=1)
    assert scores == pytest.approx([0.81, 0.01])


async def test_a_failed_judgment_fails_the_search_with_the_real_error() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(503, 'judge', 'backend down')

    chunks = [Chunk(path=f'{i}.py', line=1, end_line=1, text='x') for i in range(5)]
    with pytest.raises(ModelHTTPError, match='backend down'):
        await judge(FunctionModel(respond), 'q', chunks, concurrency=2)


def test_judge_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    run_model = TestModel()
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    assert resolve_judge_model('openai:gpt-5-mini', run_model) == 'openai:gpt-5-mini'
    assert resolve_judge_model(None, run_model) is run_model  # no TypeSafe: the run's own model
    with pytest.raises(UserError, match='could not pick a judge model'):
        resolve_judge_model(None, object())

    monkeypatch.setenv('TYPESAFE_API_KEY', 'k')

    def sdk_spec(name: str) -> ModuleSpec:
        return ModuleSpec(name, None)

    monkeypatch.setattr(_judge, 'find_spec', sdk_spec)  # the SDK is optional, so it may not be installed here
    assert resolve_judge_model(None, run_model) == TYPESAFE_MODEL  # recommended default when available
    assert resolve_judge_model('openai:gpt-5-mini', run_model) == 'openai:gpt-5-mini'  # never forced

    def no_spec(name: str) -> None:
        return None

    monkeypatch.setattr(_judge, 'find_spec', no_spec)
    assert resolve_judge_model(None, run_model) is run_model  # key but no SDK


# ---------------------------------------------------------------- search


@pytest.mark.parametrize('model', [FakeJev('expired'), language_judge('expired')], ids=['decision', 'language'])
async def test_search_code_ranks_labels_and_reports_coverage(tmp_path: Path, model: Model) -> None:
    out = await _search(_repo(tmp_path), model)
    assert {m.file_path for m in out.matches} == {'auth.py', 'tests/test_auth.py'}
    test_match = next(m for m in out.matches if m.file_path == 'tests/test_auth.py')
    assert test_match.kind == 'test'
    assert out.coverage.selection_complete and out.coverage.files == 3
    assert out.warnings == []


async def test_search_code_no_match_limit_and_shortlist(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    none = await _search(repo, FakeJev('zzz'))
    assert none.matches == [] and 'does not prove absence' in none.warnings[0]

    capped = await _search(repo, FakeJev('expired'), limit=1, candidates=2)
    assert len(capped.matches) == 1 and capped.omitted_matches == 1
    assert capped.coverage.evaluated == 2
    assert any('lexical shortlist' in w for w in capped.warnings)


async def test_empty_directory_judges_nothing(tmp_path: Path) -> None:
    model = FakeJev('x')
    out = await _search(tmp_path, model)
    assert out == SmartFileSearchResult(coverage=out.coverage)
    assert out.coverage.selection_complete and not model.requests


async def test_skipped_files_are_reported(tmp_path: Path) -> None:
    (tmp_path / 'bin.dat').write_bytes(b'\0')
    out = await _search(tmp_path, FakeJev('x'))
    assert out.coverage.skipped_files == 1
    assert out.warnings == ['1 files skipped (binary, non-UTF-8, minified or unreadable).']


@pytest.mark.parametrize(('requested', 'expected'), [(None, 128), (48, 48), (0, 1), (500, 140)])
async def test_candidate_budget_default_and_bounds(tmp_path: Path, requested: int | None, expected: int) -> None:
    for i in range(140):
        (tmp_path / f'file_{i}.py').write_text('expired = True\n')
    model = FakeJev('expired')
    out = await _search(tmp_path, model, candidates=requested)
    assert out.coverage.evaluated == expected
    assert len(model.requests) == expected


@pytest.mark.parametrize('query', ['   ', 'x' * 2001])
async def test_search_code_rejects_bad_query(tmp_path: Path, query: str) -> None:
    with pytest.raises(ModelRetry, match='1-2000'):
        await _search(tmp_path, FakeJev('x'), query)


async def test_overlapping_windows_dedupe_and_excerpt_is_capped(tmp_path: Path) -> None:
    lines = [f'filler {i}' for i in range(100)]
    lines[55] = 'the retry loop handles backoff'  # line 56: inside both windows (1-60 and 51-100)
    (tmp_path / 'log.txt').write_text('\n'.join(lines))
    out = await _search(tmp_path, FakeJev('retry'), 'retry with backoff')
    assert len(out.matches) == 1  # windows 1-60 and 51-100 overlap; one survives
    match = out.matches[0]
    assert match.end_line - match.start_line + 1 <= 12
    assert 'retry loop' in match.text  # query-focused excerpt window


async def test_excerpt_focuses_on_a_narrower_passing_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lines = [f'line {i}' for i in range(1, 41)]
    lines[29] = 'expired = True'
    whole = Chunk(path='a.py', line=1, end_line=40, text='\n'.join(lines), symbol='f')
    block = Chunk(path='a.py', line=28, end_line=32, text='\n'.join(lines[27:32]), symbol='f')
    other = Chunk(path='a.py', line=2, end_line=3, text='\n'.join(lines[1:3]), symbol='f')

    def nested(text: str, path: str) -> tuple[list[Chunk], str]:
        return [whole, block, other], 'python'

    (tmp_path / 'a.py').write_text('\n'.join(lines))
    monkeypatch.setattr(_index, 'source_chunks', nested)
    out = await _search(tmp_path, FakeJev('expired', hit=0.9, miss=0.5), 'expired')
    [match] = out.matches  # `block` collapses into `whole`, which ranks first on the tie; `other` fails
    assert (match.snippet_start_line, match.snippet_end_line) == (1, 40)
    assert (match.start_line, match.end_line) == (28, 32)  # the best narrower block that also passed
    assert 'expired = True' in match.text


async def test_long_lines_trim_the_excerpt_to_its_character_budget(tmp_path: Path) -> None:
    (tmp_path / 'wide.txt').write_text('\n'.join(f'expired {"y" * 400}' for _ in range(5)))
    out = await _search(tmp_path, FakeJev('expired'), 'expired')
    [match] = out.matches
    assert len(match.text) <= 1200 and match.end_line - match.start_line + 1 == 2


# ---------------------------------------------------------------- capability


def test_capability_validates_its_settings() -> None:
    with pytest.raises(ValueError, match='threshold'):
        SmartFileSearch[None](threshold=1.5)
    for concurrency in (0, float('nan')):
        with pytest.raises(ValueError, match='concurrency'):
            SmartFileSearch[None](concurrency=concurrency)  # pyright: ignore[reportArgumentType]


def test_guidance_replaces_or_disables_the_discovery_policy() -> None:
    default = SmartFileSearch[None]().get_instructions()
    assert isinstance(default, str) and 'smart_file_search first' in default
    assert SmartFileSearch[None](guidance='Use it.').get_instructions() == 'Use it.'
    assert SmartFileSearch[None](guidance='').get_instructions() is None


def _calls_smart_file_search(**args: object) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart('smart_file_search', {'query': 'reject expired sessions', **args})]
            )
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _tool_return(messages: list[ModelMessage]) -> ToolReturnPart:
    return next(part for message in messages for part in message.parts if isinstance(part, ToolReturnPart))


async def test_agent_searches_its_workspace_with_the_configured_judge(tmp_path: Path) -> None:
    _repo(tmp_path)
    judge_model = FakeJev('expired')
    agent = Agent(
        _calls_smart_file_search(), capabilities=[LocalWorkspace(tmp_path), SmartFileSearch(model=judge_model)]
    )
    result = await agent.run('find it')
    content = _tool_return(result.all_messages()).content
    assert isinstance(content, SmartFileSearchResult)
    assert {m.file_path for m in content.matches} == {'auth.py', 'tests/test_auth.py'}
    assert judge_model.requests


async def test_agent_judges_with_its_own_model_when_typesafe_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    (tmp_path / 'a.py').write_text('x = 1\n')
    judged: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.output_tools:  # the judge run
            judged.append(str(messages[-1]))
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'entity': 1, 'operation': 1})])
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('smart_file_search', {'query': 'assign x'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[LocalWorkspace(tmp_path), SmartFileSearch()])
    await agent.run('find it')
    assert len(judged) == 1 and 'a.py:1-1' in judged[0]


async def test_judge_failure_is_reported_to_the_model(tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text('x = 1\n')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(503, 'judge', 'backend down')

    agent = Agent(
        _calls_smart_file_search(),
        capabilities=[LocalWorkspace(tmp_path), SmartFileSearch(model=FunctionModel(respond))],
    )
    result = await agent.run('find it')
    content = str(_tool_return(result.all_messages()).content)
    assert 'smart_file_search failed: ModelHTTPError' in content


async def test_workspace_failure_is_reported_to_the_model(tmp_path: Path) -> None:
    (tmp_path / 'slow.py').write_text('x = 1\n')
    agent = Agent(_calls_smart_file_search(), capabilities=[SmartFileSearch(model=FakeJev('x'))])
    result = await agent.run('find it', workspace=Workspace(VanishingBackend(tmp_path)))
    assert 'read timed out' in str(_tool_return(result.all_messages()).content)


async def test_tool_is_hidden_on_a_workspace_that_cannot_run_commands(tmp_path: Path) -> None:
    seen: list[list[str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append([tool.name for tool in info.function_tools])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[SmartFileSearch(model=FakeJev('x'))])
    await agent.run('hi', workspace=ReadOnlyWorkspace(_workspace(tmp_path)))
    await agent.run('hi', workspace=_workspace(tmp_path))
    assert seen == [[], ['smart_file_search']]


async def test_run_without_a_workspace_fails_at_its_start() -> None:
    agent = Agent(TestModel(), capabilities=[SmartFileSearch(model=FakeJev('x'))])
    with pytest.raises(UserError, match='`SmartFileSearch` needs a workspace'):
        await agent.run('hi')


def test_toolset_advertises_the_upstream_schema() -> None:
    toolset = SmartFileSearch[None]().get_toolset()
    assert isinstance(toolset, SmartFileSearchToolset)
    schema = toolset.tools['smart_file_search'].function_schema.json_schema
    assert set(schema['properties']) == {'query', 'directory', 'glob', 'limit', 'candidates'}
    assert schema['properties']['candidates']['default'] == 128
    assert schema['required'] == ['query']


@pytest.mark.usefixtures('instrument_all_agents')
async def test_judge_runs_are_named_after_the_capability(capfire: CaptureLogfire, tmp_path: Path) -> None:
    (tmp_path / 'a.py').write_text('expired = True\n')
    agent = Agent(
        _calls_smart_file_search(),
        name='outer',
        capabilities=[LocalWorkspace(tmp_path), SmartFileSearch(model=FakeJev('x'))],
    )
    await agent.run('find it')
    assert agent_run_names(capfire).count('smart_file_search') == 1


@pytest.mark.parametrize(('cache_index', 'chunked'), [(True, 1), (False, 2)])
async def test_cache_index_keeps_the_index_across_runs(
    tmp_path: Path, chunker: CountingChunker, cache_index: bool, chunked: int
) -> None:
    (tmp_path / 'a.py').write_text('expired = True\n')
    capability = SmartFileSearch(model=FakeJev('x'), cache_index=cache_index)
    agent = Agent(_calls_smart_file_search(), capabilities=[LocalWorkspace(tmp_path), capability])
    await agent.run('find it')
    await agent.run('find it again')
    assert chunker.paths == ['a.py'] * chunked


def test_capability_loads_from_an_agent_spec() -> None:
    spec = AgentSpec.model_validate(
        {
            'model': 'test',
            'capabilities': [
                {'SmartFileSearch': {'model': 'openai:gpt-5-mini', 'threshold': 0.6, 'cache_index': True}}
            ],
        }
    )
    agent = Agent.from_spec(spec, custom_capability_types=[SmartFileSearch])
    [capability] = [c for c in agent.root_capability.capabilities if isinstance(c, SmartFileSearch)]
    assert (capability.model, capability.threshold, capability.cache_index) == ('openai:gpt-5-mini', 0.6, True)
