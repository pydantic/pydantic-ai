from __future__ import annotations

import difflib
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
import pytest
import yaml
from pydantic import BaseModel, Field

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
)

sys.path.insert(0, str(Path(__file__).parent))

import review_gates.__main__ as review_gates_main
from review_gates import gates
from review_gates.backtest import BacktestSet, Row, summarize
from review_gates.checks import sanitize
from review_gates.gates import GATES, classify
from review_gates.github import GitHub
from review_gates.locators import enumeration_dispatch, rejection_handling, state_reset, value_handling
from review_gates.prep import SourceFile, parse_patch
from review_gates.reach import reach_map, reach_of

ROOT = Path(__file__).parents[2]
REPO = 'pydantic/pydantic-ai'


class ScriptedDecisionModel(DecisionModel[None]):
    """Answers each yes/no with `yes(state, question)`, and every pick-one with its 'none' option."""

    def __init__(self, yes: Callable[[str, str], float]) -> None:
        super().__init__()
        self._yes = yes

    @property
    def model_name(self) -> str:
        return 'scripted'

    @property
    def system(self) -> str:
        return 'test-decisions'

    @property
    def base_url(self) -> str:
        return 'https://example.test/decisions'

    async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
        state = request.state if isinstance(request.state, str) else json.dumps(request.state)
        answers: dict[str, DecisionAnswer] = {}
        for name, question in request.questions.items():
            if isinstance(question, NoulQuestion):
                answers[name] = NoulAnswer(noul=self._yes(state, name))
            else:
                assert isinstance(question, ChoiceQuestion)
                answers[name] = ChoiceAnswer(
                    choice='none', confidence=0.9, probabilities={o: float(o == 'none') for o in question.criteria}
                )
        return DecisionResponse(answers=answers, model_name=self.model_name)


def changed(path: str, base: str | None, head: str) -> SourceFile:
    """A changed file whose patch is the diff from `base` to `head`, as the compare API would give it."""
    diff = difflib.unified_diff((base or '').splitlines(keepends=True), head.splitlines(keepends=True))
    return SourceFile(path, head, base, parse_patch(''.join(list(diff)[2:])))


def test_parse_patch_places_each_line_in_the_head():
    lines = parse_patch('@@ -3,3 +3,3 @@\n keep\n-old\n+new\n keep\n@@ -20 +20,2 @@\n end\n+added')
    assert [(line.kind, line.new, line.text) for line in lines] == [
        (' ', 3, 'keep'),
        ('-', 4, 'old'),
        ('+', 4, 'new'),
        (' ', 5, 'keep'),
        (' ', 20, 'end'),
        ('+', 21, 'added'),
    ]


@pytest.mark.parametrize(
    ('gate', 'yes', 'outcome'),
    [
        ('sibling-handling', {'needs_it': 0.9}, 'finding'),
        ('sibling-handling', {'needs_it': 0.75}, 'unsure'),
        ('sibling-handling', {'needs_it': 0.1}, 'clean'),
        ('unaffected-users', {'alters_results': 0.9, 'confined_to_target': 0.1}, 'finding'),
        ('unaffected-users', {'alters_results': 0.9, 'confined_to_target': 0.6}, 'unsure'),
        ('unaffected-users', {'alters_results': 0.1, 'confined_to_target': 0.5}, 'clean'),
    ],
)
def test_an_item_is_a_finding_only_when_every_answer_is_sure_of_the_defect(
    gate: str, yes: dict[str, float], outcome: str
):
    defect = next(g.defect for g in GATES.values() if g.key == gate)
    assert classify(defect, yes) == outcome


def test_locator_b_finds_an_enumerated_value_its_dispatch_never_handles():
    base = """const TOOL_CALL = /\\b(read|write)\\(/
function summarize(name: string): string {
  switch (name) {
    case 'read':
      return 'Read a file'
    case 'write':
      return 'Wrote a file'
  }
  return name
}
"""
    head = base.replace('(read|write)', '(read|write|glob)')
    [pair] = enumeration_dispatch([changed('web/tools.ts', base, head)])
    assert (pair.locator, pair.this.start, pair.sibling.start, pair.sibling.end) == ('b', 1, 4, 6)
    assert '`glob`' in pair.difference


def test_locator_c_finds_an_unhandled_call_whose_siblings_catch():
    base = """export function Panel({ client }: Props) {
  const save = () => {
    client.save(draft).catch(showError)
  }
}
"""
    head = base.replace('  }\n}', '  }\n  const resume = () => {\n    void client.resume(sessionId)\n  }\n}')
    [pair] = rejection_handling([changed('web/Panel.tsx', base, head)])
    assert (pair.locator, pair.this.start, pair.sibling.start) == ('c', 6, 3)


def test_locator_c_finds_a_void_call_inside_a_jsx_handler():
    base = """export function Banner({ api }: Props) {
  useEffect(() => { api.load().catch(report) }, [])
  return (
    <div>
    </div>
  )
}
"""
    head = base.replace(
        '    <div>\n', '    <div>\n      <button onClick={() => void api.open(id)}>Continue here</button>\n'
    )
    [pair] = rejection_handling([changed('web/Banner.tsx', base, head)])
    assert (pair.locator, pair.this.start, pair.sibling.start) == ('c', 5, 2)


def test_locator_d_finds_a_step_that_quotes_a_value_another_step_routes_through_a_file():
    base = """## Step 1
```bash
cat > body.md <<'EOF'
$TITLE
EOF
gh pr create --body-file body.md
```
"""
    head = base + '## Step 2\n```bash\ngh issue comment 1 --body "<issue title>"\n```\n'
    [pair] = value_handling([changed('.agents/skills/triage/SKILL.md', base, head)])
    assert (pair.locator, pair.this.start, pair.sibling.start) == ('d', 10, 3)
    assert '"<issue title>"' in pair.difference


def test_locator_e_finds_a_new_field_reconnect_never_resets():
    base = """class Session:
    def __init__(self):
        self._closed = False

    async def reconnect(self):
        self._closed = False
"""
    head = base.replace(
        '        self._closed = False\n\n', '        self._closed = False\n        self._resumed = False\n\n'
    )
    [pair] = state_reset([changed('pkg/session.py', base, head)])
    assert (pair.locator, pair.this.start) == ('e', 4)
    assert '`self._resumed`' in pair.difference


def test_reach_map_reads_the_providers_each_shared_file_serves():
    reach = reach_map(ROOT)
    assert '`deepseek`' in (reach_of('pydantic_ai_slim/pydantic_ai/models/openai.py', reach) or '')
    assert reach_of('pydantic_ai_slim/pydantic_ai/agent/abstract.py', reach) == 'every agent run'
    assert reach_of('pydantic_ai_slim/pydantic_ai/providers/deepseek.py', reach) is None


def test_sanitize_removes_links_and_defuses_mentions():
    text = sanitize('@octocat see https://evil.example/x and [here](https://a.example) or `code`')
    assert text == "@\N{ZERO WIDTH SPACE}octocat see <link removed> and here or 'code'"


LOCAL = """class LocalBackend:
    def run(self, argv: list[str]) -> int:
        if not argv:
            raise ValueError('empty command')
        return 0
"""
SSH = '''class SSHBackend:
    def run(self, argv: list[str]) -> int:
        """Run `argv` on the remote host and return its exit status."""
        return 0
'''


class _Annotation(BaseModel):
    path: str
    start_line: int
    end_line: int


class _Output(BaseModel):
    title: str
    summary: str
    annotations: list[_Annotation]


class _Finished(BaseModel):
    conclusion: str
    output: _Output


@dataclass
class FakeRepository:
    """The GitHub API for one PR whose head adds an SSH backend, as `httpx2.MockTransport` serves it."""

    head_sha: str = 'head1'
    head_repository: str = 'someone/pydantic-ai'
    # The head `pulls/<n>` reports once the gates have run; differs from `head_sha` when the PR moved on.
    final_head: str = 'head1'
    compare_status: int = 200
    # Leave a changed file's patch out of the compare response, as the API does for a large diff.
    unpatched: bool = False
    # Replaces `scripted_answer` for the run.
    answer: Callable[[str, str], float] | None = None
    check_runs: list[dict[str, str | int]] = field(default_factory=list[dict[str, str | int]])
    finished: dict[int, _Finished] = field(default_factory=dict[int, _Finished])

    def pull(self, head: str) -> dict[str, object]:
        return {
            'number': 7,
            'state': 'open',
            'title': 'Add an SSH backend for running commands',
            'body': 'Closes #42\n\n- [x] tests\n\nAdds an `SSHBackend` that runs commands. See https://x.example/a',
            'labels': [{'name': 'pkg:harness'}],
            'head': {'sha': head, 'ref': 'ssh', 'repo': {'full_name': self.head_repository}},
            'base': {'ref': 'main'},
        }

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix(f'/repos/{REPO}/')
        params = request.url.params
        texts = {
            'head1': {
                'pkg/backends/local.py': LOCAL.replace('return 0', 'return 0  # done'),
                'pkg/backends/ssh.py': SSH,
                'pydantic_ai_slim/pydantic_ai/_output.py': 'LIMIT = 5\n',
            },
            'base1': {'pkg/backends/local.py': LOCAL, 'pydantic_ai_slim/pydantic_ai/_output.py': 'LIMIT = 10\n'},
        }
        if path == 'pulls':
            other_fork = {**self.pull(self.head_sha), 'head': {'sha': 'x', 'ref': 'ssh', 'repo': {'full_name': 'b/c'}}}
            return httpx2.Response(200, json=[self.pull(self.head_sha), other_fork])
        if path == 'pulls/7':
            return httpx2.Response(200, json=self.pull(self.final_head))
        if path == f'compare/main...{self.head_sha}':
            files = [
                {'filename': 'pkg/backends/local.py', 'status': 'modified', 'patch': '@@ -5 +5 @@\n-        return 0\n+        return 0  # done'},
                {'filename': 'pkg/backends/ssh.py', 'status': 'added', 'patch': '@@ -0,0 +1,4 @@\n' + ''.join(f'+{line}\n' for line in SSH.splitlines())},
                {'filename': 'pydantic_ai_slim/pydantic_ai/_output.py', 'status': 'modified', 'patch': '@@ -1 +1 @@\n-LIMIT = 10\n+LIMIT = 5'},
            ]  # fmt: skip
            if self.unpatched:
                del files[0]['patch']
            body = {'merge_base_commit': {'sha': 'base1'}, 'files': files}
            return httpx2.Response(self.compare_status, json=body if self.compare_status == 200 else {})
        if path.startswith('contents/'):
            text = texts.get(params['ref'], {}).get(path.removeprefix('contents/'))
            return httpx2.Response(200, text=text) if text is not None else httpx2.Response(404)
        if path == 'issues/42':
            return httpx2.Response(200, json={'number': 42, 'title': 'Run commands over SSH', 'body': 'Please.'})
        if path == f'commits/{self.head_sha}/check-runs':
            runs = [r for r in self.check_runs if r['name'] == params['check_name']]
            return httpx2.Response(200, json={'check_runs': runs})
        if path == 'check-runs' and request.method == 'POST':
            posted = json.loads(request.content)
            self.check_runs.append({**posted, 'id': len(self.check_runs) + 1})
            return httpx2.Response(201, json=self.check_runs[-1])
        if path.startswith('check-runs/') and request.method == 'PATCH':
            check_run_id = int(path.removeprefix('check-runs/'))
            patched = json.loads(request.content)
            self.finished[check_run_id] = _Finished.model_validate(patched)
            run = self.check_runs[check_run_id - 1]
            run.update(status='completed', external_id=patched.get('external_id', run['external_id']))
            return httpx2.Response(200, json={'id': check_run_id})
        raise AssertionError(f'unexpected request: {request.method} {request.url}')

    def conclusions(self) -> dict[str, _Finished]:
        return {str(run['name']): self.finished[int(run['id'])] for run in self.check_runs}


def scripted_answer(state: str, question: str) -> float:
    """A sure `needs_it` for the SSH backend, an unsure shared change, and claims that hold."""
    if question == 'needs_it':
        return 0.95 if 'SSHBackend' in state else 0.1
    if question == 'alters_results':
        return 0.55
    if question == 'confined_to_target':
        return 0.5
    return 0.9


@pytest.fixture
def anyio_backend() -> str:
    # The gates run under `asyncio.run`, and use its task groups.
    return 'asyncio'


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeRepository:
    repository = FakeRepository()

    def connect(repo: str, token: str) -> GitHub:
        transport = httpx2.MockTransport(repository.handle)
        return GitHub(httpx2.AsyncClient(base_url='https://api.github.com', transport=transport), repo)

    def infer_model(name: str) -> ScriptedDecisionModel:
        return ScriptedDecisionModel(repository.answer or scripted_answer)

    monkeypatch.setattr(GitHub, 'connect', connect)
    monkeypatch.setattr(review_gates_main, 'infer_model', infer_model)
    return repository


def env(fake: FakeRepository) -> dict[str, str]:
    return {
        'GITHUB_REPOSITORY': REPO,
        'GITHUB_TOKEN': 'token',
        'HEAD_SHA': fake.head_sha,
        'HEAD_REPOSITORY': fake.head_repository,
        'HEAD_BRANCH': 'ssh',
    }


async def test_run_posts_one_check_run_per_gate_on_a_fork_pr(fake: FakeRepository):
    assert await review_gates_main.run(env(fake), ROOT) == 0

    results = fake.conclusions()
    assert set(results) == {gate.name for gate in GATES.values()}
    claims = results['Gate: PR claims match the diff']
    assert claims.conclusion == 'success', claims.output.summary
    shared = results['Gate: no behavior change for unaffected users']
    assert shared.conclusion == 'neutral'
    assert '### Unsure (1)' in shared.output.summary
    sibling = results['Gate: sibling code handled alike']
    assert sibling.conclusion == 'neutral'
    [annotation] = sibling.output.annotations
    assert (annotation.path, annotation.start_line, annotation.end_line) == ('pkg/backends/ssh.py', 2, 4)
    assert all(str(run['external_id']).split(':')[1] == 'head1' for run in fake.check_runs)


async def test_run_posts_nothing_for_a_head_every_gate_reported_on(fake: FakeRepository):
    await review_gates_main.run(env(fake), ROOT)
    posted = len(fake.check_runs)

    await review_gates_main.run(env(fake), ROOT)

    assert len(fake.check_runs) == posted


async def test_run_skips_the_check_runs_of_a_pr_that_moved_on(fake: FakeRepository):
    fake.final_head = 'head2'

    await review_gates_main.run(env(fake), ROOT)

    assert {result.conclusion for result in fake.conclusions().values()} == {'skipped'}


async def test_a_failed_run_closes_its_check_runs_and_a_rerun_judges_again(fake: FakeRepository):
    fake.compare_status = 502

    with pytest.raises(httpx2.HTTPStatusError):
        await review_gates_main.run(env(fake), ROOT)

    assert {r.conclusion for r in fake.conclusions().values()} == {'neutral'}
    assert all(str(run['external_id']).endswith(':failed') for run in fake.check_runs)
    fake.compare_status = 200

    await review_gates_main.run(env(fake), ROOT)

    assert len(fake.check_runs) == 2 * len(GATES)


async def test_an_item_the_model_could_not_answer_is_not_judged(fake: FakeRepository):
    def answer(state: str, question: str) -> float:
        if question == 'needs_it':
            raise ModelHTTPError(status_code=503, model_name='scripted')
        return scripted_answer(state, question)

    fake.answer = answer

    await review_gates_main.run(env(fake), ROOT)

    sibling = fake.conclusions()['Gate: sibling code handled alike']
    assert sibling.conclusion == 'neutral'
    assert '### Not judged (1)' in sibling.output.summary


async def test_items_over_the_cap_make_every_gate_neutral(fake: FakeRepository, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(gates, 'ITEM_CAP', 0)

    await review_gates_main.run(env(fake), ROOT)

    for result in fake.conclusions().values():
        assert result.conclusion == 'neutral'
        assert 'over the cap of 0 were not judged' in result.output.summary


async def test_a_file_without_a_patch_makes_every_gate_neutral(fake: FakeRepository):
    fake.unpatched = True

    await review_gates_main.run(env(fake), ROOT)

    for result in fake.conclusions().values():
        assert result.conclusion == 'neutral'
        assert 'had no patch from the API: pkg/backends/local.py' in result.output.summary


async def test_run_judges_nothing_when_ci_tested_an_older_head(fake: FakeRepository):
    await review_gates_main.run({**env(fake), 'HEAD_SHA': 'head0'}, ROOT)

    assert fake.check_runs == []


class _Step(BaseModel):
    uses: str | None = None
    run: str | None = None
    with_: dict[str, str | bool] | None = Field(None, alias='with')


class _Job(BaseModel):
    if_: str = Field(alias='if')
    permissions: dict[str, str]
    steps: list[_Step]


class _Workflow(BaseModel):
    permissions: dict[str, str]
    jobs: dict[str, _Job]


def test_workflow_runs_only_the_default_branch_code_with_least_privilege():
    workflow = _Workflow.model_validate(yaml.safe_load((ROOT / '.github/workflows/review-gates.yml').read_text()))
    [job] = workflow.jobs.values()
    assert workflow.permissions == {}
    assert job.permissions == {'checks': 'write', 'contents': 'read', 'pull-requests': 'read', 'issues': 'read'}
    assert "github.event.workflow_run.conclusion == 'success'" in job.if_
    for step in job.steps:
        assert '${{' not in (step.run or ''), 'event data reaches `run:` only through `env:`'
        assert 'download-artifact' not in (step.uses or ''), 'a PR-produced artifact is never read'
        if (step.uses or '').startswith('actions/checkout@'):
            assert step.with_ == {'ref': '${{ github.workflow_sha }}', 'persist-credentials': False}


def test_backtest_summary_says_where_each_finding_landed():
    labelled = BacktestSet.model_validate(
        {
            'window': ['2026-09-21', '2026-10-21'],
            'findings': [
                {'idx': 1, 'pr': 2, 'head': 'a', 'comment': 'https://x/1', 'path': 'p.py', 'line': 3, 'summary': 'one'},
                {'idx': 2, 'pr': 2, 'head': 'a', 'comment': 'https://x/2', 'path': 'q.py', 'line': 9, 'summary': 'two'},
            ],
            'clean': [{'pr': 5, 'head': 'b'}],
        }
    )
    rows = [
        Row(
            pr=pr,
            head='a',
            clean_pr=clean,
            gate='sibling-handling',
            path=path,
            start=1,
            end=5,
            label='l',
            text='t',
            outcome='finding',
            yes={'needs_it': 0.9},
            defect_margin=0.8,
            finding_idx=covered,
        )
        for pr, clean, path, covered in ((2, False, 'p.py', [1]), (5, True, 'r.py', []))
    ]

    report = summarize(labelled, rows)

    assert '| [1](https://x/1) | one | sibling-handling | finding |' in report
    assert '| [2](https://x/2) | two | not located | - |' in report
    assert '- #5 `r.py:1-5` (sibling-handling): "l"' in report
