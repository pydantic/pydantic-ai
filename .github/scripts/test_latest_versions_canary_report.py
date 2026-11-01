from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import latest_versions_canary_report as report

REPO = 'pydantic/pydantic-ai'
BOT = 'github-actions[bot]'
DAY = dt.timedelta(days=1)
START = dt.datetime(2026, 10, 20, 7, 23, tzinfo=dt.UTC)
UNION = 'TypeError in pydantic_ai_slim/pydantic_ai/models/anthropic.py::_map_message'
SDK_CREATE = 'TypeError in pydantic_ai_slim/pydantic_ai/models/anthropic.py::_messages_create (raised in anthropic)'
SDK_MAP = 'TypeError in pydantic_ai_slim/pydantic_ai/models/anthropic.py::_map_message (raised in anthropic)'
PREFECT = 'TypeError in tests/conftest.py::prefect_test_server (raised in prefect)'
DBOS = 'ImportError in tests/conftest.py::dbos_fixture'
SETTINGS = 'AssertionError in tests/models/test_model_settings_support.py'
COHERE = 'ImportError in pydantic_ai_slim/pydantic_ai/models/cohere.py'
REPO_FILES = (
    'pydantic_ai_slim/pydantic_ai/models/anthropic.py',
    'pydantic_ai_slim/pydantic_ai/models/cohere.py',
    'pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_model.py',
    'tests/conftest.py',
    'tests/durable_exec/temporal/test_model_and_serialization.py',
    'tests/models/test_anthropic.py',
    'tests/models/test_cohere.py',
    'tests/models/test_model_settings_support.py',
)
SEPARATOR = '_ ' * 30

# Reconstructed from the scheduled run https://github.com/pydantic/pydantic-ai/actions/runs/39741853792,
# where `anthropic`'s message params became `Union` aliases. On Python 3.11 the call raises inside
# `typing.py`, through short entries with relative paths:
UNION_311 = f"""
tests/durable_exec/temporal/test_model_and_serialization.py:3076:
{SEPARATOR}
.venv/lib/python3.11/site-packages/temporalio/testing/_activity.py:227: in run_async
    return await fn(*args)
pydantic_ai_slim/pydantic_ai/durable_exec/temporal/_model.py:121: in request_activity
    response = await self.model.request(messages, settings, params)
tests/durable_exec/temporal/test_model_and_serialization.py:3034: in render
    return await model._map_message(messages, params, settings)
pydantic_ai_slim/pydantic_ai/models/anthropic.py:2560: in _map_message
    system_prompt_parts.append(BetaTextBlockParam(text=part.content, type='text'))
../../_temp/uv-python-dir/cpython-3.11.17-linux-x86_64-gnu/lib/python3.11/typing.py:1289: in __call__
    return self._getitem(self, *args, **kwds)
{SEPARATOR}

self = typing.Union, args = ()

    @_SpecialForm
    def Union(self, parameters):
>       raise TypeError(f"Cannot instantiate {{self!r}}")
E       TypeError: Cannot instantiate typing.Union

../../_temp/uv-python-dir/cpython-3.11.17-linux-x86_64-gnu/lib/python3.11/typing.py:486: TypeError
"""


def union_315(root: Path) -> str:
    """On Python 3.15 the same call raises in C, so `_map_message` is the last, long entry."""
    model = f'{root}/pydantic_ai_slim/pydantic_ai/models/anthropic.py'
    return f"""
{model}:1225: in request
    response = await self._messages_create(messages, False, settings, params)
{model}:1424: in _messages_create
    system_prompt, anthropic_messages = await self._map_message(messages, params, settings)
{SEPARATOR}

self = AnthropicModel()

    async def _map_message(  # noqa: C901
        self,
        messages: list[ModelMessage],
    ) -> tuple[str, list[BetaMessageParam]]:
        def cache_point(part: object) -> object:
            return part

>       system_prompt_parts.append(BetaTextBlockParam(text=part.content, type='text'))
E       TypeError: 'typing.Union' object is not callable

{model}:2560: TypeError
"""


# A different cause in the same file and exception class: the SDK rejecting a keyword.
SDK_CREATE_TEXT = f"""
tests/models/test_anthropic.py:400:
{SEPARATOR}
pydantic_ai_slim/pydantic_ai/models/anthropic.py:1424: in _messages_create
    return await self.client.beta.messages.create(**kwargs)
.venv/lib/python3.11/site-packages/anthropic/_utils/_utils.py:283: in wrapper
    return func(*args, **kwargs)
{SEPARATOR}

>   ???
E   TypeError: AsyncMessages.create() got an unexpected keyword argument 'container'

.venv/lib/python3.11/site-packages/anthropic/resources/beta/messages/messages.py:1200: TypeError
"""

# And a third in the same function, raised by the SDK rather than by `typing`.
SDK_MAP_TEXT = f"""
tests/models/test_anthropic.py:510:
{SEPARATOR}
pydantic_ai_slim/pydantic_ai/models/anthropic.py:2570: in _map_message
    content = to_param(part)
.venv/lib/python3.11/site-packages/anthropic/lib/_parse.py:88: in to_param
    raise TypeError('unsupported block')
E   TypeError: unsupported block

.venv/lib/python3.11/site-packages/anthropic/lib/_parse.py:88: TypeError
"""

PREFECT_TEXT = """
tests/conftest.py:738: in prefect_test_server
    from prefect.testing.utilities import prefect_test_harness
.venv/lib/python3.11/site-packages/prefect/testing/utilities.py:34: in <module>
    from prefect.server.api.server import SubprocessASGIServer
E   TypeError: Body() got an unexpected keyword argument 'min_items'

.venv/lib/python3.11/site-packages/prefect/server/api/concurrency_limits_v2.py:305: TypeError
"""


# A second cause in the same `conftest.py`, raised by its own fixture.
DBOS_TEXT = """
    @pytest.fixture
    def dbos_fixture():
>       from dbos import _logger
E       ImportError: cannot import name '_logger' from 'dbos'

tests/conftest.py:523: ImportError
"""

# What pytest-xdist reports, as pytest's JUnit writes it: an `error` whose text is xdist's raw message.
CRASHED_WORKER = "worker 'gw3' crashed while running 'tests/models/test_anthropic.py::test_request'"
DIFFERENT_COLLECTION = """Different tests were collected between gw0 and gw1. The difference is:
--- gw0

+++ gw1

@@ -1,2 +1 @@
 tests/models/test_anthropic.py::test_request
-tests/models/test_anthropic.py::test_container
To see why this happens see 'Known limitations' in documentation for pytest-xdist"""


def assertion_text(line: int) -> str:
    return f"""
    def test_case():
>       assert settings == snapshot()
E       AssertionError: assert {{'max_retries': 3}} == {{'max_retries': 2}}

tests/models/test_model_settings_support.py:{line}: AssertionError
"""


COHERE_TEXT = """
ImportError while importing test module 'tests/models/test_cohere.py'.
Traceback:
../../_temp/uv-python-dir/cpython-3.11.17-linux-x86_64-gnu/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests/models/test_cohere.py:12: in <module>
    from pydantic_ai.models.cohere import CohereModel
pydantic_ai_slim/pydantic_ai/models/cohere.py:8: in <module>
    from cohere import AssistantChatMessageV2
E   ImportError: cannot import name 'AssistantChatMessageV2' from 'cohere'
"""

Case = tuple[str, str, str, str, str]


def case(file: str, name: str, text: str, *, tag: str = 'failure', message: str = '') -> Case:
    return file, name, tag, message or 'junit message', text


def write_leg(
    reports: Path,
    python: str,
    *,
    test: str = 'failure',
    install: str = 'success',
    resolve: str = 'success',
    legs: int = 2,
    pytest_exit: int | None = None,
    cases: list[Case] | None = None,
    moved: str | None = None,
) -> None:
    leg = reports / f'latest-versions-canary-{python}'
    leg.mkdir(parents=True)
    status = pytest_exit if pytest_exit is not None else {'failure': 1, 'success': 0}.get(test)
    outcome = {'resolve': resolve, 'install': install, 'test': test, 'legs': legs, 'pytest_exit': status}
    (leg / 'outcome.json').write_text(json.dumps(outcome))
    if cases is not None:
        suite = ElementTree.Element('testsuites')
        for file, name, tag, message, text in cases:
            module = file.removesuffix('.py').replace('/', '.')
            testcase = ElementTree.SubElement(
                suite, 'testcase', {'classname': '' if not name else module, 'name': name or module, 'file': file}
            )
            result = ElementTree.SubElement(testcase, tag, {'message': message})
            result.text = text
        (leg / 'junit.xml').write_bytes(ElementTree.tostring(suite))
    if moved is not None:
        (leg / 'moved.txt').write_text(moved)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    checkout = tmp_path / 'checkout'
    for file in REPO_FILES:
        (checkout / file).parent.mkdir(parents=True, exist_ok=True)
        (checkout / file).write_text('')
    return checkout


class FakeClient(report.attention.GitHubClient):
    """GitHub issues and comments held in memory; every write is recorded."""

    def __init__(self) -> None:
        super().__init__('token')
        self.issues: dict[int, dict[str, object]] = {}
        self.comments: dict[int, list[dict[str, object]]] = {}
        self.created: list[dict[str, object]] = []
        self.posted: list[tuple[int, str]] = []
        self.now = START

    def add_issue(self, signature: str, *, run_id: int = 1, **fields: object) -> int:
        number = 100 + len(self.issues)
        marker = report.Marker(run_id=run_id, status='failing', signature=signature)
        self.issues[number] = {
            'number': number,
            'title': f'Latest versions canary: {signature}',
            'body': f'<!-- pydantic-ai-latest-versions-canary:v1 {marker.model_dump_json()} -->\nThe canary run failed.',
            'user': {'login': BOT},
            'state': 'open',
            'created_at': self.now.isoformat(),
            'closed_at': None,
            'state_reason': None,
            **fields,
        }
        self.comments[number] = []
        return number

    def close(self, number: int, reason: str = 'completed') -> None:
        self.issues[number].update(state='closed', closed_at=self.now.isoformat(), state_reason=reason)

    def first_pages(self, path: str, *, count: int) -> tuple[list[dict[str, object]], bool]:
        assert path == f'/repos/{REPO}/issues?state=open&creator=github-actions%5Bbot%5D'
        return [issue for issue in self.issues.values() if issue['state'] == 'open'], True

    def last_pages(self, path: str, *, count: int = 1) -> list[dict[str, object]]:
        return self.comments[int(path.split('/issues/')[1].split('/')[0])]

    def get(self, path: str) -> object:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)['q'][0]
        assert query == f'repo:{REPO} is:issue is:closed author:app/github-actions in:title "Latest versions canary"'
        closed = [issue for issue in self.issues.values() if issue['state'] == 'closed']
        return {'items': sorted(closed, key=lambda issue: -int(str(issue['number'])))}

    def post(self, path: str, payload: Mapping[str, object]) -> object:
        if path == f'/repos/{REPO}/issues':
            number = 100 + len(self.issues)
            issue: dict[str, object] = {
                **payload,
                'number': number,
                'user': {'login': BOT},
                'state': 'open',
                'created_at': self.now.isoformat(),
            }
            self.issues[number] = issue
            self.comments[number] = []
            self.created.append(issue)
            return issue
        number = int(path.split('/issues/')[1].split('/')[0])
        body = str(payload['body'])
        self.comments[number].append({'body': body, 'user': {'login': BOT}, 'created_at': self.now.isoformat()})
        self.posted.append((number, body))
        return {}


def run(
    client: FakeClient, root: Path, reports: Path, run_id: int, *, green: bool = False, day: float | None = None
) -> list[str]:
    """Report one scheduled run on day `run_id`, or on `day` for a re-run, testing a commit made an hour before it."""
    client.now = START + (run_id if day is None else day) * DAY
    observation = report.Observation({}, complete=True) if green else report.collect(reports, root)
    canary = report.Run(REPO, run_id, f'{run_id:040x}', 'https://github.com', client.now - dt.timedelta(hours=1))
    return report.reconcile(client, canary, observation, report.read_moved(reports))


def failing_reports(tmp_path: Path, name: str, cases: list[Case]) -> Path:
    reports = tmp_path / name
    write_leg(reports, '3.11', cases=cases)
    write_leg(reports, '3.15', cases=cases)
    return reports


def test_one_cause_is_one_signature_across_modules_and_interpreters(tmp_path: Path, root: Path):
    reports = tmp_path / 'reports'
    temporal = 'tests/durable_exec/temporal/test_model_and_serialization.py'
    write_leg(
        reports,
        '3.11',
        cases=[
            case(temporal, 'test_reprepares_messages', UNION_311),
            case('tests/models/test_anthropic.py', 'test_request', UNION_311),
            case('tests/models/test_anthropic.py', 'test_container', SDK_CREATE_TEXT),
            case('tests/models/test_anthropic.py', 'test_blocks', SDK_MAP_TEXT),
            case('tests/conftest.py', 'test_flow', PREFECT_TEXT, tag='error', message='failed on setup'),
            case('tests/conftest.py', 'test_workflow', DBOS_TEXT, tag='error', message='failed on setup'),
            case('tests/models/test_model_settings_support.py', 'test_max_retries', assertion_text(1100)),
            case('tests/models/test_model_settings_support.py', 'test_wire', assertion_text(1180)),
            case('tests/models/test_cohere.py', '', COHERE_TEXT, tag='error', message='collection failure'),
        ],
    )
    write_leg(reports, '3.15', cases=[case('tests/models/test_anthropic.py', 'test_request', union_315(root))])

    observation = report.collect(reports, root)

    assert {signature: sorted(failure.pythons) for signature, failure in observation.failures.items()} == {
        UNION: ['3.11', '3.15'],
        SDK_CREATE: ['3.11'],
        SDK_MAP: ['3.11'],
        PREFECT: ['3.11'],
        DBOS: ['3.11'],
        SETTINGS: ['3.11'],
        COHERE: ['3.11'],
    }
    assert list(observation.failures[UNION].tests) == [
        f'{temporal}::test_reprepares_messages',
        'tests/models/test_anthropic.py::test_request',
    ]
    assert observation.failures[COHERE].tests == {
        'tests/models/test_cohere.py': "ImportError: cannot import name 'AssistantChatMessageV2' from 'cohere'"
    }
    assert observation.failures[PREFECT].tests == {
        'tests/conftest.py::test_flow': "TypeError: Body() got an unexpected keyword argument 'min_items'"
    }
    assert observation.complete


def test_stage_failures_and_missing_reports(tmp_path: Path, root: Path):
    reports = tmp_path / 'reports'
    write_leg(reports, '3.11', resolve='failure', install='skipped', test='skipped')
    write_leg(reports, '3.15', resolve='failure', install='skipped', test='skipped')
    assert sorted(report.collect(reports, root).failures['dependency resolution fails'].pythons) == ['3.11', '3.15']

    reports = tmp_path / 'more'
    write_leg(reports, '3.11', install='failure', test='skipped')
    write_leg(reports, '3.12', test='cancelled')
    write_leg(reports, '3.13', resolve='skipped', install='skipped', test='skipped')
    write_leg(reports, '3.14', cases=[])
    (reports / 'latest-versions-canary-3.15').mkdir()
    (reports / 'unrelated').mkdir()
    observation = report.collect(reports, root)
    assert list(observation.failures) == [
        'canary job on Python 3.13 fails before its resolve step',
        'canary job on Python 3.15 left no readable report',
        'install fails on Python 3.11',
        'pytest fails on Python 3.14 with no failing test reported',
        'test step times out or is cancelled on Python 3.12',
    ]
    assert not observation.complete


def test_a_failure_that_stops_a_leg_files_before_test_failures(tmp_path: Path, root: Path):
    reports = tmp_path / 'reports'
    write_leg(reports, '3.11', cases=[case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    write_leg(reports, '3.15', install='failure', test='skipped')
    assert list(report.collect(reports, root).failures) == ['install fails on Python 3.15', UNION]


def test_a_session_that_stopped_short_is_incomplete(tmp_path: Path, root: Path):
    union = case('tests/models/test_anthropic.py', 'test_request', UNION_311)
    interrupted = tmp_path / 'interrupted'
    write_leg(interrupted, '3.11', pytest_exit=3, cases=[union])
    write_leg(interrupted, '3.15', pytest_exit=2, cases=[union])
    crashed = tmp_path / 'crashed'
    write_leg(crashed, '3.11', cases=[union])
    crash = case(
        'tests/models/test_anthropic.py',
        'test_request',
        CRASHED_WORKER,
        tag='error',
        message=f'failed on setup with "{CRASHED_WORKER}"',
    )
    write_leg(crashed, '3.15', cases=[crash])
    mismatched = tmp_path / 'mismatched'
    write_leg(mismatched, '3.11', cases=[union])
    write_leg(mismatched, '3.15', cases=[case('gw1', 'gw1', DIFFERENT_COLLECTION, tag='error')])

    assert list(report.collect(interrupted, root).failures) == [UNION]
    assert not report.collect(interrupted, root).complete
    for reports, stop, detail in (
        (crashed, 'a pytest-xdist worker crashes', CRASHED_WORKER),
        (mismatched, 'pytest-xdist workers collect different tests', DIFFERENT_COLLECTION.splitlines()[0]),
    ):
        observation = report.collect(reports, root)
        assert list(observation.failures) == [stop, UNION]
        assert observation.failures[stop].tests == {}
        assert observation.failures[stop].details == [detail]
        assert not observation.complete

    # The crash issue names the node the worker was running.
    filer = FakeClient()
    assert run(filer, root, crashed, 1)[0] == '#100: filed `a pytest-xdist worker crashes`'
    assert f'```text\n{CRASHED_WORKER}\n```' in str(filer.created[0]['body'])

    # An open issue for a signature the stopped session never reached is not told it passes.
    client = FakeClient()
    run(client, root, failing_reports(tmp_path, 'prefect', [case('tests/conftest.py', 'test_flow', PREFECT_TEXT)]), 1)
    assert run(client, root, interrupted, 2) == [f'#101: filed `{UNION}`']
    assert client.posted == []


def test_a_missing_leg_makes_the_run_incomplete(tmp_path: Path, root: Path):
    reports = tmp_path / 'reports'
    write_leg(reports, '3.11', cases=[case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    assert not report.collect(reports, root).complete


def test_new_signature_files_one_labeled_issue_with_fenced_evidence(tmp_path: Path, root: Path):
    hostile = 'TypeError: ```` @maintainer --> <!-- pydantic-ai-latest-versions-canary:v1 {} -->\u202e'
    text = UNION_311.replace('E       TypeError: Cannot instantiate typing.Union', f'E       {hostile}')
    reports = tmp_path / 'reports'
    write_leg(reports, '3.11', cases=[case('tests/models/test_anthropic.py', 'test_request', text)])
    write_leg(reports, '3.15', cases=[case('tests/models/test_anthropic.py', 'test_request', union_315(root))])
    (reports / 'latest-versions-canary-3.11' / 'moved.txt').write_text('anthropic  0.60.0 -> 0.70.0\n')
    client = FakeClient()

    log = run(client, root, reports, 1)

    assert log == [f'#100: filed `{UNION}`']
    [issue] = client.created
    assert issue['title'] == f'Latest versions canary: {UNION}'
    assert issue['labels'] == ['p:1-highest', 'pydanty:triage-issue']
    body = str(issue['body'])
    assert body.startswith(
        '<!-- pydantic-ai-latest-versions-canary:v1 {"run_id":1,"status":"failing","signature":"'
        f'{UNION}"}} -->\n\nThe scheduled [canary run](https://github.com/{REPO}/actions/runs/1)'
    )
    assert f'`{UNION}` (Python 3.11, 3.15; 1 failing test)' in body
    assert '`````text\ntests/models/test_anthropic.py::test_request\n    TypeError: ```` @maintainer' in body
    assert '\u202e' not in body
    assert '```text\nanthropic  0.60.0 -> 0.70.0\n```' in body
    assert body.count('pydantic-ai-latest-versions-canary:v1') == 2


def test_a_path_outside_the_checkout_never_becomes_a_signature(tmp_path: Path, root: Path):
    planted = """
/etc/passwd.py:1: in <module>
    import os
tests/does_not_exist.py:3: in test
E   RuntimeError: planted

tests/does_not_exist.py:3: RuntimeError
"""
    reports = tmp_path / 'reports'
    write_leg(reports, '3.11', cases=[case('tests/does_not_exist.py', 'test', planted)])
    write_leg(reports, '3.15', cases=[case('../outside.py', 'test', planted)])
    assert list(report.collect(reports, root).failures) == ['RuntimeError in an unrecognized location']


def test_known_signature_is_commented_once_per_run_and_never_relabeled(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)

    assert run(client, root, reports, 2) == [f'#100: noted that `{UNION}` still fails']
    assert run(client, root, reports, 2) == ['#100: already updated by this run or a later one']
    assert run(client, root, reports, 1, green=True, day=3) == ['#100: already updated by this run or a later one']

    assert len(client.created) == 1
    [(number, comment)] = client.posted
    assert number == 100
    assert comment == (
        '<!-- pydantic-ai-latest-versions-canary:v1 {"run_id":2,"status":"failing"} -->\n'
        f'Still fails in the scheduled [canary run](https://github.com/{REPO}/actions/runs/2) at '
        f'[`000000000000`](https://github.com/{REPO}/commit/{2:040x}) (Python 3.11, 3.15; 1 failing test).\n'
    )
    assert 'labels' not in json.dumps(client.posted)


def test_new_issues_are_capped_per_run_and_the_rest_wait(tmp_path: Path, root: Path):
    cases = [case('tests/models/test_anthropic.py', f'test_request_{index}', UNION_311) for index in range(3)] + [
        case('tests/models/test_anthropic.py', 'test_container', SDK_CREATE_TEXT),
        case('tests/models/test_anthropic.py', 'test_blocks', SDK_MAP_TEXT),
        case('tests/conftest.py', 'test_flow', PREFECT_TEXT, tag='error'),
        case('tests/models/test_model_settings_support.py', 'test_max_retries', assertion_text(1100)),
        case('tests/models/test_model_settings_support.py', 'test_wire', assertion_text(1180)),
    ]
    reports = failing_reports(tmp_path, 'reports', cases)
    client = FakeClient()

    assert run(client, root, reports, 1) == [
        f'#100: filed `{UNION}`',
        f'#101: filed `{SETTINGS}`',
        f'#102: filed `{SDK_MAP}`',
        f'2 new signatures wait for a later run (limit 3): `{SDK_CREATE}`, `{PREFECT}`',
    ]
    log = run(client, root, reports, 2)
    assert log[3:] == [f'#103: filed `{SDK_CREATE}`', f'#104: filed `{PREFECT}`']
    assert [issue['title'] for issue in client.created] == [
        f'Latest versions canary: {signature}' for signature in (UNION, SETTINGS, SDK_MAP, SDK_CREATE, PREFECT)
    ]


def test_continuous_failure_after_a_completed_close_is_not_refiled(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)
    run(client, root, reports, 2)
    # Automated triage, or a fix that did not hold, closes the issue while the canary keeps failing.
    client.now = START + 2.5 * DAY
    client.close(100)

    assert run(client, root, reports, 3) == [f'#100: noted that `{UNION}` still fails after the close']
    assert run(client, root, reports, 4) == [f'`{UNION}`: still failing since #100 was closed; already noted there']
    assert run(client, root, reports, 5) == [f'`{UNION}`: still failing since #100 was closed; already noted there']

    assert len(client.created) == 1
    assert [number for number, _ in client.posted] == [100, 100]
    assert 'No canary run has passed it since its last failure, so no new issue is filed.' in client.posted[-1][1]


def test_a_rerun_of_an_older_run_leaves_what_newer_runs_reported(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 2)
    client.now = START + 2.5 * DAY
    client.close(100)
    assert run(client, root, reports, 3) == [f'#100: noted that `{UNION}` still fails after the close']

    # Run 1 passed; re-running it after the close is no recovery, so the continuing failure is not refiled.
    assert run(client, root, reports, 1, green=True, day=4) == ['#100: already updated by this run or a later one']
    assert run(client, root, reports, 5) == [f'`{UNION}`: still failing since #100 was closed; already noted there']

    # Nor does re-running a red run from before a recorded pass refile.
    assert run(client, root, reports, 6, green=True) == [f'#100: recorded that `{UNION}` passes']
    assert run(client, root, reports, 5, day=7) == ['#100: already updated by this run or a later one']

    assert len(client.created) == 1
    assert [number for number, _ in client.posted] == [100, 100]


def test_recovery_then_recurrence_files_a_linked_issue(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)
    client.now = START + 1.5 * DAY
    client.close(100)

    assert run(client, root, reports, 2, green=True) == [f'#100: recorded that `{UNION}` passes']
    assert run(client, root, reports, 3, green=True) == []
    assert run(client, root, reports, 4) == [f'#101: filed `{UNION}`']

    assert 'This recurs after #100 was closed and a later canary run passed it.' in str(client.created[-1]['body'])


def test_a_recovery_long_after_the_close_still_counts(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)
    client.now = START + 1.5 * DAY
    client.close(100)
    for run_id in range(2, 21):
        run(client, root, reports, run_id)

    assert run(client, root, reports, 21, green=True) == [f'#100: recorded that `{UNION}` passes']
    assert run(client, root, reports, 40) == [f'#101: filed `{UNION}`']
    assert [number for number, _ in client.posted] == [100, 100]


def test_a_pass_in_a_red_but_complete_run_counts_as_recovery(tmp_path: Path, root: Path):
    union = failing_reports(tmp_path, 'union', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    other = failing_reports(tmp_path, 'other', [case('tests/conftest.py', 'test_flow', PREFECT_TEXT, tag='error')])
    client = FakeClient()
    run(client, root, union, 1)
    client.now = START + 1.5 * DAY
    client.close(100)

    assert run(client, root, other, 2) == [f'#100: recorded that `{UNION}` passes', f'#101: filed `{PREFECT}`']
    assert run(client, root, union, 3)[-1] == f'#102: filed `{UNION}`'


def test_an_incomplete_run_proves_nothing_passes(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)
    broken = tmp_path / 'broken'
    write_leg(broken, '3.11', install='failure', test='skipped')
    write_leg(broken, '3.15', cases=[])

    assert run(client, root, broken, 2) == [
        '#101: filed `install fails on Python 3.11`',
        '#102: filed `pytest fails on Python 3.15 with no failing test reported`',
    ]
    assert client.posted == []


def test_a_close_as_not_planned_keeps_the_signature_quiet(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)
    client.close(100, 'not_planned')

    assert run(client, root, reports, 2) == [f'`{UNION}`: #100 was closed as not_planned; not refiled']
    assert run(client, root, reports, 3, green=True) == []
    assert run(client, root, reports, 4) == [f'`{UNION}`: #100 was closed as not_planned; not refiled']
    assert len(client.created) == 1
    assert client.posted == []


def test_a_close_after_the_tested_commit_waits_for_the_next_run(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)
    # Closed by a fix merged after run 2's commit but before its report.
    client.now = START + 2 * DAY - dt.timedelta(minutes=30)
    client.close(100)

    assert run(client, root, reports, 2) == [
        f'`{UNION}`: #100 was closed after the tested commit; waiting for the next run'
    ]
    assert client.posted == []
    assert run(client, root, reports, 3) == [f'#100: noted that `{UNION}` still fails after the close']


def test_an_open_issue_records_a_pass_once_and_a_return(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    run(client, root, reports, 1)

    assert run(client, root, reports, 2, green=True) == [f'#100: noted that `{UNION}` passes']
    assert run(client, root, reports, 3, green=True) == []
    assert run(client, root, reports, 4) == [f'#100: noted that `{UNION}` fails again after passing']
    assert 'Close this issue if the fix is meant to last.' in client.posted[0][1]
    assert len(client.created) == 1


def test_issues_this_script_did_not_file_are_ignored(tmp_path: Path, root: Path):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    # Closed, so the search returns it beside the issue below with no `closed_at`.
    client.close(client.add_issue(UNION, user={'login': 'someone'}))
    client.add_issue(UNION, pull_request={'url': 'https://example.com'})
    client.add_issue('', body='Agent workflow failure')
    client.issues[client.add_issue(UNION)].update(state='closed', closed_at=None)

    assert run(client, root, reports, 1) == [f'#104: filed `{UNION}`']


def test_main_reports_only_failed_or_green_runs(tmp_path: Path, root: Path, monkeypatch: pytest.MonkeyPatch):
    reports = failing_reports(tmp_path, 'reports', [case('tests/models/test_anthropic.py', 'test_request', UNION_311)])
    client = FakeClient()
    summary = tmp_path / 'summary.md'

    def fake_client(token: str) -> FakeClient:
        return client

    monkeypatch.setattr(report.attention, 'GitHubClient', fake_client)
    for name, value in {
        'GITHUB_TOKEN': 'token',
        'GITHUB_REPOSITORY': REPO,
        'GITHUB_RUN_ID': '7',
        'GITHUB_SHA': 'a' * 40,
        'GITHUB_WORKSPACE': str(root),
        'GITHUB_STEP_SUMMARY': str(summary),
    }.items():
        monkeypatch.setenv(name, value)
    arguments = ['report', '--reports', str(reports), '--tested-at', '2026-10-27T06:00:00+00:00', '--result']

    monkeypatch.setattr(sys, 'argv', [*arguments, 'cancelled'])
    assert report.main() == 0
    assert client.created == []

    monkeypatch.setattr(sys, 'argv', [*arguments, 'failure'])
    assert report.main() == 0
    assert summary.read_text() == f'- #100: filed `{UNION}`\n'

    (reports / 'latest-versions-canary-3.11' / 'junit.xml').unlink()
    (reports / 'latest-versions-canary-3.15' / 'junit.xml').unlink()
    monkeypatch.setenv('GITHUB_RUN_ID', '8')
    assert report.main() == 0
    assert [issue['title'] for issue in client.created[1:]] == [
        f'Latest versions canary: pytest fails on Python {python} with no failing test reported'
        for python in ('3.11', '3.15')
    ]
