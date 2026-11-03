"""The opt-in capture in `tests/refactor_capture.py`, mostly run against small in-process pytest sessions.

They test the capture tooling rather than model behavior, so no recording applies.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeAlias

import pytest
from inline_snapshot import snapshot

import pydantic_ai
from pydantic_ai.models import StreamedResponse

from .conftest import try_import

with try_import() as imports_successful:
    from . import refactor_capture
    from .refactor_capture import CAPTURE_OPTION, SDK_MOCKS, SOURCE_FILE, RefactorCapture, compare, normalized

pytest_plugins = ['pytester']

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='httpx, openai or anthropic not installed'),
    # A capture of this session would record the streams of the sessions these tests run inside it.
    pytest.mark.skipif(
        "config.getoption('--refactor-capture', None) is not None", reason='runs inside a refactor capture'
    ),
]

RunCapture: TypeAlias = Callable[..., pytest.RunResult]

STREAMS = """
import asyncio

import pytest

from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel


async def stream_text(messages, info):
    yield {text!r}


async def stream_then_fail(messages, info):
    yield 'partial'
    raise RuntimeError('provider went away')


async def drain(stream_function):
    async with model_request_stream(
        FunctionModel(stream_function=stream_function), [ModelRequest.user_text_prompt('hi')]
    ) as stream:
        async for _ in stream:
            pass
        assert aiter(stream) is aiter(stream)


async def drain_fallback():
    async with model_request_stream(
        FallbackModel(FunctionModel(stream_function=stream_text)), [ModelRequest.user_text_prompt('hi')]
    ) as stream:
        events = [event async for event in stream]
    assert events[0].event_kind == 'part_start'


class ReplayedStreamedResponse(CompletedStreamedResponse):
    def __aiter__(self):
        return super().__aiter__()


async def drain_replayed():
    stream = ReplayedStreamedResponse(
        ModelResponse(parts=[TextPart('replayed')]),
        model_request_parameters=ModelRequestParameters(),
        replay_events=True,
    )
    async for _ in stream:
        pass


def test_text():
    asyncio.run(drain(stream_text))


def test_fails_mid_stream():
    with pytest.raises(RuntimeError, match='provider went away'):
        asyncio.run(drain(stream_then_fail))


def test_fallback_peeks_the_first_event():
    asyncio.run(drain_fallback())


def test_subclass_defined_after_the_capture_started():
    asyncio.run(drain_replayed())


def test_no_stream():
    pass
"""

REQUESTS = """
import asyncio
import random

import httpx
import httpx2
from openai.types.chat import ChatCompletionMessage

from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from tests.models.mock_openai import MockOpenAI, completion_message


async def post(http, **content):
    async with http.AsyncClient(transport=http.MockTransport(lambda request: http.Response(200))) as client:
        await client.post('http://127.0.0.1:8123/v1/upload', **content)


def test_json_over_httpx():
    asyncio.run(post(httpx, json=dict(text={text!r}, pick=random.randrange(1000))))


def test_multipart_over_httpx2():
    asyncio.run(post(httpx2, files=dict(file=('notes.txt', b'notes'))))


def test_sdk_mock():
    client = MockOpenAI.create_mock(completion_message(ChatCompletionMessage(content='world', role='assistant')))
    agent = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client)))
    with override_allow_model_requests(True):
        agent.run_sync({text!r})
"""


@pytest.fixture
def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunCapture:
    """Capture an in-process pytest session over `STREAMS` and `REQUESTS`, carrying `text`, into `capture_dir`.

    It runs as `main` runs each side.
    """
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)

    def run(capture_dir: Path, *args: str, text: str = 'hello') -> pytest.RunResult:
        pytester.makepyfile(test_streams=STREAMS.format(text=text), test_requests=REQUESTS.format(text=text))
        return pytester.runpytest_inprocess(
            '-p', 'no:cacheprovider', '-p', 'tests.refactor_capture', f'{CAPTURE_OPTION}={capture_dir}', *args
        )

    return run


def test_capture_records_each_stream_per_test(run: RunCapture, tmp_path: Path):
    run(tmp_path).assert_outcomes(passed=8)

    assert json.loads((tmp_path / SOURCE_FILE).read_text()) == {'pydantic_ai': pydantic_ai.__file__}
    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    records = [record for line in lines if (record := json.loads(line))['kind'] == 'stream']
    assert [
        (
            record['node'],
            record['stream'],
            record['modules'],
            [event['event_kind'] for event in record['events']],
            record.get('error'),
            record['response']['state'],
        )
        for record in records
    ] == snapshot(
        [
            (
                'test_streams.py::test_text',
                'pydantic_ai.models.function.FunctionStreamedResponse',
                ['pydantic_ai.models.function'],
                ['part_start', 'final_result', 'part_end'],
                None,
                'complete',
            ),
            (
                'test_streams.py::test_fails_mid_stream',
                'pydantic_ai.models.function.FunctionStreamedResponse',
                ['pydantic_ai.models.function'],
                ['part_start', 'final_result'],
                'RuntimeError: provider went away',
                'incomplete',
            ),
            (
                'test_streams.py::test_fallback_peeks_the_first_event',
                'pydantic_ai.models.function.FunctionStreamedResponse',
                ['pydantic_ai.models.function'],
                ['part_start', 'final_result', 'part_end'],
                None,
                'complete',
            ),
            (
                'test_streams.py::test_subclass_defined_after_the_capture_started',
                'test_streams.ReplayedStreamedResponse',
                ['test_streams'],
                ['part_start', 'final_result', 'part_end'],
                None,
                'complete',
            ),
        ]
    )
    assert records[0]['response']['parts'] == snapshot(
        [
            {
                'content': 'hello',
                'id': None,
                'provider_name': None,
                'provider_details': None,
                'citations': None,
                'part_kind': 'text',
            }
        ]
    )


def test_capture_records_each_request_per_test(run: RunCapture, tmp_path: Path):
    """A request sent over `httpx` or `httpx2` is recorded with its body, and a call on an SDK mock with its arguments.

    `random` is seeded per test, so the value `test_json_over_httpx` draws is the same in every capture.
    """
    run(tmp_path).assert_outcomes(passed=8)

    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    records = [record for line in lines if (record := json.loads(line))['kind'] == 'request']
    assert [
        (record['node'], record['target'], record.get('body'), record.get('kwargs', {}).get('messages'))
        for record in records
    ] == snapshot(
        [
            (
                'test_requests.py::test_json_over_httpx',
                'POST http://127.0.0.1:PORT/v1/upload',
                {'text': 'hello', 'pick': 534},
                None,
            ),
            (
                'test_requests.py::test_multipart_over_httpx2',
                'POST http://127.0.0.1:PORT/v1/upload',
                """\
--BOUNDARY\r
Content-Disposition: form-data; name="file"; filename="notes.txt"\r
Content-Type: text/plain\r
\r
notes\r
--BOUNDARY--\r
""",
                None,
            ),
            (
                'test_requests.py::test_sdk_mock',
                'tests.models.mock_openai.MockOpenAI.chat_completions_create',
                None,
                [{'role': 'user', 'content': 'hello'}],
            ),
        ]
    )


def test_capture_needs_the_plugin_and_its_directory(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Each half of `-p tests.refactor_capture --refactor-capture=DIR` without the other stops the run."""
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    pytester.makepyfile(test_nothing='def test_nothing():\n    pass\n')

    without_directory = pytester.runpytest_inprocess('-p', 'tests.refactor_capture')
    assert without_directory.ret == pytest.ExitCode.USAGE_ERROR
    without_directory.stderr.fnmatch_lines(['*`-p tests.refactor_capture` needs `--refactor-capture=DIR`*'])

    without_plugin = pytester.runpytest_inprocess(f'{CAPTURE_OPTION}={pytester.path / "capture"}')
    assert without_plugin.ret == pytest.ExitCode.USAGE_ERROR
    without_plugin.stderr.fnmatch_lines(['*unrecognized arguments: --refactor-capture=*'])


def test_capture_stops_the_run_when_it_misses_the_canary_stream(tmp_path: Path):
    """A canary stream the capture doesn't see fails the run instead of the capture recording nothing.

    The capture's `__aiter__` wrappers aren't installed, standing in for streams iterated some way they don't reach.
    """
    unwrapped = StreamedResponse.__aiter__
    with pytest.raises(pytest.UsageError, match='was not recorded'):
        RefactorCapture(tmp_path).check_stream_hook()
    assert StreamedResponse.__aiter__ is unwrapped


def test_capture_stops_the_run_when_it_misses_a_canary_request(tmp_path: Path):
    """A canary request the capture doesn't see fails the run instead of the capture recording nothing.

    The capture's `send` wrappers aren't installed, standing in for SDKs that moved to an HTTP client they don't reach.
    """
    with pytest.raises(
        pytest.UsageError,
        match='with `GroqModel` over `httpx` and `AnthropicModel` over `httpx2` was not recorded',
    ):
        RefactorCapture(tmp_path).check_request_hook()


def test_capture_stops_the_run_when_a_listed_mock_is_gone(
    run: RunCapture, pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`SDK_MOCKS` entries whose module the collected tests didn't import are skipped, and one that's gone stops the run."""
    pytester.makepyfile(unimported_mock='class MockClient:\n    def create(self):\n        pass\n')
    monkeypatch.setattr(
        refactor_capture,
        'SDK_MOCKS',
        [('unimported_mock', 'MockClient', ('create',)), ('gone_mock', 'MockClient', ('create',))],
    )

    result = run(tmp_path)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*`SDK_MOCKS` lists `gone_mock.MockClient.create`, which is gone*'])


def test_sdk_mocks_exist():
    """Each method `SDK_MOCKS` lists exists, so renaming one fails here rather than when a capture starts."""
    for module_name, class_name, method_names in SDK_MOCKS:
        mock = getattr(importlib.import_module(module_name), class_name)
        assert [name for name in method_names if name not in vars(mock)] == [], f'{module_name}.{class_name}'


def test_capture_refuses_to_append_to_an_earlier_capture(
    run: RunCapture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run(tmp_path).assert_outcomes(passed=8)

    result = run(tmp_path)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*already holds a capture*'])

    # `pytest-xdist` workers write alongside the controller, which made the check.
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    run(tmp_path).assert_outcomes(passed=8)
    assert sorted(path.name for path in tmp_path.glob('*.jsonl')) == ['gw0.jsonl', 'main.jsonl']


def test_compare_flags_a_changed_event_or_request(run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    base, same, changed = tmp_path / 'base', tmp_path / 'same', tmp_path / 'changed'
    run(base).assert_outcomes(passed=8)
    run(same).assert_outcomes(passed=8)
    run(changed, '-k', 'not fails', text='goodbye').assert_outcomes(passed=7, deselected=1)
    capsys.readouterr()

    assert compare(base, tmp_path / 'missing') == 2
    assert capsys.readouterr().err.startswith('No capture in')
    assert compare(base, same) == 2
    assert capsys.readouterr().err.startswith('Both captures imported `pydantic_ai` from the same file')
    nothing = tmp_path / 'nothing'
    run(nothing, '-k', 'no_stream').assert_outcomes(passed=1, deselected=7)
    (nothing / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{nothing}/pydantic_ai/__init__.py'}))
    assert compare(base, nothing) == 2
    assert capsys.readouterr().err.startswith('Nothing recorded in')

    # Stand in for captures of two checkouts.
    for capture_dir in (same, changed):
        (capture_dir / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{capture_dir}/pydantic_ai/__init__.py'}))

    assert compare(base, same) == 0
    assert compare(base, changed) == 1
    assert capsys.readouterr().out == snapshot("""\
nodes: base 7, head 7
records: base 7, head 7
stream events: head 11
requests: head 3
nodes per stream module, head:
  pydantic_ai.models.function: 3
  test_streams: 1
nodes per request target, head:
  POST http://127.0.0.1:PORT/v1/upload: 2
  tests.models.mock_openai.MockOpenAI.chat_completions_create: 1
only in base: 0, only in head: 0, differing: 0
nodes: base 7, head 6
records: base 7, head 6
stream events: head 9
requests: head 3
nodes per stream module, head:
  pydantic_ai.models.function: 2
  test_streams: 1
nodes per request target, head:
  POST http://127.0.0.1:PORT/v1/upload: 2
  tests.models.mock_openai.MockOpenAI.chat_completions_create: 1
only in base: 1, only in head: 0, differing: 4
  only in base: test_streams.py::test_fails_mid_stream
  differing: test_requests.py::test_json_over_httpx
  differing: test_requests.py::test_sdk_mock
  differing: test_streams.py::test_fallback_peeks_the_first_event
  differing: test_streams.py::test_text
""")


def test_normalized_replaces_values_that_change_between_runs():
    now = datetime.now(UTC)
    assert normalized(
        {
            'aware_now': now,
            'naive_now': now.replace(tzinfo=None),
            'recorded': datetime(2024, 1, 1, tzinfo=UTC),
            'not_a_date': '2024-02-30T00:00:00',
            'tool_call_id': f'pyd_ai_{"0" * 32}',
            'uuid': '12345678-1234-5678-1234-567812345678',
            'repr': object(),
            'history_digest': '0' * 64,
            'duration': timedelta(seconds=1.5),
            'url': 'http://127.0.0.1:8123/v1',
            'multipart': '--0123456789abcdef0123456789abcdef--',
        }
    ) == snapshot(
        {
            'aware_now': 'NOW',
            'naive_now': 'NOW',
            'recorded': '2024-01-01T00:00:00Z',
            'not_a_date': '2024-02-30T00:00:00',
            'tool_call_id': 'pyd_ai_ID',
            'uuid': 'UUID',
            'repr': '<object object at 0x?>',
            'history_digest': 'DIGEST',
            'duration': 'DURATION',
            'url': 'http://127.0.0.1:PORT/v1',
            'multipart': '--BOUNDARY--',
        }
    )
