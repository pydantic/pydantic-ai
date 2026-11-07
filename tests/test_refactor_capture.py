"""The opt-in capture in `tests/refactor_capture.py`, mostly run against small in-process pytest sessions.

They test the capture tooling rather than model behavior, so no recording applies.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeAlias

import pytest
from inline_snapshot import snapshot

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models import StreamedResponse, override_allow_model_requests

from .conftest import IsStr, try_import

with try_import() as imports_successful:
    from . import refactor_capture
    from .refactor_capture import (
        CAPTURE_OPTION,
        SDK_MOCKS,
        SOURCE_FILE,
        RefactorCapture,
        bedrock_canary_model,
        compare,
        normalized,
    )

pytest_plugins = ['pytester']

pytestmark = [
    # The plugin imports `httpx`, `groq`, `anthropic` and `boto3`; the `REQUESTS` sessions import `openai` too.
    pytest.mark.skipif(
        not imports_successful() or importlib.util.find_spec('openai') is None,
        reason='`httpx`, `groq`, `anthropic`, `boto3` or `openai` not installed',
    ),
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
import time
import uuid
from contextlib import suppress

import httpx
import httpx2
from anthropic import AsyncAnthropic
from openai.types.chat import ChatCompletionMessage

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
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


async def send_with_beta():
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(400))) as client:
        # Keep the captured target stable across SDK versions that add or omit this query parameter.
        sdk_client = AsyncAnthropic(api_key='secret', http_client=client, default_query=dict(beta='true'))
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=sdk_client))
        with override_allow_model_requests(True), suppress(ModelHTTPError):
            await Agent(model).run('hi', model_settings=dict(anthropic_betas=[{text!r}]))


def test_anthropic_beta():
    asyncio.run(send_with_beta())


def test_function_model():
    agent = Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('hi')])), instructions='be brief')
    agent.run_sync({text!r}, model_settings=dict(temperature=0.5))


def values_that_change_between_runs():
    return dict(
        path='/tmp/pytest-of-ci/pytest-%d/popen-gw%d/test_x0' % (uuid.uuid4().int % 1000, uuid.uuid4().int % 8),
        id=uuid.uuid4().hex,
        capability='<thinking:%s>' % uuid.uuid4().hex[:6],
        latency_ms=time.perf_counter(),
    )


def test_test_model():
    agent = Agent(TestModel())
    agent.tool_plain(values_that_change_between_runs)
    agent.run_sync('hi')
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
    run(tmp_path).assert_outcomes(passed=11)

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
    run(tmp_path).assert_outcomes(passed=11)

    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    # `test_anthropic_beta`'s request is checked by `test_compare_flags_a_changed_request_header`, and the calls on
    # in-process models by `test_capture_records_each_in_process_model_call`.
    records = [
        record
        for line in lines
        if (record := json.loads(line))['kind'] == 'request'
        and record['node'] != 'test_requests.py::test_anthropic_beta'
        and 'model_request_parameters' not in record
    ]
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


def test_capture_records_each_in_process_model_call(run: RunCapture, tmp_path: Path):
    """A call on `FunctionModel` or `TestModel`, streamed or not, is recorded with what the model is handed.

    The values `values_that_change_between_runs` returns reach `TestModel`'s second call normalized.
    """
    run(tmp_path, '-k', 'test_text or function_model or test_model').assert_outcomes(passed=3, deselected=8)

    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    calls = [record for line in lines if 'model_request_parameters' in (record := json.loads(line))]
    assert [
        (
            call['node'],
            call['target'],
            [part['part_kind'] for message in call['messages'] for part in message['parts']],
            call['messages'][0]['instructions'],
            call['model_settings'],
            [tool['name'] for tool in call['model_request_parameters']['function_tools']],
        )
        for call in calls
    ] == snapshot(
        [
            (
                'test_requests.py::test_function_model',
                'pydantic_ai.models.function.FunctionModel.request',
                ['user-prompt'],
                'be brief',
                {'temperature': 0.5},
                [],
            ),
            (
                'test_requests.py::test_test_model',
                'pydantic_ai.models.test.TestModel.request',
                ['user-prompt'],
                None,
                None,
                ['values_that_change_between_runs'],
            ),
            (
                'test_requests.py::test_test_model',
                'pydantic_ai.models.test.TestModel.request',
                ['user-prompt', 'tool-call', 'tool-return'],
                None,
                None,
                ['values_that_change_between_runs'],
            ),
            (
                'test_streams.py::test_text',
                'pydantic_ai.models.function.FunctionModel.request_stream',
                ['user-prompt'],
                None,
                None,
                [],
            ),
        ]
    )
    assert calls[2]['messages'][-1]['parts'][0]['content'] == snapshot(
        {
            'path': '/tmp/pytest-of-ci/pytest-N/test_x0',
            'id': 'UUID',
            'capability': '<thinking:ID>',
            'latency_ms': 'LATENCY',
        }
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

    The capture's request wrappers aren't installed, standing in for SDKs that moved to an HTTP client they don't reach.
    """
    with pytest.raises(
        pytest.UsageError,
        match=(
            'with `GroqModel` over `httpx` and `AnthropicModel` over `httpx2` and `BedrockConverseModel` over `botocore` '
            'was not recorded'
        ),
    ):
        RefactorCapture(tmp_path).check_request_hook()


async def test_capture_records_a_request_botocore_sends(tmp_path: Path):
    """A request a botocore client sends is recorded unsigned, with the extra headers the model adds."""
    capture = RefactorCapture(tmp_path)
    capture.install()
    try:
        with override_allow_model_requests(True), suppress(ModelHTTPError):
            await Agent(bedrock_canary_model()).run('hello', model_settings={'extra_headers': {'x-canary': 'yes'}})
    finally:
        capture.pytest_unconfigure()

    assert capture.records == snapshot(
        [
            {
                'kind': 'request',
                'node': None,
                'target': 'POST https://bedrock-runtime.us-east-1.amazonaws.com/model/us.amazon.nova-micro-v1%3A0/converse',
                'headers': {'Content-Type': 'application/json', 'User-Agent': IsStr(), 'x-canary': 'yes'},
                'body': {
                    'messages': [{'role': 'user', 'content': [{'text': 'hello'}]}],
                    'system': [],
                    'inferenceConfig': {},
                },
            }
        ]
    )


def test_capture_stops_the_run_when_it_misses_a_canary_model_call(tmp_path: Path):
    """A canary call on an in-process model the capture doesn't see fails the run instead of recording nothing.

    The capture's model wrappers aren't installed, standing in for an agent that stopped calling models through them.
    """
    with pytest.raises(pytest.UsageError, match='were not recorded') as error:
        RefactorCapture(tmp_path).check_model_hook()
    assert str(error.value) == snapshot(
        'The calls a canary agent run made on `pydantic_ai.models.function.FunctionModel.request`, `pydantic_ai.models.function.FunctionModel.request_stream`, `pydantic_ai.models.test.TestModel.request`, `pydantic_ai.models.test.TestModel.request_stream` were not recorded, so the capture would miss what in-process models are handed. Update `tests/refactor_capture.py` to the methods an agent calls them through now.'
    )


def test_capture_stops_the_run_when_a_listed_mock_is_gone(
    run: RunCapture, pytester: pytest.Pytester, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`SDK_MOCKS` entries whose module collection didn't import are skipped, and one that's gone stops the run."""
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
    run(tmp_path).assert_outcomes(passed=11)

    result = run(tmp_path)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*already holds a capture*'])

    # `pytest-xdist` workers write alongside the controller, which made the check.
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    run(tmp_path).assert_outcomes(passed=11)
    assert sorted(path.name for path in tmp_path.glob('*.jsonl')) == ['gw0.jsonl', 'main.jsonl']


def test_compare_flags_a_changed_event_or_request(run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    base, same, changed = tmp_path / 'base', tmp_path / 'same', tmp_path / 'changed'
    run(base).assert_outcomes(passed=11)
    run(same).assert_outcomes(passed=11)
    run(changed, '-k', 'not fails', text='goodbye').assert_outcomes(passed=10, deselected=1)
    capsys.readouterr()

    assert compare(base, tmp_path / 'missing') == 2
    assert capsys.readouterr().err.startswith('No capture in')
    assert compare(base, same) == 2
    assert capsys.readouterr().err.startswith('Both captures imported `pydantic_ai` from the same file')
    nothing = tmp_path / 'nothing'
    run(nothing, '-k', 'no_stream').assert_outcomes(passed=1, deselected=10)
    (nothing / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{nothing}/pydantic_ai/__init__.py'}))
    assert compare(base, nothing) == 2
    assert capsys.readouterr().err.startswith('Nothing recorded in')

    # Stand in for captures of two checkouts.
    for capture_dir in (same, changed):
        (capture_dir / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{capture_dir}/pydantic_ai/__init__.py'}))

    assert compare(base, same) == 0
    assert compare(base, changed) == 1
    assert capsys.readouterr().out == snapshot("""\
nodes: base 10, head 10
records: base 14, head 14
stream events: head 11
requests: head 10
nodes per stream module, head:
  pydantic_ai.models.function: 3
  test_streams: 1
nodes per request target, head:
  POST http://127.0.0.1:PORT/v1/upload: 2
  POST https://api.anthropic.com/v1/messages?beta=true: 1
  pydantic_ai.models.function.FunctionModel.request: 1
  pydantic_ai.models.function.FunctionModel.request_stream: 3
  pydantic_ai.models.test.TestModel.request: 1
  tests.models.mock_openai.MockOpenAI.chat_completions_create: 1
nodes that recorded no request, head: 2
  test_streams.py::test_no_stream
  test_streams.py::test_subclass_defined_after_the_capture_started
only in base: 0, only in head: 0, differing: 0
nodes: base 10, head 9
records: base 14, head 12
stream events: head 9
requests: head 9
nodes per stream module, head:
  pydantic_ai.models.function: 2
  test_streams: 1
nodes per request target, head:
  POST http://127.0.0.1:PORT/v1/upload: 2
  POST https://api.anthropic.com/v1/messages?beta=true: 1
  pydantic_ai.models.function.FunctionModel.request: 1
  pydantic_ai.models.function.FunctionModel.request_stream: 2
  pydantic_ai.models.test.TestModel.request: 1
  tests.models.mock_openai.MockOpenAI.chat_completions_create: 1
nodes that recorded no request, head: 2
  test_streams.py::test_no_stream
  test_streams.py::test_subclass_defined_after_the_capture_started
only in base: 1, only in head: 0, differing: 6
  only in base: test_streams.py::test_fails_mid_stream
  differing: test_requests.py::test_anthropic_beta
  differing: test_requests.py::test_function_model
  differing: test_requests.py::test_json_over_httpx
  differing: test_requests.py::test_sdk_mock
  differing: test_streams.py::test_fallback_peeks_the_first_event
  differing: test_streams.py::test_text
""")


def test_compare_flags_a_changed_request_header(run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Request headers are recorded without credentials, so a changed `anthropic-beta` header is flagged."""
    base, head = tmp_path / 'base', tmp_path / 'head'
    run(base, '-k', 'anthropic_beta', text='files-api-2025-04-14').assert_outcomes(passed=1, deselected=10)
    run(head, '-k', 'anthropic_beta', text='code-execution-2025-08-25').assert_outcomes(passed=1, deselected=10)
    (record,) = [json.loads(line) for line in (base / 'main.jsonl').read_text().splitlines()]
    assert record['headers']['anthropic-beta'] == snapshot('files-api-2025-04-14')
    assert {'x-api-key', 'authorization'}.isdisjoint(record['headers'])
    (head / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{head}/pydantic_ai/__init__.py'}))
    capsys.readouterr()

    assert compare(base, head) == 1
    assert capsys.readouterr().out.splitlines()[-2:] == snapshot(
        ['only in base: 0, only in head: 0, differing: 1', '  differing: test_requests.py::test_anthropic_beta']
    )


def test_compare_names_the_nodes_that_recorded_no_request(
    run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A selected node that recorded no request is listed, whether it recorded a stream or nothing, without failing."""
    base, head = tmp_path / 'base', tmp_path / 'head'
    for capture_dir in (base, head):
        run(capture_dir, '-k', 'subclass or no_stream or json').assert_outcomes(passed=3, deselected=8)
    (head / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{head}/pydantic_ai/__init__.py'}))
    capsys.readouterr()

    assert compare(base, head) == 0
    assert capsys.readouterr().out.splitlines()[-4:] == snapshot(
        [
            'nodes that recorded no request, head: 2',
            '  test_streams.py::test_no_stream',
            '  test_streams.py::test_subclass_defined_after_the_capture_started',
            'only in base: 0, only in head: 0, differing: 0',
        ]
    )


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
            'content-type': 'multipart/form-data; boundary=0123456789abcdef0123456789abcdef',
            'tmp_path': '/tmp/pytest-of-ci/pytest-12/test_x0/notes.txt',
            'xdist_tmp_path': '/tmp/pytest-of-ci/pytest-12/popen-gw3/test_x0',
            'uuid4_hex': '12345678123441238123123456789abc',
            'other_hex': '0123456789abcdef0123456789abcdef',
            'capability_key': '<thinking:0a1b2c>',
            'latency_ms': 12.5,
            'plan': 'Current plan:\n1. [ ] [0a1b2c3d] a\n2. [x] [4e5f6a7b] b',
            'added_step': "Added step 'a' with id: 0a1b2c3d",
            'echoed_step_id': {'task_id': '0a1b2c3d'},
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
            'content-type': 'multipart/form-data; boundary=BOUNDARY',
            'tmp_path': '/tmp/pytest-of-ci/pytest-N/test_x0/notes.txt',
            'xdist_tmp_path': '/tmp/pytest-of-ci/pytest-N/test_x0',
            'uuid4_hex': 'UUID',
            'other_hex': '0123456789abcdef0123456789abcdef',
            'capability_key': '<thinking:ID>',
            'latency_ms': 'LATENCY',
            'plan': 'Current plan:\n1. [ ] [STEP_ID] a\n2. [x] [STEP_ID] b',
            'added_step': "Added step 'a' with id: STEP_ID",
            'echoed_step_id': {'task_id': '0a1b2c3d'},
        }
    )
