"""The opt-in capture in `tests/refactor_capture.py`, run against small in-process pytest sessions."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import TypeAlias

import pytest
from inline_snapshot import snapshot

import pydantic_ai
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import StreamedResponse

from .refactor_capture import SOURCE_FILE, Capture, compare

pytest_plugins = ['pytester']

RunCapture: TypeAlias = Callable[..., pytest.RunResult]

STREAMS = """
import asyncio

import pytest

from pydantic_ai.direct import model_request_stream
from pydantic_ai.messages import ModelRequest
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


def test_text():
    asyncio.run(drain(stream_text))


def test_fails_mid_stream():
    with pytest.raises(RuntimeError, match='provider went away'):
        asyncio.run(drain(stream_then_fail))


def test_no_stream():
    pass
"""


@pytest.fixture
def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunCapture:
    """Capture an in-process pytest session over `STREAMS`, streaming `text`, into `capture_dir`."""
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)

    def run(capture_dir: Path, *args: str, text: str = 'hello') -> pytest.RunResult:
        pytester.makepyfile(test_streams=STREAMS.format(text=text))
        return pytester.runpytest_inprocess('-p', 'no:cacheprovider', *args, plugins=[Capture(capture_dir)])

    return run


def test_capture_records_each_stream_per_test(run: RunCapture, tmp_path: Path):
    run(tmp_path).assert_outcomes(passed=3)

    assert json.loads((tmp_path / SOURCE_FILE).read_text()) == {'pydantic_ai': pydantic_ai.__file__}
    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    assert [
        (
            record['node'],
            record['stream'],
            record['modules'],
            len(record['events']),
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
                1,
                None,
                'complete',
            ),
            (
                'test_streams.py::test_fails_mid_stream',
                'pydantic_ai.models.function.FunctionStreamedResponse',
                ['pydantic_ai.models.function'],
                1,
                'RuntimeError: provider went away',
                'incomplete',
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


def test_capture_stops_the_run_when_the_hook_point_changes(
    run: RunCapture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A stream that `__aiter__` no longer reads through `_get_event_iterator` fails the run instead of going unrecorded."""

    def iterate_without_the_hook(stream: StreamedResponse) -> AsyncIterator[ModelResponseStreamEvent]:
        return type(stream)._get_event_iterator(stream)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(StreamedResponse, '__aiter__', iterate_without_the_hook)

    result = run(tmp_path)

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*A canary stream iterated through `StreamedResponse.__aiter__` was not recorded*'])
    assert StreamedResponse.__aiter__ is iterate_without_the_hook
    assert not list(tmp_path.glob('*.jsonl'))


def test_capture_refuses_to_append_to_an_earlier_capture(
    run: RunCapture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run(tmp_path).assert_outcomes(passed=3)

    result = run(tmp_path)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(['*already holds a capture*'])

    # `pytest-xdist` workers write alongside the controller, which made the check.
    monkeypatch.setenv('PYTEST_XDIST_WORKER', 'gw0')
    run(tmp_path).assert_outcomes(passed=3)
    assert sorted(path.name for path in tmp_path.glob('*.jsonl')) == ['gw0.jsonl', 'main.jsonl']


def test_compare_flags_a_changed_event(run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    base, same, changed = tmp_path / 'base', tmp_path / 'same', tmp_path / 'changed'
    run(base).assert_outcomes(passed=3)
    run(same).assert_outcomes(passed=3)
    run(changed, '-k', 'not fails', text='goodbye').assert_outcomes(passed=2, deselected=1)
    capsys.readouterr()

    assert compare(base, same) == 2
    assert capsys.readouterr().err.startswith('Both captures imported `pydantic_ai` from the same file')

    # Stand in for captures of two checkouts.
    for capture_dir in (same, changed):
        (capture_dir / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{capture_dir}/pydantic_ai/__init__.py'}))

    assert compare(base, same) == 0
    assert compare(base, changed) == 1
    assert capsys.readouterr().out == snapshot("""\
nodes: base 2, head 2
records: base 2, head 2
stream events: head 2
nodes per stream module, head:
  pydantic_ai.models.function: 2
only in base: 0, only in head: 0, differing: 0
nodes: base 2, head 1
records: base 2, head 1
stream events: head 1
nodes per stream module, head:
  pydantic_ai.models.function: 1
only in base: 1, only in head: 0, differing: 1
  only in base: test_streams.py::test_fails_mid_stream
  differing: test_streams.py::test_text
""")
