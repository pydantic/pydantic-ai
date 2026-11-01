"""Capture the events each streamed model response emits, per test node, to prove a refactor keeps them.

Copy this file out of the tree first, so the base commit, which may not have it, can load it too. Then run the
capture once on the base commit and once on the branch, into separate new directories, over the same test files:
those that reach the changed stream code, such as every file naming the changed model classes. Then compare:

    mkdir /tmp/plugin && cp scripts/stream_event_capture.py /tmp/plugin/
    STREAM_EVENT_CAPTURE_DIR=/tmp/base PYTHONPATH=/tmp/plugin uv run pytest -p stream_event_capture <test files>
    STREAM_EVENT_CAPTURE_DIR=/tmp/head PYTHONPATH=/tmp/plugin uv run pytest -p stream_event_capture <test files>
    uv run /tmp/plugin/stream_event_capture.py /tmp/base /tmp/head

As a pytest plugin, it records every event a `StreamedResponse`'s `_get_event_iterator` yields, then the response
`get()` builds once that iterator is exhausted, or the error the iterator raised. It hooks in through
`StreamedResponse.__aiter__`, so a subclass that overrides `__aiter__` without calling it, like the continuation
composite in `models/_continuation.py`, is recorded only through the streams it iterates. Each stream is one JSON
line, written per `pytest-xdist` worker when its test ends. Generated tool call ids, UUIDs, object addresses and
timestamps within a day of the capture are normalized, so two runs of the same code compare equal.

The comparison lists the test nodes whose streams differ or that only one capture has, and exits non-zero if
there are any. It also counts the nodes whose streams reach each module that defines a `StreamedResponse`
subclass, so the capture's reach over the changed modules shows.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic_core import to_json

from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import StreamedResponse

CAPTURE_DIR_ENV = 'STREAM_EVENT_CAPTURE_DIR'
GENERATED_VALUES = [
    (re.compile(r' at 0x[0-9a-f]+'), ' at 0x?'),
    (re.compile(r'pyd_ai_[0-9a-f]{32}'), 'pyd_ai_ID'),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'UUID'),
]
ISO_DATETIME = re.compile(r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"')

_current_node: list[str] = []
_node_streams: list[dict[str, object]] = []


def pytest_configure(config: pytest.Config) -> None:
    """Wrap each stream's `_get_event_iterator` when its iteration starts."""
    if not os.environ.get(CAPTURE_DIR_ENV):
        raise pytest.UsageError(f'Set `{CAPTURE_DIR_ENV}` to the directory the capture is written to.')
    capture_dir = Path(os.environ[CAPTURE_DIR_ENV])
    # Workers start once the controller has checked, and another worker may already have written by then.
    if 'PYTEST_XDIST_WORKER' not in os.environ and any(capture_dir.glob('*.jsonl')):
        raise pytest.UsageError(f'`{capture_dir}` already holds a capture; a second one would be appended to it.')
    capture_dir.mkdir(parents=True, exist_ok=True)
    original_aiter = StreamedResponse.__aiter__

    def recording_aiter(self: StreamedResponse) -> AsyncIterator[ModelResponseStreamEvent]:
        if self._event_iterator is None and '_get_event_iterator' not in vars(self):  # pyright: ignore[reportPrivateUsage]
            # `__aiter__` calls `self._get_event_iterator()`, so an instance attribute wraps it for this stream only.
            setattr(self, '_get_event_iterator', _recorded_events(self))
        return original_aiter(self)

    setattr(StreamedResponse, '__aiter__', recording_aiter)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    """Write the streams a test iterated once it ends."""
    _current_node.append(item.nodeid)
    try:
        return (yield)
    finally:
        _current_node.clear()
        if _node_streams:
            worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
            with open(Path(os.environ[CAPTURE_DIR_ENV]) / f'{worker}.jsonl', 'a', encoding='utf-8') as f:
                f.writelines(json.dumps(stream) + '\n' for stream in _node_streams)
            _node_streams.clear()


def _recorded_events(stream: StreamedResponse) -> Callable[[], AsyncIterator[ModelResponseStreamEvent]]:
    stream_type = type(stream)
    record: dict[str, object] = {
        'node': _current_node[0] if _current_node else None,
        'n': len(_node_streams),
        'stream': f'{stream_type.__module__}.{stream_type.__qualname__}',
        'modules': sorted(
            {cls.__module__ for cls in stream_type.__mro__ if issubclass(cls, StreamedResponse)}
            - {StreamedResponse.__module__}
        ),
    }
    _node_streams.append(record)
    events: list[object] = []
    record['events'] = events

    async def get_event_iterator() -> AsyncIterator[ModelResponseStreamEvent]:
        try:
            async for event in stream_type._get_event_iterator(stream):  # pyright: ignore[reportPrivateUsage]
                events.append(_normalized(event))
                yield event
        except Exception as e:
            record['error'] = _normalized(f'{type(e).__name__}: {e}')
            raise
        try:
            record['response'] = _normalized(stream.get())
        except Exception as e:
            record['response_error'] = _normalized(f'{type(e).__name__}: {e}')

    return get_event_iterator


def _normalized(value: object) -> object:
    now = datetime.now(UTC)

    def generated_timestamp(match: re.Match[str]) -> str:
        try:
            timestamp = datetime.fromisoformat(match[1])
        except ValueError:
            return match[0]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return '"NOW"' if abs(timestamp - now) < timedelta(days=1) else match[0]

    text = ISO_DATETIME.sub(generated_timestamp, to_json(value, fallback=repr, bytes_mode='base64').decode())
    for pattern, replacement in GENERATED_VALUES:
        text = pattern.sub(replacement, text)
    return json.loads(text)


def _load(capture_dir: Path) -> dict[str, list[str]]:
    nodes: defaultdict[str, list[str]] = defaultdict(list)
    for path in sorted(capture_dir.glob('*.jsonl')):
        for line in path.read_text().splitlines():
            nodes[json.loads(line)['node']].append(line)
    return nodes


def main() -> int:
    """Compare a base capture with a head capture."""
    if len(sys.argv) != 3:
        print(f'usage: {sys.argv[0]} <base capture dir> <head capture dir>', file=sys.stderr)
        return 2
    base, head = _load(Path(sys.argv[1])), _load(Path(sys.argv[2]))
    print(f'nodes: base {len(base)}, head {len(head)}')
    print(f'streams: base {sum(map(len, base.values()))}, head {sum(map(len, head.values()))}')
    events = sum(len(json.loads(line)['events']) for lines in head.values() for line in lines)
    print(f'events: head {events}')

    reach: Counter[str] = Counter()
    for lines in head.values():
        reach.update({module for line in lines for module in json.loads(line)['modules']})
    print('nodes per stream module, head:')
    for module, count in sorted(reach.items()):
        print(f'  {module}: {count}')

    only_base = sorted(base.keys() - head.keys())
    only_head = sorted(head.keys() - base.keys())
    differing = sorted(node for node in base.keys() & head.keys() if base[node] != head[node])
    print(f'only in base: {len(only_base)}, only in head: {len(only_head)}, differing: {len(differing)}')
    for label, nodes in (('only in base', only_base), ('only in head', only_head), ('differing', differing)):
        for node in nodes:
            print(f'  {label}: {node}')
    return 1 if only_base or only_head or differing else 0


if __name__ == '__main__':
    sys.exit(main())
