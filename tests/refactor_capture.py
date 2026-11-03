"""Record what streamed model responses emit, per test node, to prove a refactor keeps it.

The capture is off unless `REFACTOR_CAPTURE_DIR` is set: only then does `tests/conftest.py` register `Capture`.
To compare a base commit with the working tree, run this from the repository root, over one or two test nodes that
stream through each changed line rather than whole test files:

    uv run python -m tests.refactor_capture "$(git merge-base HEAD upstream/main)" <test nodes>

It extracts the base commit's copy of every workspace package with `git archive`, runs pytest over the nodes once with
those sources ahead of the installed ones on `PYTHONPATH` and once without, each into its own new capture directory,
then compares the two. The tests, this plugin and the environment come from the working tree both times, so the
library code is the only difference.

Each stream is one JSON line of kind `stream`: the events its `_get_event_iterator` yields, the error that iterator
raised if any, and the response `get()` builds once the test ends. It hooks in through
`StreamedResponse.__aiter__`, so a subclass that overrides `__aiter__` without calling it, like the continuation
composite in `models/_continuation.py`, is recorded only through the streams it iterates. When a capture starts, it
iterates a canary stream and stops the run if that stream wasn't recorded, so a change to the hook point fails loudly
instead of recording nothing. Lines are written per `pytest-xdist` worker when each test ends. Generated tool call
ids, UUIDs, object addresses, timestamps within a day of the capture, history digests (which hash those timestamps),
attempt durations and local server ports are normalized, so two runs of the same code compare equal.

The comparison lists the test nodes whose records differ or that only one capture has, and exits non-zero if there
are any. It refuses two captures that imported `pydantic_ai` from the same file, since the base run then didn't load
the base commit. It also counts the nodes whose streams reach each module that defines a `StreamedResponse`
subclass, so the capture's reach over the changed modules shows.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic_core import to_json

import pydantic_ai
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import ModelRequestParameters, StreamedResponse

from .continuation_utils import ScriptedStreamedResponse, StreamSegment

CAPTURE_DIR_ENV = 'REFACTOR_CAPTURE_DIR'
SOURCE_FILE = 'source.json'
GENERATED_VALUES = [
    (re.compile(r' at 0x[0-9a-f]+'), ' at 0x?'),
    (re.compile(r'pyd_ai_[0-9a-f]{32}'), 'pyd_ai_ID'),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'UUID'),
    (re.compile(r'"history_digest":"[0-9a-f]{64}"'), '"history_digest":"DIGEST"'),
    (re.compile(r'"duration":"P[0-9DTHMS.]+"'), '"duration":"DURATION"'),
    (re.compile(r'(127\.0\.0\.1|localhost):\d+'), r'\1:PORT'),
]
ISO_DATETIME = re.compile(r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"')


class Capture:
    """Records each stream a test iterates, and appends the records to the capture directory when the test ends."""

    def __init__(self, capture_dir: Path) -> None:
        self.capture_dir = capture_dir
        self.node: str | None = None
        self.records: list[tuple[dict[str, object], StreamedResponse]] = []
        self._monkeypatch = pytest.MonkeyPatch()

    def pytest_configure(self) -> None:
        # Workers start once the controller has checked, and another worker may already have written by then.
        if 'PYTEST_XDIST_WORKER' not in os.environ:
            if any(self.capture_dir.glob('*.jsonl')):
                raise pytest.UsageError(f'`{self.capture_dir}` already holds a capture; a second would be appended.')
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            (self.capture_dir / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': pydantic_ai.__file__}))
        self.install()
        self.check_hook()

    def pytest_unconfigure(self) -> None:
        self._monkeypatch.undo()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> Generator[None, object, object]:
        self.node = item.nodeid
        try:
            return (yield)
        finally:
            self.node = None
            self.write()

    def install(self) -> None:
        """Wrap each stream's `_get_event_iterator` when its iteration starts."""
        original_aiter = StreamedResponse.__aiter__

        def recording_aiter(stream: StreamedResponse) -> AsyncIterator[ModelResponseStreamEvent]:
            if '_get_event_iterator' not in vars(stream):
                # `__aiter__` calls `self._get_event_iterator()` once, so an instance attribute wraps it for this stream only.
                setattr(stream, '_get_event_iterator', lambda: self._recorded_events(stream))
            return original_aiter(stream)

        self._monkeypatch.setattr(StreamedResponse, '__aiter__', recording_aiter)

    def check_hook(self) -> None:
        """Stop the run unless a canary stream's events are recorded."""
        canary = ScriptedStreamedResponse(
            ModelRequestParameters(),
            _segment=StreamSegment(
                texts=['canary'], state='complete', provider_response_id='canary', input_tokens=1, output_tokens=1
            ),
        )

        async def drain() -> None:
            async for _ in canary:
                pass

        asyncio.run(drain())
        recorded = any(stream is canary and record['events'] for record, stream in self.records)
        self.records.clear()
        if not recorded:
            self._monkeypatch.undo()
            raise pytest.UsageError(
                'A canary stream iterated through `StreamedResponse.__aiter__` was not recorded, so the capture would '
                'miss every stream. Update `tests/refactor_capture.py` to the new hook point.'
            )

    def write(self) -> None:
        """Append the streams recorded so far, with the response each builds now, to this worker's file."""
        if self.records:
            worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
            with open(self.capture_dir / f'{worker}.jsonl', 'a', encoding='utf-8') as f:
                f.writelines(
                    json.dumps({**record, 'response': _normalized(stream.get())}) + '\n'
                    for record, stream in self.records
                )
            self.records.clear()

    def _recorded_events(self, stream: StreamedResponse) -> AsyncIterator[ModelResponseStreamEvent]:
        stream_type = type(stream)
        events: list[object] = []
        record: dict[str, object] = {
            'kind': 'stream',
            'node': self.node,
            'stream': f'{stream_type.__module__}.{stream_type.__qualname__}',
            'modules': sorted(
                {cls.__module__ for cls in stream_type.__mro__ if issubclass(cls, StreamedResponse)}
                - {StreamedResponse.__module__}
            ),
            'events': events,
        }
        self.records.append((record, stream))

        async def recorded() -> AsyncIterator[ModelResponseStreamEvent]:
            try:
                async for event in stream_type._get_event_iterator(stream):  # pyright: ignore[reportPrivateUsage]
                    events.append(_normalized(event))
                    yield event
            except Exception as e:
                record['error'] = _normalized(f'{type(e).__name__}: {e}')
                raise

        return recorded()


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
            record: dict[str, object] = json.loads(line)
            nodes[str(record['node'])].append(line)
    return nodes


def compare(base_dir: Path, head_dir: Path) -> int:
    """Print how a head capture differs from a base capture. Return 1 if it does, 2 if they can't be compared."""
    base_source, head_source = (json.loads((d / SOURCE_FILE).read_text()) for d in (base_dir, head_dir))
    if base_source == head_source:
        print(f'Both captures imported `pydantic_ai` from the same file: {base_source}', file=sys.stderr)
        return 2
    base, head = _load(base_dir), _load(head_dir)
    head_records = [json.loads(line) for lines in head.values() for line in lines]
    print(f'nodes: base {len(base)}, head {len(head)}')
    print(f'records: base {sum(map(len, base.values()))}, head {len(head_records)}')
    print(f'stream events: head {sum(len(record["events"]) for record in head_records if record["kind"] == "stream")}')

    reach: Counter[str] = Counter()
    for lines in head.values():
        reach.update({module for line in lines for module in json.loads(line).get('modules', [])})
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


# Runs `git` and two pytest sessions in subprocesses; the tests cover what it calls.
def main(argv: list[str] | None = None) -> int:  # pragma: no cover
    """Capture the selected tests on a base commit and on the working tree, then compare the captures."""
    parser = argparse.ArgumentParser(prog='python -m tests.refactor_capture', description=main.__doc__)
    parser.add_argument('base', help='the commit to compare against, usually the merge base with `main`')
    parser.add_argument('pytest_args', nargs=argparse.REMAINDER, help='test nodes and pytest options')
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parent.parent
    venv = Path(sys.prefix).resolve()
    # The workspace packages are installed editable, which puts each package root on `sys.path`.
    entries = {Path(entry).resolve() for entry in sys.path if entry}
    roots = sorted(
        entry.relative_to(repo)
        for entry in entries
        if entry.is_relative_to(repo) and entry != repo and not entry.is_relative_to(venv)
    )
    out = Path(tempfile.mkdtemp(prefix='refactor-capture-'))
    archive = subprocess.run(
        ['git', 'archive', args.base, '--', *map(str, roots)], cwd=repo, check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out / 'base-src', filter='data')

    for side, sources in (('base', [out / 'base-src' / root for root in roots]), ('head', [])):
        pythonpath = os.pathsep.join([*map(str, sources), *filter(None, [os.environ.get('PYTHONPATH')])])
        env = os.environ | {CAPTURE_DIR_ENV: str(out / side), 'PYTHONPATH': pythonpath}
        code = subprocess.run([sys.executable, '-m', 'pytest', *args.pytest_args], cwd=repo, env=env).returncode
        print(f'{side} capture: pytest exited {code}')
    print(f'captures: {out}')
    return compare(out / 'base', out / 'head')


if __name__ == '__main__':
    sys.exit(main())
