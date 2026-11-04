"""Record what streamed model responses emit and the requests tests send, per test node, to prove a refactor keeps them.

The capture is off unless pytest loads this module as a plugin, `-p tests.refactor_capture --refactor-capture=DIR`, so
other runs never import it. To compare a base commit with the working tree, run this from the repository root, over
one or two test nodes that reach each changed line rather than whole test files:

    uv run python -m tests.refactor_capture "$(git merge-base HEAD upstream/main)" <test nodes>

It extracts the base commit's copy of every workspace package with `git archive`, runs pytest over the nodes once with
those sources ahead of the installed ones on `PYTHONPATH` and once without, each into its own new capture directory,
then compares the two. The tests, this plugin and the environment come from the working tree both times, so the
library code is the only difference.

Each stream is one JSON line of kind `stream`: the events its `__aiter__` hands the consumer, `PartEndEvent`s and
`FinalResultEvent`s included, the error iteration raised if any, and the response `get()` builds once the test ends.
The capture wraps `__aiter__` on `StreamedResponse` and on every subclass that overrides it, whether the subclass is
defined before the capture starts or after; one that overrides `__init_subclass__` without calling `super()`, or
assigns `__aiter__` once it's defined, escapes it. A stream whose `__aiter__` iterates other streams, like the
continuation composite in `models/_continuation.py`, is recorded once itself and once for each stream it iterates.
When a capture starts, it runs an agent over a streamed canary response and stops the run if that stream wasn't
recorded, so a consumer that stops iterating through `__aiter__` fails loudly instead of recording nothing.

Each request is one JSON line of kind `request`. A request an `httpx` or `httpx2` `AsyncClient` sends is recorded from
`send`, with its method and URL as the target, its headers and its body, parsed when it's JSON. Two kinds of header are
left out: credentials (`authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `x-goog-api-key`, `cookie`), so a
capture never writes a key to disk, and trace context (`traceparent`, `tracestate`, `baggage`), whose ids are random per
run. A test that hands a model one of the SDK client mocks `SDK_MOCKS` lists sends no HTTP request, so each call the
model makes on a listed method is recorded instead, with the method as the target and the arguments as passed, SDK
sentinels for an omitted argument included. When a capture starts, it runs an agent over a Groq model on `httpx` and
over an Anthropic model on `httpx2`, and stops the run if either request wasn't recorded, so an SDK that moves to an
HTTP client the capture doesn't wrap fails loudly, as Anthropic's did moving from `httpx` to `httpx2`. Once collection
has imported the tests, a listed mock method that's gone stops the run too. Other transports, like botocore for Bedrock,
and client methods a test patches by hand, like the `generate_content` Google's tests patch on a model's client, escape
it.

Each test starts with `random` seeded from its node id, so a test that draws from it draws the same values on both
sides. Lines, and the id of each test node run, are written per `pytest-xdist` worker when each test ends. Generated
tool call ids, UUIDs, object addresses, timestamps within a day of the capture, history digests (which hash those
timestamps), attempt durations, local server ports and multipart boundaries are normalized, so two runs of the same
code compare equal.

The comparison lists the test nodes whose records differ or that only one capture has, and exits non-zero if there
are any. It refuses two captures that imported `pydantic_ai` from the same file, since the base run then didn't load
the base commit, and a capture that recorded nothing, since it proves nothing. It also counts the nodes whose streams
reach each module that defines a `StreamedResponse` subclass and the nodes that send to each request target, and names
the selected nodes that recorded no request, so the capture's reach over the changed code shows.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Awaitable, Callable, Generator
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import httpx2
import pytest
from pydantic_core import to_json

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponseStreamEvent
from pydantic_ai.models import Model, StreamedResponse, override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.groq import GroqProvider

CAPTURE_OPTION = '--refactor-capture'
SOURCE_FILE = 'source.json'
NODES_SUFFIX = '.nodes'
# Request headers left out of a capture; the module docstring says why.
UNRECORDED_HEADERS = frozenset(
    {
        'authorization',
        'proxy-authorization',
        'x-api-key',
        'api-key',
        'x-goog-api-key',
        'cookie',
        'traceparent',
        'tracestate',
        'baggage',
    }
)
GENERATED_VALUES = [
    (re.compile(r' at 0x[0-9a-f]+'), ' at 0x?'),
    (re.compile(r'pyd_ai_[0-9a-f]{32}'), 'pyd_ai_ID'),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'UUID'),
    (re.compile(r'"history_digest":"[0-9a-f]{64}"'), '"history_digest":"DIGEST"'),
    (re.compile(r'"duration":"P[0-9DTHMS.]+"'), '"duration":"DURATION"'),
    (re.compile(r'(127\.0\.0\.1|localhost):\d+'), r'\1:PORT'),
    (re.compile(r'(--|boundary=)[0-9a-f]{32}'), r'\1BOUNDARY'),
]
ISO_DATETIME = re.compile(r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)"')
# Test doubles that stand in for an SDK client, by module and class, with the methods a model sends a request through.
SDK_MOCKS: list[tuple[str, str, tuple[str, ...]]] = [
    ('tests.models.test_anthropic', 'MockAnthropic', ('messages_create', 'messages_count_tokens')),
    ('tests.models.mock_openai', 'MockOpenAI', ('chat_completions_create',)),
    (
        'tests.models.mock_openai',
        'MockOpenAIResponses',
        ('responses_create', 'responses_input_tokens_count', 'responses_retrieve', 'responses_cancel'),
    ),
    ('tests.models.mock_xai', 'MockXai', ('chat_create', 'files_upload')),
    ('tests.models.test_groq', 'MockGroq', ('chat_completions_create',)),
    ('tests.models.test_huggingface', 'MockHuggingFace', ('chat_completions_create',)),
    ('tests.models.test_mistral', 'MockMistralAI', ('chat_completions_create',)),
    ('tests.models.test_cohere', 'MockAsyncClientV2', ('chat', 'chat_stream')),
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        CAPTURE_OPTION, metavar='DIR', help='new directory to record the streams and requests of each test into'
    )


def pytest_configure(config: pytest.Config) -> None:
    capture_dir: str | None = config.getoption(CAPTURE_OPTION)
    if capture_dir is None:
        raise pytest.UsageError(f'`-p tests.refactor_capture` needs `{CAPTURE_OPTION}=DIR`.')
    config.pluginmanager.register(RefactorCapture(Path(capture_dir)))


class RefactorCapture:
    """Records each stream a test iterates and each request it sends, and appends them to the capture directory."""

    def __init__(self, capture_dir: Path) -> None:
        self.capture_dir = capture_dir
        self.node: str | None = None
        self.records: list[dict[str, object]] = []
        # Each stream record with its stream, whose response `write` adds.
        self._streams: list[tuple[dict[str, object], StreamedResponse]] = []
        # Per stream, by `id`: the iterator `__aiter__` first returned and the recording that wraps it. `_streams` keeps
        # the stream alive meanwhile, so its `id` isn't reused.
        self._iterators: dict[
            int, tuple[AsyncIterator[ModelResponseStreamEvent], AsyncIterator[ModelResponseStreamEvent]]
        ] = {}
        self._monkeypatch = pytest.MonkeyPatch()

    def pytest_configure(self) -> None:
        # Workers start once the controller has checked, and another worker may already have written by then.
        if 'PYTEST_XDIST_WORKER' not in os.environ:
            if any(self.capture_dir.glob('*.jsonl')) or any(self.capture_dir.glob(f'*{NODES_SUFFIX}')):
                raise pytest.UsageError(f'`{self.capture_dir}` already holds a capture; a second would be appended.')
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            (self.capture_dir / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': pydantic_ai.__file__}))
        self.install()
        self.check_stream_hook()
        self.check_request_hook()

    def pytest_collection_finish(self) -> None:
        """Wrap each method `SDK_MOCKS` lists on the mocks collection imported, and stop the run if one is gone."""
        for module_name, class_name, method_names in SDK_MOCKS:
            module = sys.modules.get(module_name)
            if module is None and importlib.util.find_spec(module_name) is not None:
                continue
            mock = getattr(module, class_name, None)
            for name in method_names:
                if not isinstance(mock, type) or name not in vars(mock):
                    raise pytest.UsageError(
                        f'`SDK_MOCKS` lists `{module_name}.{class_name}.{name}`, which is gone, so the capture would '
                        'miss the requests sent through it. Update `SDK_MOCKS` in `tests/refactor_capture.py`.'
                    )
                self._wrap_mock(mock, name)

    def pytest_unconfigure(self) -> None:
        self._monkeypatch.undo()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item) -> Generator[None, object, object]:
        self.node = item.nodeid
        random.seed(item.nodeid)
        try:
            return (yield)
        finally:
            self.node = None
            self.write(item.nodeid)

    def install(self) -> None:
        """Wrap `__aiter__` on `StreamedResponse` and every subclass that overrides it, now and when one is defined.

        Also wrap `send` on the `httpx` and `httpx2` `AsyncClient`s.
        """
        pending: list[type[StreamedResponse]] = [StreamedResponse]
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            self._wrap_aiter(cls)
        self._monkeypatch.setattr(StreamedResponse, '__init_subclass__', classmethod(self._wrap_aiter))
        self._wrap_send(httpx.AsyncClient)
        self._wrap_send(httpx2.AsyncClient)

    def check_stream_hook(self) -> None:
        """Stop the run unless the stream an agent iterates is recorded."""

        async def stream_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'canary'

        async def run() -> None:
            async with Agent(FunctionModel(stream_function=stream_text)).run_stream('canary') as result:
                await result.get_output()

        asyncio.run(run())
        recorded = any(record['events'] for record, _ in self._streams)
        self.clear()
        if not recorded:
            self._monkeypatch.undo()
            raise pytest.UsageError(
                'The stream a canary agent run iterated was not recorded, so the capture would miss streams. '
                'Update `tests/refactor_capture.py` to the way streams are iterated now.'
            )

    def check_request_hook(self) -> None:
        """Stop the run unless the request an agent sends through each of `httpx` and `httpx2` is recorded."""

        async def run() -> list[str]:
            missed: list[str] = []
            async with (
                httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(400))) as http_client,
                httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(400))) as http2_client,
            ):
                models: list[tuple[str, Model]] = [
                    (
                        '`GroqModel` over `httpx`',
                        GroqModel(
                            'llama-3.3-70b-versatile', provider=GroqProvider(api_key='canary', http_client=http_client)
                        ),
                    ),
                    (
                        '`AnthropicModel` over `httpx2`',
                        AnthropicModel(
                            'claude-sonnet-4-5', provider=AnthropicProvider(api_key='canary', http_client=http2_client)
                        ),
                    ),
                ]
                for label, model in models:
                    recorded = len(self.records)
                    with override_allow_model_requests(True), suppress(ModelHTTPError):
                        await Agent(model).run('canary')
                    if len(self.records) == recorded:
                        missed.append(label)
            return missed

        missed = asyncio.run(run())
        self.clear()
        if missed:
            self._monkeypatch.undo()
            raise pytest.UsageError(
                f'The request a canary agent run sent with {" and ".join(missed)} was not recorded, so the capture '
                'would miss requests. Update `tests/refactor_capture.py` to the HTTP client the SDK sends with now.'
            )

    def write(self, node: str) -> None:
        """Append `node` and the records so far, with the response each stream builds now, to this worker's files."""
        worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
        with open(self.capture_dir / f'{worker}{NODES_SUFFIX}', 'a', encoding='utf-8') as f:
            f.write(f'{node}\n')
        if self.records:
            for record, stream in self._streams:
                record['response'] = normalized(stream.get())
            with open(self.capture_dir / f'{worker}.jsonl', 'a', encoding='utf-8') as f:
                f.writelines(json.dumps(record) + '\n' for record in self.records)
            self.clear()

    def clear(self) -> None:
        """Drop the records so far."""
        self.records.clear()
        self._streams.clear()
        self._iterators.clear()

    def _wrap_aiter(self, cls: type[StreamedResponse]) -> None:
        if '__aiter__' not in vars(cls):
            return
        original: Callable[[StreamedResponse], AsyncIterator[ModelResponseStreamEvent]] = vars(cls)['__aiter__']

        def recording_aiter(stream: StreamedResponse) -> AsyncIterator[ModelResponseStreamEvent]:
            iterator = original(stream)
            if (recording := self._iterators.get(id(stream))) is None:
                recording = self._iterators[id(stream)] = (iterator, self._recorded(stream, iterator))
            # Anything else already reads through the recording: an override's `super().__aiter__()` result, or the
            # peeking stream `request_stream_until_first_event` stores over it.
            return recording[1] if iterator is recording[0] else iterator

        self._monkeypatch.setattr(cls, '__aiter__', recording_aiter)

    def _recorded(
        self, stream: StreamedResponse, iterator: AsyncIterator[ModelResponseStreamEvent]
    ) -> AsyncIterator[ModelResponseStreamEvent]:
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
        self.records.append(record)
        self._streams.append((record, stream))

        # Closing this doesn't close `iterator`: every `__aiter__` in the library returns a guard that does no cleanup
        # on close, so only its garbage collection moves.
        async def recorded() -> AsyncIterator[ModelResponseStreamEvent]:
            try:
                async for event in iterator:
                    events.append(normalized(event))
                    yield event
            except Exception as e:
                record['error'] = normalized(f'{type(e).__name__}: {e}')
                raise

        return recorded()

    def _wrap_send(self, client_class: type[httpx.AsyncClient] | type[httpx2.AsyncClient]) -> None:
        original: Callable[..., Awaitable[object]] = vars(client_class)['send']

        async def recording_send(
            client: object, request: httpx.Request | httpx2.Request, *args: object, **kwargs: object
        ) -> object:
            text = (await request.aread()).decode(errors='backslashreplace')
            try:
                body: object = json.loads(text)
            except ValueError:
                body = text
            self.records.append(
                {
                    'kind': 'request',
                    'node': self.node,
                    'target': normalized(f'{request.method} {request.url}'),
                    'headers': normalized(
                        {name: value for name, value in request.headers.items() if name not in UNRECORDED_HEADERS}
                    ),
                    'body': normalized(body),
                }
            )
            return await original(client, request, *args, **kwargs)

        self._monkeypatch.setattr(client_class, 'send', recording_send)

    def _wrap_mock(self, mock: type, name: str) -> None:
        original: Callable[..., object] = vars(mock)[name]
        target = f'{mock.__module__}.{mock.__qualname__}.{name}'

        # Whatever the method returns, a coroutine, an async generator or a plain value, is handed back as is.
        def recording_method(client: object, *args: object, **kwargs: object) -> object:
            self.records.append(
                {
                    'kind': 'request',
                    'node': self.node,
                    'target': target,
                    'args': normalized(args),
                    'kwargs': normalized(kwargs),
                }
            )
            return original(client, *args, **kwargs)

        self._monkeypatch.setattr(mock, name, recording_method)


def normalized(value: object) -> object:
    """`value` as JSON-compatible data, with the values that change from run to run replaced by placeholders."""
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
    if missing := [d for d in (base_dir, head_dir) if not (d / SOURCE_FILE).exists()]:
        print(
            f'No capture in {", ".join(map(str, missing))}; its pytest run stopped before it started.', file=sys.stderr
        )
        return 2
    base_source, head_source = (json.loads((d / SOURCE_FILE).read_text()) for d in (base_dir, head_dir))
    if base_source == head_source:
        print(f'Both captures imported `pydantic_ai` from the same file: {base_source}', file=sys.stderr)
        return 2
    base, head = _load(base_dir), _load(head_dir)
    if empty := [d for d, nodes in ((base_dir, base), (head_dir, head)) if not nodes]:
        print(
            f'Nothing recorded in {", ".join(map(str, empty))}; the selected tests may not stream or send requests.',
            file=sys.stderr,
        )
        return 2
    head_nodes = {node: [json.loads(line) for line in lines] for node, lines in head.items()}
    head_records = [record for records in head_nodes.values() for record in records]
    print(f'nodes: base {len(base)}, head {len(head)}')
    print(f'records: base {sum(map(len, base.values()))}, head {len(head_records)}')
    print(f'stream events: head {sum(len(record["events"]) for record in head_records if record["kind"] == "stream")}')
    print(f'requests: head {sum(record["kind"] == "request" for record in head_records)}')

    modules: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    for records in head_nodes.values():
        modules.update({module for record in records for module in record.get('modules', [])})
        targets.update({record['target'] for record in records if record['kind'] == 'request'})
    for label, counts in (('stream module', modules), ('request target', targets)):
        print(f'nodes per {label}, head:')
        for name, count in sorted(counts.items()):
            print(f'  {name}: {count}')
    ran = {node for path in head_dir.glob(f'*{NODES_SUFFIX}') for node in path.read_text().splitlines()}
    requesting = {
        node for node, records in head_nodes.items() if any(record['kind'] == 'request' for record in records)
    }
    print(f'nodes that recorded no request, head: {len(ran - requesting)}')
    for node in sorted(ran - requesting):
        print(f'  {node}')

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

    failed = False
    for side, sources in (('base', [out / 'base-src' / root for root in roots]), ('head', [])):
        pythonpath = os.pathsep.join([*map(str, sources), *filter(None, [os.environ.get('PYTHONPATH')])])
        command = ['-m', 'pytest', '-p', 'tests.refactor_capture', f'{CAPTURE_OPTION}={out / side}', *args.pytest_args]
        env = os.environ | {'PYTHONPATH': pythonpath}
        code = subprocess.run([sys.executable, *command], cwd=repo, env=env).returncode
        print(f'{side} capture: pytest exited {code}')
        failed = failed or code != 0
    print(f'captures: {out}')
    # Tests that fail alike on both sides compare equal, but they don't prove the refactor.
    return compare(out / 'base', out / 'head') or int(failed)


if __name__ == '__main__':
    sys.exit(main())
