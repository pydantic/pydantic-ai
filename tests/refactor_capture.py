"""Record what streamed model responses and UI adapters emit and the requests tests send, per test node, to prove a
refactor keeps them.

The capture is off unless pytest loads this module as a plugin, `-p tests.refactor_capture --refactor-capture=DIR`, so
other runs never import it. To compare a base commit with the working tree, run this from the repository root, over
one or two test nodes that reach each changed line rather than whole test files:

    uv run python -m tests.refactor_capture "$(git merge-base HEAD upstream/main)" <test nodes>

It extracts the base commit's copy of every workspace package with `git archive`, into a temporary directory it reaches
through no symlink, runs pytest over the nodes once with those sources ahead of the installed ones on `PYTHONPATH` and
once without, each into its own new capture directory, then compares the two. The tests, this plugin and the
environment come from the working tree both times, so the library code is the only difference.

Each stream is one JSON line of kind `stream`: the events its `__aiter__` hands the consumer, `PartEndEvent`s and
`FinalResultEvent`s included, the error iteration raised if any, and the response `get()` builds once the test ends.
The capture wraps `__aiter__` on `StreamedResponse` and on every subclass that overrides it, whether the subclass is
defined before the capture starts or after; one that overrides `__init_subclass__` without calling `super()`, or
assigns `__aiter__` once it's defined, escapes it. A stream whose `__aiter__` iterates other streams, like the
continuation composite in `models/_continuation.py`, is recorded once itself and once for each stream it iterates.
When a capture starts, it runs an agent over a streamed canary response and stops the run if that stream wasn't
recorded, so a consumer that stops iterating through `__aiter__` fails loudly instead of recording nothing.

Each request is one JSON line of kind `request`. A request an `httpx` or `httpx2` `AsyncClient` sends is recorded from
`send`, once per attempt except for Prefect's client, below, and one a botocore client sends, like Bedrock's, from
`Endpoint.make_request`, before it's signed and after a model's extra headers are added, once per call: botocore
retries a call below `make_request`, so its retries aren't recorded. Either is recorded with its method and URL as the
target, its headers and its body, parsed when it's JSON. Two kinds of header are left out: credentials
(`authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `x-goog-api-key`, `x-amz-security-token`, `cookie`),
so a capture never writes a key to disk, and trace context (`traceparent`, `tracestate`, `baggage`), whose ids are
random per run. Prefect's client, which marks each request it sends with its API version, retries a request by sending
it again when its test server is busy, so each request it sends is recorded once. Its checks of the server's version and
fetches of a CSRF token carry nothing from the code under test and are sent wherever the load puts them, so they aren't
recorded. Neither are the log batches its background worker sends during whichever test is running: no node can claim
them, so the task run names and failure messages they hold go uncompared. A test that hands a model one of the SDK
client mocks `SDK_MOCKS` lists sends no HTTP request, so each call the model makes on a listed method is recorded
instead, with the method as the target and the arguments as passed, SDK sentinels for an omitted argument included. When
a capture starts, it runs an agent over a Groq model on `httpx`, an Anthropic model on `httpx2` and a Bedrock model on
botocore, and stops the run if any of those requests wasn't recorded, so an SDK that moves to an HTTP client the capture
doesn't wrap fails loudly, as Anthropic's did moving from `httpx` to `httpx2`. Once collection has imported the tests, a
listed mock method that's gone stops the run too. Client stand-ins a test hands a model, like the stub clients Bedrock's
tests define, and client methods a test patches by hand, like the `generate_content` Google's tests patch on a model's
client, escape it.

A test that drives `FunctionModel` or `TestModel` sends no request either, since the model answers in process, so each
call on a method `IN_PROCESS_MODELS` lists is recorded as a request too, with the method as the target and the
messages, model settings and model request parameters the model is handed. A model that wraps another, like
`FallbackModel`, is recorded through the call it makes on the model it wraps; a subclass that overrides a listed method
without calling `super()` escapes it. When a capture starts, it runs an agent over each of those models, streamed and
not, and stops the run if a call on a listed method wasn't recorded, so an agent that stops calling a model through
them fails loudly.

What the UI adapters produce is recorded as lines of kind `ui`, each with the target that produced it. A
`UIEventStream.transform_stream` call is recorded under the class of the event stream it's called on, Vercel AI's,
AG-UI's or a test's own, with the events it hands the consumer and the error iteration raised if any. The capture wraps
`transform_stream` on `UIEventStream` and on every subclass that overrides it, whether the subclass is defined before
the capture starts or after, and closes the stream it wraps when the consumer closes the recording, so the stream's
cleanup runs when it would without the capture. An override that calls `super()` is recorded once itself and once for
the call it makes. A call on a method `UI_MESSAGE_METHODS` lists, wrapped the same way on `UIAdapter` and its
subclasses, is recorded with what it returns or the error it raises, and with the fields equality skips on each
dataclass in what it returns, like `FileUrl._media_type` and `_identifier`: serializing a file shows the media type and
identifier it resolves to, not whether they were set. Each request an app `create_api_app` built answers, like the one
`create_web_app` mounts under `/api` and a test drives with Starlette's `TestClient`, is recorded with its method and
path as the target and the status, headers and body of the response, the body parsed when it's JSON.

Each test starts with `random` seeded from its node id, so a test that draws from it draws the same values on both
sides, and both sides run with one `PYTHONHASHSEED`, so a set, like the tool names in model request parameters, is
recorded in the same order. Lines, and the id of each test node run, are written per `pytest-xdist` worker when each
test ends. Generated tool call ids, UUIDs (version 4 ones without dashes too), the trace and span ids of a W3C trace
context, like the `traceparent` an MCP client or Prefect puts in a request body, object addresses (the decimal one a
`unittest.mock` repr gives too), timestamps within a day of the capture (ones in the basic format of a SigV4
`x-amz-date` header, an AG-UI event's milliseconds since the epoch, and ones in JSON a value holds as text, like an
AG-UI message's `encryptedValue` or a web API response's event stream, too), history digests (which hash those
timestamps), attempt durations, local server ports, multipart boundaries, pytest's numbered temporary directories, the
random suffix of the key a capability without an `id` gets, `PromptInjectionDefender`'s measured `latency_ms`,
`Planning`'s step ids where its plan and its reply to an added step render them, the code hash a Prefect `task_key` ends
in, which hashes the file the task function was compiled from, and the `custom_id` of each batch request
`InMemoryBatchModel` encodes, which hashes a random token, are normalized, so two runs of the same code compare equal. A
step id a model echoes back isn't, in a tool call's arguments or where `Planning` renders it from them, as a subtask's
parent or a dependency. Cassette playback hands a response's headers back in a different order each run, so the
`HTTPHeaders` botocore copies into an error response's `ResponseMetadata` are sorted where an error message renders
them.

The comparison lists the test nodes whose records differ or that only one capture has, and exits non-zero if there
are any. It refuses two captures that imported `pydantic_ai` from the same file, since the base run then didn't load
the base commit, and a capture that recorded nothing, since it proves nothing. It also counts the nodes whose streams
reach each module that defines a `StreamedResponse` subclass, the nodes that send to each request target and the nodes
with a UI record from each UI target, and names the selected nodes that recorded no request, so the capture's reach
over the changed code shows.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
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
import weakref
from collections import Counter, defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Generator, Iterable, Mapping
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import httpx2
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse, HTTPHeaders
from botocore.endpoint import Endpoint
from pydantic_core import to_json
from starlette.applications import Starlette
from starlette.types import Message, Receive, Scope, Send
from typing_extensions import TypedDict, TypeVar
from urllib3 import HTTPResponse

import pydantic_ai
import pydantic_ai.ui._web.api as web_api
import pydantic_ai.ui._web.app as web_app
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponseStreamEvent, TextPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.ui import UIAdapter, UIEventStream

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
        'x-amz-security-token',
        'cookie',
        'traceparent',
        'tracestate',
        'baggage',
    }
)
GENERATED_VALUES = [
    # A W3C trace context, like the `traceparent` an MCP client or Prefect puts in a request body. It comes before the
    # UUID patterns, which a trace ID can match.
    (
        re.compile(r'(?<![0-9a-f])([0-9a-f]{2})-[0-9a-f]{32}-[0-9a-f]{16}-([0-9a-f]{2})(?![0-9a-f])'),
        r'\1-TRACE_ID-SPAN_ID-\2',
    ),
    (re.compile(r' at 0x[0-9a-f]+'), ' at 0x?'),
    # A `unittest.mock` repr gives the mock's `id()` in decimal.
    (re.compile(r"(<\w*Mock\b[^<>]* id=')\d+'"), r"\1ID'"),
    (re.compile(r'pyd_ai_[0-9a-f]{32}'), 'pyd_ai_ID'),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'UUID'),
    (re.compile(r'"history_digest":"[0-9a-f]{64}"'), '"history_digest":"DIGEST"'),
    (re.compile(r'"duration":"P[0-9DTHMS.]+"'), '"duration":"DURATION"'),
    (re.compile(r'(127\.0\.0\.1|localhost):\d+'), r'\1:PORT'),
    (re.compile(r'(--|boundary=)[0-9a-f]{32}'), r'\1BOUNDARY'),
    # Each pytest session numbers its temporary directory, and puts each `pytest-xdist` worker's in its own.
    (re.compile(r'(pytest-of-[^/"]+/pytest-)\d+(?:/popen-gw\d+)?'), r'\1N'),
    # `uuid4().hex` ids, like `AskUserQuestion.id`; the version and variant digits spare most other hex digests.
    (re.compile(r'(?<![0-9a-f])[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}(?![0-9a-f])'), 'UUID'),
    # The key the framework gives a capability without an `id` ends in a random suffix.
    (re.compile(r'<(\w+):[0-9a-f]{6}>'), r'<\1:ID>'),
    # `PromptInjectionDefender` times each screening it runs.
    (re.compile(r'"latency_ms":[0-9.eE+-]+'), '"latency_ms":"LATENCY"'),
    # `Planning`'s `uuid4().hex[:8]` step ids, in a plan line after the status marker and in the reply that adds a step.
    (re.compile(r'(\[[ ~x!-]\] \[)[0-9a-f]{8}(?=\])'), r'\1STEP_ID'),
    (re.compile(r'(with id: )[0-9a-f]{8}(?![0-9a-f])'), r'\1STEP_ID'),
    # Prefect ends a `task_key` in a hash of the task function's code object, which holds the file it was compiled from.
    (re.compile(r'("task_key":"[^"]*-)[0-9a-f]{8}"'), r'\1CODE_HASH"'),
    # `InMemoryBatchModel`, which `TestModel` subclasses, encodes each request as a random token, and a batch request's
    # `custom_id` hashes the encoded request. Its batches are the ones whose ID is a UUID; one of their `custom_id`s can
    # itself have matched the UUID pattern. A suffix that tells apart requests with the same hash stays.
    (
        re.compile(
            r'(?<=batch:batch_UUID:)(?:[0-9a-f]{32}|UUID)'
            r'|(?<="id":"batch_UUID","custom_id":")(?:[0-9a-f]{32}|UUID)'
            r'|(?<="custom_id":")(?:[0-9a-f]{32}|UUID)(?=(?:-\d+)?","batch":\{"id":"batch_UUID")'
        ),
        'CUSTOM_ID',
    ),
]
# Prefect's client marks each request it sends with its API version. These paths are its log batches, version checks
# and CSRF token fetches; the module docstring says why a capture leaves them and the client's retries out.
PREFECT_API_VERSION_HEADER = 'x-prefect-api-version'
PREFECT_OWN_PATHS = re.compile(r'/api/(?:logs/|admin/version|csrf-token)$')
# The second form is ISO 8601's basic format, which SigV4 signs a request's `x-amz-date` header with. A timestamp quoted
# `\"` is in JSON a value holds as text, like an AG-UI message's `encryptedValue` or a web API response's event stream.
ISO_DATETIME = re.compile(r'(\\?")(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?|\d{8}T\d{6}Z)\1')
# AG-UI stamps each event with the milliseconds since the epoch.
EPOCH_MILLISECONDS = re.compile(r'"timestamp":(\d{13})(?![\d.])')
# Cassette playback hands a response's headers back in a different order each run, and botocore copies them into an
# error response's `ResponseMetadata`, which an error message renders with `repr`.
BOTOCORE_ERROR_HEADERS = re.compile(r"'HTTPHeaders': \{([^{}]*)\}")
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
# Models that answer in process and send no request, with the methods an agent calls them through.
IN_PROCESS_MODELS: list[tuple[type[Model], tuple[str, ...]]] = [
    (FunctionModel, ('request', 'request_stream')),
    (TestModel, ('request', 'request_stream')),
]
# The `UIAdapter` class methods that convert messages, whose results a capture records.
UI_MESSAGE_METHODS = ('load_messages', 'dump_messages')

WrappedT = TypeVar('WrappedT')


class BotocoreRequest(TypedDict):
    """What a capture reads from the request dict botocore's `Endpoint.make_request` is handed."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: object


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
        # The requests Prefect's client has sent, so a retry, which sends one again, isn't recorded twice.
        self._prefect_requests: weakref.WeakSet[httpx.Request | httpx2.Request] = weakref.WeakSet()
        # The apps `create_api_app` has built, whose responses are recorded.
        self._api_apps: weakref.WeakSet[Starlette] = weakref.WeakSet()
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
        self.check_model_hook()

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

        Also wrap `send` on the `httpx` and `httpx2` `AsyncClient`s, botocore's `Endpoint.make_request`, each method
        `IN_PROCESS_MODELS` lists, `transform_stream` and the methods `UI_MESSAGE_METHODS` lists where `UIEventStream`,
        `UIAdapter` or a subclass defines them, and the app `create_api_app` builds.
        """
        self._wrap_subclasses(StreamedResponse, self._wrap_aiter)
        self._wrap_subclasses(UIEventStream, self._wrap_transform_stream)
        self._wrap_subclasses(UIAdapter, self._wrap_message_methods)
        self._wrap_api_app()
        self._wrap_send(httpx.AsyncClient)
        self._wrap_send(httpx2.AsyncClient)
        self._wrap_make_request()
        for model_class, method_names in IN_PROCESS_MODELS:
            for name in method_names:
                self._wrap_model(model_class, name)

    def check_stream_hook(self) -> None:
        """Stop the run unless the stream an agent iterates is recorded."""

        async def run() -> None:
            async with Agent(FunctionModel(stream_function=stream_canary)).run_stream('canary') as result:
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
        """Stop the run unless the request an agent sends through each of `httpx`, `httpx2` and botocore is recorded."""

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
                    ('`BedrockConverseModel` over `botocore`', bedrock_canary_model()),
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

    def check_model_hook(self) -> None:
        """Stop the run unless a canary agent run's call on each method `IN_PROCESS_MODELS` lists is recorded."""

        async def run() -> None:
            text_canary = FunctionModel(
                lambda messages, info: ModelResponse(parts=[TextPart('canary')]), stream_function=stream_canary
            )
            for model in (text_canary, TestModel()):
                await Agent(model).run('canary')
                async with Agent(model).run_stream('canary') as result:
                    await result.get_output()

        asyncio.run(run())
        recorded = {record['target'] for record in self.records if record['kind'] == 'request'}
        self.clear()
        targets = [method_target(model_class, name) for model_class, names in IN_PROCESS_MODELS for name in names]
        if missed := [target for target in targets if target not in recorded]:
            self._monkeypatch.undo()
            raise pytest.UsageError(
                f'The calls a canary agent run made on `{"`, `".join(missed)}` were not recorded, so the capture would '
                'miss what in-process models are handed. Update `tests/refactor_capture.py` to the methods an agent '
                'calls them through now.'
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

    def _wrap_subclasses(self, base: type[WrappedT], wrap: Callable[[type[WrappedT]], None]) -> None:
        """Call `wrap` on `base` and each of its subclasses, now and when one is defined."""
        pending = [base]
        while pending:
            cls = pending.pop()
            pending.extend(cls.__subclasses__())
            wrap(cls)

        # `Generic.__init_subclass__`, which `super()` reaches, sets up a generic subclass's type parameters.
        def init_subclass(cls: type[WrappedT], **kwargs: object) -> None:
            super(base, cls).__init_subclass__(**kwargs)
            wrap(cls)

        self._monkeypatch.setattr(base, '__init_subclass__', classmethod(init_subclass))

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

    def _wrap_transform_stream(self, cls: type) -> None:
        if 'transform_stream' not in vars(cls):
            return
        original: Callable[..., AsyncGenerator[object]] = vars(cls)['transform_stream']

        @functools.wraps(original)
        def recording_transform_stream(event_stream: object, *args: object, **kwargs: object) -> AsyncIterator[object]:
            return self._recorded_ui_events(event_stream, original(event_stream, *args, **kwargs))

        self._monkeypatch.setattr(cls, 'transform_stream', recording_transform_stream)

    def _recorded_ui_events(self, event_stream: object, iterator: AsyncGenerator[object]) -> AsyncIterator[object]:
        events: list[object] = []
        record: dict[str, object] = {
            'kind': 'ui',
            'node': self.node,
            'target': method_target(type(event_stream), 'transform_stream'),
            'events': events,
        }
        self.records.append(record)

        async def recorded() -> AsyncIterator[object]:
            try:
                async for event in iterator:
                    events.append(normalized(event))
                    yield event
            except Exception as e:
                record['error'] = normalized(f'{type(e).__name__}: {e}')
                raise
            finally:
                # `transform_stream` runs its cleanup when it's closed, so it's closed with this rather than collected.
                await iterator.aclose()

        return recorded()

    def _wrap_message_methods(self, cls: type) -> None:
        for name in UI_MESSAGE_METHODS:
            if name in vars(cls):
                self._wrap_message_method(cls, name)

    def _wrap_message_method(self, cls: type, name: str) -> None:
        original: classmethod[object, ..., object] = vars(cls)[name]
        target = method_target(cls, name)

        # `functools.wraps` carries over `__isabstractmethod__`, which tells an abstract subclass apart.
        @functools.wraps(original.__func__)
        def recording_method(adapter_class: type[object], *args: object, **kwargs: object) -> object:
            record: dict[str, object] = {'kind': 'ui', 'node': self.node, 'target': target}
            self.records.append(record)
            try:
                result = original.__get__(None, adapter_class)(*args, **kwargs)
            except Exception as e:
                record['error'] = normalized(f'{type(e).__name__}: {e}')
                raise
            record['result'] = normalized(result)
            record['uncompared_fields'] = uncompared_fields(result)
            return result

        self._monkeypatch.setattr(cls, name, classmethod(recording_method))

    def _wrap_api_app(self) -> None:
        original_create: Callable[..., Starlette] = vars(web_api)['create_api_app']
        original_call: Callable[[Starlette, Scope, Receive, Send], Awaitable[None]] = vars(Starlette)['__call__']

        @functools.wraps(original_create)
        def recording_create_api_app(*args: object, **kwargs: object) -> Starlette:
            app = original_create(*args, **kwargs)
            self._api_apps.add(app)
            return app

        async def recording_call(app: Starlette, scope: Scope, receive: Receive, send: Send) -> None:
            if scope['type'] != 'http' or app not in self._api_apps:
                return await original_call(app, scope, receive, send)
            record: dict[str, object] = {
                'kind': 'ui',
                'node': self.node,
                'target': f'{scope["method"]} {scope["path"]}',
            }
            self.records.append(record)
            body = bytearray()

            async def recording_send(message: Message) -> None:
                if message['type'] == 'http.response.start':
                    record['status'] = message['status']
                    record['headers'] = normalized(
                        {name.decode(): value.decode() for name, value in message['headers']}
                    )
                body.extend(message.get('body', b''))
                await send(message)

            try:
                await original_call(app, scope, receive, recording_send)
            finally:
                record['body'] = normalized(parsed_json(body.decode(errors='backslashreplace')))

        # `create_web_app` calls it through the name its module imported.
        for module in (web_api, web_app):
            self._monkeypatch.setattr(module, 'create_api_app', recording_create_api_app)
        self._monkeypatch.setattr(Starlette, '__call__', recording_call)

    def _wrap_send(self, client_class: type[httpx.AsyncClient] | type[httpx2.AsyncClient]) -> None:
        original: Callable[..., Awaitable[object]] = vars(client_class)['send']

        async def recording_send(
            client: object, request: httpx.Request | httpx2.Request, *args: object, **kwargs: object
        ) -> object:
            if PREFECT_API_VERSION_HEADER in request.headers:
                if request in self._prefect_requests or PREFECT_OWN_PATHS.search(request.url.path):
                    return await original(client, request, *args, **kwargs)
                self._prefect_requests.add(request)
            text = (await request.aread()).decode(errors='backslashreplace')
            self._record_request(f'{request.method} {request.url}', request.headers.items(), text)
            return await original(client, request, *args, **kwargs)

        self._monkeypatch.setattr(client_class, 'send', recording_send)

    def _wrap_make_request(self) -> None:
        original: Callable[..., object] = vars(Endpoint)['make_request']

        def recording_make_request(
            endpoint: Endpoint, operation_model: object, request_dict: BotocoreRequest
        ) -> object:
            body = request_dict['body']
            text = body.decode(errors='backslashreplace') if isinstance(body, bytes) else str(body)
            self._record_request(
                f'{request_dict["method"]} {request_dict["url"]}', request_dict['headers'].items(), text
            )
            return original(endpoint, operation_model, request_dict)

        self._monkeypatch.setattr(Endpoint, 'make_request', recording_make_request)

    def _record_request(self, target: str, headers: Iterable[tuple[str, str]], text: str) -> None:
        self.records.append(
            {
                'kind': 'request',
                'node': self.node,
                'target': normalized(target),
                'headers': normalized(
                    {name: value for name, value in headers if name.lower() not in UNRECORDED_HEADERS}
                ),
                'body': normalized(parsed_json(text)),
            }
        )

    def _wrap_mock(self, mock: type, name: str) -> None:
        original: Callable[..., object] = vars(mock)[name]
        target = method_target(mock, name)

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

    def _wrap_model(self, model_class: type[Model], name: str) -> None:
        original: Callable[..., object] = vars(model_class)[name]
        target = method_target(model_class, name)

        # `request` returns a coroutine and `request_stream` an async context manager; either is handed back as is.
        def recording_method(
            model: Model,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            *args: object,
            **kwargs: object,
        ) -> object:
            self.records.append(
                {
                    'kind': 'request',
                    'node': self.node,
                    'target': target,
                    'messages': normalized(messages),
                    'model_settings': normalized(model_settings),
                    'model_request_parameters': normalized(model_request_parameters),
                }
            )
            return original(model, messages, model_settings, model_request_parameters, *args, **kwargs)

        self._monkeypatch.setattr(model_class, name, recording_method)


async def stream_canary(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'canary'


def bedrock_canary_model() -> BedrockConverseModel:
    """A Bedrock model whose client answers each request with a 400 instead of sending it."""
    provider = BedrockProvider(region_name='us-east-1', aws_access_key_id='canary', aws_secret_access_key='canary')
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=provider)

    def answer_400(request: AWSPreparedRequest, **_: object) -> AWSResponse:
        return AWSResponse(request.url, 400, HTTPHeaders(), HTTPResponse(body=io.BytesIO(b'{}'), preload_content=False))

    # botocore sends the response a `before-send` handler returns instead of the request, though the stubs type every
    # handler as returning `None`.
    model.client.meta.events.register_last('before-send.bedrock-runtime', answer_400)  # pyright: ignore[reportArgumentType]
    return model


def method_target(owner: type, name: str) -> str:
    """The request target of a call on `owner`'s method `name`."""
    return f'{owner.__module__}.{owner.__qualname__}.{name}'


def parsed_json(text: str) -> object:
    """`text` parsed when it's JSON, else `text`."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def uncompared_fields(value: object) -> list[object]:
    """The fields equality skips, like `FileUrl._media_type`, of each dataclass `value` holds, in order.

    Serialization shows what they resolve to, like the media type a URL's extension implies, not what they hold.
    """
    found: list[object] = []
    items: list[object] = []
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        if skipped := {field.name: getattr(value, field.name) for field in fields if not field.compare}:
            found.append({'type': type(value).__qualname__, 'fields': normalized(skipped)})
        items = [getattr(value, field.name) for field in fields]
    elif isinstance(value, Mapping):
        items = [*value.values()]
    elif isinstance(value, list | tuple):
        items = [*value]
    for item in items:
        found.extend(uncompared_fields(item))
    return found


def normalized(value: object) -> object:
    """`value` as JSON-compatible data, with the values that change from run to run replaced by placeholders."""
    now = datetime.now(UTC)

    def generated_timestamp(match: re.Match[str]) -> str:
        try:
            timestamp = datetime.fromisoformat(match[2])
        except ValueError:
            return match[0]
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return f'{match[1]}NOW{match[1]}' if abs(timestamp - now) < timedelta(days=1) else match[0]

    def generated_epoch_milliseconds(match: re.Match[str]) -> str:
        timestamp = datetime.fromtimestamp(int(match[1]) / 1000, UTC)
        return '"timestamp":"NOW"' if abs(timestamp - now) < timedelta(days=1) else match[0]

    # Split only between two quoted strings, so a value with a comma in it, like an HTTP date, stays whole.
    def sorted_headers(match: re.Match[str]) -> str:
        return "'HTTPHeaders': {" + ', '.join(sorted(re.split(r"(?<='), (?=')", match[1]))) + '}'

    text = ISO_DATETIME.sub(generated_timestamp, to_json(value, fallback=repr, bytes_mode='base64').decode())
    text = EPOCH_MILLISECONDS.sub(generated_epoch_milliseconds, text)
    text = BOTOCORE_ERROR_HEADERS.sub(sorted_headers, text)
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
            f'Nothing recorded in {", ".join(map(str, empty))}; the selected tests may not stream, send requests or '
            'run a UI adapter.',
            file=sys.stderr,
        )
        return 2
    head_nodes = {node: [json.loads(line) for line in lines] for node, lines in head.items()}
    head_records = [record for records in head_nodes.values() for record in records]
    print(f'nodes: base {len(base)}, head {len(head)}')
    print(f'records: base {sum(map(len, base.values()))}, head {len(head_records)}')
    print(f'stream events: head {sum(len(record["events"]) for record in head_records if record["kind"] == "stream")}')
    print(f'requests: head {sum(record["kind"] == "request" for record in head_records)}')
    print(f'ui records: head {sum(record["kind"] == "ui" for record in head_records)}')

    modules: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    ui_targets: Counter[str] = Counter()
    for records in head_nodes.values():
        modules.update({module for record in records for module in record.get('modules', [])})
        targets.update({record['target'] for record in records if record['kind'] == 'request'})
        ui_targets.update({record['target'] for record in records if record['kind'] == 'ui'})
    for label, counts in (('stream module', modules), ('request target', targets), ('ui target', ui_targets)):
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


def main(argv: list[str] | None = None) -> int:
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
    # DBOS resolves a workflow function's file with `os.path.realpath`, which reads each symlink on the way with
    # `os.readlink`, and blockbuster raises on that in the event loop. On macOS, the temporary directory is reached
    # through `/var`, a symlink, so only the base side's sources would be.
    out = Path(tempfile.mkdtemp(prefix='refactor-capture-')).resolve()
    archive = subprocess.run(
        ['git', 'archive', args.base, '--', *map(str, roots)], cwd=repo, check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out / 'base-src', filter='data')

    failed = False
    for side, sources in (('base', [out / 'base-src' / root for root in roots]), ('head', [])):
        pythonpath = os.pathsep.join([*map(str, sources), *filter(None, [os.environ.get('PYTHONPATH')])])
        command = ['-m', 'pytest', '-p', 'tests.refactor_capture', f'{CAPTURE_OPTION}={out / side}', *args.pytest_args]
        env = os.environ | {'PYTHONPATH': pythonpath, 'PYTHONHASHSEED': '0'}
        code = subprocess.run([sys.executable, *command], cwd=repo, env=env).returncode
        print(f'{side} capture: pytest exited {code}')
        failed = failed or code != 0
    print(f'captures: {out}')
    # Tests that fail alike on both sides compare equal, but they don't prove the refactor.
    return compare(out / 'base', out / 'head') or int(failed)


if __name__ == '__main__':
    sys.exit(main())
