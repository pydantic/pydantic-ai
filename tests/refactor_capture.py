"""Record what streamed model responses, UI adapters and realtime connections emit, the requests tests send, the tool
hooks their runs call, the event streams their runs hand consumers, what `AgentStream` methods yield and return, the
spans tests end, the exceptions they catch and warnings they raise, and the names durable execution registers, the
workflows it runs and the cache keys it computes, per test node, to prove a refactor keeps them.

The capture is off unless pytest loads this module as a plugin, `-p tests.refactor_capture --refactor-capture=DIR`, so
other runs never import it. To compare a base commit with the working tree, run this from the repository root, over
one or two test nodes that reach each changed line rather than whole test files:

    uv run python -m tests.refactor_capture "$(git merge-base HEAD upstream/main)" <test nodes>

It extracts the base commit's copy of every workspace package with `git archive`, into a temporary directory it reaches
through no symlink, runs pytest over the nodes once with those sources ahead of the installed ones on `PYTHONPATH` and
once without, each into its own new capture directory, then compares the two. The tests, this plugin and the
environment come from the working tree both times, so the library code is the only difference.

Each stream is one JSON line of kind `stream`: the events its `__aiter__` hands the consumer, `PartEndEvent`s and
`FinalResultEvent`s included, the error iteration raised if any, and the response `get()` builds once iteration ends,
or once the test ends if the stream still lives then, as it is when the test ends, so what the run sets on it once
iteration ended, like its usage's cost, shows; a response `get()` can't build, or the capture can't write, is recorded
as the error. A stream holds what the run iterating it holds, like its
dependencies, so the capture holds a stream only while the recording of its iteration lives.
The capture wraps `__aiter__` on `StreamedResponse` and on every subclass that overrides it, whether the subclass is
defined before the capture starts or after; one that overrides `__init_subclass__` without calling `super()`, or
assigns `__aiter__` once it's defined, escapes it. A stream whose `__aiter__` iterates other streams, like the
continuation composite in `models/_continuation.py`, is recorded once itself and once for each stream it iterates.
When a capture starts, it runs an instrumented agent over a streamed canary response and stops the run if that stream,
the run's event stream, its `AgentStream` calls or its spans weren't recorded, so a consumer that stops iterating
through `__aiter__`, or a run that stops reaching the other methods the capture wraps, fails loudly instead of recording
nothing.

Each request is one JSON line of kind `request`. A request an `httpx` or `httpx2` `AsyncClient` sends is recorded from
`send`, once per attempt except for Prefect's client, below, and one a botocore client sends, like Bedrock's, from
`Endpoint.make_request`, before it's signed and after a model's extra headers are added, once per call: botocore
retries a call below `make_request`, so its retries aren't recorded. Either is recorded with its method and URL as the
target, its headers and its body, parsed when it's JSON. Two kinds of header are left out: credentials
(`authorization`, `proxy-authorization`, `x-api-key`, `api-key`, `x-goog-api-key`, `xi-api-key`, `x-amz-security-token`,
`cookie`), so a capture never writes a key to disk, and trace context (`traceparent`, `tracestate`, `baggage`), whose
ids are random per run. Prefect's client, which marks each request it sends with its API version, retries a request by
sending it again when its test server is busy, so each request it sends is recorded once. Its checks of the server's
version and fetches of a CSRF token carry nothing from the code under test and are sent wherever the load puts them, so
they aren't recorded. Neither are the log batches its background worker sends during whichever test is running: no node
can claim them, so the task run names and failure messages they hold go uncompared. A test that hands a model one of the
SDK client mocks `SDK_MOCKS` lists sends no HTTP request, so each call the model makes on a listed method is recorded
instead, with the method as the target and the arguments as passed, SDK sentinels for an omitted argument included. When
a capture starts, it runs an agent over a Groq model on `httpx`, an Anthropic model on `httpx2` and a Bedrock model on
botocore, and stops the run if any of those requests wasn't recorded, so an SDK that moves to an HTTP client the capture
doesn't wrap fails loudly, as Anthropic's did moving from `httpx` to `httpx2`. Once collection has imported the tests, a
listed mock method that's gone stops the run too. Client stand-ins a test hands a model, like the stub clients Bedrock's
tests define, and client methods a test patches by hand, like the `generate_content` Google's tests patch on a model's
client, escape it.

A test that drives `FunctionModel` or `TestModel` sends no request either, since the model answers in process, so each
call on a method `IN_PROCESS_MODELS` lists is recorded as a request too, with the method as the target and the
messages, model settings and model request parameters the model is handed. A tool's metadata isn't sent to the model,
so a metadata value `json` can't write, like a model, whose serializer can be a test's, is recorded as `unrecorded`,
with its class. A model that wraps another, like `FallbackModel`, is recorded through the call it makes on the model it
wraps; a subclass that overrides a listed method without calling `super()` escapes it. When a capture starts, it runs
an agent over each of those models, streamed and not, and stops the run if a call on a listed method wasn't recorded,
so an agent that stops calling a model through them fails loudly.

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

What realtime connections do is recorded as lines of kind `realtime`, in the order it happens, so a send a connection
makes while it handles a frame comes before the frame. Each frame `RealtimeConnection.frames` yields is one line, under
the connection's own class, with its events in order, each named by its class, lifecycle events and `StaleEvent`s
included: iterating the connection drops the lifecycle events and unwraps the stale ones. So is the error iteration
raised, if any. The capture wraps `frames` and `send` on `RealtimeConnection` and on every subclass that overrides them,
whether the subclass is defined before the capture starts or after, and closes the generator `frames` returned when the
consumer closes the recording, so the connection settles what's open when it would without the capture. A `send` call is
recorded with its input, named by its class, and the lifecycle events it returns or the error it raises. A frame
reaches the provider through a test double: a cassette's `ReplayWebSocket`, or a test's own socket or Google
`AsyncSession`. Each call on a method `REALTIME_SEND_METHODS` lists is recorded, under the double's class, with its
arguments, a frame parsed when it's JSON, and the error it raises, if any, where a class in a package
`REALTIME_DOUBLE_PACKAGES` lists defines it, whether before the capture starts or after. That records each frame a
connection sends, its handshake included, and each call on a fake `AsyncSession`; a real one sends its frames on its
socket. A call that an override or a double makes on another while it's recorded, through `super()` or to a socket it
wraps, isn't recorded again; a `frames` override that calls `super()` is recorded once itself and once for the call it
makes.

What a realtime session does above its connection is recorded as lines of kind `session`, in the order it happens.
Each call on a method `SESSION_CORE_METHODS` lists is one line, under the class of the `SessionCore` it's called on,
with its arguments and its result or the error it raises: each input the session applies, the effects `take_effects`
hands back, and what the session reads. A call the core makes on itself isn't recorded again. Each iteration of a
`RealtimeSession` is one line, with the events `__aiter__` yields, each named by its class, and the error iteration
raised, if any. The recording is an async generator, as `__aiter__` is, and closes the one it wraps when the consumer
closes it. Each view `stream_audio` or `stream_transcripts` returns is one line, with the arguments it was opened with
and what it yields, and so is the error each public session call raises, unless another public call on the same
session, in the same task, made it. Once the first `close` call on a session ends, unless it was cancelled, what the
session holds is one line: `all_messages()`, `new_messages()`, `usage`, `played_audio_bytes` or the error reading it
raises, and the counters `SESSION_DROP_COUNTERS` lists. Each `infer_realtime_model` call is one line, with its
arguments and the type, `model_name`, `system` and `base_url` of the model it returns, or the error it raises. A call
on a method `LIFECYCLE_METHODS` lists is recorded as a core's is where a test calls it directly; what one a
connection's `frames` or `send` makes returns is in that connection's `realtime` lines. The capture wraps these where
`RealtimeSession`, `SessionCore`, `InferredLifecycle` or a subclass defines them, whether the subclass is defined
before the capture starts or after. Each session, core and lifecycle a test uses is numbered in its lines, per kind, in
the order the test first used it.

An `MCPSamplingModel` sends its requests through an MCP session, so each call on its `request` is recorded as a request
too, with the messages, model settings and model request parameters it's handed and the response it returns or the
error it raises. Where a `unittest.mock` mock stands in for the session's `create_message`, each call the mock gets is
a request after it, with the arguments the mock kept. A real session's calls aren't: what one sends shows where its
peer handles it, like the sampling model an `MCPToolset` is given.

Each call the tool manager makes on a hook `TOOL_HOOKS` lists is one line of kind `hook`, under the class of the
capability it's called on, with the arguments it's handed, an exception among them as its class and message, and the
result it returns or the error it raises. The run context and a wrap hook's handler are left out. The capture wraps the
hooks where `AbstractCapability` or a subclass defines them, whether the subclass is defined before the capture starts
or after. A call on a hook while a recorded call on the same hook runs in the same agent run, like the one a combined
capability makes on each capability it combines, isn't recorded again; one an agent run inside a hook or a tool makes
is. The calls a run step makes run concurrently, and a sync tool on a worker thread, so the hook lines of each step's
tool calls are put in the order the calls started, each call's own in the order they were made: how one call's hooks
interleave with another's isn't compared.

Each event stream an agent run's root capability hands its consumer from `wrap_run_event_stream` is one line of kind
`run_event_stream`, with the events it yields, in order, and the error iteration raised, if any. A run hands one for each
node it streams: the model request node's, which `AgentStream.__aiter__` iterates, and the tool call node's, which a run
streams whether or not its caller does. So the line holds what `run_stream`, `run_stream_events` and an
`event_stream_handler` get, with what each capability does to the stream, like what `ProcessEventStream` passes on, and
the `EnqueuedMessagesEvent`s that delivering a message adds. The stream of each capability the root combines is inside
the root's, so it isn't recorded again. A call with a run context that has no root capability, as a test makes to call
a capability directly, is recorded too, and so is each call a capability called that way makes on the capabilities it
combines. The capture wraps `wrap_run_event_stream` where `AbstractCapability` or a subclass defines it, whether the
subclass is defined before the capture starts or after, and closes the stream it wraps when the consumer closes the
recording.

Each call on a method `AGENT_STREAM_ITERATORS` or `AGENT_STREAM_CALLS` lists, on `AgentStream`, is one line of kind
`agent_stream`, with its arguments, defaults filled in, and what it yields, in order, or returns, or the error it raises:
the text `stream_text` yields, the partial outputs `stream_output` yields, the output `get_output` returns and what each
`validate_response_output` call returns or raises, the error of a partial output `stream_output` swallows included. The
response a call validates is the stream's own, which the stream's `stream` line holds, so it's left out. Lines come in
the order the calls started, so one a call makes, like `get_output`'s on `validate_response_output`, comes after it.
`StreamedRunResult` and `StreamedRunResultSync` call these methods, so their calls are recorded through them. A
`debounce_by`, which `stream_text` and `stream_output` default to, groups what arrives within it into one yield, so
what a test that leaves it set records depends on how fast the stream arrives.

Each span the OpenTelemetry SDK ends is one line of kind `span`, with its name, number, parent, kind, the name of its
instrumentation scope, its attributes, its status, and its events, each with its name and attributes. An exception
event's stack trace is left out, since a refactor moves the lines it names. A test's spans are numbered in the order
they started, and a parent is named by its number; a parent that didn't start in the test the span ends in, like a
remote one, is `external`. Concurrent tool calls end their spans in any order, so a test's span lines come after its
others, ahead of the durable execution ones below, in the order the spans started. A span ended again is recorded once.
Span IDs, timestamps and durations aren't recorded; the ones its attributes hold are normalized, below. The capture wraps
`start` and `end` on the SDK's `Span`, so a span that isn't recording, like one a sampler drops or a tracer other than
the SDK's starts, records nothing, and only its absence shows.

Each exception a `pytest.raises` or `pytest.RaisesGroup` block catches is one line of kind `exception`, with its class,
message and notes, whether it suppresses its context, and, described alike, the exceptions it groups, its `__cause__`
and its `__context__`. Each warning the filters let through while a test runs is one line of kind `warning`, whether
it's shown or recorded, by `pytest.warns` or `warnings.catch_warnings`, with its category, its message and the frame it
reports: that frame's module and function, and its line unless a workspace package holds it, since a refactor moves the
lines of the code it edits. One the filters raise as an error is recorded only as an exception a block catches, and one
they ignore isn't. Neither is a `ResourceWarning`, which a finalizer raises whenever garbage collection reaches it, so
in whichever test the allocations made put it. A recorded hook or `MCPSamplingModel.request` call is one more frame on
the stack, so a warning raised inside one with a `stacklevel` that reaches past it would report the capture's frame.
Wherever the capture records an exception's or a warning's message, one whose `__str__` raises is recorded as
`unrecorded`, with the class of the error it raised, which the code under test never sees.

What durable execution does is recorded once a test imports Temporal, DBOS or Prefect: the capture wraps what it needs
from a module once the module first runs, and imports none of them itself, since tests configure Prefect through
environment variables before they import it. Each name the code in `pydantic_ai.durable_exec` registers with
`temporalio.activity.defn`, `temporalio.workflow.defn`, `DBOS.step`, `DBOS.workflow`, `prefect.task` or `prefect.flow`
is one line of kind `durable_name`, with the decorator as the target, the name the engine registers it under, and the
other options the call passes. One registered while a module is collected, like by an agent the module defines, is
recorded under the module's node. A name a test registers itself isn't recorded, and neither is one registered again
in Temporal's workflow sandbox, which imports a workflow's module, and the agents it defines, again for each run.

Each Temporal workflow run is one line of kind `temporal`, with its workflow type as the target, its workflow ID, and
each command its workflow activations issue, in order, as `MessageToDict` gives it, with each payload's metadata, and
its data when it's JSON, decoded. So is the payload of each item a workflow stream publishes, which the stream sends
serialized and base64 encoded. A failure's stack trace is left out, since a refactor moves the lines it names. A
serialized run context sends each set it holds, like `active_capability_ids`, as a list in the set's order, which
differs from run to run where its members end in the random suffix of a capability's key, so those lists are sorted.
The capture reads a run's type and ID from the activation that starts it, which the bridge worker's
`poll_workflow_activation` returns, and its commands from each completion `complete_workflow_activation` is handed. A
completion that issues nothing, like the one for the activation that evicts a run from the worker's cache, adds
nothing. Each DBOS workflow is one line of kind `dbos`, with its name as the target, its ID, class name, config name and
parent workflow's ID, and each operation it checks for a recorded result, in order: its function ID and name, and
whether a result was recorded, unless the check raised, like on a step that isn't the one recorded, whose error reaches
the test. A forked workflow's line also has the ID of the workflow it was forked from, whose name, class name and config
name it takes, when the test started that one. The capture wraps the system database's `_insert_workflow_status`,
`_check_operation_execution_txn` and `fork_workflow`.

Each `compute_key` call on a Prefect cache policy is one line of kind `cache_key`, with the policy's class as the
target, the inputs it's handed, and the key it returns or the error it raises. The capture wraps `compute_key` where
`CachePolicy` or a subclass defines it, whether the subclass is defined before the capture starts or after. A call made
in another, like the one a compound policy makes on each policy it combines, comes after it, and leaves out its inputs
when they're the ones the other was handed. A key that holds a value each run draws anew is recorded as `RUN_SCOPED`,
and so is each key computed from it: the `RunId` policy's, which is the flow run's ID, and the key of a call whose
recorded inputs hold a value normalized below, like a generated ID. A policy handed the inputs its caller was, like
`TaskSource`, which hashes the task's code, keeps its key. Inputs or options that can't be written as JSON, like ones that
hold a reference cycle, as a run context does, are recorded as the error writing them raised, which the code under
test never sees, and so is a Temporal command whose payload can't be decoded.

Durable names and cache keys are made on other threads than the test's, like the one Prefect runs a task on, and
workflows run concurrently, so a test's `durable_name`, `cache_key`, `temporal` and `dbos` lines come after its others,
each kind sorted, a policy's calls kept together.

Each test starts with `random` seeded from its node id, so a test that draws from it draws the same values on both
sides, and both sides run with one `PYTHONHASHSEED`, so a set, like the tool names in model request parameters, is
recorded in the same order. Lines, and the id of each test node run, are written per `pytest-xdist` worker when each
test ends. Generated tool call ids, UUIDs (version 4 ones without dashes too), the trace and span ids of a W3C trace
context, like the `traceparent` an MCP client or Prefect puts in a request body, object addresses (the decimal one a
`unittest.mock` repr gives, and the hex one anyio names a cancel scope by, too), timestamps within a day of the capture
(ones in the basic format of a SigV4 `x-amz-date` header, an AG-UI event's milliseconds since the epoch, a naive
datetime's `str()`, like the local time the `clai` agent's system prompt states, and ones in JSON a value holds as text,
like an AG-UI message's `encryptedValue` or a web API response's event stream, too), history digests (which hash those
timestamps), attempt durations, local server ports, multipart boundaries, pytest's numbered temporary directories, the
random suffix of the key a capability without an `id` gets, `PromptInjectionDefender`'s measured `latency_ms`,
`Planning`'s step ids where its plan and its reply to an added step render them, the code hash a Prefect `task_key` ends
in, which hashes the file the task function was compiled from, the digest of a run's inputs that keys the task
`PrefectDurability` draws the run's default ID from, which hashes the timestamps of the history the run is handed,
the process ID a Temporal worker's identity starts with, the trace and span IDs an online evaluation's dispatch
references, durations a clock measured, like the evaluated call's in that dispatch and a delegation's that
`DelegationFinishedEvent` reports, the time a streamed request's chat span records its first chunk took, the id a
sub-agent's question to its parent gets, Temporal timers computed from a clock reading, and the `custom_id` of each batch
request `InMemoryBatchModel` encodes, which hashes a random token, are normalized, so two runs of the same code compare
equal. A
step id a model echoes back isn't, in a tool call's arguments or where `Planning` renders it from them, as a subtask's
parent or a dependency. Cassette playback hands a response's headers back in a different order each run, so the
`HTTPHeaders` botocore copies into an error response's `ResponseMetadata` are sorted where an error message renders
them. So are the native tools the error for an unsupported one lists as supported, a set of classes, which hash by
identity.

The comparison lists the test nodes whose records differ or that only one capture has, and exits non-zero if there are
any. It refuses two captures that imported `pydantic_ai` from the same file, since the base run then didn't load the
base commit, and a capture that recorded nothing, since it proves nothing. It also counts the nodes whose streams reach
each module that defines a `StreamedResponse` subclass, the nodes that send to each request target and the nodes with a
UI, realtime, session, hook, Temporal, DBOS, durable name or cache key record from each of their targets, the nodes
with each run event kind, `AgentStream` method and span name, and names the selected nodes that recorded no request,
realtime or session record, so the capture's reach over the changed code
shows. It first names the nodes that failed on each side: a test that fails records only what it did before it failed,
so matching records don't prove it. A test that waits on a fixed timeout, like the five seconds an MCP server started from `--mcp-config` has to
initialize, can fail on both sides on a loaded machine; run it again once the load is down.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import builtins
import copy
import dataclasses
import functools
import importlib.abc
import importlib.util
import inspect
import io
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import warnings
import weakref
from collections import Counter, defaultdict
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Sequence,
    Set,
)
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from datetime import UTC, datetime, timedelta
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, TracebackType
from unittest.mock import NonCallableMock

import httpx
import httpx2
import pytest
from botocore.awsrequest import AWSPreparedRequest, AWSResponse, HTTPHeaders
from botocore.endpoint import Endpoint
from opentelemetry.sdk.trace import ReadableSpan, Span as SdkSpan, TracerProvider
from pydantic_core import to_json
from starlette.applications import Starlette
from starlette.types import Message, Receive, Scope, Send
from typing_extensions import TypedDict, TypeVar
from urllib3 import HTTPResponse

import pydantic_ai
import pydantic_ai.realtime
import pydantic_ai.realtime.model as realtime_model
import pydantic_ai.ui._web.api as web_api
import pydantic_ai.ui._web.app as web_app
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Instrumentation
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.groq import GroqModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.mcp_sampling import MCPSamplingModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.bedrock import BedrockProvider
from pydantic_ai.providers.groq import GroqProvider
from pydantic_ai.realtime import RealtimeModel, RealtimeSession
from pydantic_ai.realtime._core import SessionCore
from pydantic_ai.realtime.codec import RealtimeConnection
from pydantic_ai.realtime.lifecycle import InferredLifecycle, RealtimeFrame, StaleEvent
from pydantic_ai.result import AgentStream
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.ui import UIAdapter, UIEventStream

REPO = Path(__file__).resolve().parent.parent
CAPTURE_OPTION = '--refactor-capture'
SOURCE_FILE = 'source.json'
NODES_SUFFIX = '.nodes'
FAILED_SUFFIX = '.failed'
# Request headers left out of a capture; the module docstring says why.
UNRECORDED_HEADERS = frozenset(
    {
        'authorization',
        'proxy-authorization',
        'x-api-key',
        'api-key',
        'x-goog-api-key',
        'xi-api-key',
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
    # The span an online evaluation's dispatch references.
    (re.compile(r'("trace_id":")[0-9a-f]{32}"'), r'\1TRACE_ID"'),
    (re.compile(r'("span_id":")[0-9a-f]{16}"'), r'\1SPAN_ID"'),
    # A duration a clock measured, like the evaluated call's in an online evaluation's dispatch, or the delegation's a
    # `DelegationFinishedEvent` reports. One a test sets has fewer decimal places.
    (re.compile(r'("duration(?:_ms)?":)\d+\.\d{10,}(?:e-\d+)?(?=[,}])'), r'\1"DURATION"'),
    # A Temporal timer given to the nanosecond, which a `timedelta` can't be, was computed from a clock reading, like
    # the time left until a deadline set in an earlier activation: a workflow's time moves on between activations.
    (re.compile(r'"start_to_fire_timeout":"\d+\.\d{9}s"'), '"start_to_fire_timeout":"DURATION"'),
    (re.compile(r' at 0x[0-9a-f]+'), ' at 0x?'),
    # anyio names a cancel scope by its address, in hex without `0x`.
    (re.compile(r'(Cancelled via cancel scope )[0-9a-f]+'), r'\1ADDRESS'),
    # A `unittest.mock` repr gives the mock's `id()` in decimal.
    (re.compile(r"(<\w*Mock\b[^<>]* id=')\d+'"), r"\1ID'"),
    (re.compile(r'pyd_ai_[0-9a-f]{32}'), 'pyd_ai_ID'),
    (re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'), 'UUID'),
    # Also in JSON a value holds as text, like a Temporal workflow's result.
    (re.compile(r'(\\?)"history_digest\1":\1"[0-9a-f]{64}\1"'), r'\1"history_digest\1":\1"DIGEST\1"'),
    (re.compile(r'"duration":"P[0-9DTHMS.]+"'), '"duration":"DURATION"'),
    (re.compile(r'(127\.0\.0\.1|localhost):\d+'), r'\1:PORT'),
    (re.compile(r'(--|boundary=)[0-9a-f]{32}'), r'\1BOUNDARY'),
    # Each pytest session numbers its temporary directory, and puts each `pytest-xdist` worker's in its own.
    (re.compile(r'(pytest-of-[^/"]+/pytest-)\d+(?:/popen-gw\d+)?'), r'\1N'),
    # The `uuid4().hex[:12]` id a sub-agent's question to its parent gets, which its task question and answer carry.
    (re.compile(r'("question_id":")[0-9a-f]{12}"'), r'\1QUESTION_ID"'),
    # `uuid4().hex` ids, like `AskUserQuestion.id`; the version and variant digits spare most other hex digests.
    (re.compile(r'(?<![0-9a-f])[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}(?![0-9a-f])'), 'UUID'),
    # The key the framework gives a capability without an `id` ends in a random suffix.
    (re.compile(r'<(\w+):[0-9a-f]{6}>'), r'<\1:ID>'),
    # `PromptInjectionDefender` times each screening it runs.
    (re.compile(r'"latency_ms":[0-9.eE+-]+'), '"latency_ms":"LATENCY"'),
    # A streamed model request's chat span records how long its first chunk took to arrive.
    (re.compile(r'("gen_ai\.client\.operation\.time_to_first_chunk":)[0-9.eE+-]+'), r'\1"DURATION"'),
    # `Planning`'s `uuid4().hex[:8]` step ids, in a plan line after the status marker and in the reply that adds a step.
    (re.compile(r'(\[[ ~x!-]\] \[)[0-9a-f]{8}(?=\])'), r'\1STEP_ID'),
    (re.compile(r'(with id: )[0-9a-f]{8}(?![0-9a-f])'), r'\1STEP_ID'),
    # Prefect ends a `task_key` in a hash of the task function's code object, which holds the file it was compiled from.
    (re.compile(r'("task_key":"[^"]*-)[0-9a-f]{8}"'), r'\1CODE_HASH"'),
    # The task `PrefectDurability` draws a run's default ID from is keyed by a digest of the run's inputs, which hold the
    # timestamps of the history the run is handed.
    (re.compile(r'(pydantic_ai:default_run_id:)[0-9a-f]{32}'), r'\1DIGEST'),
    # A Temporal worker's default identity, which an activity failure names, starts with its process ID.
    (re.compile(r'("identity":")\d+@'), r'\1PID@'),
    # `InMemoryBatchModel`, which `TestModel` subclasses, encodes each request as a random token, and a batch request's
    # `custom_id` hashes the encoded request. Its batches are the ones whose ID is a UUID; one of their `custom_id`s can
    # itself have matched the UUID pattern. A suffix that tells apart requests with the same hash stays.
    (
        re.compile(
            r'(?<=batch:batch_UUID:)(?:[0-9a-f]{32}|UUID)'
            r'|(?<="id":"batch_UUID","custom_id":")(?:[0-9a-f]{32}|UUID)'
            r'|(?<="custom_id":")(?:[0-9a-f]{32}|UUID)(?=(?:-\d+)?","batch":\{"id":"batch_UUID")'
            r'|(?<="custom_id":")(?:[0-9a-f]{32}|UUID)(?=","id":"batch_UUID")'
        ),
        'CUSTOM_ID',
    ),
    # Temporal's JSON converter sorts a payload's keys, so there a batch request's `custom_id` comes before its batch's
    # ID, which the alternative above it reads, and after the batch in the ticket a pending response keeps.
    (
        re.compile(
            r'("batch":\{(?:[^{}]|\{[^{}]*\})*"id":"batch_UUID"(?:[^{}]|\{[^{}]*\})*\},"custom_id":")(?:[0-9a-f]{32}|UUID)"'
        ),
        r'\1CUSTOM_ID"',
    ),
]
# Prefect's client marks each request it sends with its API version. These paths are its log batches, version checks
# and CSRF token fetches; the module docstring says why a capture leaves them and the client's retries out.
PREFECT_API_VERSION_HEADER = 'x-prefect-api-version'
PREFECT_OWN_PATHS = re.compile(r'/api/(?:logs/|admin/version|csrf-token)$')
# The second form is ISO 8601's basic format, which SigV4 signs a request's `x-amz-date` header with. A timestamp quoted
# `\"` is in JSON a value holds as text, like an AG-UI message's `encryptedValue` or a web API response's event stream.
ISO_DATETIME = re.compile(r'(\\?")(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?|\d{8}T\d{6}Z)\1')
# A naive datetime's `str()`, like the local time the `clai` agent's system prompt states.
DATETIME_STR = re.compile(r'(?<![\d-])\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{6})?(?![\d.])')
# AG-UI stamps each event with the milliseconds since the epoch.
EPOCH_MILLISECONDS = re.compile(r'"timestamp":(\d{13})(?![\d.])')
# Cassette playback hands a response's headers back in a different order each run, and botocore copies them into an
# error response's `ResponseMetadata`, which an error message renders with `repr`.
BOTOCORE_ERROR_HEADERS = re.compile(r"'HTTPHeaders': \{([^{}]*)\}")
# The native tools a model supports are a set of classes, which hash by identity, so the error that lists them for an
# unsupported one lists them in a different order each run.
SUPPORTED_NATIVE_TOOLS = re.compile(r'(not supported by this model\. Supported: \[)([^\[\]]*)\]')
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
# The packages whose classes stand in for what a realtime connection sends on, and the methods it sends through: a
# socket's `send`, and the ones Google's `AsyncSession` sends with.
REALTIME_DOUBLE_PACKAGES = ('tests.realtime',)
REALTIME_SEND_METHODS = ('send', 'send_realtime_input', 'send_client_content', 'send_tool_response')
# The call a class statement makes, which the capture runs from a frame of its own: see `_wrap_realtime_doubles`.
BUILD_CLASS = compile(
    '__refactor_capture_build_class__(*__refactor_capture_args__, **__refactor_capture_kwargs__)', __file__, 'eval'
)
# The `SessionCore` methods a capture records the results of: what it does with what it's applied, and what the session
# reads from it.
SESSION_CORE_METHODS = (
    'apply',
    'take_effects',
    'wait_tokens',
    'still_owed',
    'all_messages',
    'replayable_messages',
    'conversation_so_far',
)
# The `InferredLifecycle` methods a connection calls, which a capture records where a test calls them directly.
LIFECYCLE_METHODS = (
    'input_sent',
    'input_failed',
    'input_unanswerable',
    'take_pending',
    'speech_heard',
    'frame_for',
    'takes_turn',
    'connection_lost',
)
# What a realtime session counts as dropped, which its span reports and a capture records once it closes.
SESSION_DROP_COUNTERS = (
    '_audio_tap_drops',
    '_transcript_tap_drops',
    '_queue_dropped_deltas',
    '_queue_dropped_structural',
)
# The dunder methods of `RealtimeSession` a caller calls, besides `__aiter__`, whose events a capture records.
SESSION_DUNDER_CALLS = ('__init__', '__aenter__', '__aexit__')
# The kinds of record durable execution makes, with the label the comparison counts them under.
DURABLE_KINDS = {
    'temporal': 'temporal workflow run',
    'dbos': 'dbos workflow',
    'durable_name': 'durable name',
    'cache_key': 'cache key',
}
# The `AgentStream` methods whose calls a capture records, with what each yields or returns.
AGENT_STREAM_ITERATORS = ('stream_text', 'stream_output')
AGENT_STREAM_CALLS = ('get_output', 'validate_response_output')
# The `AgentStream` method arguments a call's record leaves out: the response a call validates is the stream's own,
# which its stream record holds.
UNRECORDED_AGENT_STREAM_ARGUMENTS = frozenset({'self', 'message'})
# What a canary agent run makes that a capture records, by record kind.
CANARY_RECORDS = {
    'stream': 'stream events',
    'run_event_stream': 'run event stream',
    'agent_stream': '`AgentStream` calls',
    'span': 'spans',
}
# The span event attributes left out of a capture: a stack trace names the lines a refactor moves.
UNRECORDED_SPAN_EVENT_ATTRIBUTES = frozenset({'exception.stacktrace'})
# The capability hooks the tool manager runs a tool call through, whose calls a capture records.
TOOL_HOOKS = (
    'wrap_tool_validate',
    'before_tool_validate',
    'on_tool_validate_error',
    'after_tool_validate',
    'wrap_tool_execute',
    'before_tool_execute',
    'on_tool_execute_error',
    'after_tool_execute',
)
# The tool hooks whose recorded calls are running, by name and agent run: the calls a combined capability makes on the
# ones it combines aren't recorded again, and the ones an agent run inside a hook or a tool makes are.
_RUNNING_TOOL_HOOKS: ContextVar[frozenset[tuple[str, str | None]]] = ContextVar(
    '_RUNNING_TOOL_HOOKS', default=frozenset()
)
# Set while a recorded connection `send`, or a recorded call on a realtime double, runs.
_RECORDING_CONNECTION_SEND: ContextVar[bool] = ContextVar('_RECORDING_CONNECTION_SEND', default=False)
_RECORDING_DOUBLE_SEND: ContextVar[bool] = ContextVar('_RECORDING_DOUBLE_SEND', default=False)
# Set while a recorded connection's `frames` works out its next frame.
_READING_CONNECTION_FRAMES: ContextVar[bool] = ContextVar('_READING_CONNECTION_FRAMES', default=False)
# Set while a recorded `SessionCore` or `InferredLifecycle` call runs, or the capture reads a closed session.
_RECORDING_CORE_CALL: ContextVar[bool] = ContextVar('_RECORDING_CORE_CALL', default=False)
_RECORDING_LIFECYCLE_CALL: ContextVar[bool] = ContextVar('_RECORDING_LIFECYCLE_CALL', default=False)
# The task and session of the public session call running, by `id`: a task copies the context it's created in, so the
# value would otherwise keep both alive through a task the session holds.
_RECORDING_SESSION_CALL: ContextVar[tuple[int, int] | None] = ContextVar('_RECORDING_SESSION_CALL', default=None)
# The package whose calls on a durable engine's decorators register the names a capture records.
DURABLE_EXEC_PACKAGE = 'pydantic_ai.durable_exec.'
# What a capture records for a Prefect cache key computed from a value each run draws anew, like the flow run's ID.
RUN_SCOPED_KEY = 'RUN_SCOPED'
# The sets a serialized run context sends as lists, in their own order, which `sorted_sets` sorts.
RUN_CONTEXT_SETS = frozenset(
    {
        'loaded_capability_ids',
        'discovered_tool_names',
        'available_tool_names',
        'active_capability_ids',
        '_deferred_capability_ids',
    }
)

WrappedT = TypeVar('WrappedT')


class BotocoreRequest(TypedDict):
    """What a capture reads from the request dict botocore's `Endpoint.make_request` is handed."""

    method: str
    url: str
    headers: Mapping[str, str]
    body: object


@dataclasses.dataclass
class CacheKeyCall:
    """A recorded `compute_key` call on a Prefect cache policy, while it runs."""

    inputs: object
    # The records of the outermost call running and of the calls it makes, which are written together.
    records: list[dict[str, object]]
    outer: CacheKeyCall | None
    # Whether its key, or a key it's computed from, holds a value each run draws anew.
    run_scoped: bool = False


_COMPUTING_CACHE_KEY: ContextVar[CacheKeyCall | None] = ContextVar('_COMPUTING_CACHE_KEY', default=None)


class ImportWatcher(importlib.abc.MetaPathFinder):
    """Calls the wrap given for a module once the module first runs.

    A module is found by the finders after this one on `sys.meta_path`. Its loader's `exec_module` runs the wrap
    after the module, so the capture wraps what a module defines without importing it any earlier. A spec found but
    never run, like one `importlib.util.find_spec` returns, leaves the wrap for the module's import.
    """

    def __init__(self, wraps: dict[str, Callable[[], None]]) -> None:
        self.wraps = wraps

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname not in self.wraps:
            return None
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            if (spec := finder.find_spec(fullname, path, target)) is not None:
                break
        else:
            return None
        # Only a namespace package has no loader of its own, and none of the modules wrapped is one.
        loader = spec.loader
        assert loader is not None
        exec_module = type(loader).exec_module
        loader_ref = weakref.ref(loader)

        # Set on the loader the spec holds, and deleted again once called, which leaves the loader's own method. It
        # holds the loader weakly, so a loader whose spec never runs holds no reference cycle.
        def exec_then_wrap(module: ModuleType) -> None:
            loader = loader_ref()
            assert loader is not None
            delattr(loader, 'exec_module')
            exec_module(loader, module)
            self.wraps.pop(fullname, lambda: None)()

        setattr(loader, 'exec_module', exec_then_wrap)
        return spec


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
        # The collector collecting, whose node a durable name registered meanwhile is recorded under.
        self.collecting: str | None = None
        self.records: list[dict[str, object]] = []
        # Each stream record with its stream, held weakly: a stream holds what the run iterating it holds, like its
        # dependencies. While the stream lives, `write` records the response it builds then.
        self._streams: list[tuple[dict[str, object], weakref.ref[StreamedResponse]]] = []
        # Each hook record of a tool call in a run, with the run and step it's in and the call's ID, by which `write`
        # orders them.
        self._tool_calls: list[tuple[dict[str, object], tuple[str | None, int], str]] = []
        # Each span record with its span's number, and the number of each span started since the records were last
        # written, by span ID, in the order they started, which `write` records the spans in: concurrent tool calls
        # end theirs in any order.
        self._spans: list[tuple[int, dict[str, object]]] = []
        self._span_numbers: dict[int, int] = {}
        # Per stream, by `id`: the stream, the iterator `__aiter__` first returned and the recording that wraps it, all
        # held weakly, as the stream is, and the stream's record with its events. The stream's own reference tells a
        # reused `id` apart.
        self._iterators: dict[
            int,
            tuple[
                weakref.ref[StreamedResponse],
                weakref.ref[AsyncIterator[ModelResponseStreamEvent]],
                weakref.ref[AsyncIterator[ModelResponseStreamEvent]],
                tuple[dict[str, object], list[object]],
            ],
        ] = {}
        # The Temporal workflow runs and DBOS workflows the test ran, by ID, each record with the list of its commands or
        # operations it holds, which `write` records in a set order.
        self._temporal_runs: dict[str, tuple[dict[str, object], list[object]]] = {}
        self._dbos_workflows: dict[str, tuple[dict[str, object], list[object]]] = {}
        # The durable names registered, and the records of each outermost `compute_key` call on a Prefect cache policy
        # with the calls it made, which `write` records in a set order: Prefect runs a task on another thread than the
        # requests its flow sends.
        self._durable_names: list[dict[str, object]] = []
        self._cache_key_calls: list[list[dict[str, object]]] = []
        self._import_watcher = ImportWatcher({})
        # The requests Prefect's client has sent, so a retry, which sends one again, isn't recorded twice.
        self._prefect_requests: weakref.WeakSet[httpx.Request | httpx2.Request] = weakref.WeakSet()
        # The apps `create_api_app` has built, whose responses are recorded.
        self._api_apps: weakref.WeakSet[Starlette] = weakref.WeakSet()
        # Each realtime session, `SessionCore` and `InferredLifecycle` the test has recorded, numbered per kind in the
        # order the test first used it.
        self._numbers: weakref.WeakKeyDictionary[object, int] = weakref.WeakKeyDictionary()
        self._next_numbers: Counter[str] = Counter()
        # Each view a session's `stream_audio` or `stream_transcripts` returned, with what it yields, kept past its test
        # while the view lives, and the classes of those views, whose `__anext__` is wrapped.
        self._taps: weakref.WeakKeyDictionary[object, list[object]] = weakref.WeakKeyDictionary()
        self._tap_classes: set[type] = set()
        # The sessions whose state was recorded once they closed.
        self._closed_sessions: weakref.WeakSet[RealtimeSession] = weakref.WeakSet()
        # The workspace's top-level packages, whose lines a warning's reported frame leaves out.
        self._workspace_packages = frozenset(
            child.name
            for root in workspace_roots()
            for child in (REPO / root).iterdir()
            if (child / '__init__.py').exists()
        )
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
        with suppress(ValueError):
            sys.meta_path.remove(self._import_watcher)
        self._monkeypatch.undo()

    @pytest.hookimpl(wrapper=True)
    def pytest_make_collect_report(self, collector: pytest.Collector) -> Generator[None, object, object]:
        """Record the durable names collecting a module, like the agents it defines, registers under its node."""
        collecting, self.collecting = self.collecting, collector.nodeid
        try:
            return (yield)
        finally:
            self.collecting = collecting
            self._append_records()

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.failed:
            worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
            with open(self.capture_dir / f'{worker}{FAILED_SUFFIX}', 'a', encoding='utf-8') as f:
                f.write(f'{report.nodeid}\n')

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
        `UIAdapter` or a subclass defines them, the app `create_api_app` builds, `frames` and `send` where
        `RealtimeConnection` or a subclass defines them, the realtime doubles' send methods, the public methods of
        `RealtimeSession`, the methods `SESSION_CORE_METHODS` and `LIFECYCLE_METHODS` list where `SessionCore`,
        `InferredLifecycle` or a subclass defines them, `infer_realtime_model`, the hooks `TOOL_HOOKS` lists where
        `AbstractCapability` or a subclass defines them, `MCPSamplingModel.request`, the point where `pytest.raises` and
        `pytest.RaisesGroup` keep the exception they caught, and the one where a warning is shown or recorded. Once
        Temporal, DBOS or Prefect runs, wrap what registers a durable name, Temporal's bridge worker, DBOS's system
        database, and `compute_key` where Prefect's `CachePolicy` or a subclass defines it.
        """
        self.wrap_on_import(
            {
                'temporalio.activity': self._wrap_temporal_activity_defn,
                'temporalio.workflow': self._wrap_temporal_workflow_defn,
                'temporalio.bridge.worker': self._wrap_temporal_worker,
                'dbos._dbos': self._wrap_dbos_decorators,
                'dbos._sys_db': self._wrap_dbos_system_database,
                'prefect.main': self._wrap_prefect_task,
                'prefect.flows': self._wrap_prefect_flow,
                'prefect.cache_policies': self._wrap_cache_policies,
            }
        )
        self._wrap_subclasses(StreamedResponse, self._wrap_aiter)
        self._wrap_subclasses(AbstractCapability, self._wrap_capability)
        self._wrap_agent_stream()
        self._wrap_spans()
        self._wrap_sampling_request()
        self._wrap_raises()
        self._wrap_warnings()
        self._wrap_subclasses(UIEventStream, self._wrap_transform_stream)
        self._wrap_subclasses(UIAdapter, self._wrap_message_methods)
        self._wrap_subclasses(RealtimeConnection, self._wrap_connection)
        self._wrap_subclasses(RealtimeSession, self._wrap_session)
        self._wrap_subclasses(SessionCore, self._wrap_core)
        self._wrap_subclasses(InferredLifecycle, self._wrap_lifecycle)
        self._wrap_infer_realtime_model()
        self._wrap_realtime_doubles()
        self._wrap_api_app()
        self._wrap_send(httpx.AsyncClient)
        self._wrap_send(httpx2.AsyncClient)
        self._wrap_make_request()
        for model_class, method_names in IN_PROCESS_MODELS:
            for name in method_names:
                self._wrap_model(model_class, name)

    def check_stream_hook(self) -> None:
        """Stop the run unless the streams, run event stream, `AgentStream` calls and spans of an instrumented
        agent's streamed run are recorded.
        """

        async def run() -> None:
            tracer_provider = TracerProvider(shutdown_on_exit=False)
            instrumentation = Instrumentation(settings=InstrumentationSettings(tracer_provider=tracer_provider))
            agent = Agent(FunctionModel(stream_function=stream_canary), capabilities=[instrumentation])
            async with agent.run_stream('canary') as result:
                await result.get_output()

        asyncio.run(run())
        # A stream or run event stream counts only with the events it handed the consumer.
        recorded = {record['kind'] for record in self.records if record.get('events', True)}
        if self._spans:
            recorded.add('span')
        self.clear()
        if missed := [label for kind, label in CANARY_RECORDS.items() if kind not in recorded]:
            self._monkeypatch.undo()
            raise pytest.UsageError(
                f'The {" and ".join(missed)} of a canary agent run were not recorded, so the capture would miss '
                'them. Update `tests/refactor_capture.py` to the way they are made now.'
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
        """Append `node` and the records so far to this worker's files."""
        worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
        with open(self.capture_dir / f'{worker}{NODES_SUFFIX}', 'a', encoding='utf-8') as f:
            f.write(f'{node}\n')
        self._append_records()

    def _append_records(self) -> None:
        """Append the records so far to this worker's records file, with the response each stream that lives builds
        now, the spans in the order they started, and the durable names, cache keys and workflows last, each in a set
        order: they're made on other threads than the test's, or concurrently, and a workflow's ID is drawn anew each
        run.
        """
        for record, stream_ref in self._streams:
            if (stream := stream_ref()) is not None:
                record['response'] = built_response(stream)
            record['response'] = recorded_value(record['response'])
        self._order_tool_calls()
        self.records.extend(record for _, record in sorted(self._spans, key=lambda span: span[0]))
        self.records.extend(sorted(self._durable_names, key=json.dumps))
        for calls in sorted(self._cache_key_calls, key=json.dumps):
            self.records.extend(calls)
        for workflows in (self._temporal_runs, self._dbos_workflows):
            self.records.extend(sorted((record for record, _ in workflows.values()), key=json.dumps))
        if self.records:
            worker = os.environ.get('PYTEST_XDIST_WORKER', 'main')
            with open(self.capture_dir / f'{worker}.jsonl', 'a', encoding='utf-8') as f:
                f.writelines(json.dumps(record) + '\n' for record in self.records)
        self.clear()

    def _order_tool_calls(self) -> None:
        """Put the hook records of the tool calls each run step makes in the order the calls started, each call's own
        in the order they were made, in the places the step's hook records take among the others.

        The calls of a step run concurrently, and a sync tool on a worker thread, so how the hooks of one call called
        after its tool returns interleave with another's is the threads' order.
        """
        steps: dict[tuple[str | None, int], list[tuple[dict[str, object], str]]] = {}
        for record, step, call_id in self._tool_calls:
            steps.setdefault(step, []).append((record, call_id))
        places = {id(record): place for place, record in enumerate(self.records)}
        for calls in steps.values():
            started: dict[str, int] = {}
            for index, (_, call_id) in enumerate(calls):
                started.setdefault(call_id, index)
            ordered = sorted(range(len(calls)), key=lambda index: (started[calls[index][1]], index))
            for (record, _), index in zip(calls, ordered):
                self.records[places[id(record)]] = calls[index][0]

    def clear(self) -> None:
        """Drop the records so far."""
        self.records.clear()
        self._streams.clear()
        self._tool_calls.clear()
        self._spans.clear()
        self._span_numbers.clear()
        self._iterators.clear()
        self._temporal_runs.clear()
        self._dbos_workflows.clear()
        self._durable_names.clear()
        self._cache_key_calls.clear()
        self._numbers.clear()
        self._next_numbers.clear()
        self._closed_sessions.clear()

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
            known = self._iterators.get(id(stream))
            if known is None or known[0]() is not stream:
                record = self._stream_record(stream)
            elif iterator is not known[1]():
                # It already reads through the recording: an override's `super().__aiter__()` result, or the peeking
                # stream `request_stream_until_first_event` stores over it.
                return iterator
            elif (recording := known[2]()) is not None:
                return recording
            else:
                # The consumer dropped the recording, and goes on reading the same iterator, into the same record.
                record = known[3]
            recording = self._recorded(stream, iterator, *record)
            self._iterators[id(stream)] = (weakref.ref(stream), weakref.ref(iterator), weakref.ref(recording), record)
            return recording

        self._monkeypatch.setattr(cls, '__aiter__', recording_aiter)

    def _stream_record(self, stream: StreamedResponse) -> tuple[dict[str, object], list[object]]:
        """A new record of `stream`, with the response it builds so far, and the list of events it holds."""
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
            'response': built_response(stream),
        }
        self.records.append(record)
        self._streams.append((record, weakref.ref(stream)))
        return record, events

    def _recorded(
        self,
        stream: StreamedResponse,
        iterator: AsyncIterator[ModelResponseStreamEvent],
        record: dict[str, object],
        events: list[object],
    ) -> AsyncIterator[ModelResponseStreamEvent]:

        # Closing this doesn't close `iterator`: every `__aiter__` in the library returns a guard that does no cleanup
        # on close, so only its garbage collection moves. Once iteration ends, the response the stream builds is
        # recorded, which `write` records again if the stream still lives then.
        async def recorded() -> AsyncIterator[ModelResponseStreamEvent]:
            try:
                async for event in iterator:
                    events.append(normalized(event))
                    yield event
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                record['response'] = built_response(stream)

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
                record['error'] = error_text(e)
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
                record['error'] = error_text(e)
                raise
            record['result'] = normalized(result)
            record['uncompared_fields'] = uncompared_fields(result)
            return result

        self._monkeypatch.setattr(cls, name, classmethod(recording_method))

    def _wrap_connection(self, cls: type[RealtimeConnection]) -> None:
        if 'frames' in vars(cls):
            self._wrap_frames(cls)
        if 'send' in vars(cls):
            self._wrap_connection_send(cls)

    def _wrap_frames(self, cls: type[RealtimeConnection]) -> None:
        original: Callable[[RealtimeConnection], AsyncIterator[RealtimeFrame]] = vars(cls)['frames']

        @functools.wraps(original)
        def recording_frames(connection: RealtimeConnection) -> AsyncIterator[RealtimeFrame]:
            return self._recorded_frames(connection, original(connection))

        self._monkeypatch.setattr(cls, 'frames', recording_frames)

    def _recorded_frames(
        self, connection: RealtimeConnection, iterator: AsyncIterator[RealtimeFrame]
    ) -> AsyncIterator[RealtimeFrame]:
        target = method_target(type(connection), 'frames')

        async def recorded() -> AsyncIterator[RealtimeFrame]:
            frames = aiter(iterator)
            try:
                while True:
                    token = _READING_CONNECTION_FRAMES.set(True)
                    try:
                        frame = await anext(frames)
                    except StopAsyncIteration:
                        break
                    finally:
                        _READING_CONNECTION_FRAMES.reset(token)
                    self.records.append(
                        {'kind': 'realtime', 'node': self.node, 'target': target, 'frame': list(map(tagged, frame))}
                    )
                    yield frame
            except Exception as e:
                self.records.append(
                    {
                        'kind': 'realtime',
                        'node': self.node,
                        'target': target,
                        'error': error_text(e),
                    }
                )
                raise
            finally:
                # A connection's `frames` settles what's open when it's closed, so it's closed with this.
                if isinstance(iterator, AsyncGenerator):
                    await iterator.aclose()

        return recorded()

    def _wrap_connection_send(self, cls: type[RealtimeConnection]) -> None:
        original: Callable[..., Awaitable[Iterable[object] | None]] = vars(cls)['send']

        # `functools.wraps` carries over `__isabstractmethod__`, which tells an abstract subclass apart.
        @functools.wraps(original)
        async def recording_send(connection: RealtimeConnection, content: object) -> Iterable[object] | None:
            if _RECORDING_CONNECTION_SEND.get():
                return await original(connection, content)
            record: dict[str, object] = {
                'kind': 'realtime',
                'node': self.node,
                'target': method_target(type(connection), 'send'),
                'input': tagged(content),
            }
            self.records.append(record)
            token = _RECORDING_CONNECTION_SEND.set(True)
            try:
                result = await original(connection, content)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                _RECORDING_CONNECTION_SEND.reset(token)
            record['result'] = None if result is None else list(map(tagged, result))
            return result

        self._monkeypatch.setattr(cls, 'send', recording_send)

    def _wrap_realtime_doubles(self) -> None:
        """Wrap the realtime doubles' send methods on the classes defined so far, and on each one defined later."""
        for module_name, module in [*sys.modules.items()]:
            if is_realtime_double_module(module_name):
                for value in [*vars(module).values()]:
                    if isinstance(value, type) and value.__module__ == module_name:
                        self._wrap_realtime_double(value)
        original: Callable[..., object] = vars(builtins)['__build_class__']

        # A class statement builds its class through `__build_class__`, wherever it runs. Pydantic resolves a model's
        # annotations in the namespace of the frame its metaclass is called from, past a PEP 695 annotation scope,
        # unless that frame runs a module, and a warning a class raises as it's defined, like one from
        # `__init_subclass__`, reports that frame. So the class is built from a frame with the locals, module, name,
        # file and line of the frame the statement runs in, rather than from this one. Its globals are a copy of that
        # frame's, which shares the module's warning registry once the module has one.
        def build_class(*args: object, **kwargs: object) -> object:
            frame = inspect.currentframe()
            assert frame is not None and frame.f_back is not None
            caller = frame.f_back
            # A frame that holds itself would be a reference cycle.
            del frame
            if caller.f_code.co_name.startswith('<generic parameters of') and caller.f_back is not None:
                caller = caller.f_back
            code = BUILD_CLASS.replace(
                co_name=caller.f_code.co_name,
                co_qualname=caller.f_code.co_qualname,
                co_filename=caller.f_code.co_filename,
                co_firstlineno=caller.f_lineno or caller.f_code.co_firstlineno,
            )
            names = {
                **caller.f_globals,
                '__refactor_capture_build_class__': original,
                '__refactor_capture_args__': args,
                '__refactor_capture_kwargs__': kwargs,
            }
            try:
                cls = eval(code, names, caller.f_locals)
            finally:
                del caller
            if isinstance(cls, type) and is_realtime_double_module(cls.__module__):
                self._wrap_realtime_double(cls)
            return cls

        self._monkeypatch.setattr(builtins, '__build_class__', build_class)

    def _wrap_realtime_double(self, cls: type[object]) -> None:
        if issubclass(cls, RealtimeConnection):
            return
        for name in REALTIME_SEND_METHODS:
            if inspect.iscoroutinefunction(vars(cls).get(name)):
                self._wrap_double_send(cls, name)

    def _wrap_double_send(self, cls: type[object], name: str) -> None:
        original: Callable[..., Awaitable[object]] = vars(cls)[name]

        @functools.wraps(original)
        async def recording_send(double: object, *args: object, **kwargs: object) -> object:
            if _RECORDING_DOUBLE_SEND.get():
                return await original(double, *args, **kwargs)
            record: dict[str, object] = {
                'kind': 'realtime',
                'node': self.node,
                'target': method_target(type(double), name),
                'args': normalized([parsed_json(arg) if isinstance(arg, str) else arg for arg in args]),
                'kwargs': normalized(kwargs),
            }
            self.records.append(record)
            token = _RECORDING_DOUBLE_SEND.set(True)
            try:
                return await original(double, *args, **kwargs)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                _RECORDING_DOUBLE_SEND.reset(token)

        self._monkeypatch.setattr(cls, name, recording_send)

    def _number(self, label: str, value: object) -> int:
        """`value`'s number among the objects of kind `label` the test has recorded, in the order it first used them."""
        if (number := self._numbers.get(value)) is None:
            number = self._numbers[value] = self._next_numbers[label]
            self._next_numbers[label] += 1
        return number

    def _wrap_session(self, cls: type[RealtimeSession]) -> None:
        for name, value in [*vars(cls).items()]:
            if name == '__aiter__':
                self._wrap_session_events(cls)
            elif name in ('stream_audio', 'stream_transcripts'):
                self._wrap_session_tap(cls, name)
            elif name == 'close':
                self._wrap_session_close(cls)
            elif name.startswith('_') and name not in SESSION_DUNDER_CALLS:
                continue
            elif isinstance(value, property):
                assert value.fget is not None
                fget = self._recorded_session_call(name, value.fget)
                self._monkeypatch.setattr(cls, name, property(fget, value.fset, value.fdel, value.__doc__))
            elif inspect.isfunction(value):
                self._monkeypatch.setattr(cls, name, self._recorded_session_call(name, value))

    @contextmanager
    def _session_call(self, session: RealtimeSession, name: str) -> Generator[None]:
        """Record the error a public call on `session` raises, unless another public call on it in this task made it."""
        key = (running_task_id(), id(session))
        if _RECORDING_SESSION_CALL.get() == key:
            yield
            return
        token = _RECORDING_SESSION_CALL.set(key)
        try:
            yield
        except Exception as e:
            self.records.append(
                {
                    'kind': 'session',
                    'node': self.node,
                    'target': method_target(type(session), name),
                    'session': self._number('session', session),
                    'error': error_text(e),
                }
            )
            raise
        finally:
            _RECORDING_SESSION_CALL.reset(token)

    def _recorded_session_call(self, name: str, original: Callable[..., object]) -> Callable[..., object]:
        if inspect.iscoroutinefunction(original):

            @functools.wraps(original)
            async def recording_async_call(session: RealtimeSession, *args: object, **kwargs: object) -> object:
                with self._session_call(session, name):
                    return await original(session, *args, **kwargs)

            return recording_async_call

        @functools.wraps(original)
        def recording_call(session: RealtimeSession, *args: object, **kwargs: object) -> object:
            with self._session_call(session, name):
                return original(session, *args, **kwargs)

        return recording_call

    def _wrap_session_events(self, cls: type[RealtimeSession]) -> None:
        original: Callable[[RealtimeSession], AsyncGenerator[object]] = vars(cls)['__aiter__']

        # An async generator, as `__aiter__` is, so closing the recording closes the generator it wraps.
        @functools.wraps(original)
        async def recording_aiter(session: RealtimeSession) -> AsyncIterator[object]:
            events: list[object] = []
            record: dict[str, object] = {
                'kind': 'session',
                'node': self.node,
                'target': method_target(type(session), '__aiter__'),
                'session': self._number('session', session),
                'events': events,
            }
            self.records.append(record)
            iterator = original(session)
            try:
                async for event in iterator:
                    events.append(tagged(event))
                    yield event
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                await iterator.aclose()

        self._monkeypatch.setattr(cls, '__aiter__', recording_aiter)

    def _wrap_session_tap(self, cls: type[RealtimeSession], name: str) -> None:
        original: Callable[..., AsyncIterator[object]] = vars(cls)[name]

        @functools.wraps(original)
        def recording_tap(session: RealtimeSession, **kwargs: object) -> AsyncIterator[object]:
            with self._session_call(session, name):
                view = original(session, **kwargs)
            # An override that calls `super()` hands back the view it's handed.
            if view not in self._taps:
                items: list[object] = []
                record: dict[str, object] = {
                    'kind': 'session',
                    'node': self.node,
                    'target': method_target(type(session), name),
                    'session': self._number('session', session),
                    'kwargs': normalized(kwargs),
                    'items': items,
                }
                self._taps[view] = items
                self.records.append(record)
                self._wrap_tap_view(type(view))
            return view

        self._monkeypatch.setattr(cls, name, recording_tap)

    def _wrap_tap_view(self, view_class: type[AsyncIterator[object]]) -> None:
        """Wrap `__anext__` on the class of a view a session's tap returned, which tests check the type of."""
        if view_class in self._tap_classes:
            return
        self._tap_classes.add(view_class)
        original = view_class.__anext__

        async def recording_anext(view: AsyncIterator[object]) -> object:
            item = await original(view)
            self._taps[view].append(normalized(item))
            return item

        self._monkeypatch.setattr(view_class, '__anext__', recording_anext)

    def _wrap_session_close(self, cls: type[RealtimeSession]) -> None:
        original: Callable[[RealtimeSession], Awaitable[None]] = vars(cls)['close']

        @functools.wraps(original)
        async def recording_close(session: RealtimeSession) -> None:
            try:
                with self._session_call(session, 'close'):
                    await original(session)
            except Exception:
                self._record_closed(session)
                raise
            self._record_closed(session)

        self._monkeypatch.setattr(cls, 'close', recording_close)

    def _record_closed(self, session: RealtimeSession) -> None:
        """Record what `session` holds once the first `close` call on it ends, unless that call was cancelled."""
        if session in self._closed_sessions:
            return
        self._closed_sessions.add(session)
        record: dict[str, object] = {
            'kind': 'session',
            'node': self.node,
            'target': method_target(type(session), 'close'),
            'session': self._number('session', session),
        }

        # What the capture reads is no call the test or the session made.
        core_token = _RECORDING_CORE_CALL.set(True)
        session_token = _RECORDING_SESSION_CALL.set((running_task_id(), id(session)))
        try:
            record['all_messages'] = normalized(session.all_messages())
            record['new_messages'] = normalized(session.new_messages())
            record['usage'] = normalized(session.usage)
            # Reading `played_audio_bytes` can raise the error that ended the session again. Raised where another
            # exception is being handled, as when `close` raises, Python chains it to that one, and cuts the other's
            # chain where it held the error, to keep it from looping back. The chain being handled is put back as it
            # was. One thing can't be: the error's own `__context__`, where the chain didn't hold it. The reads stay in
            # the event loop, where waiting on a thread would block it.
            handled_chain: list[tuple[BaseException, BaseException | None]] = []
            handled = sys.exception()
            while handled is not None and all(handled is not seen for seen, _ in handled_chain):
                handled_chain.append((handled, handled.__context__))
                handled = handled.__context__
            try:
                record['played_audio_bytes'] = session.played_audio_bytes
            except Exception as e:
                record['played_audio_bytes'] = {'error': error_text(e)}
            for exception, context in handled_chain:
                exception.__context__ = context
            record['drops'] = {name: getattr(session, name) for name in SESSION_DROP_COUNTERS}
        finally:
            _RECORDING_SESSION_CALL.reset(session_token)
            _RECORDING_CORE_CALL.reset(core_token)
        self.records.append(record)

    def _wrap_core(self, cls: type[SessionCore]) -> None:
        for name in SESSION_CORE_METHODS:
            if name in vars(cls):
                self._wrap_state_call(cls, name, 'core', _RECORDING_CORE_CALL)

    def _wrap_lifecycle(self, cls: type[InferredLifecycle]) -> None:
        for name in LIFECYCLE_METHODS:
            if name in vars(cls):
                self._wrap_state_call(cls, name, 'lifecycle', _RECORDING_LIFECYCLE_CALL, outside_connections=True)

    def _wrap_state_call(
        self, cls: type, name: str, label: str, running: ContextVar[bool], *, outside_connections: bool = False
    ) -> None:
        """Wrap a `SessionCore` or `InferredLifecycle` method, to record its arguments and result.

        A call it makes on itself isn't recorded. With `outside_connections`, neither is one a connection's `frames`
        or `send` makes, whose `realtime` records hold what it returns.
        """
        original: Callable[..., object] = vars(cls)[name]

        @functools.wraps(original)
        def recording_call(owner: object, *args: object, **kwargs: object) -> object:
            if running.get() or (
                outside_connections and (_READING_CONNECTION_FRAMES.get() or _RECORDING_CONNECTION_SEND.get())
            ):
                return original(owner, *args, **kwargs)
            record: dict[str, object] = {
                'kind': 'session',
                'node': self.node,
                'target': method_target(type(owner), name),
                label: self._number(label, owner),
                'args': recorded(args),
                'kwargs': {key: recorded(value) for key, value in kwargs.items()},
            }
            self.records.append(record)
            token = running.set(True)
            try:
                result = original(owner, *args, **kwargs)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                running.reset(token)
            record['result'] = recorded(result)
            return result

        self._monkeypatch.setattr(cls, name, recording_call)

    def _wrap_infer_realtime_model(self) -> None:
        original: Callable[..., RealtimeModel] = vars(realtime_model)['infer_realtime_model']
        target = f'{realtime_model.__name__}.infer_realtime_model'

        @functools.wraps(original)
        def recording_infer_realtime_model(*args: object, **kwargs: object) -> RealtimeModel:
            record: dict[str, object] = {
                'kind': 'session',
                'node': self.node,
                'target': target,
                'args': normalized(args),
                'kwargs': normalized(kwargs),
            }
            self.records.append(record)
            try:
                model = original(*args, **kwargs)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            record['result'] = {
                'type': f'{type(model).__module__}.{type(model).__qualname__}',
                'model_name': model.model_name,
                'system': model.system,
                'base_url': model.base_url,
            }
            return model

        # The agent, and most tests, import it from `pydantic_ai.realtime`, which re-exports it.
        for module in (realtime_model, pydantic_ai.realtime):
            self._monkeypatch.setattr(module, 'infer_realtime_model', recording_infer_realtime_model)

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
            recorded_parameters = copy.copy(model_request_parameters)
            recorded_parameters.function_tools = self._recorded_tools(model_request_parameters.function_tools)
            recorded_parameters.output_tools = self._recorded_tools(model_request_parameters.output_tools)
            self.records.append(
                {
                    'kind': 'request',
                    'node': self.node,
                    'target': target,
                    'messages': normalized(messages),
                    'model_settings': normalized(model_settings),
                    'model_request_parameters': normalized(recorded_parameters),
                }
            )
            return original(model, messages, model_settings, model_request_parameters, *args, **kwargs)

        self._monkeypatch.setattr(model_class, name, recording_method)

    @staticmethod
    def _recorded_tools(tools: list[ToolDefinition]) -> list[ToolDefinition]:
        """Copies of `tools` that keep each metadata value `json` can write, and record each other one as `unrecorded`,
        with its class.

        A tool's metadata isn't sent to the model, so the code under test may never serialize it, and serializing a value
        like a model can run a test's serializer.
        """
        recorded_tools: list[ToolDefinition] = []
        for tool in tools:
            recorded = copy.copy(tool)
            if tool.metadata is not None:
                metadata: dict[str, object] = tool.metadata
                recorded.metadata = {}
                for key, value in metadata.items():
                    try:
                        json.dumps(value)
                    except (TypeError, ValueError):
                        recorded.metadata[key] = {'unrecorded': qualified_name(type(value))}
                    else:
                        recorded.metadata[key] = value
            recorded_tools.append(recorded)
        return recorded_tools

    def _wrap_capability(self, cls: type[AbstractCapability[object]]) -> None:
        self._wrap_tool_hooks(cls)
        if 'wrap_run_event_stream' in vars(cls):
            self._wrap_run_event_stream(cls)

    def _wrap_run_event_stream(self, cls: type[AbstractCapability[object]]) -> None:
        original: Callable[..., AsyncIterable[AgentStreamEvent]] = vars(cls)['wrap_run_event_stream']

        @functools.wraps(original)
        def recording_wrap_run_event_stream(
            capability: AbstractCapability[object], *args: object, **kwargs: object
        ) -> AsyncIterable[AgentStreamEvent]:
            stream = original(capability, *args, **kwargs)
            ctx = args[0] if args else kwargs.get('ctx')
            if not isinstance(ctx, RunContext):
                return stream
            # The root capability's stream is the one the consumer gets, with the streams of the capabilities it
            # combines inside it. A run context with no root is one a test built to call a capability directly.
            # A run context's deps type is the run's, which this can't name.
            if ctx.root_capability is not None and ctx.root_capability is not capability:  # pyright: ignore[reportUnknownMemberType]
                return stream
            return self._recorded_run_events(stream)

        self._monkeypatch.setattr(cls, 'wrap_run_event_stream', recording_wrap_run_event_stream)

    def _recorded_run_events(self, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
        events: list[object] = []
        record: dict[str, object] = {'kind': 'run_event_stream', 'node': self.node, 'events': events}
        self.records.append(record)

        async def recorded() -> AsyncIterator[AgentStreamEvent]:
            iterator = aiter(stream)
            try:
                async for event in iterator:
                    events.append(normalized(event))
                    yield event
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                # The consumer closes the stream once it's done, which closes the streams of the capabilities inside
                # it, so it's closed with this.
                if isinstance(iterator, AsyncGenerator):
                    await iterator.aclose()

        return recorded()

    def _wrap_agent_stream(self) -> None:
        """Wrap the methods `AGENT_STREAM_ITERATORS` and `AGENT_STREAM_CALLS` list on `AgentStream`."""
        for name in AGENT_STREAM_ITERATORS:
            self._wrap_agent_stream_iterator(name)
        for name in AGENT_STREAM_CALLS:
            self._wrap_agent_stream_call(name)

    def _agent_stream_record(self, name: str, *args: object, **kwargs: object) -> dict[str, object]:
        """A new record of a call on the `AgentStream` method `name`, with its arguments, defaults filled in."""
        signature = inspect.signature(vars(AgentStream)[name])
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        record: dict[str, object] = {
            'kind': 'agent_stream',
            'node': self.node,
            'target': method_target(AgentStream, name),
            'arguments': normalized(
                {
                    argument: value
                    for argument, value in arguments.arguments.items()
                    if argument not in UNRECORDED_AGENT_STREAM_ARGUMENTS
                }
            ),
        }
        self.records.append(record)
        return record

    def _wrap_agent_stream_iterator(self, name: str) -> None:
        original: Callable[..., AsyncGenerator[object]] = vars(AgentStream)[name]

        @functools.wraps(original)
        def recording_iterator(
            stream: AgentStream[object, object], *args: object, **kwargs: object
        ) -> AsyncIterator[object]:
            yields: list[object] = []
            record = self._agent_stream_record(name, stream, *args, **kwargs)
            record['yields'] = yields
            iterator = original(stream, *args, **kwargs)

            async def recorded() -> AsyncIterator[object]:
                try:
                    async for item in iterator:
                        yields.append(normalized(item))
                        yield item
                except Exception as e:
                    record['error'] = error_text(e)
                    raise
                finally:
                    # What it holds open, like the stream of text deltas `stream_text` reads, it closes when it's
                    # closed, so it's closed with this.
                    await iterator.aclose()

            return recorded()

        self._monkeypatch.setattr(AgentStream, name, recording_iterator)

    def _wrap_agent_stream_call(self, name: str) -> None:
        original: Callable[..., Awaitable[object]] = vars(AgentStream)[name]

        @functools.wraps(original)
        async def recording_call(stream: AgentStream[object, object], *args: object, **kwargs: object) -> object:
            record = self._agent_stream_record(name, stream, *args, **kwargs)
            try:
                result = await original(stream, *args, **kwargs)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            record['result'] = normalized(result)
            return result

        self._monkeypatch.setattr(AgentStream, name, recording_call)

    def _wrap_spans(self) -> None:
        """Number each span the OpenTelemetry SDK starts, in the order they start, and record each once it ends.

        A span that isn't recording, like one a tracer that's not the SDK's starts, has nothing to record.
        """
        original_start: Callable[..., None] = vars(SdkSpan)['start']
        original_end: Callable[..., None] = vars(SdkSpan)['end']

        @functools.wraps(original_start)
        def recording_start(span: SdkSpan, *args: object, **kwargs: object) -> None:
            self._span_numbers.setdefault(span.get_span_context().span_id, len(self._span_numbers))
            original_start(span, *args, **kwargs)

        @functools.wraps(original_end)
        def recording_end(span: SdkSpan, *args: object, **kwargs: object) -> None:
            # A span ended again stays as it was.
            ended = span.end_time is not None
            original_end(span, *args, **kwargs)
            if not ended:
                number = self._span_numbers.setdefault(span.get_span_context().span_id, len(self._span_numbers))
                self._spans.append((number, self._span_record(span, number)))

        self._monkeypatch.setattr(SdkSpan, 'start', recording_start)
        self._monkeypatch.setattr(SdkSpan, 'end', recording_end)

    def _span_record(self, span: ReadableSpan, number: int) -> dict[str, object]:
        """A record of `span`, which has ended, numbered `number`, with its parent named by its number."""
        parent = span.parent
        return {
            'kind': 'span',
            'node': self.node,
            'name': span.name,
            'number': number,
            # A parent no span started since the records were last written, like a remote one, is `external`.
            'parent': None if parent is None else self._span_numbers.get(parent.span_id, 'external'),
            'span_kind': span.kind.name,
            'scope': None if span.instrumentation_scope is None else span.instrumentation_scope.name,
            'attributes': normalized(dict(span.attributes or {})),
            'status': {'code': span.status.status_code.name, 'description': normalized(span.status.description)},
            'events': [
                {
                    'name': event.name,
                    'attributes': normalized(
                        {
                            key: value
                            for key, value in (event.attributes or {}).items()
                            if key not in UNRECORDED_SPAN_EVENT_ATTRIBUTES
                        }
                    ),
                }
                for event in span.events
            ],
        }

    def _wrap_tool_hooks(self, cls: type[AbstractCapability[object]]) -> None:
        for name in TOOL_HOOKS:
            if inspect.iscoroutinefunction(vars(cls).get(name)):
                self._wrap_tool_hook(cls, name)

    def _wrap_tool_hook(self, cls: type[AbstractCapability[object]], name: str) -> None:
        original: Callable[..., Awaitable[object]] = vars(cls)[name]

        @functools.wraps(original)
        async def recording_hook(capability: AbstractCapability[object], *args: object, **kwargs: object) -> object:
            ctx = args[0] if args else kwargs.get('ctx')
            key = (name, ctx.run_id if isinstance(ctx, RunContext) else None)
            try:
                running = _RUNNING_TOOL_HOOKS.get()
                if key in running:
                    return await original(capability, *args, **kwargs)
                record: dict[str, object] = {
                    'kind': 'hook',
                    'node': self.node,
                    'target': method_target(type(capability), name),
                    # The run context is the run's, and the handler a function.
                    'kwargs': {
                        arg: error_text(value) if isinstance(value, BaseException) else normalized(value)
                        for arg, value in kwargs.items()
                        if arg not in ('ctx', 'handler')
                    },
                }
                self.records.append(record)
                if isinstance(ctx, RunContext) and isinstance(call := kwargs.get('call'), ToolCallPart):
                    self._tool_calls.append((record, (ctx.run_id, ctx.run_step), call.tool_call_id))
                token = _RUNNING_TOOL_HOOKS.set(running | {key})
                try:
                    result = await original(capability, *args, **kwargs)
                except Exception as e:
                    record['error'] = error_text(e)
                    raise
                finally:
                    _RUNNING_TOOL_HOOKS.reset(token)
                record['result'] = normalized(result)
                return result
            finally:
                # An `on_*_error` hook can raise the error it's handed, whose traceback holds this frame, so holding
                # the error here would keep the run context, and its dependencies, in a cycle with it.
                del args, kwargs

        self._monkeypatch.setattr(cls, name, recording_hook)

    def _wrap_sampling_request(self) -> None:
        original: Callable[..., Awaitable[ModelResponse]] = vars(MCPSamplingModel)['request']
        target = method_target(MCPSamplingModel, 'request')

        @functools.wraps(original)
        async def recording_request(
            model: MCPSamplingModel,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            record: dict[str, object] = {
                'kind': 'request',
                'node': self.node,
                'target': target,
                'messages': normalized(messages),
                'model_settings': normalized(model_settings),
                'model_request_parameters': normalized(model_request_parameters),
            }
            self.records.append(record)
            # A mock keeps the calls it's handed; what a real session sends shows where its peer handles it.
            create_message = getattr(model.session, 'create_message', None)
            mock_calls = len(create_message.call_args_list) if isinstance(create_message, NonCallableMock) else 0
            try:
                response = await original(model, messages, model_settings, model_request_parameters)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            finally:
                if isinstance(create_message, NonCallableMock):
                    self.records.extend(
                        {
                            'kind': 'request',
                            'node': self.node,
                            'target': method_target(type(model.session), 'create_message'),
                            'args': normalized(call.args),
                            'kwargs': normalized(call.kwargs),
                        }
                        for call in create_message.call_args_list[mock_calls:]
                    )
            record['response'] = normalized(response)
            return response

        self._monkeypatch.setattr(MCPSamplingModel, 'request', recording_request)

    def _wrap_raises(self) -> None:
        """Record each exception `pytest.raises` or `pytest.RaisesGroup` catches, which both keep with `fill_unfilled`."""
        original: Callable[..., None] = vars(pytest.ExceptionInfo)['fill_unfilled']

        @functools.wraps(original)
        def recording_fill_unfilled(
            excinfo: pytest.ExceptionInfo[BaseException],
            exc_info: tuple[type[BaseException], BaseException, TracebackType],
        ) -> None:
            original(excinfo, exc_info)
            self.records.append({'kind': 'exception', 'node': self.node, 'exception': described_exception(exc_info[1])})

        self._monkeypatch.setattr(pytest.ExceptionInfo, 'fill_unfilled', recording_fill_unfilled)

    def _wrap_warnings(self) -> None:
        """Record each warning the filters let through, which `warnings` shows, or records, with `_showwarnmsg`."""
        original: Callable[[warnings.WarningMessage], None] = vars(warnings)['_showwarnmsg']

        @functools.wraps(original)
        def recording_showwarnmsg(message: warnings.WarningMessage) -> None:
            # One raised while no test runs, like an import during collection, belongs to no test node. A finalizer
            # raises a `ResourceWarning` whenever garbage collection reaches it, which the allocations made decide.
            if self.node is not None and not issubclass(message.category, ResourceWarning):
                self.records.append(
                    {
                        'kind': 'warning',
                        'node': self.node,
                        'category': qualified_name(message.category),
                        'message': readable_text(message.message),
                        'frame': self._reported_frame(message),
                    }
                )
            original(message)

        self._monkeypatch.setattr(warnings, '_showwarnmsg', recording_showwarnmsg)

    def _reported_frame(self, message: warnings.WarningMessage) -> dict[str, object]:
        """The module and function of the frame on the stack `message` reports, and its line outside the workspace.

        A line in a workspace package moves with each edit above it, so a refactor would move it. A frame the stack
        doesn't hold, like the one `warnings.warn_explicit` is handed, is reported by its file and line.
        """
        location = (message.filename, message.lineno)
        frame = inspect.currentframe()
        try:
            while frame is not None and (frame.f_code.co_filename, frame.f_lineno) != location:
                frame = frame.f_back
            if frame is None:
                return {'file': normalized(message.filename), 'line': message.lineno}
            module = str(frame.f_globals.get('__name__'))
            reported: dict[str, object] = {'module': module, 'function': frame.f_code.co_qualname}
            if module.partition('.')[0] not in self._workspace_packages:
                reported['line'] = message.lineno
            return reported
        finally:
            # A frame that holds itself would be a reference cycle.
            del frame

    def wrap_on_import(self, wraps: dict[str, Callable[[], None]]) -> None:
        """Call each module's wrap now if the module has run, and otherwise once it first runs."""
        for name in [*wraps]:
            if name in sys.modules:
                wraps.pop(name)()
        self._import_watcher.wraps = wraps
        sys.meta_path.insert(0, self._import_watcher)

    def _recorded_registration(
        self,
        target: str,
        decorator: Callable[..., object],
        registered_name: Callable[[object], object],
        *,
        owner_args: int = 0,
        unrecorded: Callable[[], bool] = lambda: False,
    ) -> Callable[..., object]:
        """`decorator`, recording the name each call `pydantic_ai.durable_exec` makes on it registers, and the options
        it passes, unless `unrecorded()` says otherwise.

        The first `owner_args` positional arguments are what `decorator`, a method, is bound to. A call handed what it
        decorates registers it, and one handed only options returns the decorator that does.
        """

        @functools.wraps(decorator)
        def recording(*args: object, **kwargs: object) -> object:
            result = decorator(*args, **kwargs)
            if not calling_module().startswith(DURABLE_EXEC_PACKAGE) or unrecorded():
                return result
            node = self.node or self.collecting
            options = recorded_value({key: value for key, value in kwargs.items() if key != 'name'})

            def record(registered: object) -> object:
                self._durable_names.append(
                    {
                        'kind': 'durable_name',
                        'node': node,
                        'target': target,
                        'name': registered_name(registered),
                        'options': options,
                    }
                )
                return registered

            if len(args) > owner_args:
                return record(result)
            assert callable(result)
            register: Callable[[object], object] = result

            def recording_register(decorated: object) -> object:
                return record(register(decorated))

            return recording_register

        return recording

    def _wrap_temporal_activity_defn(self) -> None:
        from temporalio import activity

        def registered_name(function: object) -> object:
            return getattr(getattr(function, '__temporal_activity_definition'), 'name')

        recording = self._recorded_registration(
            'temporalio.activity.defn', activity.defn, registered_name, unrecorded=in_workflow_sandbox
        )
        self._monkeypatch.setattr(activity, 'defn', recording)

    def _wrap_temporal_workflow_defn(self) -> None:
        from temporalio import workflow

        def registered_name(cls: object) -> object:
            return getattr(getattr(cls, '__temporal_workflow_definition'), 'name')

        recording = self._recorded_registration(
            'temporalio.workflow.defn', workflow.defn, registered_name, unrecorded=in_workflow_sandbox
        )
        self._monkeypatch.setattr(workflow, 'defn', recording)

    def _wrap_temporal_worker(self) -> None:
        """Record the commands each workflow activation a Temporal worker completes issues, per workflow run."""
        from google.protobuf.json_format import MessageToDict
        from temporalio.bridge.proto.workflow_activation import WorkflowActivation
        from temporalio.bridge.proto.workflow_completion import WorkflowActivationCompletion
        from temporalio.bridge.worker import Worker

        poll: Callable[[Worker], Awaitable[WorkflowActivation]] = vars(Worker)['poll_workflow_activation']
        complete: Callable[[Worker, WorkflowActivationCompletion], Awaitable[None]] = vars(Worker)[
            'complete_workflow_activation'
        ]

        @functools.wraps(poll)
        async def recording_poll(worker: Worker) -> WorkflowActivation:
            activation = await poll(worker)
            for job in activation.jobs:
                if job.HasField('initialize_workflow'):
                    started = job.initialize_workflow
                    self._workflow_steps(
                        self._temporal_runs,
                        activation.run_id,
                        {
                            'kind': 'temporal',
                            'node': self.node,
                            'target': started.workflow_type,
                            'workflow_id': started.workflow_id,
                        },
                        'commands',
                    )
            return activation

        @functools.wraps(complete)
        async def recording_complete(worker: Worker, completion: WorkflowActivationCompletion) -> None:
            if completion.HasField('failed'):
                commands = [{'failed': MessageToDict(completion.failed, preserving_proto_field_name=True)}]
            else:
                commands = [
                    MessageToDict(command, preserving_proto_field_name=True)
                    for command in completion.successful.commands
                ]
            # An activation that issues nothing, like the one that evicts a run from the worker's cache once it ended,
            # is left out.
            if commands:
                steps = self._workflow_steps(
                    self._temporal_runs,
                    completion.run_id,
                    {'kind': 'temporal', 'node': self.node, 'target': None},
                    'commands',
                )
                steps.extend(recorded_value(command, decoded_payloads) for command in commands)
            return await complete(worker, completion)

        self._monkeypatch.setattr(Worker, 'poll_workflow_activation', recording_poll)
        self._monkeypatch.setattr(Worker, 'complete_workflow_activation', recording_complete)

    def _workflow_steps(
        self,
        workflows: dict[str, tuple[dict[str, object], list[object]]],
        workflow_id: str,
        record: dict[str, object],
        steps: str,
    ) -> list[object]:
        """The list `steps` of the record of the workflow with `workflow_id` in `workflows`, made from `record` if the
        test hasn't recorded it yet.
        """
        if (known := workflows.get(workflow_id)) is None:
            listed: list[object] = []
            known = workflows[workflow_id] = (
                {**{key: normalized(value) for key, value in record.items()}, steps: listed},
                listed,
            )
        return known[1]

    def _wrap_dbos_decorators(self) -> None:
        # The `dbos` package is still running its `__init__` when the module it imports `DBOS` from has run.
        from dbos._dbos import DBOS
        from dbos._registrations import get_dbos_func_name

        for name in ('step', 'workflow'):
            original: classmethod[DBOS, ..., object] = vars(DBOS)[name]
            recording = self._recorded_registration(
                f'dbos.DBOS.{name}', original.__func__, get_dbos_func_name, owner_args=1
            )
            self._monkeypatch.setattr(DBOS, name, classmethod(recording))

    def _wrap_dbos_system_database(self) -> None:
        """Record each workflow DBOS starts or forks, with the operations it checks for a recorded result, in order."""
        from dbos._sys_db import SystemDatabase

        insert: Callable[..., object] = vars(SystemDatabase)['_insert_workflow_status']
        check: Callable[..., object] = vars(SystemDatabase)['_check_operation_execution_txn']
        insert_signature, check_signature = inspect.signature(insert), inspect.signature(check)

        @functools.wraps(insert)
        def recording_insert(database: SystemDatabase, *args: object, **kwargs: object) -> object:
            status: Mapping[str, object] = insert_signature.bind(database, *args, **kwargs).arguments['status']
            record: dict[str, object] = {
                'kind': 'dbos',
                'node': self.node,
                'target': status.get('name'),
                'workflow_id': status.get('workflow_uuid'),
                'class_name': status.get('class_name'),
                'config_name': status.get('config_name'),
                'parent_workflow_id': status.get('parent_workflow_id'),
            }
            self._workflow_steps(self._dbos_workflows, str(status.get('workflow_uuid')), record, 'operations')
            return insert(database, *args, **kwargs)

        @functools.wraps(check)
        def recording_check(database: SystemDatabase, *args: object, **kwargs: object) -> object:
            arguments = check_signature.bind(database, *args, **kwargs).arguments
            workflow_id = str(arguments['workflow_id'])
            operation: dict[str, object] = {
                'function_id': arguments['function_id'],
                'function_name': arguments['function_name'],
            }
            unknown: dict[str, object] = {'kind': 'dbos', 'node': self.node, 'target': None, 'workflow_id': workflow_id}
            self._workflow_steps(self._dbos_workflows, workflow_id, unknown, 'operations').append(operation)
            # A check that raises, like on a step that isn't the one recorded, leaves its operation without `recorded`;
            # the error reaches the test, as an exception record when it catches it.
            result = check(database, *args, **kwargs)
            operation['recorded'] = result is not None
            return result

        fork: Callable[..., object] = vars(SystemDatabase)['fork_workflow']
        fork_signature = inspect.signature(fork)

        @functools.wraps(fork)
        def recording_fork(database: SystemDatabase, *args: object, **kwargs: object) -> object:
            # A fork's status row takes its original's name, class and config, so its record takes them from the
            # original's, when the test started it. It's made before the fork, which can run before this returns.
            arguments = fork_signature.bind(database, *args, **kwargs).arguments
            original_ids: list[str] = arguments['original_workflow_ids']
            forked_ids: list[str] = arguments['forked_workflow_ids']
            for original_id, forked_id in zip(original_ids, forked_ids):
                original = self._dbos_workflows.get(original_id, ({}, []))[0]
                record: dict[str, object] = {
                    'kind': 'dbos',
                    'node': self.node,
                    'target': original.get('target'),
                    'workflow_id': forked_id,
                    'class_name': original.get('class_name'),
                    'config_name': original.get('config_name'),
                    'parent_workflow_id': None,
                    'forked_from': original_id,
                }
                self._workflow_steps(self._dbos_workflows, forked_id, record, 'operations')
            return fork(database, *args, **kwargs)

        self._monkeypatch.setattr(SystemDatabase, '_insert_workflow_status', recording_insert)
        self._monkeypatch.setattr(SystemDatabase, '_check_operation_execution_txn', recording_check)
        self._monkeypatch.setattr(SystemDatabase, 'fork_workflow', recording_fork)

    def _wrap_prefect_task(self) -> None:
        # `from prefect import task` reads it from `prefect.main` each time.
        import prefect.main

        original = prefect.main.task
        recording = self._recorded_registration('prefect.task', original, registered_attribute_name)
        self._monkeypatch.setattr(prefect.main, 'task', recording)
        # A module imported before the capture started holds the decorator it imported.
        for name, module in [*sys.modules.items()]:
            if name.startswith(DURABLE_EXEC_PACKAGE) and vars(module).get('task') is original:
                self._monkeypatch.setattr(module, 'task', recording)

    def _wrap_prefect_flow(self) -> None:
        # `prefect.flow` is a `FlowDecorator`, whose other methods, like `from_source`, stay as they are.
        from prefect.flows import FlowDecorator

        original: Callable[..., object] = vars(FlowDecorator)['__call__']
        recording = self._recorded_registration('prefect.flow', original, registered_attribute_name, owner_args=1)
        self._monkeypatch.setattr(FlowDecorator, '__call__', recording)

    def _wrap_cache_policies(self) -> None:
        from prefect.cache_policies import CachePolicy, RunId

        self._wrap_subclasses(CachePolicy, functools.partial(self._wrap_compute_key, run_id_policy=RunId))

    def _wrap_compute_key(self, cls: type[object], run_id_policy: type[object]) -> None:
        """Record each `compute_key` call on a Prefect cache policy, with the inputs it's handed and the key it returns.

        A call made in another, like the one a compound policy makes on each policy it combines, is recorded after it,
        and one handed the same inputs leaves them out. A key that holds a value each run draws anew is recorded as
        `RUN_SCOPED_KEY`, and so is each key computed from it: one the flow run's ID went into, and one computed from
        inputs the call records that hold such a value, like a generated ID. A policy handed the inputs its caller was,
        like `TaskSource`, which hashes the task's code, keeps its key.
        """
        if 'compute_key' not in vars(cls):
            return
        original: Callable[..., object] = vars(cls)['compute_key']
        signature = inspect.signature(original)

        @functools.wraps(original)
        def recording_compute_key(policy: object, *args: object, **kwargs: object) -> object:
            inputs = signature.bind(policy, *args, **kwargs).arguments.get('inputs')
            outer = _COMPUTING_CACHE_KEY.get()
            record: dict[str, object] = {
                'kind': 'cache_key',
                'node': self.node,
                'target': method_target(type(policy), 'compute_key'),
            }
            call = CacheKeyCall(inputs, [] if outer is None else outer.records, outer)
            if outer is None or inputs is not outer.inputs:
                record['inputs'] = recorded_value(inputs)
                call.run_scoped = holds_run_values(inputs, record['inputs'])
            call.records.append(record)
            token = _COMPUTING_CACHE_KEY.set(call)
            try:
                key = original(policy, *args, **kwargs)
            except Exception as e:
                record['error'] = error_text(e)
                raise
            else:
                if call.run_scoped or (isinstance(policy, run_id_policy) and key is not None):
                    scoped: CacheKeyCall | None = call
                    while scoped is not None:
                        scoped.run_scoped, scoped = True, scoped.outer
                # A key is a digest, which no normalization may touch.
                record['key'] = RUN_SCOPED_KEY if call.run_scoped else key
                return key
            finally:
                _COMPUTING_CACHE_KEY.reset(token)
                # The calls a policy makes on the ones it combines are written with it, as one.
                if outer is None:
                    self._cache_key_calls.append(call.records)

        self._monkeypatch.setattr(cls, 'compute_key', recording_compute_key)


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
    return f'{qualified_name(owner)}.{name}'


def qualified_name(cls: type) -> str:
    return f'{cls.__module__}.{cls.__qualname__}'


def calling_module() -> str:
    """The module of the function that called the function calling this one."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back.f_back if frame is not None and frame.f_back is not None else None
        return '' if caller is None else str(caller.f_globals.get('__name__'))
    finally:
        # A frame that holds itself would be a reference cycle.
        del frame


def in_workflow_sandbox() -> bool:
    """Whether Temporal's workflow sandbox runs this, which imports a workflow's module again for each run, and with it
    the agents the module defines.
    """
    from temporalio import workflow

    return workflow.unsafe.in_sandbox()


def readable_text(value: object, prefix: str = '') -> object:
    """`prefix` and `str(value)`, normalized, or, when `value`'s `__str__` raises, `unrecorded` with the class of the
    error it raised, which is never raised into the code under test.
    """
    try:
        text = f'{prefix}{value}'
    except Exception as e:
        return {'unrecorded': f'{prefix}`str()` raised {type(e).__name__}'}
    return normalized(text)


def error_text(e: BaseException) -> object:
    """`e`'s class and message, as `readable_text` records them."""
    return readable_text(e, f'{type(e).__name__}: ')


def recorded_value(value: object, decode: Callable[[object], object] | None = None) -> object:
    """`value`, decoded by `decode` if given, normalized, or the error decoding or normalizing it raised, like on a
    reference cycle, which is never raised into the code under test.
    """
    try:
        return normalized(value if decode is None else decode(value))
    except Exception as e:
        return {'unrecorded': error_text(e)}


def built_response(stream: StreamedResponse) -> object:
    """The response `stream` builds, or the error building it raised, which is left to the consumer, who reads it
    again, rather than raised into the code under test. It's normalized when the test ends, so what the run sets on it
    once iteration ended, like its usage's cost, shows.
    """
    try:
        return stream.get()
    except Exception as e:
        return {'unrecorded': error_text(e)}


def holds_run_values(value: object, recorded: object) -> bool:
    """Whether normalizing `value` into `recorded` replaced a value that changes between runs, like a generated ID.

    One that can't be written as JSON is taken not to.
    """
    try:
        return json.loads(to_json(value, fallback=repr, bytes_mode='base64')) != recorded
    except Exception:
        return False


def registered_attribute_name(registered: object) -> object:
    """The name of a Prefect task or flow, which is what it's registered under."""
    return getattr(registered, 'name')


def decoded_payloads(value: object) -> object:
    """`value`, a Temporal command as `MessageToDict` gives it, with each payload's metadata and data decoded and the
    stack traces of failures left out, since a refactor moves the lines they name.

    Data encoded as JSON is parsed; other data stays as `MessageToDict` gives it, base64 encoded.
    """
    if isinstance(value, list):
        items: list[object] = [*value]
        return [decoded_payloads(item) for item in items]
    if not isinstance(value, dict):
        return value
    fields: dict[str, object] = {**value}
    metadata = fields.get('metadata')
    if isinstance(metadata, dict) and 'encoding' in metadata:
        encoded: dict[str, object] = {**metadata}
        decoded = {key: base64.b64decode(str(item)).decode(errors='backslashreplace') for key, item in encoded.items()}
        data = base64.b64decode(str(fields.get('data', '')))
        return {
            **fields,
            'metadata': decoded,
            'data': sorted_sets(decoded_stream_items(json.loads(data)))
            if decoded['encoding'] == 'json/plain'
            else fields.get('data'),
        }
    return {key: decoded_payloads(item) for key, item in fields.items() if key != 'stack_trace'}


def decoded_stream_items(value: object) -> object:
    """`value`, a payload's JSON data, with each item a Temporal workflow stream publishes decoded like a payload.

    The stream sends an item as its topic, its offset, and its payload serialized and base64 encoded.
    """
    if isinstance(value, list):
        items: list[object] = [*value]
        return [decoded_stream_items(item) for item in items]
    if not isinstance(value, dict):
        return value
    fields: dict[str, object] = {**value}
    if fields.keys() == {'topic', 'data', 'offset'}:
        from google.protobuf.json_format import MessageToDict
        from temporalio.api.common.v1 import Payload

        item = Payload.FromString(base64.b64decode(str(fields['data'])))
        return {**fields, 'data': decoded_payloads(MessageToDict(item, preserving_proto_field_name=True))}
    return {key: decoded_stream_items(item) for key, item in fields.items()}


def sorted_sets(value: object) -> object:
    """`value`, a payload's JSON data, with each set a serialized run context holds sorted.

    The run context sends each as a list in the set's order, which differs from run to run where its members end in the
    random suffix of a capability's key, and reads it back as a set.
    """
    if isinstance(value, list):
        items: list[object] = [*value]
        return [sorted_sets(item) for item in items]
    if not isinstance(value, dict):
        return value
    fields: dict[str, object] = {**value}
    for key, item in fields.items():
        if key in RUN_CONTEXT_SETS and isinstance(item, list):
            members: list[object] = [*item]
            fields[key] = sorted(map(str, members))
        else:
            fields[key] = sorted_sets(item)
    return fields


def workspace_roots() -> list[Path]:
    """The root of each workspace package, relative to the repository, which its editable install puts on `sys.path`."""
    venv = Path(sys.prefix).resolve()
    entries = {Path(entry).resolve() for entry in sys.path if entry}
    return sorted(
        entry.relative_to(REPO)
        for entry in entries
        if entry.is_relative_to(REPO) and entry != REPO and not entry.is_relative_to(venv)
    )


def described_exception(exception: BaseException, chain: frozenset[int] = frozenset()) -> dict[str, object]:
    """`exception`'s class, message and notes, and, described alike, the exceptions it groups, its cause and context.

    A context that is the cause is described as `'cause'`. An exception met again along the chain, which a chain set by
    hand can loop back to, is described without its own.
    """
    description: dict[str, object] = {'type': qualified_name(type(exception)), 'message': readable_text(exception)}
    if (notes := getattr(exception, '__notes__', None)) is not None:
        description['notes'] = normalized(notes)
    if id(exception) in chain:
        return description
    chain |= {id(exception)}
    if isinstance(exception, BaseExceptionGroup):
        # `isinstance` can't tell pyright what the group holds, which is always exceptions.
        group: BaseExceptionGroup[BaseException] = exception  # pyright: ignore[reportUnknownVariableType]
        description['exceptions'] = [described_exception(member, chain) for member in group.exceptions]
    if exception.__cause__ is not None:
        description['cause'] = described_exception(exception.__cause__, chain)
    if (context := exception.__context__) is not None:
        description['context'] = 'cause' if context is exception.__cause__ else described_exception(context, chain)
    if exception.__suppress_context__:
        description['suppress_context'] = True
    return description


def is_realtime_double_module(name: str) -> bool:
    """Whether module `name` is in a package `REALTIME_DOUBLE_PACKAGES` lists."""
    return any(name == package or name.startswith(f'{package}.') for package in REALTIME_DOUBLE_PACKAGES)


def running_task_id() -> int:
    """The `id` of the asyncio task running, or of `None` outside one."""
    try:
        return id(asyncio.current_task())
    except RuntimeError:
        return id(None)


def recorded(value: object) -> object:
    """`value` normalized, with each object in it, or in a list, tuple or set it is, tagged with its class.

    A set is sorted, so two equal sets, like the `Owed` tokens `SessionCore.wait_tokens` returns, are recorded alike.
    """
    if isinstance(value, list | tuple):
        items: list[object] = [*value]
        return [recorded(item) for item in items]
    if isinstance(value, Set):
        members: list[object] = [*value]
        return sorted((recorded(member) for member in members), key=json.dumps)
    if value is None or isinstance(value, bool | int | float | str):
        return normalized(value)
    return tagged(value)


def tagged(value: object) -> object:
    """`value` normalized, with its class named, which no field of a realtime event or input names."""
    if isinstance(value, StaleEvent):
        return {'type': 'StaleEvent', 'event': tagged(value.event)}
    return {'type': type(value).__qualname__, 'value': normalized(value)}


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

    def generated_local_timestamp(match: re.Match[str]) -> str:
        try:
            timestamp = datetime.fromisoformat(match[0])
        except ValueError:
            return match[0]
        return 'NOW' if abs(timestamp - datetime.now()) < timedelta(days=1) else match[0]

    def generated_epoch_milliseconds(match: re.Match[str]) -> str:
        timestamp = datetime.fromtimestamp(int(match[1]) / 1000, UTC)
        return '"timestamp":"NOW"' if abs(timestamp - now) < timedelta(days=1) else match[0]

    # Split only between two quoted strings, so a value with a comma in it, like an HTTP date, stays whole.
    def sorted_headers(match: re.Match[str]) -> str:
        return "'HTTPHeaders': {" + ', '.join(sorted(re.split(r"(?<='), (?=')", match[1]))) + '}'

    text = ISO_DATETIME.sub(generated_timestamp, to_json(value, fallback=repr, bytes_mode='base64').decode())
    text = DATETIME_STR.sub(generated_local_timestamp, text)
    text = EPOCH_MILLISECONDS.sub(generated_epoch_milliseconds, text)
    text = BOTOCORE_ERROR_HEADERS.sub(sorted_headers, text)
    text = SUPPORTED_NATIVE_TOOLS.sub(lambda match: f'{match[1]}{", ".join(sorted(match[2].split(", ")))}]', text)
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
    # A test that fails records only what it did before it failed, so it proves nothing even when both sides match.
    for side, capture_dir in (('base', base_dir), ('head', head_dir)):
        if failed := sorted(
            {node for path in capture_dir.glob(f'*{FAILED_SUFFIX}') for node in path.read_text().split()}
        ):
            print(f'nodes that failed, {side}: {len(failed)}')
            for node in failed:
                print(f'  {node}')
    base, head = _load(base_dir), _load(head_dir)
    if empty := [d for d, nodes in ((base_dir, base), (head_dir, head)) if not nodes]:
        print(
            f'Nothing recorded in {", ".join(map(str, empty))}; the selected tests may not stream, send requests, '
            'run a UI adapter, a realtime connection or session or a tool hook, catch an exception, warn, end a span, '
            'or run durable execution code.',
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
    print(f'realtime records: head {sum(record["kind"] == "realtime" for record in head_records)}')
    print(f'session records: head {sum(record["kind"] == "session" for record in head_records)}')
    print(f'hook records: head {sum(record["kind"] == "hook" for record in head_records)}')
    print(f'run event stream records: head {sum(record["kind"] == "run_event_stream" for record in head_records)}')
    print(f'agent stream records: head {sum(record["kind"] == "agent_stream" for record in head_records)}')
    print(f'span records: head {sum(record["kind"] == "span" for record in head_records)}')
    print(f'exception records: head {sum(record["kind"] == "exception" for record in head_records)}')
    print(f'warning records: head {sum(record["kind"] == "warning" for record in head_records)}')
    for kind, label in DURABLE_KINDS.items():
        print(f'{label} records: head {sum(record["kind"] == kind for record in head_records)}')

    modules: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    ui_targets: Counter[str] = Counter()
    realtime_targets: Counter[str] = Counter()
    session_targets: Counter[str] = Counter()
    hook_targets: Counter[str] = Counter()
    run_events: Counter[str] = Counter()
    agent_stream_targets: Counter[str] = Counter()
    span_names: Counter[str] = Counter()
    durable_targets: dict[str, Counter[str]] = {kind: Counter() for kind in DURABLE_KINDS}
    for records in head_nodes.values():
        modules.update({module for record in records for module in record.get('modules', [])})
        targets.update({record['target'] for record in records if record['kind'] == 'request'})
        ui_targets.update({record['target'] for record in records if record['kind'] == 'ui'})
        realtime_targets.update({record['target'] for record in records if record['kind'] == 'realtime'})
        session_targets.update({record['target'] for record in records if record['kind'] == 'session'})
        hook_targets.update({record['target'] for record in records if record['kind'] == 'hook'})
        run_events.update(
            {
                event['event_kind']
                for record in records
                if record['kind'] == 'run_event_stream'
                for event in record['events']
            }
        )
        agent_stream_targets.update({record['target'] for record in records if record['kind'] == 'agent_stream'})
        span_names.update({record['name'] for record in records if record['kind'] == 'span'})
        for kind, counts in durable_targets.items():
            counts.update({str(record['target']) for record in records if record['kind'] == kind})
    for label, counts in (
        ('stream module', modules),
        ('request target', targets),
        ('ui target', ui_targets),
        ('realtime target', realtime_targets),
        ('session target', session_targets),
        ('hook target', hook_targets),
        ('run event kind', run_events),
        ('agent stream target', agent_stream_targets),
        ('span name', span_names),
        *((f'{DURABLE_KINDS[kind]} target', counts) for kind, counts in durable_targets.items()),
    ):
        print(f'nodes per {label}, head:')
        for name, count in sorted(counts.items()):
            print(f'  {name}: {count}')
    ran = {node for path in head_dir.glob(f'*{NODES_SUFFIX}') for node in path.read_text().splitlines()}
    requesting = {
        node
        for node, records in head_nodes.items()
        if any(record['kind'] in ('request', 'realtime', 'session') for record in records)
    }
    print(f'nodes that recorded no request, realtime or session record, head: {len(ran - requesting)}')
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

    roots = workspace_roots()
    # DBOS resolves a workflow function's file with `os.path.realpath`, which reads each symlink on the way with
    # `os.readlink`, and blockbuster raises on that in the event loop. On macOS, the temporary directory is reached
    # through `/var`, a symlink, so only the base side's sources would be.
    out = Path(tempfile.mkdtemp(prefix='refactor-capture-')).resolve()
    archive = subprocess.run(
        ['git', 'archive', args.base, '--', *map(str, roots)], cwd=REPO, check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out / 'base-src', filter='data')

    failed = False
    for side, sources in (('base', [out / 'base-src' / root for root in roots]), ('head', [])):
        pythonpath = os.pathsep.join([*map(str, sources), *filter(None, [os.environ.get('PYTHONPATH')])])
        command = ['-m', 'pytest', '-p', 'tests.refactor_capture', f'{CAPTURE_OPTION}={out / side}', *args.pytest_args]
        env = os.environ | {'PYTHONPATH': pythonpath, 'PYTHONHASHSEED': '0'}
        code = subprocess.run([sys.executable, *command], cwd=REPO, env=env).returncode
        # Flushed, so a run stopped from outside, like by `timeout`, still says which side finished.
        print(f'{side} capture: pytest exited {code}', flush=True)
        failed = failed or code != 0
    print(f'captures: {out}')
    # Tests that fail alike on both sides compare equal, but they don't prove the refactor.
    return compare(out / 'base', out / 'head') or int(failed)


if __name__ == '__main__':
    sys.exit(main())
