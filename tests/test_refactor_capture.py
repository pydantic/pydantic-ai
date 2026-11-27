"""The opt-in capture in `tests/refactor_capture.py`, mostly run against small in-process pytest sessions.

They test the capture tooling rather than model behavior, so no recording applies.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import warnings
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeAlias
from unittest.mock import Mock

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.direct import model_request_stream
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models import StreamedResponse, override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime._core import Owed

from .batch_utils import FAST_BATCH_SETTINGS
from .conftest import IsStr, try_import

with try_import() as imports_successful:
    import httpx
    import httpx2
    from anthropic import AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.ui import UIAdapter
    from pydantic_ai.ui.vercel_ai import VercelAIEventStream

    from . import refactor_capture
    from .refactor_capture import (
        CAPTURE_OPTION,
        SDK_MOCKS,
        SOURCE_FILE,
        ImportWatcher,
        RefactorCapture,
        bedrock_canary_model,
        compare,
        main,
        normalized,
        recorded,
    )

# Imported before the `REALTIME` session's capture starts, so the capture finds `ReplayWebSocket` among the classes
# already defined.
from .realtime import ws_cassettes

pytest_plugins = ['pytester']

pytestmark = [
    # The plugin imports `httpx`, `groq`, `anthropic`, `boto3`, `starlette` and `mcp`; the `REQUESTS` sessions import
    # `openai` too, and the `UI` sessions `ag_ui`.
    pytest.mark.skipif(
        not imports_successful() or any(importlib.util.find_spec(name) is None for name in ('openai', 'ag_ui')),
        reason='`httpx`, `groq`, `anthropic`, `boto3`, `starlette`, `mcp`, `openai` or `ag_ui` not installed',
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
import threading
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
    # A thread an earlier test left running can draw from `random` after the capture seeds it, as the keepalive ping of
    # a `websockets` connection does.
    pinging = threading.Thread(target=random.getrandbits, args=(32,))
    pinging.start()
    pinging.join()
    content = dict(json=dict(text={text!r}, pick=random.randrange(1000)), headers={{'xi-api-key': 'secret'}})
    asyncio.run(post(httpx, **content))


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

# The sessions run on this process's main thread, where `run_sync` installs an event loop and leaves it open for the
# next sync call, and the next `asyncio.run()` replaces it. Left to the garbage collector, a replaced loop is
# finalized in this test's teardown, where its self-pipe socket can be finalized before it and fail the test with
# `ValueError: Invalid file descriptor: -1`. So each session test closes the loop it leaves behind.
CONFTEST = """
import asyncio
import warnings

import pytest


@pytest.fixture(autouse=True)
def close_the_event_loop_left_behind():
    yield
    with warnings.catch_warnings():
        # When none was set, Python 3.11 to 3.13 create a loop to return here (3.12 and 3.13 also warn), and 3.14
        # raises.
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
    loop.close()
    asyncio.set_event_loop(None)
"""

UI = """
import asyncio

import pytest
from ag_ui.core import RunAgentInput, UserMessage
from starlette.testclient import TestClient

from pydantic_ai import Agent
from pydantic_ai.messages import ImageUrl, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.ui import UIAdapter
from pydantic_ai.ui._web import create_web_app
from pydantic_ai.ui.ag_ui import AGUIAdapter
from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
from pydantic_ai.ui.vercel_ai.request_types import FileUIPart, SubmitMessage, TextUIPart, UIMessage
from pydantic_ai.ui.vercel_ai.response_types import StartChunk

AGENT = Agent(TestModel(custom_output_text={text!r}))
HISTORY = [
    ModelRequest(parts=[UserPromptPart([{text!r}, ImageUrl('https://example.com/a.png', media_type='image/png')])]),
    ModelResponse(parts=[TextPart('hi')]),
]


def vercel_ai_input():
    return SubmitMessage(
        id='request', messages=[UIMessage(id='message', role='user', parts=[TextUIPart(text={text!r})])]
    )


async def drain(adapter):
    async for _ in adapter.run_stream():
        pass


def test_vercel_ai_messages():
    VercelAIAdapter.load_messages(
        [
            UIMessage(
                id='message',
                role='user',
                parts=[TextUIPart(text={text!r}), FileUIPart(url='https://example.com/a.png', media_type='image/png')],
            )
        ]
    )
    VercelAIAdapter.dump_messages(HISTORY)


def test_ag_ui_messages():
    AGUIAdapter.load_messages([UserMessage(id='message', content={text!r})])
    AGUIAdapter.dump_messages(HISTORY)


def test_vercel_ai_events():
    asyncio.run(drain(VercelAIAdapter(agent=AGENT, run_input=vercel_ai_input())))


def test_ag_ui_events():
    run_input = RunAgentInput(
        thread_id='thread',
        run_id='run',
        messages=[UserMessage(id='message', content={text!r})],
        state={{}},
        context=[],
        tools=[],
        forwarded_props=None,
    )
    asyncio.run(drain(AGUIAdapter(agent=AGENT, run_input=run_input)))


class ClosingEventStream(VercelAIEventStream[None, str]):
    closed = False

    async def transform_stream(self, stream, on_complete=None, on_cancel=None):
        try:
            async for event in super().transform_stream(stream, on_complete, on_cancel):
                yield event
        finally:
            ClosingEventStream.closed = True


async def close_after_the_first_event():
    events = ClosingEventStream(vercel_ai_input()).transform_stream(AGENT.run_stream_events({text!r}))
    await anext(events)
    await events.aclose()
    assert ClosingEventStream.closed


def test_subclass_defined_after_the_capture_started():
    asyncio.run(close_after_the_first_event())


class FailingEventStream(VercelAIEventStream[None, str]):
    async def transform_stream(self, stream, on_complete=None, on_cancel=None):
        yield StartChunk()
        raise RuntimeError('client went away')


async def no_events():
    return
    yield


async def drain_failing():
    async for _ in FailingEventStream(vercel_ai_input()).transform_stream(no_events()):
        pass


def test_errors():
    with pytest.raises(RuntimeError, match='client went away'):
        asyncio.run(drain_failing())
    with pytest.raises(NotImplementedError):
        UIAdapter.dump_messages([])


def test_web_api():
    with TestClient(create_web_app(AGENT), base_url='http://127.0.0.1:7932') as client:
        client.get('/api/health')
        client.post('/api/chat', json=vercel_ai_input().model_dump(by_alias=True))


class MediaTypeAdapter(VercelAIAdapter):
    @classmethod
    def load_messages(cls, messages):
        # Set to the media type the URL's extension implies, or left to be implied: equal, and serialized alike.
        media_type = 'image/png' if {text!r} == 'hello' else None
        return [ModelRequest(parts=[UserPromptPart([ImageUrl('https://example.com/a.png', media_type=media_type)])])]


def test_media_type_set_or_implied():
    MediaTypeAdapter.load_messages([])
"""

REALTIME = """
import asyncio
import json
from contextlib import aclosing

import pytest

from pydantic_ai.realtime.codec import OutputTranscript, RealtimeConnection
from pydantic_ai.realtime.google import GoogleRealtimeConnection
from pydantic_ai.realtime.lifecycle import ResponseStarted, StaleEvent
from pydantic_ai.realtime.openai import OpenAIRealtimeConnection

from tests.realtime.ws_cassettes import CassetteMessage, RealtimeCassette, ReplayWebSocket


class Socket:
    close_code = 1000
    close_reason = ''

    def __init__(self, *frames):
        self.frames = [json.dumps(frame) for frame in frames]

    async def send(self, data):
        pass

    async def __aiter__(self):
        for frame in self.frames:
            yield frame


class WrappingSocket(Socket):
    async def send(self, data):
        await super().send(data)


class BrokenSocket(Socket):
    async def send(self, data):
        raise ConnectionError('socket closed')


async def openai_turn():
    connection = OpenAIRealtimeConnection(
        WrappingSocket(
            dict(type='response.created', event_id='event-1', response=dict(id='resp-1')),
            dict(
                type='response.output_text.delta',
                event_id='event-2',
                response_id='resp-1',
                item_id='item-1',
                output_index=0,
                content_index=0,
                delta={text!r},
            ),
        )
    )
    await connection.send('hi')
    [event async for event in connection]


def test_openai_turn():
    asyncio.run(openai_turn())


def test_broken_socket():
    with pytest.raises(ConnectionError, match='socket closed'):
        asyncio.run(OpenAIRealtimeConnection(BrokenSocket()).send('hi'))


class Session:
    async def send_client_content(self, *, turns=None, turn_complete=True):
        pass


def test_google_session():
    asyncio.run(GoogleRealtimeConnection(Session()).send({text!r}))


async def replay():
    frame = dict(type='session.update')
    await ReplayWebSocket(RealtimeCassette(interactions=[CassetteMessage(direction='sent', data=frame)])).send(
        json.dumps(frame)
    )


def test_cassette_replay():
    asyncio.run(replay())


class Scripted(RealtimeConnection):
    closed = False

    async def send(self, content):
        return [ResponseStarted(response_id='r1')]

    async def frames(self):
        try:
            yield [ResponseStarted(response_id='r1'), StaleEvent(event=OutputTranscript(text={text!r}))]
            raise RuntimeError('provider went away')
        finally:
            Scripted.closed = True


class Chained(Scripted):
    async def send(self, content):
        return await super().send(content)


class ListedFrames:
    def __init__(self):
        self.frames = [[OutputTranscript(text='listed')]]

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop()


class Listed(Scripted):
    def frames(self):
        return ListedFrames()


async def scripted():
    assert await Chained().send('hi') == [ResponseStarted(response_id='r1')]
    async with aclosing(Chained().frames()) as frames:
        await anext(frames)
    assert Scripted.closed
    with pytest.raises(RuntimeError, match='provider went away'):
        [event async for event in Scripted()]
    [frame async for frame in Listed().frames()]


def test_connection_defined_after_the_capture_started():
    asyncio.run(scripted())
"""

SESSION = """
import asyncio
from dataclasses import dataclass

import pytest

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.realtime import RealtimeSession, infer_realtime_model
from pydantic_ai.realtime._core import SessionCore
from pydantic_ai.realtime.codec import AudioDelta, OutputTranscript, ResponseDone
from pydantic_ai.realtime.lifecycle import InferredLifecycle, LifecycleEvent
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage

from tests.realtime_connections import InferredConnection


class Scripted(InferredConnection):
    def __init__(self, *, fails=False):
        self.fails = fails

    async def send_input(self, content):
        pass

    async def codec_events(self):
        if self.fails:
            raise RuntimeError('provider went away')
        yield AudioDelta(data=b'\\x01\\x00')
        yield OutputTranscript(text={text!r}, is_final=True)
        yield ResponseDone()


class Session(RealtimeSession):
    label = 'not a method'

    def stream_audio(self):
        return super().stream_audio()


class Unlisted(Session):
    def new_messages(self):
        raise RuntimeError('cannot list the new messages')


def session(cls=Session, *, fails=False, **kwargs):
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_step=0)
    ctx.tool_manager = ToolManager(FunctionToolset(), ctx=ctx, tools={{}})
    return cls(Scripted(fails=fails), tool_manager=ctx.tool_manager, **kwargs)


async def converse():
    talk = session()
    with pytest.raises(UserError, match='async with'):
        talk.stream_audio()
    async with talk:
        audio = talk.stream_audio()
        transcripts = talk.stream_transcripts()
        deltas = talk.stream_transcripts(delta=True)
        [event async for event in talk]
        [item for view in (audio, transcripts, deltas) async for item in view]
        with pytest.raises(UserError, match='exactly one active'):
            talk.played_audio_bytes
        await talk.close()
    with pytest.raises(UserError, match='closed'):
        [event async for event in talk]


async def fail():
    talk = session(fails=True)
    async with talk:
        with pytest.raises(RuntimeError, match='provider went away') as failure:
            [event async for event in talk]
        handled = KeyError('handled')
        handled.__context__ = failure.value
        try:
            raise handled
        except KeyError:
            await talk.close()
    # Recording what the closed session holds raises its failure again, which must leave the chain of the error being
    # handled, which holds it, as it was.
    assert handled.__context__ is failure.value
    assert failure.value.__context__ is None


async def close_with_audio_unread():
    async with session() as talk:
        audio = talk.stream_audio()
    assert audio


async def close_with_a_failing_read():
    # A read the capture can't make stops the test, rather than leave a record without it.
    with pytest.raises(RuntimeError, match='cannot list the new messages'):
        async with session(Unlisted):
            pass


async def close_with_an_unseen_failure():
    # `close` raises the failure no consumer of the event stream saw, and what the session holds is still recorded.
    with pytest.raises(RuntimeError, match='provider went away'):
        async with session(fails=True) as talk:
            [item async for item in talk.stream_transcripts()]


def test_session():
    asyncio.run(converse())
    asyncio.run(fail())
    asyncio.run(close_with_audio_unread())
    asyncio.run(close_with_a_failing_read())
    asyncio.run(close_with_an_unseen_failure())


def test_session_out_of_a_loop():
    with pytest.raises(UserError, match='retain_images_max'):
        session(retain_images_max=-1)


@dataclass(frozen=True, kw_only=True)
class Unknown(LifecycleEvent):
    pass


class Lifecycle(InferredLifecycle):
    def take_pending(self):
        return super().take_pending()


class Core(SessionCore):
    def take_effects(self):
        return super().take_effects()


def test_core_and_lifecycle():
    lifecycle = Lifecycle()
    core = Core(model_name=lambda: None, provider_name=None, provider_url=None, conversation_id=None, run_id=None)
    for event in lifecycle.frame_for([OutputTranscript(text={text!r}, is_final=True), ResponseDone()]):
        core.apply(event)
        core.wait_tokens()
    core.take_effects()
    core.all_messages()
    with pytest.raises(TypeError, match='no handling'):
        core.apply(Unknown())
    lifecycle.take_pending()


def test_infer_realtime_model(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    infer_realtime_model('openai:gpt-realtime')
    with pytest.raises(UserError, match='provider:model'):
        infer_realtime_model('openai')
"""

TOOL_PLUMBING = """
import asyncio
import threading
import warnings
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from mcp.types import CreateMessageResult, TextContent

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.mcp_sampling import MCPSamplingModel


class Rewrites(AbstractCapability[None]):
    async def before_tool_execute(self, ctx, *, call, tool_def, args):
        return {{'n': args['n'] + 1}}

    async def on_tool_execute_error(self, ctx, *, call, tool_def, args, error):
        return f'recovered from {{error}}'


class Untouched(AbstractCapability[None]):
    pass


def call_add(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('add', {{'n': 1}}, tool_call_id='big')])
    if len(messages) == 3:
        return ModelResponse(parts=[ToolCallPart('add', {{'n': 'x'}}, tool_call_id='invalid')])
    return ModelResponse(parts=[TextPart({text!r})])


def test_tool_hooks():
    agent = Agent(FunctionModel(call_add), capabilities=[Rewrites(), Untouched()])

    @agent.tool_plain
    def add(n: int) -> int:
        if n > 1:
            raise RuntimeError('too big')
        return n

    agent.run_sync('add')
    # Outside a run, the call on `Rewrites` itself is the outermost.
    call = ToolCallPart('add', tool_call_id='direct')
    asyncio.run(Rewrites().before_tool_execute(None, call=call, tool_def=None, args={{'n': 0}}))


def call_once(tool_name):
    def respond(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, tool_call_id=tool_name)])
        return ModelResponse(parts=[TextPart('done')])

    return respond


def test_tool_hooks_of_concurrent_calls():
    # `first` returns only once the hooks after `second` have been called.
    second_done = threading.Event()

    class Signals(AbstractCapability[None]):
        async def after_tool_execute(self, ctx, *, call, tool_def, args, result):
            if call.tool_name == 'second':
                second_done.set()
            return result

    def respond(messages, info):
        if len(messages) == 1:
            calls = [ToolCallPart('first', tool_call_id='a'), ToolCallPart('second', tool_call_id='b')]
            return ModelResponse(parts=calls)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[Signals()])

    @agent.tool_plain
    def first() -> str:
        assert second_done.wait(5)
        return 'first'

    @agent.tool_plain
    def second() -> str:
        return 'second'

    agent.run_sync('both')


def test_tool_hooks_of_a_nested_run():
    inner = Agent(FunctionModel(call_once('echo')))
    inner.tool_plain(lambda: 'echoed', name='echo')
    outer = Agent(FunctionModel(call_once('delegate')))

    @outer.tool_plain
    async def delegate() -> str:
        return (await inner.run('echo')).output

    outer.run_sync('delegate')


@dataclass
class FakeSession:
    create_message: AsyncMock


class ScriptedSession:
    async def create_message(self, messages, **kwargs):
        return CreateMessageResult(role='assistant', content=TextContent(type='text', text={text!r}), model='scripted')


def test_sampling():
    result = CreateMessageResult(role='assistant', content=TextContent(type='text', text={text!r}), model='sampler')
    model = MCPSamplingModel(FakeSession(AsyncMock(return_value=result)))
    assert Agent(model, instructions='Be brief.').run_sync('hi', model_settings={{'max_tokens': 5}}).output == {text!r}


def test_sampling_through_a_session():
    assert Agent(MCPSamplingModel(ScriptedSession())).run_sync('hi').output == {text!r}


def test_sampling_fails():
    result = CreateMessageResult(role='user', content=TextContent(type='text', text='?'), model='sampler')
    with pytest.raises(UnexpectedModelBehavior):
        Agent(MCPSamplingModel(FakeSession(AsyncMock(return_value=result)))).run_sync('hi')


def raise_noted():
    noted = LookupError('noted')
    noted.add_note('a note')
    raise noted


def test_raises():
    with pytest.raises(ValueError, match='outer'):
        try:
            raise KeyError('inner')
        except KeyError as e:
            raise ValueError('outer') from e
    with pytest.raises(TypeError):
        try:
            raise KeyError('handled')
        except KeyError:
            raise TypeError('replaced') from None
    pytest.raises(LookupError, raise_noted)
    with pytest.RaisesGroup(ValueError, RuntimeError):
        raise ExceptionGroup('both', [ValueError('a'), RuntimeError('b')])


def warn_from_here():
    warnings.warn('reported in the test module', UserWarning, stacklevel=2)


async def warn_from_the_model(messages, info):
    warnings.warn('reported in the model', DeprecationWarning, stacklevel=2)
    return ModelResponse(parts=[TextPart({text!r})])


def test_warnings():
    with pytest.warns(UserWarning):
        warn_from_here()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        Agent(FunctionModel(warn_from_the_model)).run_sync('hi')
        warnings.warn_explicit('reported where told', RuntimeWarning, 'elsewhere.py', 7)
        warnings.warn('raised by a finalizer', ResourceWarning)
        warnings.simplefilter('ignore')
        warnings.warn('filtered out')
"""

AGENT_RUNS = """
import asyncio

import pytest
from mcp.types import CreateMessageResult, TextContent
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel, model_serializer

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, ProcessEventStream
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import ModelResponse, PartEndEvent, PartStartEvent, TextPart, ToolCallPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.mcp_sampling import MCPSamplingModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage


async def call_then_answer(messages, info):
    if len(messages) == 1:
        yield {{0: DeltaToolCall('echo', '{{}}', tool_call_id='echo')}}
    else:
        yield {text!r}[:2]
        yield {text!r}[2:]


async def drop_part_ends(ctx, stream):
    async for event in stream:
        if event.event_kind != 'part_end':
            yield event


async def fail_after_the_first(ctx, stream):
    async for event in stream:
        yield event
        raise RuntimeError('processor failed')


def test_run_event_streams():
    agent = Agent(FunctionModel(stream_function=call_then_answer), capabilities=[ProcessEventStream(drop_part_ends)])
    agent.tool_plain(lambda: 'echoed', name='echo')

    async def run():
        async with agent.run_stream_events('echo') as events:
            async for _ in events:
                pass

    asyncio.run(run())


def test_enqueued_message():
    agent = Agent(FunctionModel(stream_function=call_then_answer))

    @agent.tool(name='echo')
    def echo(ctx: RunContext) -> str:
        ctx.enqueue('follow up')
        return 'echoed'

    async def run():
        async with agent.run_stream_events('echo') as events:
            async for _ in events:
                pass

    asyncio.run(run())


def test_run_event_stream_fails():
    agent = Agent(FunctionModel(stream_function=call_then_answer), capabilities=[ProcessEventStream(fail_after_the_first)])
    with pytest.raises(RuntimeError):
        agent.run_sync('echo')


class Passthrough(AbstractCapability[None]):
    def wrap_run_event_stream(self, ctx, *, stream):
        return stream


class Events:
    def __init__(self, *events):
        self.events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.events)
        except StopIteration:
            raise StopAsyncIteration


def test_run_event_stream_called_directly():
    async def events():
        yield PartStartEvent(index=0, part=TextPart({text!r}))
        yield PartEndEvent(index=0, part=TextPart({text!r}))

    async def run():
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
        [event async for event in ProcessEventStream(drop_part_ends).wrap_run_event_stream(ctx, stream=events())]
        # A stream that isn't a generator has nothing to close.
        stream = Events(PartStartEvent(index=0, part=TextPart({text!r})))
        [event async for event in Passthrough().wrap_run_event_stream(ctx, stream=stream)]

    asyncio.run(run())


class Answer(BaseModel):
    value: int


async def stream_answer(messages, info):
    name = info.output_tools[0].name
    yield {{0: DeltaToolCall(name, '{{"value": ', tool_call_id='out')}}
    yield {{0: DeltaToolCall(json_args='4')}}
    yield {{0: DeltaToolCall(json_args='2}}')}}


async def fail_mid_stream(messages, info):
    yield {text!r}[:2]
    raise RuntimeError('stream failed')


async def stream_no_answer(messages, info):
    yield {{0: DeltaToolCall(info.output_tools[0].name, '{{"value": "x"}}', tool_call_id='out')}}


def test_agent_stream_calls():
    async def run():
        text_agent = Agent(FunctionModel(stream_function=call_then_answer))
        async with text_agent.run_stream('hi', message_history=[]) as result:
            [text async for text in result.stream_text(debounce_by=None)]
            await result.get_output()
        async with text_agent.run_stream('hi') as result:
            [text async for text in result.stream_text(delta=True, debounce_by=None)]
        async with Agent(FunctionModel(stream_function=stream_answer), output_type=Answer).run_stream('hi') as result:
            [output async for output in result.stream_output(debounce_by=None)]
        async with Agent(FunctionModel(stream_function=stream_no_answer), output_type=Answer).run_stream('hi') as result:
            with pytest.raises(UnexpectedModelBehavior):
                await result.get_output()
        with pytest.raises(RuntimeError):
            async with Agent(FunctionModel(stream_function=fail_mid_stream)).run_stream('hi') as result:
                [text async for text in result.stream_text(debounce_by=None)]

    asyncio.run(run())


def call_and_fail(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('fail', tool_call_id='fail')])
    return ModelResponse(parts=[TextPart({text!r})])


def test_spans():
    settings = InstrumentationSettings(tracer_provider=TracerProvider(shutdown_on_exit=False))
    agent = Agent(FunctionModel(call_and_fail), capabilities=[Instrumentation(settings=settings)])

    @agent.tool_plain
    def fail() -> str:
        raise LookupError('tool failed')

    with pytest.raises(LookupError):
        agent.run_sync('fail')


def test_spans_of_a_tracer():
    tracer = TracerProvider(shutdown_on_exit=False).get_tracer('test')
    sampled = trace.TraceFlags(trace.TraceFlags.SAMPLED)
    remote = trace.NonRecordingSpan(trace.SpanContext(trace_id=1, span_id=2, is_remote=True, trace_flags=sampled))
    with tracer.start_as_current_span('outer', context=trace.set_span_in_context(remote)) as outer:
        with tracer.start_as_current_span('inner', attributes={{'answer': 42}}):
            pass
    outer.end()


class Unformattable(Exception):
    def __str__(self):
        raise RuntimeError('formatting blew up')


async def fail_unformattably(ctx, stream):
    async for event in stream:
        yield event
        raise Unformattable()


def test_unformattable_error():
    agent = Agent(FunctionModel(stream_function=call_then_answer), capabilities=[ProcessEventStream(fail_unformattably)])
    with pytest.raises(Unformattable):
        agent.run_sync('echo')


class ScriptedSession:
    async def create_message(self, messages, **kwargs):
        return CreateMessageResult(role='assistant', content=TextContent(type='text', text={text!r}), model='scripted')


@pytest.mark.parametrize('model', [TestModel(), MCPSamplingModel(ScriptedSession())], ids=['test', 'sampling'])
def test_tool_metadata(model):
    serializations = []

    class Tracked(BaseModel):
        @model_serializer
        def serialize(self) -> str:
            serializations.append('tracked')
            return 'tracked'

    agent = Agent(model)

    @agent.tool_plain(metadata={{'tracked': Tracked(), 'plain': {{'kind': 'lookup', 'tags': ['a']}}}})
    def lookup(key: str) -> str:
        return key

    agent.run_sync('hello')
    assert serializations == []
"""

DURABLE = """
import asyncio
import base64
import json
import threading
from types import SimpleNamespace
from uuid import uuid4

import pytest
from dbos import DBOS
from prefect.cache_policies import INPUTS, Inputs
from temporalio import activity
from temporalio.api.common.v1 import Payload
from temporalio.api.failure.v1 import Failure
from temporalio.bridge.proto.workflow_activation import (
    InitializeWorkflow,
    RemoveFromCache,
    WorkflowActivation,
    WorkflowActivationJob,
)
from temporalio.bridge.proto.workflow_commands import CompleteWorkflowExecution, ScheduleActivity, WorkflowCommand
from temporalio.bridge.proto.workflow_completion import Failure as Failed, Success, WorkflowActivationCompletion
from temporalio.bridge.worker import Worker

from pydantic_ai import Agent, PydanticAIDeprecationWarning
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.durable_exec.prefect import DEFAULT_PYDANTIC_AI_CACHE_POLICY, PrefectAgent
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.models.test import TestModel

COLLECTED = Agent(TestModel(), name='collected', capabilities=[TemporalDurability()])


def test_durable_names():
    Agent(TestModel(), name='dbos_agent', capabilities=[DBOSDurability(register_legacy_workflows=True)])
    with pytest.warns(PydanticAIDeprecationWarning):
        PrefectAgent(Agent(TestModel(), name='prefect_agent'))
    activity.defn(name='by_the_test')(lambda: None)


class Core:
    def __init__(self, activations):
        self.activations = [activation.SerializeToString() for activation in activations]

    async def poll_workflow_activation(self):
        return self.activations.pop(0)

    async def complete_workflow_activation(self, completion):
        pass


def payload(value):
    return Payload(metadata={{'encoding': b'json/plain'}}, data=json.dumps(value).encode())


def started(run_id, workflow_type, workflow_id):
    job = WorkflowActivationJob(
        initialize_workflow=InitializeWorkflow(workflow_type=workflow_type, workflow_id=workflow_id)
    )
    return WorkflowActivation(run_id=run_id, jobs=[job])


def test_temporal_commands():
    first, second = str(uuid4()), str(uuid4())
    event = payload({{'tool_call_id': f'pyd_ai_{{uuid4().hex}}'}})
    stream_item = {{'topic': 'agent_events', 'data': base64.b64encode(event.SerializeToString()).decode(), 'offset': 0}}
    activations = [
        started(second, 'Second', 'second'),
        started(first, 'First', f'first-{{uuid4()}}'),
        WorkflowActivation(run_id=first, jobs=[WorkflowActivationJob(remove_from_cache=RemoveFromCache())]),
    ]
    failure = Failure(message='boom', stack_trace='File "workflow.py", line 7, in run')
    commands = [
        WorkflowCommand(
            schedule_activity=ScheduleActivity(
                seq=1,
                activity_id='1',
                activity_type='agent__model_request',
                arguments=[
                    payload({{'serialized_run_context': {{'active_capability_ids': ['tool_search', 'agent']}}}}),
                    payload({{'items': [stream_item], 'topic': 'not a stream item'}}),
                ],
            )
        ),
        WorkflowCommand(
            schedule_activity=ScheduleActivity(
                seq=2,
                activity_id='2',
                activity_type='raw',
                arguments=[Payload(metadata={{'encoding': b'binary/plain'}}, data=b'raw')],
            )
        ),
        WorkflowCommand(
            schedule_activity=ScheduleActivity(
                seq=3,
                activity_id='3',
                activity_type='not_json',
                arguments=[Payload(metadata={{'encoding': b'json/plain'}}, data=b'not json')],
            )
        ),
        WorkflowCommand(complete_workflow_execution=CompleteWorkflowExecution(result=payload({text!r}))),
    ]
    completions = [
        WorkflowActivationCompletion(run_id=second, failed=Failed(failure=failure)),
        WorkflowActivationCompletion(run_id=first, successful=Success(commands=commands)),
        WorkflowActivationCompletion(run_id=first, successful=Success()),
    ]
    worker = Worker(Core(activations))

    async def drive():
        for completion in completions:
            await worker.poll_workflow_activation()
            await worker.complete_workflow_activation(completion)

    asyncio.run(drive())


def test_dbos_workflow(tmp_path):
    DBOS(config={{'name': 'refactor_capture', 'system_database_url': f'sqlite:///{{tmp_path / "dbos.sqlite"}}'}})

    @DBOS.step()
    def double(n: int) -> int:
        return n * 2

    @DBOS.workflow()
    def quadruple(n: int) -> int:
        return double(double(n))

    DBOS.launch()
    try:
        handle = DBOS.start_workflow(quadruple, 1)
        assert handle.get_result() == 4
        # The fork replays the first step's recorded result and runs the second.
        assert DBOS.fork_workflow(handle.workflow_id, 2).get_result() == 4
    finally:
        DBOS.destroy()


def test_cache_keys():
    def task_function():
        pass

    task_ctx = SimpleNamespace(task_run=SimpleNamespace(flow_run_id=uuid4(), id=uuid4()), task=task_function)
    DEFAULT_PYDANTIC_AI_CACHE_POLICY.compute_key(task_ctx, {{'prompt': {text!r}}}, {{}})
    with pytest.raises(ValueError):
        INPUTS.compute_key(task_ctx, {{'lock': threading.Lock()}}, {{}})

    class Inherited(Inputs):
        pass

    Inherited().compute_key(task_ctx, {{'n': 1}}, {{}})
    cyclic = []
    cyclic.append(cyclic)
    INPUTS.compute_key(task_ctx, {{'cyclic': cyclic}}, {{}})
    INPUTS.compute_key(task_ctx, {{'generated': str(uuid4())}}, {{}})
"""


@pytest.fixture
def run(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> RunCapture:
    """Capture an in-process pytest session over `STREAMS` and `REQUESTS`, `UI`, `REALTIME`, `SESSION`,
    `TOOL_PLUMBING`, `AGENT_RUNS` or `DURABLE`, carrying `text`, into `capture_dir`.

    It runs as `main` runs each side.
    """
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)
    # The Anthropic SDK traces each request through the global tracer provider, which an earlier test in this worker
    # may have set, like one that calls `logfire.configure()`, so the inner sessions would record its spans or not
    # depending on what ran before them.
    monkeypatch.setenv('ANTHROPIC_OPEN_TELEMETRY', 'false')
    # The inner sessions share this worker's `random` with every thread an earlier test left running, and such a
    # thread can draw from it between the capture's seed and an inner test's draw: Prefect's events client keeps a
    # `websockets` connection open whose pings draw `random.getrandbits(32)`. So the capture's seed also seeds `rng`,
    # and `random.randrange`, the draw the inner tests assert on, reads `rng`: a thread that draws through
    # `random.getrandbits`, as a `websockets` ping does, can't shift it.
    rng = random.Random()
    seed = random.seed
    # Pyright can't infer a lambda parameter's type with nothing to infer it from.
    monkeypatch.setattr(random, 'seed', lambda a: (seed(a), rng.seed(a)))  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    monkeypatch.setattr(random, 'randrange', rng.randrange)

    def run(
        capture_dir: Path,
        *args: str,
        text: str = 'hello',
        ui: bool = False,
        realtime: bool = False,
        session: bool = False,
        tool_plumbing: bool = False,
        agent_runs: bool = False,
        durable: bool = False,
    ) -> pytest.RunResult:
        pytester.makeconftest(CONFTEST)
        if durable:
            pytester.makepyfile(test_capture_durable=DURABLE.format(text=text))
        elif ui:
            pytester.makepyfile(test_ui=UI.format(text=text))
        elif realtime:
            pytester.makepyfile(test_realtime=REALTIME.format(text=text))
        elif session:
            pytester.makepyfile(test_session=SESSION.format(text=text))
        elif tool_plumbing:
            pytester.makepyfile(test_tool_plumbing=TOOL_PLUMBING.format(text=text))
        elif agent_runs:
            pytester.makepyfile(test_agent_runs=AGENT_RUNS.format(text=text))
        else:
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

    `random` is seeded per test, so the value `test_json_over_httpx` draws is the same in every capture, even after a
    thread drew through `random.getrandbits`. The API keys the session's requests carry, under ElevenLabs' `xi-api-key`
    and Anthropic's `x-api-key`, are left out.
    """
    run(tmp_path).assert_outcomes(passed=11)

    capture = (tmp_path / 'main.jsonl').read_text()
    assert 'secret' not in capture
    lines = capture.splitlines()
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


def test_capture_records_what_ui_adapters_produce(run: RunCapture, tmp_path: Path):
    """`transform_stream`'s events, what `load_messages` and `dump_messages` return, and the web API's responses.

    A request to the app `create_web_app` mounts the API app in is recorded once, for the API app.
    `test_subclass_defined_after_the_capture_started` checks that closing the recording closes the stream it records.
    """
    run(tmp_path, '-k', 'not media_type', ui=True).assert_outcomes(passed=7, deselected=1)

    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    records = [record for line in lines if (record := json.loads(line))['kind'] == 'ui']
    assert [
        (
            record['node'],
            record['target'],
            [event['type'] for event in record.get('events', [])],
            record.get('error'),
            record.get('status'),
        )
        for record in records
    ] == snapshot(
        [
            (
                'test_ui.py::test_vercel_ai_messages',
                'pydantic_ai.ui.vercel_ai._adapter.VercelAIAdapter.load_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_vercel_ai_messages',
                'pydantic_ai.ui.vercel_ai._adapter.VercelAIAdapter.dump_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_ag_ui_messages',
                'pydantic_ai.ui.ag_ui._adapter.AGUIAdapter.load_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_ag_ui_messages',
                'pydantic_ai.ui.ag_ui._adapter.AGUIAdapter.dump_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_vercel_ai_events',
                'pydantic_ai.ui.vercel_ai._adapter.VercelAIAdapter.load_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_vercel_ai_events',
                'pydantic_ai.ui.vercel_ai._event_stream.VercelAIEventStream.transform_stream',
                [
                    'start',
                    'start-step',
                    'text-start',
                    'text-delta',
                    'text-delta',
                    'text-end',
                    'message-metadata',
                    'finish-step',
                    'finish',
                    'done',
                ],
                None,
                None,
            ),
            (
                'test_ui.py::test_ag_ui_events',
                'pydantic_ai.ui.ag_ui._adapter.AGUIAdapter.load_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_ag_ui_events',
                'pydantic_ai.ui.ag_ui._event_stream.AGUIEventStream.transform_stream',
                [
                    'RUN_STARTED',
                    'TEXT_MESSAGE_START',
                    'TEXT_MESSAGE_CONTENT',
                    'TEXT_MESSAGE_CONTENT',
                    'TEXT_MESSAGE_END',
                    'RUN_FINISHED',
                ],
                None,
                None,
            ),
            (
                'test_ui.py::test_subclass_defined_after_the_capture_started',
                'test_ui.ClosingEventStream.transform_stream',
                ['start'],
                None,
                None,
            ),
            (
                'test_ui.py::test_subclass_defined_after_the_capture_started',
                'test_ui.ClosingEventStream.transform_stream',
                ['start'],
                None,
                None,
            ),
            (
                'test_ui.py::test_errors',
                'test_ui.FailingEventStream.transform_stream',
                ['start'],
                'RuntimeError: client went away',
                None,
            ),
            (
                'test_ui.py::test_errors',
                'pydantic_ai.ui._adapter.UIAdapter.dump_messages',
                [],
                'NotImplementedError: ',
                None,
            ),
            ('test_ui.py::test_web_api', 'GET /api/health', [], None, 200),
            ('test_ui.py::test_web_api', 'POST /api/chat', [], None, 200),
            (
                'test_ui.py::test_web_api',
                'pydantic_ai.ui.vercel_ai._adapter.VercelAIAdapter.load_messages',
                [],
                None,
                None,
            ),
            (
                'test_ui.py::test_web_api',
                'pydantic_ai.ui.vercel_ai._event_stream.VercelAIEventStream.transform_stream',
                [
                    'start',
                    'start-step',
                    'text-start',
                    'text-delta',
                    'text-delta',
                    'text-end',
                    'message-metadata',
                    'finish-step',
                    'finish',
                    'done',
                ],
                None,
                None,
            ),
        ]
    )
    loaded = next(record for record in records if record['target'].endswith('VercelAIAdapter.load_messages'))
    assert loaded['result'] == snapshot(
        [
            {
                'parts': [
                    {
                        'content': [
                            'hello',
                            {
                                'url': 'https://example.com/a.png',
                                'force_download': False,
                                'vendor_metadata': None,
                                'kind': 'image-url',
                                'media_type': 'image/png',
                                'identifier': 'b86daf',
                            },
                        ],
                        'timestamp': 'NOW',
                        'part_kind': 'user-prompt',
                    }
                ],
                'timestamp': None,
                'instructions': None,
                'kind': 'request',
                'run_id': None,
                'conversation_id': None,
                'metadata': {'__pydantic_ai__': {'ui_message_id': 'message'}},
                'state': 'complete',
            }
        ]
    )
    assert loaded['uncompared_fields'] == snapshot(
        [{'type': 'ImageUrl', 'fields': {'_media_type': 'image/png', '_identifier': None}}]
    )
    assert [(record['headers'], record['body']) for record in records if 'status' in record] == snapshot(
        [
            ({'content-length': '11', 'content-type': 'application/json'}, {'ok': True}),
            (
                {'x-vercel-ai-ui-message-stream': 'v1', 'content-type': 'text/event-stream; charset=utf-8'},
                """\
data: {"type":"start"}

data: {"type":"start-step"}

data: {"type":"text-start","id":"UUID"}

data: {"type":"text-delta","delta":"he","id":"UUID"}

data: {"type":"text-delta","delta":"llo","id":"UUID"}

data: {"type":"text-end","id":"UUID"}

data: {"type":"message-metadata","messageMetadata":{"pydantic_ai":{"timestamp":"NOW"}}}

data: {"type":"finish-step"}

data: {"type":"finish"}

data: [DONE]

""",
            ),
        ]
    )


@pytest.mark.skipif(not ws_cassettes.ws_cassettes_available(), reason='`websockets` or `PyYAML` not installed')
def test_capture_records_what_realtime_connections_send_and_yield(
    run: RunCapture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Each frame `frames` yields, each `send` call and each call on a realtime double, in the order they happen.

    The session's own classes stand in for doubles defined under `tests/realtime/` once the capture started, and
    `ReplayWebSocket` for one defined before. `test_connection_defined_after_the_capture_started` checks that closing
    the recording closes what `frames` returned.
    """
    pytest.importorskip('google.genai')
    monkeypatch.setattr(refactor_capture, 'REALTIME_DOUBLE_PACKAGES', ('tests.realtime', 'test_realtime'))
    run(tmp_path, realtime=True).assert_outcomes(passed=5)

    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    records = [record for line in lines if (record := json.loads(line))['kind'] == 'realtime']
    assert [
        (
            record['node'].split('::')[1],
            record['target'],
            [
                f'StaleEvent({event["event"]["type"]})' if event['type'] == 'StaleEvent' else event['type']
                for event in record.get('frame', [])
            ],
            record.get('input'),
            [event['type'] for event in record.get('result') or ()],
            record.get('args'),
            sorted(record.get('kwargs', {})),
            record.get('error'),
        )
        for record in records
    ] == snapshot(
        [
            (
                'test_openai_turn',
                'pydantic_ai.realtime.openai.OpenAIRealtimeConnection.send',
                [],
                {'type': 'str', 'value': 'hi'},
                [],
                None,
                [],
                None,
            ),
            (
                'test_openai_turn',
                'test_realtime.WrappingSocket.send',
                [],
                None,
                [],
                [
                    {
                        'type': 'conversation.item.create',
                        'event_id': 'pydantic_ai.content.0',
                        'item': {
                            'id': 'pydantic_ai_item_0',
                            'type': 'message',
                            'role': 'user',
                            'content': [{'type': 'input_text', 'text': 'hi'}],
                        },
                    }
                ],
                [],
                None,
            ),
            (
                'test_openai_turn',
                'test_realtime.WrappingSocket.send',
                [],
                None,
                [],
                [
                    {
                        'type': 'response.create',
                        'event_id': 'pydantic_ai.response.0',
                        'response': {'metadata': {'pydantic_ai_inputs': '0'}},
                    }
                ],
                [],
                None,
            ),
            (
                'test_openai_turn',
                'pydantic_ai.realtime.openai.OpenAIRealtimeConnection.frames',
                ['ResponseStarted'],
                None,
                [],
                None,
                [],
                None,
            ),
            (
                'test_openai_turn',
                'pydantic_ai.realtime.openai.OpenAIRealtimeConnection.frames',
                ['OutputTranscript'],
                None,
                [],
                None,
                [],
                None,
            ),
            (
                'test_openai_turn',
                'pydantic_ai.realtime.openai.OpenAIRealtimeConnection.frames',
                ['ResponseEnded', 'RealtimeSessionErrorEvent'],
                None,
                [],
                None,
                [],
                None,
            ),
            (
                'test_broken_socket',
                'pydantic_ai.realtime.openai.OpenAIRealtimeConnection.send',
                [],
                {'type': 'str', 'value': 'hi'},
                [],
                None,
                [],
                'ConnectionError: socket closed',
            ),
            (
                'test_broken_socket',
                'test_realtime.BrokenSocket.send',
                [],
                None,
                [],
                [
                    {
                        'type': 'conversation.item.create',
                        'event_id': 'pydantic_ai.content.0',
                        'item': {
                            'id': 'pydantic_ai_item_0',
                            'type': 'message',
                            'role': 'user',
                            'content': [{'type': 'input_text', 'text': 'hi'}],
                        },
                    }
                ],
                [],
                'ConnectionError: socket closed',
            ),
            (
                'test_google_session',
                'pydantic_ai.realtime.google.GoogleRealtimeConnection.send',
                [],
                {'type': 'str', 'value': 'hello'},
                ['InputAdded'],
                None,
                [],
                None,
            ),
            (
                'test_google_session',
                'test_realtime.Session.send_client_content',
                [],
                None,
                [],
                [],
                ['turn_complete', 'turns'],
                None,
            ),
            (
                'test_cassette_replay',
                'tests.realtime.ws_cassettes.ReplayWebSocket.send',
                [],
                None,
                [],
                [{'type': 'session.update'}],
                [],
                None,
            ),
            (
                'test_connection_defined_after_the_capture_started',
                'test_realtime.Chained.send',
                [],
                {'type': 'str', 'value': 'hi'},
                ['ResponseStarted'],
                None,
                [],
                None,
            ),
            (
                'test_connection_defined_after_the_capture_started',
                'test_realtime.Chained.frames',
                ['ResponseStarted', 'StaleEvent(OutputTranscript)'],
                None,
                [],
                None,
                [],
                None,
            ),
            (
                'test_connection_defined_after_the_capture_started',
                'test_realtime.Scripted.frames',
                ['ResponseStarted', 'StaleEvent(OutputTranscript)'],
                None,
                [],
                None,
                [],
                None,
            ),
            (
                'test_connection_defined_after_the_capture_started',
                'test_realtime.Scripted.frames',
                [],
                None,
                [],
                None,
                [],
                'RuntimeError: provider went away',
            ),
            (
                'test_connection_defined_after_the_capture_started',
                'test_realtime.Listed.frames',
                ['OutputTranscript'],
                None,
                [],
                None,
                [],
                None,
            ),
        ]
    )
    (stale_frame,) = [
        record['frame']
        for record in records
        if record['target'] == 'test_realtime.Scripted.frames' and 'frame' in record
    ]
    assert stale_frame == snapshot(
        [
            {
                'type': 'ResponseStarted',
                'value': {
                    'response_id': 'r1',
                    'answers': [],
                    'basis': 'protocol',
                    'user_turn_id': None,
                    'continues': None,
                    'provider_id': True,
                },
            },
            {
                'type': 'StaleEvent',
                'event': {
                    'type': 'OutputTranscript',
                    'value': {
                        'text': 'hello',
                        'is_final': False,
                        'output_text': False,
                        'item_id': None,
                        'response_id': None,
                    },
                },
            },
        ]
    )


def test_capture_records_what_realtime_sessions_do(run: RunCapture, tmp_path: Path):
    """Each core call, session iteration, tap and error, what a closed session holds, and each model inferred.

    `Session` and `Core` stand in for subclasses defined once the capture started, which override some of the methods
    recorded, and `Lifecycle` for a lifecycle a test calls directly; the calls the session's connection makes on its own
    lifecycle are in its `realtime` records instead.
    """
    run(tmp_path, session=True).assert_outcomes(passed=4)

    lines = (tmp_path / 'main.jsonl').read_text().splitlines()
    records = [record for line in lines if (record := json.loads(line))['kind'] == 'session']
    assert [
        (
            record['node'].split('::')[1],
            record['target'].rsplit('.', 2)[-2] + '.' + record['target'].rsplit('.', 1)[-1],
            {key: record[key] for key in ('session', 'core', 'lifecycle') if key in record},
            sorted(set(record) - {'kind', 'node', 'target', 'session', 'core', 'lifecycle', 'error'}),
            record.get('error'),
        )
        for record in records
    ] == snapshot(
        [
            (
                'test_session',
                'Session.stream_audio',
                {'session': 0},
                [],
                'UserError: Enter the realtime session with `async with` before streaming it.',
            ),
            ('test_session', 'Session.stream_audio', {'session': 0}, ['items', 'kwargs'], None),
            ('test_session', 'Session.stream_transcripts', {'session': 0}, ['items', 'kwargs'], None),
            ('test_session', 'Session.stream_transcripts', {'session': 0}, ['items', 'kwargs'], None),
            ('test_session', 'Session.__aiter__', {'session': 0}, ['events'], None),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.wait_tokens', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            (
                'test_session',
                'Session.played_audio_bytes',
                {'session': 0},
                [],
                'UserError: `played_audio_bytes` needs exactly one active `stream_audio()` iterator to report the playback position of, not 0; keep your own accounting instead.',
            ),
            ('test_session', 'SessionCore.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 0}, ['args', 'kwargs', 'result'], None),
            (
                'test_session',
                'Session.close',
                {'session': 0},
                ['all_messages', 'drops', 'new_messages', 'played_audio_bytes', 'usage'],
                None,
            ),
            (
                'test_session',
                'Session.__aiter__',
                {'session': 0},
                ['events'],
                'UserError: This realtime session is closed and cannot be iterated.',
            ),
            ('test_session', 'Session.__aiter__', {'session': 1}, ['events'], 'RuntimeError: provider went away'),
            ('test_session', 'SessionCore.apply', {'core': 1}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 1}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 1}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 1}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 1}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 1}, ['args', 'kwargs', 'result'], None),
            (
                'test_session',
                'Session.close',
                {'session': 1},
                ['all_messages', 'drops', 'new_messages', 'played_audio_bytes', 'usage'],
                None,
            ),
            ('test_session', 'Session.stream_audio', {'session': 2}, ['items', 'kwargs'], None),
            ('test_session', 'SessionCore.apply', {'core': 2}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 2}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 2}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 2}, ['args', 'kwargs', 'result'], None),
            (
                'test_session',
                'Session.close',
                {'session': 2},
                ['all_messages', 'drops', 'new_messages', 'played_audio_bytes', 'usage'],
                None,
            ),
            ('test_session', 'SessionCore.apply', {'core': 3}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 3}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 3}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 3}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'Unlisted.__aexit__', {'session': 3}, [], 'RuntimeError: cannot list the new messages'),
            ('test_session', 'Session.stream_transcripts', {'session': 4}, ['items', 'kwargs'], None),
            ('test_session', 'SessionCore.apply', {'core': 4}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 4}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.apply', {'core': 4}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.take_effects', {'core': 4}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 4}, ['args', 'kwargs', 'result'], None),
            ('test_session', 'SessionCore.all_messages', {'core': 4}, ['args', 'kwargs', 'result'], None),
            (
                'test_session',
                'Session.close',
                {'session': 4},
                ['all_messages', 'drops', 'new_messages', 'played_audio_bytes', 'usage'],
                None,
            ),
            ('test_session', 'Session.__aexit__', {'session': 4}, [], 'RuntimeError: provider went away'),
            (
                'test_session_out_of_a_loop',
                'Session.__init__',
                {'session': 0},
                [],
                'UserError: `retain_images_max` must be at least 0, or `None` for no limit.',
            ),
            ('test_core_and_lifecycle', 'Lifecycle.frame_for', {'lifecycle': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.wait_tokens', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.wait_tokens', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.wait_tokens', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.apply', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.wait_tokens', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.take_effects', {'core': 0}, ['args', 'kwargs', 'result'], None),
            ('test_core_and_lifecycle', 'Core.all_messages', {'core': 0}, ['args', 'kwargs', 'result'], None),
            (
                'test_core_and_lifecycle',
                'Core.apply',
                {'core': 0},
                ['args', 'kwargs'],
                'TypeError: The session core has no handling for the lifecycle event Unknown().',
            ),
            ('test_core_and_lifecycle', 'Lifecycle.take_pending', {'lifecycle': 0}, ['args', 'kwargs', 'result'], None),
            ('test_infer_realtime_model', 'model.infer_realtime_model', {}, ['args', 'kwargs', 'result'], None),
            (
                'test_infer_realtime_model',
                'model.infer_realtime_model',
                {},
                ['args', 'kwargs'],
                "UserError: Realtime model identifiers use the `provider:model` format (e.g. `openai:gpt-realtime`); got 'openai'.",
            ),
        ]
    )
    assert [
        (record['target'].rsplit('.', 1)[-1], record['kwargs'], record['items'])
        for record in records
        if 'items' in record
    ] == snapshot(
        [
            ('stream_audio', {}, ['AQA=']),
            (
                'stream_transcripts',
                {},
                [
                    {
                        'speaker': 'assistant',
                        'transcript': 'hello',
                        'audio': None,
                        'interrupted_at_ms': None,
                        'timestamps': None,
                        'id': None,
                        'provider_name': None,
                        'provider_details': None,
                        'part_kind': 'speech',
                    }
                ],
            ),
            (
                'stream_transcripts',
                {'delta': True},
                [{'index': 0, 'speaker': 'assistant', 'delta': 'hello', 'transcript': 'hello'}],
            ),
            ('stream_audio', {}, []),
            ('stream_transcripts', {}, []),
        ]
    )
    assert [
        ([event['type'] for event in record['events']], record.get('error')) for record in records if 'events' in record
    ] == snapshot(
        [
            (['PartStartEvent', 'PartDeltaEvent', 'PartDeltaEvent', 'PartEndEvent', 'RealtimeTurnCompleteEvent'], None),
            ([], 'UserError: This realtime session is closed and cannot be iterated.'),
            ([], 'RuntimeError: provider went away'),
        ]
    )
    assert [
        {
            'session': record['session'],
            'all_messages': [message['kind'] for message in record['all_messages']],
            'new_messages': [message['kind'] for message in record['new_messages']],
            'requests': record['usage']['requests'],
            'played_audio_bytes': record['played_audio_bytes'],
            'drops': record['drops'],
        }
        for record in records
        if record['target'].endswith('.close')
    ] == snapshot(
        [
            {
                'session': 0,
                'all_messages': ['response'],
                'new_messages': ['response'],
                'requests': 1,
                'played_audio_bytes': {
                    'error': 'UserError: `played_audio_bytes` needs exactly one active `stream_audio()` iterator to report the playback position of, not 0; keep your own accounting instead.'
                },
                'drops': {
                    '_audio_tap_drops': 0,
                    '_transcript_tap_drops': 0,
                    '_queue_dropped_deltas': 0,
                    '_queue_dropped_structural': 0,
                },
            },
            {
                'session': 1,
                'all_messages': [],
                'new_messages': [],
                'requests': 0,
                'played_audio_bytes': {'error': 'RuntimeError: provider went away'},
                'drops': {
                    '_audio_tap_drops': 0,
                    '_transcript_tap_drops': 0,
                    '_queue_dropped_deltas': 0,
                    '_queue_dropped_structural': 0,
                },
            },
            {
                'session': 2,
                'all_messages': [],
                'new_messages': [],
                'requests': 0,
                'played_audio_bytes': 0,
                'drops': {
                    '_audio_tap_drops': 0,
                    '_transcript_tap_drops': 0,
                    '_queue_dropped_deltas': 0,
                    '_queue_dropped_structural': 0,
                },
            },
            {
                'session': 4,
                'all_messages': [],
                'new_messages': [],
                'requests': 0,
                'played_audio_bytes': {'error': 'RuntimeError: provider went away'},
                'drops': {
                    '_audio_tap_drops': 0,
                    '_transcript_tap_drops': 0,
                    '_queue_dropped_deltas': 0,
                    '_queue_dropped_structural': 0,
                },
            },
        ]
    )
    assert [
        record['result'] for record in records if record['target'].endswith('.wait_tokens') and record['result']
    ] == snapshot(
        [
            [{'type': 'Owed', 'value': {'kind': 'response', 'key': 'pydantic_ai_response_1', 'epoch': 0}}],
            [{'type': 'Owed', 'value': {'kind': 'response', 'key': 'pydantic_ai_response_1', 'epoch': 0}}],
            [{'type': 'Owed', 'value': {'kind': 'response', 'key': 'pydantic_ai_response_1', 'epoch': 0}}],
        ]
    )
    (frame_for,) = [record for record in records if record['target'].endswith('.frame_for')]
    assert (
        [event['type'] for event in frame_for['args'][0]],
        [event['type'] for event in frame_for['result']],
    ) == snapshot(
        (['OutputTranscript', 'ResponseDone'], ['ResponseStarted', 'OutputTranscript', 'ResponseDone', 'ResponseEnded'])
    )
    assert [record.get('result') for record in records if record['target'].endswith('.infer_realtime_model')] == (
        snapshot(
            [
                {
                    'type': 'pydantic_ai.realtime.openai.OpenAIRealtimeModel',
                    'model_name': 'gpt-realtime',
                    'system': 'openai',
                    'base_url': 'https://api.openai.com/v1/',
                },
                None,
            ]
        )
    )


def test_capture_records_tool_hooks_sampling_exceptions_and_warnings(run: RunCapture, tmp_path: Path):
    """Each tool hook the tool manager calls, each `MCPSamplingModel` request and the calls its mocked session gets,
    each exception `pytest.raises` or `pytest.RaisesGroup` catches, and each warning the filters let through but a
    `ResourceWarning`.

    The agent combines `Rewrites`, a capability defined once the capture started, with `Untouched`; the calls the
    combined capability makes on them aren't recorded again, and the one the test makes on `Rewrites` directly is. The
    first tool call fails once `Rewrites` raised its argument, and `Rewrites` recovers it; the second fails validation.
    The calls of an agent run inside a tool, while the outer run's `wrap_tool_execute` call runs, are recorded too. Each
    call's hooks come before the next call's, though the hooks after `second` were called before `first` returned. A
    session that isn't a mock keeps no calls to record.
    """
    run(tmp_path, tool_plumbing=True).assert_outcomes(passed=8)

    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    assert [
        (
            record['target'].rsplit('.', 2)[-2] + '.' + record['target'].rsplit('.', 1)[-1],
            record['kwargs']['call']['tool_call_id'],
            sorted(record['kwargs']),
            record['kwargs']['args'],
            record.get('result'),
            record.get('error'),
        )
        for record in records
        if record['kind'] == 'hook'
    ] == snapshot(
        [
            ('CombinedCapability.wrap_tool_validate', 'big', ['args', 'call', 'tool_def'], {'n': 1}, {'n': 1}, None),
            ('CombinedCapability.before_tool_validate', 'big', ['args', 'call', 'tool_def'], {'n': 1}, {'n': 1}, None),
            ('CombinedCapability.after_tool_validate', 'big', ['args', 'call', 'tool_def'], {'n': 1}, {'n': 1}, None),
            (
                'CombinedCapability.wrap_tool_execute',
                'big',
                ['args', 'call', 'tool_def'],
                {'n': 1},
                'recovered from too big',
                None,
            ),
            ('CombinedCapability.before_tool_execute', 'big', ['args', 'call', 'tool_def'], {'n': 1}, {'n': 2}, None),
            (
                'CombinedCapability.on_tool_execute_error',
                'big',
                ['args', 'call', 'error', 'tool_def'],
                {'n': 2},
                'recovered from too big',
                None,
            ),
            (
                'CombinedCapability.after_tool_execute',
                'big',
                ['args', 'call', 'result', 'tool_def'],
                {'n': 2},
                'recovered from too big',
                None,
            ),
            (
                'CombinedCapability.wrap_tool_validate',
                'invalid',
                ['args', 'call', 'tool_def'],
                {'n': 'x'},
                None,
                """\
ValidationError: 1 validation error for add
n
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', input_type=str]
    For further information visit https://errors.pydantic.dev/2.14/v/int_parsing\
""",
            ),
            (
                'CombinedCapability.before_tool_validate',
                'invalid',
                ['args', 'call', 'tool_def'],
                {'n': 'x'},
                {'n': 'x'},
                None,
            ),
            (
                'CombinedCapability.on_tool_validate_error',
                'invalid',
                ['args', 'call', 'error', 'tool_def'],
                {'n': 'x'},
                None,
                """\
ValidationError: 1 validation error for add
n
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', input_type=str]
    For further information visit https://errors.pydantic.dev/2.14/v/int_parsing\
""",
            ),
            ('Rewrites.before_tool_execute', 'direct', ['args', 'call', 'tool_def'], {'n': 0}, {'n': 1}, None),
            ('CombinedCapability.wrap_tool_validate', 'a', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.before_tool_validate', 'a', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.after_tool_validate', 'a', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.wrap_tool_execute', 'a', ['args', 'call', 'tool_def'], {}, 'first', None),
            ('CombinedCapability.before_tool_execute', 'a', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.after_tool_execute', 'a', ['args', 'call', 'result', 'tool_def'], {}, 'first', None),
            ('CombinedCapability.wrap_tool_validate', 'b', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.before_tool_validate', 'b', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.after_tool_validate', 'b', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.wrap_tool_execute', 'b', ['args', 'call', 'tool_def'], {}, 'second', None),
            ('CombinedCapability.before_tool_execute', 'b', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.after_tool_execute', 'b', ['args', 'call', 'result', 'tool_def'], {}, 'second', None),
            ('CombinedCapability.wrap_tool_validate', 'delegate', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.before_tool_validate', 'delegate', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.after_tool_validate', 'delegate', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.wrap_tool_execute', 'delegate', ['args', 'call', 'tool_def'], {}, 'done', None),
            ('CombinedCapability.before_tool_execute', 'delegate', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.wrap_tool_validate', 'echo', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.before_tool_validate', 'echo', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.after_tool_validate', 'echo', ['args', 'call', 'tool_def'], {}, {}, None),
            ('CombinedCapability.wrap_tool_execute', 'echo', ['args', 'call', 'tool_def'], {}, 'echoed', None),
            ('CombinedCapability.before_tool_execute', 'echo', ['args', 'call', 'tool_def'], {}, {}, None),
            (
                'CombinedCapability.after_tool_execute',
                'echo',
                ['args', 'call', 'result', 'tool_def'],
                {},
                'echoed',
                None,
            ),
            (
                'CombinedCapability.after_tool_execute',
                'delegate',
                ['args', 'call', 'result', 'tool_def'],
                {},
                'done',
                None,
            ),
        ]
    )
    assert [
        {key: value for key, value in record.items() if key not in ('kind', 'messages', 'model_request_parameters')}
        for record in records
        if record['kind'] == 'request' and 'sampling' in record['node']
    ] == snapshot(
        [
            {
                'node': 'test_tool_plumbing.py::test_sampling',
                'target': 'pydantic_ai.models.mcp_sampling.MCPSamplingModel.request',
                'model_settings': {'max_tokens': 5},
                'response': {
                    'parts': [
                        {
                            'content': 'hello',
                            'id': None,
                            'provider_name': None,
                            'provider_details': None,
                            'citations': None,
                            'part_kind': 'text',
                        }
                    ],
                    'usage': {
                        'input_tokens': 0,
                        'cache_write_tokens': 0,
                        'cache_write_1h_tokens': 0,
                        'cache_read_tokens': 0,
                        'output_tokens': 0,
                        'output_reasoning_tokens': 0,
                        'input_audio_tokens': 0,
                        'cache_audio_read_tokens': 0,
                        'output_audio_tokens': 0,
                        'audio_seconds': 0.0,
                        'details': {},
                        'cost': None,
                    },
                    'model_name': 'sampler',
                    'model_id': None,
                    'timestamp': 'NOW',
                    'kind': 'response',
                    'provider_name': None,
                    'provider_url': None,
                    'canonical_system': None,
                    'provider_details': None,
                    'provider_response_id': None,
                    'finish_reason': None,
                    'run_id': None,
                    'conversation_id': None,
                    'metadata': None,
                    'workspace_ref': None,
                    'history_digest': None,
                    'failed_attempts': None,
                    'state': 'complete',
                },
            },
            {
                'node': 'test_tool_plumbing.py::test_sampling',
                'target': 'test_tool_plumbing.FakeSession.create_message',
                'args': [
                    [
                        {
                            'role': 'user',
                            'content': {'type': 'text', 'text': 'hi', 'annotations': None, '_meta': None},
                            '_meta': None,
                        }
                    ]
                ],
                'kwargs': {
                    'max_tokens': 5,
                    'system_prompt': 'Be brief.',
                    'temperature': None,
                    'model_preferences': None,
                    'stop_sequences': None,
                },
            },
            {
                'node': 'test_tool_plumbing.py::test_sampling_through_a_session',
                'target': 'pydantic_ai.models.mcp_sampling.MCPSamplingModel.request',
                'model_settings': None,
                'response': {
                    'parts': [
                        {
                            'content': 'hello',
                            'id': None,
                            'provider_name': None,
                            'provider_details': None,
                            'citations': None,
                            'part_kind': 'text',
                        }
                    ],
                    'usage': {
                        'input_tokens': 0,
                        'cache_write_tokens': 0,
                        'cache_write_1h_tokens': 0,
                        'cache_read_tokens': 0,
                        'output_tokens': 0,
                        'output_reasoning_tokens': 0,
                        'input_audio_tokens': 0,
                        'cache_audio_read_tokens': 0,
                        'output_audio_tokens': 0,
                        'audio_seconds': 0.0,
                        'details': {},
                        'cost': None,
                    },
                    'model_name': 'scripted',
                    'model_id': None,
                    'timestamp': 'NOW',
                    'kind': 'response',
                    'provider_name': None,
                    'provider_url': None,
                    'canonical_system': None,
                    'provider_details': None,
                    'provider_response_id': None,
                    'finish_reason': None,
                    'run_id': None,
                    'conversation_id': None,
                    'metadata': None,
                    'workspace_ref': None,
                    'history_digest': None,
                    'failed_attempts': None,
                    'state': 'complete',
                },
            },
            {
                'node': 'test_tool_plumbing.py::test_sampling_fails',
                'target': 'pydantic_ai.models.mcp_sampling.MCPSamplingModel.request',
                'model_settings': None,
                'error': 'ModelInvalidResponseError: Unexpected result from MCP sampling, expected "assistant" role, got user.',
            },
            {
                'node': 'test_tool_plumbing.py::test_sampling_fails',
                'target': 'test_tool_plumbing.FakeSession.create_message',
                'args': [
                    [
                        {
                            'role': 'user',
                            'content': {'type': 'text', 'text': 'hi', 'annotations': None, '_meta': None},
                            '_meta': None,
                        }
                    ]
                ],
                'kwargs': {
                    'max_tokens': 16384,
                    'system_prompt': '',
                    'temperature': None,
                    'model_preferences': None,
                    'stop_sequences': None,
                },
            },
        ]
    )
    assert [record['exception'] for record in records if record['kind'] == 'exception'] == snapshot(
        [
            {
                'type': 'pydantic_ai.exceptions.ModelInvalidResponseError',
                'message': 'Unexpected result from MCP sampling, expected "assistant" role, got user.',
                'suppress_context': True,
            },
            {
                'type': 'builtins.ValueError',
                'message': 'outer',
                'cause': {'type': 'builtins.KeyError', 'message': "'inner'"},
                'context': 'cause',
                'suppress_context': True,
            },
            {
                'type': 'builtins.TypeError',
                'message': 'replaced',
                'context': {'type': 'builtins.KeyError', 'message': "'handled'"},
                'suppress_context': True,
            },
            {'type': 'builtins.LookupError', 'message': 'noted', 'notes': ['a note']},
            {
                'type': 'builtins.ExceptionGroup',
                'message': 'both (2 sub-exceptions)',
                'exceptions': [
                    {'type': 'builtins.ValueError', 'message': 'a'},
                    {'type': 'builtins.RuntimeError', 'message': 'b'},
                ],
            },
        ]
    )
    assert [
        {key: value for key, value in record.items() if key != 'kind'}
        for record in records
        if record['kind'] == 'warning'
    ] == snapshot(
        [
            {
                'node': 'test_tool_plumbing.py::test_warnings',
                'category': 'builtins.UserWarning',
                'message': 'reported in the test module',
                'frame': {'module': 'test_tool_plumbing', 'function': 'test_warnings', 'line': 163},
            },
            {
                'node': 'test_tool_plumbing.py::test_warnings',
                'category': 'builtins.DeprecationWarning',
                'message': 'reported in the model',
                'frame': {'module': 'pydantic_ai._utils', 'function': 'await_maybe'},
            },
            {
                'node': 'test_tool_plumbing.py::test_warnings',
                'category': 'builtins.RuntimeWarning',
                'message': 'reported where told',
                'frame': {'file': 'elsewhere.py', 'line': 7},
            },
        ]
    )


def test_capture_records_run_event_streams_agent_stream_calls_and_spans(run: RunCapture, tmp_path: Path):
    """Each event stream a run's root capability hands its consumer, or a capability called directly does, each call
    on an `AgentStream` method that `AGENT_STREAM_ITERATORS` or `AGENT_STREAM_CALLS` lists, and each span the
    OpenTelemetry SDK ends.

    The events are the ones `ProcessEventStream` passes on, without the `PartEndEvent`s it drops, in one record per
    node stream: the stream of the capability it is, inside the root's, isn't recorded again. A processor that fails
    leaves the error in its stream's record. A message a tool enqueues adds the `EnqueuedMessagesEvent` its delivery
    emits. A call on a capability with a run context that has no root capability, as a test makes to call it directly,
    is recorded too. A call's record has its arguments, defaults filled in, and what it yields, returns or raises:
    `stream_output` validates each partial output, and swallows the error of one that's invalid. The spans are in the
    order they started, each parent named by its number, or as `external` when no span started for it; an exception
    event leaves out its stack trace, and a span ended twice is recorded once.
    """
    run(tmp_path, agent_runs=True).assert_outcomes(passed=10)

    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    assert [
        (record['node'].split('::')[1], [event['event_kind'] for event in record['events']], record.get('error'))
        for record in records
        if record['kind'] == 'run_event_stream'
    ] == snapshot(
        [
            ('test_run_event_streams', ['model_response_start', 'part_start', 'model_response_end'], None),
            ('test_run_event_streams', ['function_tool_call', 'function_tool_result'], None),
            (
                'test_run_event_streams',
                ['model_response_start', 'part_start', 'final_result', 'part_delta', 'model_response_end'],
                None,
            ),
            ('test_run_event_streams', [], None),
            ('test_enqueued_message', ['model_response_start', 'part_start', 'part_end', 'model_response_end'], None),
            ('test_enqueued_message', ['function_tool_call', 'function_tool_result'], None),
            (
                'test_enqueued_message',
                [
                    'enqueued_messages',
                    'model_response_start',
                    'part_start',
                    'final_result',
                    'part_delta',
                    'part_end',
                    'model_response_end',
                ],
                None,
            ),
            ('test_enqueued_message', [], None),
            ('test_run_event_stream_fails', ['model_response_start'], 'RuntimeError: processor failed'),
            ('test_run_event_stream_called_directly', ['part_start'], None),
            ('test_run_event_stream_called_directly', ['part_start'], None),
            ('test_agent_stream_calls', ['model_response_start', 'part_start', 'part_end', 'model_response_end'], None),
            ('test_agent_stream_calls', ['function_tool_call', 'function_tool_result'], None),
            (
                'test_agent_stream_calls',
                ['model_response_start', 'part_start', 'final_result', 'part_delta', 'part_end', 'model_response_end'],
                None,
            ),
            ('test_agent_stream_calls', ['model_response_start', 'part_start', 'part_end', 'model_response_end'], None),
            ('test_agent_stream_calls', ['function_tool_call', 'function_tool_result'], None),
            (
                'test_agent_stream_calls',
                ['model_response_start', 'part_start', 'final_result', 'part_delta', 'part_end', 'model_response_end'],
                None,
            ),
            (
                'test_agent_stream_calls',
                [
                    'model_response_start',
                    'part_start',
                    'final_result',
                    'part_delta',
                    'part_delta',
                    'part_end',
                    'model_response_end',
                ],
                None,
            ),
            (
                'test_agent_stream_calls',
                ['model_response_start', 'part_start', 'final_result', 'part_end', 'model_response_end'],
                None,
            ),
            (
                'test_agent_stream_calls',
                ['model_response_start', 'part_start', 'final_result', 'model_response_end'],
                'RuntimeError: stream failed',
            ),
            ('test_spans', ['function_tool_call'], 'LookupError: tool failed'),
            (
                'test_unformattable_error',
                ['model_response_start'],
                {'unrecorded': 'Unformattable: `str()` raised RuntimeError'},
            ),
            ('test_tool_metadata[test]', ['function_tool_call', 'function_tool_result'], None),
            ('test_tool_metadata[test]', [], None),
            ('test_tool_metadata[sampling]', [], None),
        ]
    )
    assert [
        (
            record['node'].split('::')[1],
            record['target'].rsplit('.', 1)[1],
            record['arguments'],
            record.get('yields'),
            record.get('result'),
            record.get('error'),
        )
        for record in records
        if record['kind'] == 'agent_stream'
    ] == snapshot(
        [
            (
                'test_agent_stream_calls',
                'stream_text',
                {'delta': False, 'debounce_by': None},
                ['he', 'hello'],
                None,
                None,
            ),
            ('test_agent_stream_calls', 'get_output', {}, None, 'hello', None),
            ('test_agent_stream_calls', 'validate_response_output', {'allow_partial': False}, None, 'hello', None),
            ('test_agent_stream_calls', 'get_output', {}, None, 'hello', None),
            ('test_agent_stream_calls', 'stream_text', {'delta': True, 'debounce_by': None}, ['he', 'llo'], None, None),
            ('test_agent_stream_calls', 'get_output', {}, None, 'hello', None),
            ('test_agent_stream_calls', 'validate_response_output', {'allow_partial': False}, None, 'hello', None),
            (
                'test_agent_stream_calls',
                'stream_output',
                {'debounce_by': None},
                [{'value': 4}, {'value': 42}, {'value': 42}],
                None,
                None,
            ),
            (
                'test_agent_stream_calls',
                'validate_response_output',
                {'allow_partial': True},
                None,
                None,
                """\
ValidationError: 1 validation error for Answer
value
  Field required [type=missing, input_value={}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.14/v/missing\
""",
            ),
            ('test_agent_stream_calls', 'validate_response_output', {'allow_partial': True}, None, {'value': 4}, None),
            ('test_agent_stream_calls', 'validate_response_output', {'allow_partial': True}, None, {'value': 42}, None),
            (
                'test_agent_stream_calls',
                'validate_response_output',
                {'allow_partial': False},
                None,
                {'value': 42},
                None,
            ),
            ('test_agent_stream_calls', 'get_output', {}, None, {'value': 42}, None),
            (
                'test_agent_stream_calls',
                'get_output',
                {},
                None,
                None,
                'UnexpectedModelBehavior: Output validation failed during streaming, and retries are not supported in `run_stream()`',
            ),
            (
                'test_agent_stream_calls',
                'validate_response_output',
                {'allow_partial': False},
                None,
                None,
                'UnexpectedModelBehavior: Output validation failed during streaming, and retries are not supported in `run_stream()`',
            ),
            (
                'test_agent_stream_calls',
                'stream_text',
                {'delta': False, 'debounce_by': None},
                ['he'],
                None,
                'RuntimeError: stream failed',
            ),
        ]
    )
    spans = [record for record in records if record['kind'] == 'span']
    assert [
        (
            record['node'].split('::')[1],
            record['number'],
            record['parent'],
            record['name'],
            record['span_kind'],
            record['scope'],
            record['status'],
            record['events'],
        )
        for record in spans
    ] == snapshot(
        [
            (
                'test_spans',
                0,
                None,
                'invoke_agent agent',
                'INTERNAL',
                'pydantic-ai',
                {'code': 'ERROR', 'description': 'LookupError: tool failed'},
                [
                    {
                        'name': 'exception',
                        'attributes': {
                            'exception.type': 'LookupError',
                            'exception.message': 'tool failed',
                            'exception.escaped': 'False',
                        },
                    }
                ],
            ),
            (
                'test_spans',
                1,
                0,
                'chat function:call_and_fail:',
                'CLIENT',
                'pydantic-ai',
                {'code': 'UNSET', 'description': None},
                [],
            ),
            (
                'test_spans',
                2,
                0,
                'execute_tool fail',
                'INTERNAL',
                'pydantic-ai',
                {'code': 'ERROR', 'description': None},
                [
                    {
                        'name': 'exception',
                        'attributes': {
                            'exception.type': 'LookupError',
                            'exception.message': 'tool failed',
                            'exception.escaped': 'True',
                        },
                    }
                ],
            ),
            (
                'test_spans_of_a_tracer',
                0,
                'external',
                'outer',
                'INTERNAL',
                'test',
                {'code': 'UNSET', 'description': None},
                [],
            ),
            ('test_spans_of_a_tracer', 1, 0, 'inner', 'INTERNAL', 'test', {'code': 'UNSET', 'description': None}, []),
        ]
    )
    assert next(record['attributes'] for record in spans if record['name'] == 'execute_tool fail') == snapshot(
        {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': 'fail',
            'gen_ai.tool.call.id': 'fail',
            'gen_ai.tool.call.arguments': '{}',
            'gen_ai.agent.name': 'agent',
            'gen_ai.agent.call.id': 'UUID',
            'gen_ai.conversation.id': 'UUID',
            'logfire.msg': 'running tool: fail',
            'logfire.json_schema': '{"type":"object","properties":{"gen_ai.tool.call.arguments":{"type":"object"},"gen_ai.tool.call.result":{"type":"object"},"gen_ai.tool.name":{},"gen_ai.tool.call.id":{}}}',
        }
    )


def test_capture_records_an_unreadable_error_message_as_unrecorded(run: RunCapture, tmp_path: Path):
    """An exception whose `__str__` raises, which a run event stream raises and a `pytest.raises` block catches, has its
    message recorded as `unrecorded`, with the class of the error `__str__` raised, instead of the capture raising that
    error into the test.
    """
    run(tmp_path, agent_runs=True).assert_outcomes(passed=10)

    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    assert [
        {key: value for key, value in record.items() if key in ('kind', 'error', 'exception')}
        for record in records
        if record['node'].endswith('::test_unformattable_error') and record['kind'] in ('run_event_stream', 'exception')
    ] == snapshot(
        [
            {'kind': 'run_event_stream', 'error': {'unrecorded': 'Unformattable: `str()` raised RuntimeError'}},
            {
                'kind': 'exception',
                'exception': {
                    'type': 'test_agent_runs.Unformattable',
                    'message': {'unrecorded': '`str()` raised RuntimeError'},
                    'suppress_context': True,
                },
            },
        ]
    )


def test_capture_records_only_the_tool_metadata_json_can_write(run: RunCapture, tmp_path: Path):
    """A request record of an in-process model or an `MCPSamplingModel`, and a hook record's tool definition, keep each
    tool metadata value `json` can write, and record a model as `unrecorded`, with its class, without serializing it:
    the test that counts its serializer's calls passes.
    """
    run(tmp_path, agent_runs=True).assert_outcomes(passed=10)

    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    assert [
        (
            record['node'].split('::')[1],
            record['target'].rsplit('.', 2)[1:],
            [tool['metadata'] for tool in record['model_request_parameters']['function_tools']]
            if record['kind'] == 'request'
            else record['kwargs']['tool_def']['metadata'],
        )
        for record in records
        if '::test_tool_metadata[' in record['node'] and record['kind'] in ('request', 'hook')
    ] == snapshot(
        [
            (
                'test_tool_metadata[test]',
                ['TestModel', 'request'],
                [
                    {
                        'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                        'plain': {'kind': 'lookup', 'tags': ['a']},
                    }
                ],
            ),
            (
                'test_tool_metadata[test]',
                ['CombinedCapability', 'wrap_tool_validate'],
                {
                    'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                    'plain': {'kind': 'lookup', 'tags': ['a']},
                },
            ),
            (
                'test_tool_metadata[test]',
                ['CombinedCapability', 'before_tool_validate'],
                {
                    'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                    'plain': {'kind': 'lookup', 'tags': ['a']},
                },
            ),
            (
                'test_tool_metadata[test]',
                ['CombinedCapability', 'after_tool_validate'],
                {
                    'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                    'plain': {'kind': 'lookup', 'tags': ['a']},
                },
            ),
            (
                'test_tool_metadata[test]',
                ['CombinedCapability', 'wrap_tool_execute'],
                {
                    'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                    'plain': {'kind': 'lookup', 'tags': ['a']},
                },
            ),
            (
                'test_tool_metadata[test]',
                ['CombinedCapability', 'before_tool_execute'],
                {
                    'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                    'plain': {'kind': 'lookup', 'tags': ['a']},
                },
            ),
            (
                'test_tool_metadata[test]',
                ['CombinedCapability', 'after_tool_execute'],
                {
                    'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                    'plain': {'kind': 'lookup', 'tags': ['a']},
                },
            ),
            (
                'test_tool_metadata[test]',
                ['TestModel', 'request'],
                [
                    {
                        'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                        'plain': {'kind': 'lookup', 'tags': ['a']},
                    }
                ],
            ),
            (
                'test_tool_metadata[sampling]',
                ['MCPSamplingModel', 'request'],
                [
                    {
                        'tracked': {'unrecorded': 'test_agent_runs.test_tool_metadata.<locals>.Tracked'},
                        'plain': {'kind': 'lookup', 'tags': ['a']},
                    }
                ],
            ),
        ]
    )


@pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ('temporalio', 'dbos', 'prefect')),
    reason='`temporalio`, `dbos` or `prefect` not installed',
)
# `DBOS` is one instance per process, which the DBOS test modules launch once per module on their worker.
@pytest.mark.xdist_group(name='dbos')
def test_capture_records_durable_names_workflows_and_cache_keys(
    run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Each name `pydantic_ai.durable_exec` registers, each Temporal workflow run's commands and DBOS workflow's
    operations, and each `compute_key` call on a Prefect cache policy.

    A name registered while the module is collected is recorded under the module's node; one the test registers itself
    isn't recorded. The Temporal worker polls and completes activations through a stand-in for its core: the runs are
    recorded in a set order, a run context's set sorted, the activation that evicts a run issues nothing, and a
    failure's stack trace is left out. A forked DBOS workflow is recorded with its original's name, and its operations
    show which steps it replayed. The compound default policy's key, the key of the policy for the flow run's ID it
    combines, and a key computed from a generated ID are run-scoped; the calls handed the same inputs as the call
    they're made in leave them out, and inputs that hold a reference cycle are recorded as the error writing them
    raised.

    The Prefect modules are imported before the capture starts, as a conftest can import them, so they hold the `task`
    decorator they imported, which the capture rebinds.

    Compared with a capture of the Temporal and cache key tests carrying other text, the summary counts each durable
    kind and its targets, and lists the tests whose durable records changed.
    """
    importlib.import_module('pydantic_ai.durable_exec.prefect')
    run(tmp_path, durable=True).assert_outcomes(passed=4)

    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    assert [
        (record['node'], record['target'], record['name'], record['options'])
        for record in records
        if record['kind'] == 'durable_name'
    ] == snapshot(
        [
            ('test_capture_durable.py', 'temporalio.activity.defn', 'agent__collected__external_event_ask', {}),
            (
                'test_capture_durable.py',
                'temporalio.activity.defn',
                'agent__collected__model_cancel_suspended_response',
                {},
            ),
            ('test_capture_durable.py', 'temporalio.activity.defn', 'agent__collected__model_compact_messages', {}),
            ('test_capture_durable.py', 'temporalio.activity.defn', 'agent__collected__model_count_tokens', {}),
            ('test_capture_durable.py', 'temporalio.activity.defn', 'agent__collected__model_request', {}),
            ('test_capture_durable.py', 'temporalio.activity.defn', 'agent__collected__model_request_stream', {}),
            (
                'test_capture_durable.py',
                'temporalio.activity.defn',
                'agent__collected__toolset__<agent>__call_tool',
                {},
            ),
            (
                'test_capture_durable.py',
                'temporalio.activity.defn',
                'agent__collected__toolset__<agent>__validate_args',
                {},
            ),
            (
                'test_capture_durable.py',
                'temporalio.workflow.defn',
                'agent__collected__toolset__<agent>__call_tool__child_workflow',
                {},
            ),
            (
                'test_capture_durable.py::test_durable_names',
                'dbos.DBOS.step',
                'dbos_agent__model.cancel_suspended_response',
                {},
            ),
            ('test_capture_durable.py::test_durable_names', 'dbos.DBOS.step', 'dbos_agent__model.compact_messages', {}),
            ('test_capture_durable.py::test_durable_names', 'dbos.DBOS.step', 'dbos_agent__model.count_tokens', {}),
            ('test_capture_durable.py::test_durable_names', 'dbos.DBOS.step', 'dbos_agent__model.request', {}),
            ('test_capture_durable.py::test_durable_names', 'dbos.DBOS.step', 'dbos_agent__model.request_stream', {}),
            ('test_capture_durable.py::test_durable_names', 'dbos.DBOS.workflow', 'dbos_agent.run', {}),
            ('test_capture_durable.py::test_durable_names', 'dbos.DBOS.workflow', 'dbos_agent.run_sync', {}),
            ('test_capture_durable.py::test_durable_names', 'prefect.task', 'call_tool_task', {}),
            ('test_capture_durable.py::test_durable_names', 'prefect.task', 'cancel_suspended_response_task', {}),
            ('test_capture_durable.py::test_durable_names', 'prefect.task', 'request_stream_task', {}),
            ('test_capture_durable.py::test_durable_names', 'prefect.task', 'wrapped_request', {}),
        ]
    )
    assert [
        {key: value for key, value in record.items() if key != 'kind'}
        for record in records
        if record['kind'] in ('temporal', 'dbos')
    ] == snapshot(
        [
            {
                'node': 'test_capture_durable.py::test_temporal_commands',
                'target': 'First',
                'workflow_id': 'first-UUID',
                'commands': [
                    {
                        'schedule_activity': {
                            'seq': 1,
                            'activity_id': '1',
                            'activity_type': 'agent__model_request',
                            'arguments': [
                                {
                                    'metadata': {'encoding': 'json/plain'},
                                    'data': {
                                        'serialized_run_context': {'active_capability_ids': ['agent', 'tool_search']}
                                    },
                                },
                                {
                                    'metadata': {'encoding': 'json/plain'},
                                    'data': {
                                        'items': [
                                            {
                                                'topic': 'agent_events',
                                                'data': {
                                                    'metadata': {'encoding': 'json/plain'},
                                                    'data': {'tool_call_id': 'pyd_ai_ID'},
                                                },
                                                'offset': 0,
                                            }
                                        ],
                                        'topic': 'not a stream item',
                                    },
                                },
                            ],
                        }
                    },
                    {
                        'schedule_activity': {
                            'seq': 2,
                            'activity_id': '2',
                            'activity_type': 'raw',
                            'arguments': [{'metadata': {'encoding': 'binary/plain'}, 'data': 'cmF3'}],
                        }
                    },
                    {'unrecorded': 'JSONDecodeError: Expecting value: line 1 column 1 (char 0)'},
                    {
                        'complete_workflow_execution': {
                            'result': {'metadata': {'encoding': 'json/plain'}, 'data': 'hello'}
                        }
                    },
                ],
            },
            {
                'node': 'test_capture_durable.py::test_temporal_commands',
                'target': 'Second',
                'workflow_id': 'second',
                'commands': [{'failed': {'failure': {'message': 'boom'}}}],
            },
            {
                'node': 'test_capture_durable.py::test_dbos_workflow',
                'target': 'test_dbos_workflow.<locals>.quadruple',
                'workflow_id': 'UUID',
                'class_name': None,
                'config_name': None,
                'parent_workflow_id': None,
                'forked_from': 'UUID',
                'operations': [
                    {'function_id': 1, 'function_name': 'test_dbos_workflow.<locals>.double', 'recorded': True},
                    {'function_id': 2, 'function_name': 'test_dbos_workflow.<locals>.double', 'recorded': False},
                ],
            },
            {
                'node': 'test_capture_durable.py::test_dbos_workflow',
                'target': 'test_dbos_workflow.<locals>.quadruple',
                'workflow_id': 'UUID',
                'class_name': None,
                'config_name': None,
                'parent_workflow_id': None,
                'operations': [
                    {'function_id': 1, 'function_name': 'test_dbos_workflow.<locals>.double', 'recorded': False},
                    {'function_id': 2, 'function_name': 'test_dbos_workflow.<locals>.double', 'recorded': False},
                ],
            },
        ]
    )
    assert [
        {key: value for key, value in record.items() if key not in ('kind', 'node')}
        for record in records
        if record['kind'] == 'cache_key'
    ] == snapshot(
        [
            {
                'target': 'prefect.cache_policies.CompoundCachePolicy.compute_key',
                'inputs': {'prompt': 'hello'},
                'key': 'RUN_SCOPED',
            },
            {
                'target': 'pydantic_ai.durable_exec.prefect._cache_policies.PrefectAgentInputs.compute_key',
                'key': 'afa8b1aa08166a4e863e973bd4acb59f',
            },
            {
                'target': 'prefect.cache_policies.Inputs.compute_key',
                'inputs': {'prompt': 'hello'},
                'key': 'afa8b1aa08166a4e863e973bd4acb59f',
            },
            {'target': 'prefect.cache_policies.TaskSource.compute_key', 'key': 'b75d1a1f59ed47f1109c2d723e3866c3'},
            {'target': 'prefect.cache_policies.RunId.compute_key', 'key': 'RUN_SCOPED'},
            {
                'target': 'prefect.cache_policies.Inputs.compute_key',
                'inputs': {'generated': 'UUID'},
                'key': 'RUN_SCOPED',
            },
            {
                'target': 'prefect.cache_policies.Inputs.compute_key',
                'inputs': {'lock': '<unlocked _thread.lock object at 0x?>'},
                'error': """\
ValueError: Unable to create hash - objects could not be serialized.
  JSON error: Unable to serialize unknown type: <class '_thread.lock'>
  Pickle error: cannot pickle '_thread.lock' object

This often occurs when task inputs contain objects that cannot be cached like locks, file handles, or other system resources.

To resolve this, you can:
  1. Exclude these arguments by defining a custom `cache_key_fn`
  2. Disable caching by passing `cache_policy=NO_CACHE`
""",
            },
            {
                'target': 'prefect.cache_policies.Inputs.compute_key',
                'inputs': {
                    'unrecorded': 'PydanticSerializationError: Error serializing to JSON: ValueError: Circular reference detected (id repeated)'
                },
                'key': 'b4e74d9d48bc91343a5a0b0553ebf41e',
            },
            {
                'target': 'test_capture_durable.test_cache_keys.<locals>.Inherited.compute_key',
                'inputs': {'n': 1},
                'key': 'ca4f55a38abb6a072c674e3ffedc365a',
            },
        ]
    )

    head = tmp_path / 'head'
    run(head, '-k', 'temporal_commands or cache_keys', durable=True, text='goodbye').assert_outcomes(
        passed=2, deselected=2
    )
    (head / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{head}/pydantic_ai/__init__.py'}))
    capsys.readouterr()
    assert compare(tmp_path, head) == 1
    assert capsys.readouterr().out == snapshot("""\
nodes: base 5, head 3
records: base 35, head 21
stream events: head 0
requests: head 0
ui records: head 0
realtime records: head 0
session records: head 0
hook records: head 0
run event stream records: head 0
agent stream records: head 0
span records: head 0
exception records: head 1
warning records: head 0
temporal workflow run records: head 2
dbos workflow records: head 0
durable name records: head 9
cache key records: head 9
nodes per stream module, head:
nodes per request target, head:
nodes per ui target, head:
nodes per realtime target, head:
nodes per session target, head:
nodes per hook target, head:
nodes per run event kind, head:
nodes per agent stream target, head:
nodes per span name, head:
nodes per temporal workflow run target, head:
  First: 1
  Second: 1
nodes per dbos workflow target, head:
nodes per durable name target, head:
  temporalio.activity.defn: 1
  temporalio.workflow.defn: 1
nodes per cache key target, head:
  prefect.cache_policies.CompoundCachePolicy.compute_key: 1
  prefect.cache_policies.Inputs.compute_key: 1
  prefect.cache_policies.RunId.compute_key: 1
  prefect.cache_policies.TaskSource.compute_key: 1
  pydantic_ai.durable_exec.prefect._cache_policies.PrefectAgentInputs.compute_key: 1
  test_capture_durable.test_cache_keys.<locals>.Inherited.compute_key: 1
nodes that recorded no request, realtime or session record, head: 2
  test_capture_durable.py::test_cache_keys
  test_capture_durable.py::test_temporal_commands
only in base: 2, only in head: 0, differing: 2
  only in base: test_capture_durable.py::test_dbos_workflow
  only in base: test_capture_durable.py::test_durable_names
  differing: test_capture_durable.py::test_cache_keys
  differing: test_capture_durable.py::test_temporal_commands
""")


async def test_capture_holds_a_stream_only_while_it_lives(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A stream holds what the run iterating it holds, like its dependencies, so the capture holds it only while the
    recording of its iteration lives. The response it builds is recorded once iteration ends, and again when the test
    ends if the stream still lives then, as it is when the test ends, so what the run sets on it in between shows, even
    once the stream is gone; one that can't be built or written then is recorded as the error, never raised.

    Garbage collection is off, as in `tests/test_run_reference_cycles.py`, so only reference counting frees them.
    """
    # The capture names its records file after the xdist worker it runs on.
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)

    class Deps:
        pass

    class Unwritable:
        def __repr__(self) -> str:
            raise RuntimeError('unwritable')

    def unbuildable() -> ModelResponse:
        raise RuntimeError('unbuildable')

    async def stream_text(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello'

    agent = Agent(FunctionModel(stream_function=stream_text), deps_type=Deps)
    capture = RefactorCapture(tmp_path)
    capture.install()
    gc.disable()
    try:
        deps = Deps()
        ref = weakref.ref(deps)
        async with agent.run_stream('hello', deps=deps, infer_name=False) as result:
            assert await result.get_output() == 'hello'
        del result, deps
        assert ref() is None
        async with model_request_stream(
            FunctionModel(stream_function=stream_text), [ModelRequest.user_text_prompt('hi')]
        ) as kept:
            async for _ in kept:
                pass
        kept.provider_details = {'set': 'once iteration ended'}
        async with model_request_stream(
            FunctionModel(stream_function=stream_text), [ModelRequest.user_text_prompt('hi')]
        ) as dropped:
            async for _ in dropped:
                pass
        dropped.usage.details['counted_once_iteration_ended'] = 1
        dropped_ref = weakref.ref(dropped)
        del dropped
        assert dropped_ref() is None
        async with model_request_stream(
            FunctionModel(stream_function=stream_text), [ModelRequest.user_text_prompt('hi')]
        ) as unwritable:
            unwritable.provider_details = {'unwritable': Unwritable()}
            async for _ in unwritable:
                pass
        async with model_request_stream(
            FunctionModel(stream_function=stream_text), [ModelRequest.user_text_prompt('hi')]
        ) as unbuilt:
            async for _ in unbuilt:
                pass
        # Built once iteration ended, its response can't be built when the test ends.
        monkeypatch.setattr(unbuilt, 'get', unbuildable)
        capture.write('test_streams.py::test_kept')
    finally:
        gc.enable()
        capture.pytest_unconfigure()
    records = [json.loads(line) for line in (tmp_path / 'main.jsonl').read_text().splitlines()]
    *streams, counted, unwritten, unbuilt = [record for record in records if record['kind'] == 'stream']
    assert counted['response']['usage']['details'] == {'counted_once_iteration_ended': 1}
    assert [unwritten['response'], unbuilt['response']] == snapshot(
        [
            {'unrecorded': 'PydanticSerializationError: Error serializing to JSON: RuntimeError: unwritable'},
            {'unrecorded': 'RuntimeError: unbuildable'},
        ]
    )
    assert [
        (record['stream'], record['response']['parts'][0]['content'], record['response']['provider_details'])
        for record in streams
    ] == snapshot(
        [
            ('pydantic_ai.models._continuation._ContinuationStreamedResponse', 'hello', None),
            ('pydantic_ai.models.function.FunctionStreamedResponse', 'hello', None),
            ('pydantic_ai.models.function.FunctionStreamedResponse', 'hello', {'set': 'once iteration ended'}),
        ]
    )


def test_capture_wraps_a_module_now_or_once_it_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A module imported already is wrapped at once, and one imported later once it has run, leaving its loader as it
    was; a spec found but never run leaves the wrap for the import. Other imports pass through, and a module that can't
    be found isn't wrapped.
    """
    (tmp_path / 'watched_module.py').write_text('VALUE = 1\n')
    (tmp_path / 'unwatched_module.py').write_text('')
    monkeypatch.syspath_prepend(str(tmp_path))  # pyright: ignore[reportUnknownMemberType]
    wrapped: list[object] = []
    capture = RefactorCapture(tmp_path)
    capture.wrap_on_import(
        {
            'json': lambda: wrapped.append('json'),
            'watched_module': lambda: wrapped.append(vars(sys.modules['watched_module'])['VALUE']),
            'missing_module': lambda: wrapped.append('missing'),
        }
    )
    try:
        assert wrapped == ['json']
        importlib.import_module('unwatched_module')
        spec = importlib.util.find_spec('watched_module')
        assert spec is not None and wrapped == ['json']
        # Garbage collection is off, so only reference counting frees the loader of a spec that never runs.
        gc.disable()
        try:
            not_run = weakref.ref(spec.loader)
            del spec
            assert not_run() is None
        finally:
            gc.enable()
        module = importlib.import_module('watched_module')
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module('missing_module')
    finally:
        capture.pytest_unconfigure()
        for name in ('watched_module', 'unwatched_module'):
            sys.modules.pop(name, None)
    assert wrapped == ['json', 1]
    assert 'exec_module' not in vars(module.__loader__)
    assert not any(isinstance(finder, ImportWatcher) for finder in sys.meta_path)


async def test_capture_keeps_what_ui_adapter_subclasses_are(tmp_path: Path):
    """A subclass defined once the capture started keeps its type parameters, and its abstract methods."""
    capture = RefactorCapture(tmp_path)
    capture.install()
    try:

        class Concrete(VercelAIEventStream[None, str]):
            pass

        class Partial(UIAdapter[None, None, None, None, None]):
            pass

    finally:
        capture.pytest_unconfigure()

    class Uncaptured(UIAdapter[None, None, None, None, None]):
        pass

    # `Generic.__init_subclass__` sets them, from the bases the class was defined with.
    assert vars(Concrete)['__parameters__'] == ()
    assert Partial.__abstractmethods__ == Uncaptured.__abstractmethods__
    assert sorted(Partial.__abstractmethods__) == snapshot(['build_event_stream', 'build_run_input', 'load_messages'])


def test_capture_builds_a_class_in_the_namespace_its_statement_runs_in(tmp_path: Path):
    """A model keeps the namespace of the frame its class statement runs in, or none for one that runs in a module, as
    without the capture: Pydantic resolves the model's annotations in it. A namespace that held the capture couldn't be
    pickled, so Prefect couldn't hash the model. A warning a class raises as it's defined reports the statement's line.
    """

    class Warns:
        def __init_subclass__(cls) -> None:
            warnings.warn('defined', UserWarning, stacklevel=2)

    def define() -> type[Warns]:
        class Subclass(Warns):
            pass

        return Subclass

    with pytest.warns(UserWarning, match='defined') as uncaptured:
        define()
    module: dict[str, object] = {}
    capture = RefactorCapture(tmp_path)
    capture.install()
    try:

        class Inner(BaseModel):
            value: int

        class Outer(BaseModel):
            inner: Inner

        exec('from pydantic import BaseModel\nclass AtModuleLevel(BaseModel):\n    value: int\n', module)
        with pytest.warns(UserWarning, match='defined') as captured:
            define()
    finally:
        capture.pytest_unconfigure()
    assert [(warning.filename, warning.lineno) for warning in captured] == [
        (warning.filename, warning.lineno) for warning in uncaptured
    ]
    assert 'Inner' in vars(Outer)['__pydantic_parent_namespace__']
    assert vars(module['AtModuleLevel'])['__pydantic_parent_namespace__'] is None


@pytest.mark.skipif(sys.version_info < (3, 12), reason='a PEP 695 generic class needs Python 3.12')
def test_capture_builds_a_generic_class_in_the_namespace_its_statement_runs_in(tmp_path: Path):
    """A PEP 695 generic class statement runs in an annotation scope, which Pydantic looks past to the frame the
    statement runs in, as without the capture.
    """
    module: dict[str, object] = {}
    exec(
        'from pydantic import BaseModel\n'
        'def define():\n'
        '    class Inner(BaseModel):\n'
        '        value: int\n'
        '    class Generic[T](BaseModel):\n'
        '        inner: Inner\n'
        '        value: T\n'
        '    return Generic\n',
        module,
    )
    define = module['define']
    assert callable(define)
    capture = RefactorCapture(tmp_path)
    capture.install()
    try:
        generic = define()
    finally:
        capture.pytest_unconfigure()
    assert 'Inner' in vars(generic)['__pydantic_parent_namespace__']


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
    """A canary stream, run event stream, `AgentStream` call or span the capture doesn't see fails the run instead of
    the capture recording nothing.

    The capture's wrappers aren't installed, standing in for streams iterated, and spans ended, some way they don't
    reach.
    """
    unwrapped = StreamedResponse.__aiter__
    with pytest.raises(pytest.UsageError) as exc_info:
        RefactorCapture(tmp_path).check_stream_hook()
    assert str(exc_info.value) == snapshot(
        'The stream events and run event stream and `AgentStream` calls and spans of a canary agent run were not recorded, so the capture would miss them. Update `tests/refactor_capture.py` to the way they are made now.'
    )
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


async def test_capture_leaves_out_a_sigv4_session_token(tmp_path: Path):
    """A request SigV4 signs with a session token is recorded without the token, which is a credential."""
    sent: list[httpx2.Request] = []

    def answer_400(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(400)

    capture = RefactorCapture(tmp_path)
    capture.install()
    try:
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(answer_400)) as http_client:
            client = AsyncAnthropicBedrock(
                aws_access_key='canary',
                aws_secret_key='canary',
                aws_session_token='canary-session-token',
                aws_region='us-east-1',
                http_client=http_client,
            )
            model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
            with override_allow_model_requests(True), suppress(ModelHTTPError):
                await Agent(model).run('hello')
    finally:
        capture.pytest_unconfigure()

    assert [request.headers['x-amz-security-token'] for request in sent] == ['canary-session-token']
    assert len(capture.records) == 1
    assert 'canary-session-token' not in json.dumps(capture.records)


@pytest.mark.skipif(importlib.util.find_spec('prefect') is None, reason='`prefect` not installed')
async def test_capture_records_a_prefect_request_once_and_not_the_ones_prefect_sends_itself(tmp_path: Path):
    """Prefect's client retries a request by sending it again, which the capture records once.

    The log batches, server version checks and CSRF token fetches it sends of its own accord aren't recorded at all. When
    its test server is busy, the client retries, and its background worker sends logs, checks the version and fetches a
    token in whichever test is running.
    """
    from prefect.client.orchestration import PrefectClient

    sent: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        if request.url.path == '/api/csrf-token':
            # What a server without CSRF tokens answers, so the client stops asking for one.
            return httpx.Response(404)
        if sent.count(request.url.path) == 1:
            return httpx.Response(503, headers={'Retry-After': '0'})
        return httpx.Response(200, json={})

    capture = RefactorCapture(tmp_path)
    capture.install()
    try:
        async with PrefectClient(
            'http://127.0.0.1:4200/api', httpx_settings={'transport': httpx.MockTransport(answer)}
        ) as client:
            assert await client.api_healthcheck() is None
            await client.api_version()
            await client.create_logs([])
    finally:
        capture.pytest_unconfigure()

    assert sent == snapshot(
        [
            '/api/health',
            '/api/health',
            '/api/admin/version',
            '/api/admin/version',
            '/api/csrf-token',
            '/api/logs/',
            '/api/logs/',
        ]
    )
    assert [record['target'] for record in capture.records] == snapshot(['GET http://127.0.0.1:PORT/api/health'])


async def test_capture_holds_no_error_a_tool_hook_raises(tmp_path: Path):
    """The error a tool raises passes through the recorded `on_tool_execute_error` hook, whose frame its traceback
    holds, so a frame that held the error would keep the run's dependencies in a cycle with it.

    Garbage collection is off, as in `tests/test_run_reference_cycles.py`, so only reference counting frees them.
    """

    class Deps:
        pass

    async def call_fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('fail')])

    agent = Agent(FunctionModel(call_fail), deps_type=Deps)

    # A sync model function or tool runs on a worker thread, which can drop the context it copied only once the run has
    # returned.
    @agent.tool_plain
    async def fail() -> str:
        raise RuntimeError('tool failed')

    capture = RefactorCapture(tmp_path)
    capture.install()
    gc.disable()
    try:
        deps = Deps()
        ref = weakref.ref(deps)
        try:
            await agent.run('hello', deps=deps, infer_name=False)
        except RuntimeError:
            pass
        del deps
        assert ref() is None
    finally:
        gc.enable()
        capture.pytest_unconfigure()
    assert [
        str(record['target']).rsplit('.', 1)[-1] for record in capture.records if record['kind'] == 'hook'
    ] == snapshot(
        [
            'wrap_tool_validate',
            'before_tool_validate',
            'after_tool_validate',
            'wrap_tool_execute',
            'before_tool_execute',
            'on_tool_execute_error',
        ]
    )


def test_capture_records_no_warning_while_no_test_runs(tmp_path: Path):
    """A warning raised outside a test, like one an import during collection raises, belongs to no test node."""
    capture = RefactorCapture(tmp_path)
    capture.install()
    try:
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            warnings.warn('outside a test', UserWarning)
    finally:
        capture.pytest_unconfigure()

    assert [str(warning.message) for warning in shown] == ['outside a test']
    assert capture.records == []


async def test_capture_normalizes_the_custom_id_of_an_in_memory_batch_request(tmp_path: Path):
    """`TestModel`'s batches encode each request as a random token, which the `custom_id` it's sent with hashes."""
    agent = Agent(TestModel(), capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    @agent.tool_plain
    def pick() -> str:
        return 'a'

    capture = RefactorCapture(tmp_path)
    capture.install()
    try:
        await agent.run('hello')
    finally:
        capture.pytest_unconfigure()

    # The second request carries the first response, which the batch answered.
    assert re.findall(r'"(?:custom_id|provider_response_id)": "[^"]*"', json.dumps(capture.records)) == snapshot(
        ['"custom_id": "CUSTOM_ID"', '"provider_response_id": "batch:batch_UUID:CUSTOM_ID"']
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
records: base 26, head 26
stream events: head 11
requests: head 10
ui records: head 0
realtime records: head 0
session records: head 0
hook records: head 6
run event stream records: head 4
agent stream records: head 0
span records: head 0
exception records: head 1
warning records: head 1
temporal workflow run records: head 0
dbos workflow records: head 0
durable name records: head 0
cache key records: head 0
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
nodes per ui target, head:
nodes per realtime target, head:
nodes per session target, head:
nodes per hook target, head:
  pydantic_ai.capabilities.combined.CombinedCapability.after_tool_execute: 1
  pydantic_ai.capabilities.combined.CombinedCapability.after_tool_validate: 1
  pydantic_ai.capabilities.combined.CombinedCapability.before_tool_execute: 1
  pydantic_ai.capabilities.combined.CombinedCapability.before_tool_validate: 1
  pydantic_ai.capabilities.combined.CombinedCapability.wrap_tool_execute: 1
  pydantic_ai.capabilities.combined.CombinedCapability.wrap_tool_validate: 1
nodes per run event kind, head:
  function_tool_call: 1
  function_tool_result: 1
nodes per agent stream target, head:
nodes per span name, head:
nodes per temporal workflow run target, head:
nodes per dbos workflow target, head:
nodes per durable name target, head:
nodes per cache key target, head:
nodes that recorded no request, realtime or session record, head: 2
  test_streams.py::test_no_stream
  test_streams.py::test_subclass_defined_after_the_capture_started
only in base: 0, only in head: 0, differing: 0
nodes: base 10, head 9
records: base 26, head 23
stream events: head 9
requests: head 9
ui records: head 0
realtime records: head 0
session records: head 0
hook records: head 6
run event stream records: head 4
agent stream records: head 0
span records: head 0
exception records: head 0
warning records: head 1
temporal workflow run records: head 0
dbos workflow records: head 0
durable name records: head 0
cache key records: head 0
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
nodes per ui target, head:
nodes per realtime target, head:
nodes per session target, head:
nodes per hook target, head:
  pydantic_ai.capabilities.combined.CombinedCapability.after_tool_execute: 1
  pydantic_ai.capabilities.combined.CombinedCapability.after_tool_validate: 1
  pydantic_ai.capabilities.combined.CombinedCapability.before_tool_execute: 1
  pydantic_ai.capabilities.combined.CombinedCapability.before_tool_validate: 1
  pydantic_ai.capabilities.combined.CombinedCapability.wrap_tool_execute: 1
  pydantic_ai.capabilities.combined.CombinedCapability.wrap_tool_validate: 1
nodes per run event kind, head:
  function_tool_call: 1
  function_tool_result: 1
nodes per agent stream target, head:
nodes per span name, head:
nodes per temporal workflow run target, head:
nodes per dbos workflow target, head:
nodes per durable name target, head:
nodes per cache key target, head:
nodes that recorded no request, realtime or session record, head: 2
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
    # The model's deprecation warning is recorded too.
    (record,) = [
        record
        for line in (base / 'main.jsonl').read_text().splitlines()
        if (record := json.loads(line))['kind'] == 'request'
    ]
    assert record['headers']['anthropic-beta'] == snapshot('files-api-2025-04-14')
    assert {'x-api-key', 'authorization'}.isdisjoint(record['headers'])
    (head / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{head}/pydantic_ai/__init__.py'}))
    capsys.readouterr()

    assert compare(base, head) == 1
    assert capsys.readouterr().out.splitlines()[-2:] == snapshot(
        ['only in base: 0, only in head: 0, differing: 1', '  differing: test_requests.py::test_anthropic_beta']
    )


FAILING = """
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


def test_passes():
    Agent(TestModel()).run_sync('hello')


def test_fails_after_a_request():
    Agent(TestModel()).run_sync('hello')
    assert False


def test_fails_first():
    assert False
"""


def test_compare_names_the_nodes_that_failed(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A node that fails is listed for each side it failed on, first, so it shows even when a side recorded nothing.

    A test that fails records only what it did before it failed, like one whose MCP server missed its startup timeout,
    so matching records don't prove it. Listing it doesn't fail the comparison; the failed pytest run does.
    """
    # As `run` sets them, so the session runs as `main` runs each side, even under xdist.
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD', '1')
    monkeypatch.delenv('PYTEST_XDIST_WORKER', raising=False)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_failing=FAILING)
    base, head, nothing = tmp_path / 'base', tmp_path / 'head', tmp_path / 'nothing'
    for capture_dir, args in ((base, ()), (head, ('-k', 'request')), (nothing, ('-k', 'first'))):
        pytester.runpytest_inprocess(
            '-p', 'no:cacheprovider', '-p', 'tests.refactor_capture', f'{CAPTURE_OPTION}={capture_dir}', *args
        )
        (capture_dir / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{capture_dir}/pydantic_ai/__init__.py'}))
    capsys.readouterr()

    assert compare(base, head) == 1
    assert capsys.readouterr().out.splitlines()[:4] == snapshot(
        [
            'nodes that failed, base: 2',
            '  test_failing.py::test_fails_after_a_request',
            '  test_failing.py::test_fails_first',
            'nodes that failed, head: 1',
        ]
    )
    assert compare(base, nothing) == 2
    output = capsys.readouterr()
    assert output.out.splitlines() == snapshot(
        [
            'nodes that failed, base: 2',
            '  test_failing.py::test_fails_after_a_request',
            '  test_failing.py::test_fails_first',
            'nodes that failed, head: 1',
            '  test_failing.py::test_fails_first',
        ]
    )
    assert output.err.startswith('Nothing recorded in')


def test_compare_names_the_nodes_that_recorded_no_request(
    run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """A selected node that recorded no request, realtime or session record is listed, whether it recorded a stream or
    nothing.

    Listing it doesn't fail the comparison.
    """
    base, head = tmp_path / 'base', tmp_path / 'head'
    for capture_dir in (base, head):
        run(capture_dir, '-k', 'subclass or no_stream or json').assert_outcomes(passed=3, deselected=8)
    (head / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{head}/pydantic_ai/__init__.py'}))
    capsys.readouterr()

    assert compare(base, head) == 0
    assert capsys.readouterr().out.splitlines()[-4:] == snapshot(
        [
            'nodes that recorded no request, realtime or session record, head: 2',
            '  test_streams.py::test_no_stream',
            '  test_streams.py::test_subclass_defined_after_the_capture_started',
            'only in base: 0, only in head: 0, differing: 0',
        ]
    )


def test_compare_flags_a_file_field_equality_skips(run: RunCapture, tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A file whose media type is set on one side and implied on the other differs, though equal and serialized alike."""
    base, head = tmp_path / 'base', tmp_path / 'head'
    run(base, '-k', 'media_type', ui=True).assert_outcomes(passed=1, deselected=7)
    run(head, '-k', 'media_type', ui=True, text='goodbye').assert_outcomes(passed=1, deselected=7)
    (base_record,), (head_record,) = (
        [json.loads(line) for line in (capture_dir / 'main.jsonl').read_text().splitlines()]
        for capture_dir in (base, head)
    )
    assert base_record['result'] == head_record['result']
    assert [record['uncompared_fields'] for record in (base_record, head_record)] == snapshot(
        [
            [{'type': 'ImageUrl', 'fields': {'_media_type': 'image/png', '_identifier': None}}],
            [{'type': 'ImageUrl', 'fields': {'_media_type': None, '_identifier': None}}],
        ]
    )
    (head / SOURCE_FILE).write_text(json.dumps({'pydantic_ai': f'{head}/pydantic_ai/__init__.py'}))
    capsys.readouterr()

    assert compare(base, head) == 1
    assert capsys.readouterr().out == snapshot("""\
nodes: base 1, head 1
records: base 1, head 1
stream events: head 0
requests: head 0
ui records: head 1
realtime records: head 0
session records: head 0
hook records: head 0
run event stream records: head 0
agent stream records: head 0
span records: head 0
exception records: head 0
warning records: head 0
temporal workflow run records: head 0
dbos workflow records: head 0
durable name records: head 0
cache key records: head 0
nodes per stream module, head:
nodes per request target, head:
nodes per ui target, head:
  test_ui.MediaTypeAdapter.load_messages: 1
nodes per realtime target, head:
nodes per session target, head:
nodes per hook target, head:
nodes per run event kind, head:
nodes per agent stream target, head:
nodes per span name, head:
nodes per temporal workflow run target, head:
nodes per dbos workflow target, head:
nodes per durable name target, head:
nodes per cache key target, head:
nodes that recorded no request, realtime or session record, head: 1
  test_ui.py::test_media_type_set_or_implied
only in base: 0, only in head: 0, differing: 1
  differing: test_ui.py::test_media_type_set_or_implied
""")


def test_main_extracts_the_base_commit_where_no_symlink_leads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Each pytest session is handed paths that go through no symlink, though the temporary directory is reached by one.

    DBOS resolves a workflow function's file with `os.path.realpath`, which reads a symlink on the way with
    `os.readlink`, and blockbuster raises on that in the event loop. On macOS the temporary directory is reached through
    `/var`, a symlink, so DBOS tests failed on the base side alone.
    """
    (tmp_path / 'real').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'real')
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'link'))
    monkeypatch.delenv('PYTHONPATH', raising=False)
    # `main` extracts each workspace package on `sys.path`; only `pydantic_graph`, the smallest, is left there.
    repo, venv = Path(refactor_capture.__file__).resolve().parent.parent, Path(sys.prefix).resolve()
    kept = [
        entry
        for entry in sys.path
        if not (path := Path(entry).resolve()).is_relative_to(repo) or path == repo or path.is_relative_to(venv)
    ]
    monkeypatch.setattr(sys, 'path', [*kept, str(repo / 'pydantic_graph')])
    run_git = subprocess.run
    sessions: list[tuple[list[str], str]] = []

    def run(
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        check: bool = False,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[bytes]:
        if env is None:
            return run_git(command, cwd=cwd, check=check, capture_output=capture_output)
        sessions.append((command, env['PYTHONPATH']))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, 'run', run)

    # Neither pytest session ran, so there's no capture to compare.
    assert main(['HEAD', 'tests/test_nothing.py']) == 2

    (out,) = (tmp_path / 'real').iterdir()
    sources = out / 'base-src' / 'pydantic_graph'
    assert (sources / 'pydantic_graph' / '__init__.py').is_file()
    pytest_command = [sys.executable, '-m', 'pytest', '-p', 'tests.refactor_capture']
    assert sessions == [
        ([*pytest_command, f'{CAPTURE_OPTION}={out / "base"}', 'tests/test_nothing.py'], str(sources)),
        ([*pytest_command, f'{CAPTURE_OPTION}={out / "head"}', 'tests/test_nothing.py'], ''),
    ]
    assert os.path.realpath(out) == str(out)


def test_described_exception_stops_where_a_chain_loops():
    """A chain set by hand can loop back to an exception already described, which is then described without its own."""
    first, second = ValueError('first'), KeyError('second')
    first.__cause__, second.__cause__ = second, first
    assert refactor_capture.described_exception(first) == snapshot(
        {
            'type': 'builtins.ValueError',
            'message': 'first',
            'cause': {
                'type': 'builtins.KeyError',
                'message': "'second'",
                'cause': {'type': 'builtins.ValueError', 'message': 'first'},
                'suppress_context': True,
            },
            'suppress_context': True,
        }
    )


def test_recorded_sorts_a_set():
    """A set comes out in one order, whatever order it iterates in, as two equal sets can on the two sides."""
    assert recorded(frozenset({'b', 'c', 'a'})) == snapshot(['a', 'b', 'c'])
    assert recorded({Owed('response', 'r1', 0), Owed('input', '0', 0)}) == snapshot(
        [
            {'type': 'Owed', 'value': {'kind': 'input', 'key': '0', 'epoch': 0}},
            {'type': 'Owed', 'value': {'kind': 'response', 'key': 'r1', 'epoch': 0}},
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
            'mock_repr': Mock(name='mock.service_tier'),
            'cancel_scope': 'Cancelled via cancel scope 11dc504b0; reason: deadline exceeded',
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
            'gen_ai.client.operation.time_to_first_chunk': 0.000741666997782886,
            'plan': 'Current plan:\n1. [ ] [0a1b2c3d] a\n2. [x] [4e5f6a7b] b',
            'added_step': "Added step 'a' with id: 0a1b2c3d",
            'echoed_step_id': {'task_id': '0a1b2c3d'},
            'amz_date': now.strftime('%Y%m%dT%H%M%SZ'),
            'recorded_amz_date': '20240101T000000Z',
            'clai_prompt': f'The current date and time is {datetime.now()} -05.',
            'recorded_clai_prompt': 'The current date and time is 2024-01-01 00:00:00 UTC.',
            'not_a_local_date': 'due 2024-02-30 00:00:00',
            'ag_ui_event': {'type': 'RUN_STARTED', 'timestamp': int(now.timestamp() * 1000)},
            'recorded_ag_ui_event': {'type': 'RUN_STARTED', 'timestamp': 1704067200000},
            # An AG-UI message's `encryptedValue`, and a web API response's event stream, hold JSON as text.
            'json_in_text': json.dumps({'timestamp': now.isoformat()}),
            'recorded_json_in_text': 'data: {"timestamp":"2024-01-01T00:00:00Z"}',
            'botocore_error': (
                "ModelHTTPError: status_code: 400, body: {'ResponseMetadata': {'HTTPStatusCode': 400, 'HTTPHeaders': "
                "{'date': 'Mon, 01 Jan 2024 00:00:00 GMT', 'content-type': 'application/json', 'connection': "
                "'keep-alive'}, 'RetryAttempts': 0}}"
            ),
            'native_tools_error': (
                "UserError: Native tool(s) ['ToolSearchTool'] not supported by this model. Supported: ['WebSearchTool', "
                "'AdvisorTool', 'MemoryTool']. To use these tools ..."
            ),
            # The trace ID has the shape of a version 4 UUID without dashes.
            'mcp_meta': {'traceparent': '00-12345678123441238123123456789abc-0123456789abcdef-01', 'progressToken': 2},
            'prefect_labels': {'__OTEL_TRACEPARENT': '00-01a1cb2f55a38937d7e2c3ba122f0f00-367bccfae831323d-01'},
            'prefect_task_run': {'name': 'agent Run-0', 'task_key': 'wrapped_run_flow-095f908e', 'dynamic_key': '0'},
            'default_run_id_inputs': {'sequence_key': 'pydantic_ai:default_run_id:02be28f320dd3c5469019298e187e2cb'},
            'activity_failure_info': {'identity': '8844@worker.local'},
            'evaluation_dispatch': {
                'duration': 0.010735334130004048,
                'span_reference': {'trace_id': '01a20f4cca618c0447a135792da95fe7', 'span_id': 'ff67063ae64898cb'},
                'tiny': {'duration': 9.5367431640625e-07},
            },
            'set_duration': {'action': 'wait', 'duration': 2.5},
            'delegation_finished': {'outcome': 'ok', 'duration_ms': 16.999249997752486},
            'task_question': {'question_id': 'cb05e2415528', 'question': 'a'},
            'timers': [{'start_to_fire_timeout': '0.099999624s'}, {'start_to_fire_timeout': '0.100s'}],
            'workflow_result': '[{"history_digest":"35a4e6e45d1d07a2fc4e901f819028c8915756b6e7dfae805874a8fa012d69b7"}]',
            'sorted_ticket': {
                'batch': {'id': 'batch_12345678-1234-4123-8123-123456789abc', 'provider_details': {}, 'status': 'done'},
                'custom_id': '027dd8a70deea5cb190b0995f0b6c37e',
                'polls': 0,
            },
            'sorted_details': {
                'custom_id': '7435ac26f85dbab292c80d8b26546f38',
                'id': 'batch_12345678-1234-4123-8123-123456789abc',
            },
            'in_memory_ticket': {
                'custom_id': '52a31e60e1afe0ff4a1058d5fcf539a6-1',
                'batch': {'id': 'batch_12345678123441238123123456789abc', 'model_name': 'test'},
            },
            'in_memory_response': {
                'provider_details': {
                    'batch': {
                        'id': 'batch_12345678123441238123123456789abc',
                        'custom_id': '12345678123441238123123456789abc',
                    }
                },
                'provider_response_id': 'batch:batch_12345678123441238123123456789abc:52a31e60e1afe0ff4a1058d5fcf539a6',
            },
            # A provider's batch ID isn't a UUID, so the `custom_id` it was sent, which hashes the request, stays.
            'provider_response': {
                'provider_details': {
                    'batch': {
                        'id': 'batch_6ad154c0fd50819086f8336d5b30addb',
                        'custom_id': '400d94ba99d83e31c153e0e49776c1a5',
                    }
                },
                'provider_response_id': 'batch:batch_6ad154c0fd50819086f8336d5b30addb:400d94ba99d83e31c153e0e49776c1a5',
            },
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
            'mock_repr': "<Mock name='mock.service_tier' id='ID'>",
            'cancel_scope': 'Cancelled via cancel scope ADDRESS; reason: deadline exceeded',
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
            'gen_ai.client.operation.time_to_first_chunk': 'DURATION',
            'plan': 'Current plan:\n1. [ ] [STEP_ID] a\n2. [x] [STEP_ID] b',
            'added_step': "Added step 'a' with id: STEP_ID",
            'echoed_step_id': {'task_id': '0a1b2c3d'},
            'amz_date': 'NOW',
            'recorded_amz_date': '20240101T000000Z',
            'clai_prompt': 'The current date and time is NOW -05.',
            'recorded_clai_prompt': 'The current date and time is 2024-01-01 00:00:00 UTC.',
            'not_a_local_date': 'due 2024-02-30 00:00:00',
            'ag_ui_event': {'type': 'RUN_STARTED', 'timestamp': 'NOW'},
            'recorded_ag_ui_event': {'type': 'RUN_STARTED', 'timestamp': 1704067200000},
            'json_in_text': '{"timestamp": "NOW"}',
            'recorded_json_in_text': 'data: {"timestamp":"2024-01-01T00:00:00Z"}',
            'botocore_error': (
                "ModelHTTPError: status_code: 400, body: {'ResponseMetadata': {'HTTPStatusCode': 400, 'HTTPHeaders': "
                "{'connection': 'keep-alive', 'content-type': 'application/json', 'date': 'Mon, 01 Jan 2024 00:00:00 "
                "GMT'}, 'RetryAttempts': 0}}"
            ),
            'native_tools_error': (
                "UserError: Native tool(s) ['ToolSearchTool'] not supported by this model. Supported: ['AdvisorTool', "
                "'MemoryTool', 'WebSearchTool']. To use these tools ..."
            ),
            'mcp_meta': {'traceparent': '00-TRACE_ID-SPAN_ID-01', 'progressToken': 2},
            'prefect_labels': {'__OTEL_TRACEPARENT': '00-TRACE_ID-SPAN_ID-01'},
            'prefect_task_run': {'name': 'agent Run-0', 'task_key': 'wrapped_run_flow-CODE_HASH', 'dynamic_key': '0'},
            'default_run_id_inputs': {'sequence_key': 'pydantic_ai:default_run_id:DIGEST'},
            'activity_failure_info': {'identity': 'PID@worker.local'},
            'evaluation_dispatch': {
                'duration': 'DURATION',
                'span_reference': {'trace_id': 'TRACE_ID', 'span_id': 'SPAN_ID'},
                'tiny': {'duration': 'DURATION'},
            },
            'set_duration': {'action': 'wait', 'duration': 2.5},
            'delegation_finished': {'outcome': 'ok', 'duration_ms': 'DURATION'},
            'task_question': {'question_id': 'QUESTION_ID', 'question': 'a'},
            'timers': [{'start_to_fire_timeout': 'DURATION'}, {'start_to_fire_timeout': '0.100s'}],
            'workflow_result': '[{"history_digest":"DIGEST"}]',
            'sorted_ticket': {
                'batch': {'id': 'batch_UUID', 'provider_details': {}, 'status': 'done'},
                'custom_id': 'CUSTOM_ID',
                'polls': 0,
            },
            'sorted_details': {'custom_id': 'CUSTOM_ID', 'id': 'batch_UUID'},
            'in_memory_ticket': {'custom_id': 'CUSTOM_ID-1', 'batch': {'id': 'batch_UUID', 'model_name': 'test'}},
            'in_memory_response': {
                'provider_details': {'batch': {'id': 'batch_UUID', 'custom_id': 'CUSTOM_ID'}},
                'provider_response_id': 'batch:batch_UUID:CUSTOM_ID',
            },
            'provider_response': {
                'provider_details': {
                    'batch': {
                        'id': 'batch_6ad154c0fd50819086f8336d5b30addb',
                        'custom_id': '400d94ba99d83e31c153e0e49776c1a5',
                    }
                },
                'provider_response_id': 'batch:batch_6ad154c0fd50819086f8336d5b30addb:400d94ba99d83e31c153e0e49776c1a5',
            },
        }
    )
