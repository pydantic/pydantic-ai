"""Tests for `IPythonCodeMode`, driven through `Agent(capabilities=[...])` against real IPython kernels.

Most tests share one module-scoped kernel, which keeps them to a single process launch. Tests that
stop, restart, or replace a kernel launch their own and carry the `subprocess` marker.
"""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, Tool, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, ToolSearch
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    BinaryImage,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, WrapperToolset
from pydantic_ai.usage import UsageLimits
from tests.conftest import try_import
from tests.harness.conftest import IsDatetime, IsStr

from .._recording_durability import RecordingDurability

with try_import() as imports_successful:
    from pydantic_ai_harness import IPythonCodeMode
    from pydantic_ai_harness.ipython_code_mode import IPythonKernel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='jupyter-client not installed')

_PNG = base64.b64decode(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='
)


@dataclass
class _NewEachStep(WrapperToolset[None]):
    """A toolset that is a new instance every step, like one whose tools depend on the step."""

    async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        return replace(self)


class Trip(BaseModel):
    city: str
    nights: int


@pytest.fixture(scope='module')
async def kernel() -> AsyncIterator[IPythonKernel]:
    async with IPythonKernel() as kernel:
        yield kernel


def _scripted(*codes: str | dict[str, Any], seen: list[AgentInfo] | None = None) -> FunctionModel:
    """A model that sends each `run_code` call in turn, then answers `done`."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append(info)
        turn = sum(isinstance(message, ModelResponse) for message in messages)
        if turn < len(codes):
            code = codes[turn]
            args = {'code': code} if isinstance(code, str) else code
            return ModelResponse(parts=[ToolCallPart('run_code', args, tool_call_id=f'call_{turn}')])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


def _assert_process_gone(pid: int) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def _returns(messages: Sequence[ModelMessage]) -> list[Any]:
    """What `run_code` sent back for each call: the return value, or the retry message."""
    found: list[Any] = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                found.append(part.content)
    return found


async def _run(
    capability: IPythonCodeMode[None],
    *codes: str | dict[str, Any],
    tools: Sequence[Tool[None]] = (),
    capabilities: Sequence[AbstractCapability[None]] = (),
) -> list[Any]:
    agent = Agent(_scripted(*codes), deps_type=type(None), capabilities=[capability, *capabilities], tools=tools)
    result = await agent.run('Run the code')
    return _returns(result.all_messages())


class TestIPythonCodeMode:
    async def test_results_follow_the_code_mode_return_shapes(self, kernel: IPythonKernel) -> None:
        assert await _run(
            IPythonCodeMode(kernel=kernel),
            "{'answer': 42, 'items': [1, 2.5, None]}",
            "print('hello')",
            "print('hello')\n21 * 2",
            'x = 1',
            "'text'",
            '{1, 2}',
        ) == snapshot(
            [
                {'answer': 42, 'items': [1, 2.5, None]},
                {'output': 'hello\n'},
                {'output': 'hello\n', 'result': 42},
                {},
                'text',
                '{1, 2}',
            ]
        )

    async def test_stderr_and_displayed_text_are_part_of_the_output(self, kernel: IPythonKernel) -> None:
        assert await _run(
            IPythonCodeMode(kernel=kernel),
            "import sys\nprint('out')\nprint('err', file=sys.stderr)\nfrom IPython.display import display\ndisplay([1, 2])\n"
            "display({'text/html': '<b>only html</b>'}, raw=True)\n"
            "display({'image/png': 'not base64!', 'text/plain': 'fallback'}, raw=True)",
        ) == snapshot([{'output': 'out\nerr\n[1, 2]\nfallback\n'}])

    async def test_state_persists_between_calls_in_a_run(self, kernel: IPythonKernel) -> None:
        assert await _run(
            IPythonCodeMode(kernel=kernel),
            'import math\ndef area(r):\n    return math.pi * r ** 2',
            'round(area(2), 3)',
        ) == snapshot([{}, 12.566])

    async def test_a_shared_kernel_keeps_state_across_runs(self, kernel: IPythonKernel) -> None:
        capability = IPythonCodeMode[None](kernel=kernel)
        assert await _run(capability, "shared_across_runs = 'kept'") == [{}]
        assert await _run(capability, 'shared_across_runs') == ['kept']

    async def test_ipython_syntax_and_top_level_await(self, kernel: IPythonKernel) -> None:
        assert await _run(
            IPythonCodeMode(kernel=kernel),
            'import asyncio\nawait asyncio.sleep(0)\n%time 1 + 1\nlisting = !echo hi\nlen(listing)',
        ) == snapshot([{'output': IsStr(), 'result': 1}])

    async def test_input_returns_an_empty_string(self, kernel: IPythonKernel) -> None:
        assert await _run(IPythonCodeMode(kernel=kernel), "input('name? ')") == ['']

    async def test_displayed_images_are_returned_as_images(self, kernel: IPythonKernel) -> None:
        png = repr(_PNG)
        image = BinaryImage(data=_PNG, media_type='image/png')
        assert await _run(
            IPythonCodeMode(kernel=kernel),
            f'from IPython.display import Image, display\ndisplay(Image(data={png}))',
            f"print('chart')\ndisplay(Image(data={png}))\n'caption'",
            f'Image(data={png})',
            f"print('chart')\nImage(data={png})",
        ) == [[image], ['chart\n', image, 'caption'], image, ['chart\n', image]]

    async def test_long_output_is_truncated(self, kernel: IPythonKernel, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.ipython_code_mode._kernel.MAX_OUTPUT_CHARS', 10)
        assert await _run(
            IPythonCodeMode(kernel=kernel), "import sys\nprint('a' * 25, flush=True)\nprint('b', file=sys.stderr)"
        ) == snapshot([{'output': 'aaaaaaaaaa\n[output truncated: 18 more characters were dropped]'}])

    async def test_an_exception_becomes_a_retry_with_the_traceback(self, kernel: IPythonKernel) -> None:
        (retry, after) = await _run(
            IPythonCodeMode(kernel=kernel), "print('before')\nraise ValueError('boom')", "'recovered'"
        )
        assert retry.startswith('Runtime error:\nbefore\n')
        assert 'ValueError: boom' in retry
        assert '\x1b[' not in retry
        assert after == 'recovered'

    async def test_errors_past_max_retries_end_the_run(self, kernel: IPythonKernel) -> None:
        agent = Agent(_scripted('1 / 0', '1 / 0'), capabilities=[IPythonCodeMode(kernel=kernel, max_retries=1)])
        with pytest.raises(Exception, match='exceeded max retries count of 1'):
            await agent.run('Run the code')

    async def test_timeout_interrupts_the_code_and_keeps_the_state(self, kernel: IPythonKernel) -> None:
        (_, retry, after) = await _run(
            IPythonCodeMode(kernel=kernel, timeout=0.5),
            "survives_timeout = 'yes'",
            "print('working')\nimport time\ntime.sleep(30)",
            'survives_timeout',
        )
        assert retry.startswith('working\n')
        assert 'KeyboardInterrupt' in retry
        assert retry.endswith(
            'The code ran longer than the 0.5-second timeout and was interrupted. The kernel and its state are kept. '
            'Make the code do less work per call.'
        )
        assert after == 'yes'

    async def test_timeout_interrupts_code_that_keeps_printing(self, kernel: IPythonKernel) -> None:
        (retry,) = await _run(
            IPythonCodeMode(kernel=kernel, timeout=0.5),
            "import time\nfor _ in range(50):\n    print('tick')\n    time.sleep(0.1)",
        )
        assert 'KeyboardInterrupt' in retry
        assert 'The code ran longer than the 0.5-second timeout and was interrupted.' in retry

    async def test_timeout_is_reported_when_the_code_catches_the_interrupt(self, kernel: IPythonKernel) -> None:
        (retry,) = await _run(
            IPythonCodeMode(kernel=kernel, timeout=0.5),
            "import time\ntry:\n    time.sleep(30)\nexcept KeyboardInterrupt:\n    print('caught')",
        )
        assert retry.startswith('caught\n\nThe code ran longer than the 0.5-second timeout and was interrupted.')

    async def test_time_in_tool_calls_does_not_count_toward_the_timeout(self, kernel: IPythonKernel) -> None:
        async def slow() -> str:
            await asyncio.sleep(0.6)
            return 'slow'

        assert await _run(IPythonCodeMode(kernel=kernel, timeout=0.5), 'slow()', tools=[Tool(slow, name='slow')]) == [
            'slow'
        ]


class TestIPythonCodeModeTools:
    async def test_tools_are_functions_in_the_kernel(self, kernel: IPythonKernel) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(_scripted('add(a=1, b=2) + add(a=3, b=4)'), capabilities=[IPythonCodeMode(kernel=kernel)])
        agent.tool_plain(add)
        result = await agent.run('Run the code')
        (tool_return,) = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]
        assert tool_return.content == 10
        assert tool_return.metadata == snapshot(
            {
                'code_mode': True,
                'tool_calls': {
                    'call_0__1': ToolCallPart(tool_name='add', args={'a': 1, 'b': 2}, tool_call_id='call_0__1'),
                    'call_0__2': ToolCallPart(tool_name='add', args={'a': 3, 'b': 4}, tool_call_id='call_0__2'),
                },
                'tool_returns': {
                    'call_0__1': ToolReturnPart(
                        tool_name='add', content=3, tool_call_id='call_0__1', timestamp=IsDatetime()
                    ),
                    'call_0__2': ToolReturnPart(
                        tool_name='add', content=7, tool_call_id='call_0__2', timestamp=IsDatetime()
                    ),
                },
            }
        )

    async def test_the_description_lists_the_functions(self, kernel: IPythonKernel) -> None:
        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return f'Sunny in {city}'  # pragma: no cover

        def book(trip: Trip, travellers: int) -> str:
            """Book a trip."""
            return f'Booked {trip.city}'  # pragma: no cover

        seen: list[AgentInfo] = []
        agent = Agent(
            _scripted(seen=seen),
            capabilities=[IPythonCodeMode(kernel=kernel)],
            tools=[Tool(get_weather, name='get-weather'), Tool(book)],
        )
        await agent.run('Hi')
        (run_code,) = seen[0].function_tools
        assert run_code.name == 'run_code'
        assert run_code.description == snapshot('''\
Run Python code in a persistent IPython kernel.

The kernel is a full CPython interpreter: any installed package can be imported, and code runs with the host process's permissions, including its filesystem and network access. Variables, imports, functions, and loaded data persist between calls. IPython syntax works, including magics such as `%timeit`, shell escapes such as `!ls`, and top-level `await`. `input()` returns an empty string. Set `restart: true` to restart the kernel and clear its state before the code runs.

Each call may run for 60 seconds, not counting time spent in the functions below; code still running then is interrupted with `KeyboardInterrupt`.

The last expression's value is automatically captured as the return value -- you do **not** need to `print()` it. A final expression that evaluates to `None` is treated as no result. Values with a JSON form are returned as data; other values are returned as their `repr`, and images (for example `matplotlib` figures or PIL images) are returned as images. Without a result, printed output, or displayed images, `run_code` returns `{}`.

Returns a non-`None` last expression's value directly when nothing is printed or displayed. With printed output and no result, returns `{"output": "<printed text>"}`. With printed output and a plain result, returns `{"output": "<printed text>", "result": <last expression>}`. With displayed images, returns a list: the printed text if any, then the images, then the result if any. With printed output and an image result, returns a list of the printed text and the image.

The following functions are already defined in the kernel; call them by name without redefining or importing them. All parameters are keyword-only. They are synchronous, so call them directly, e.g. `tool_name(arg=value)`, without `await`. Calls run one at a time. A failed call raises `ToolCallError` with the tool's error message.

```python
class Trip(TypedDict):
    city: str
    nights: int
```

```python
def get_weather(*, city: str) -> str:
    """Get the weather for a city."""
    ...

def book(*, trip: Trip, travellers: int) -> str:
    """Book a trip."""
    ...
```\
''')
        assert run_code.metadata == {'code_arg_name': 'code', 'code_arg_language': 'python'}
        assert run_code.sequential

    async def test_without_a_timeout_the_description_states_none(self, kernel: IPythonKernel) -> None:
        seen: list[AgentInfo] = []
        agent = Agent(_scripted(seen=seen), capabilities=[IPythonCodeMode(kernel=kernel, timeout=None)])
        await agent.run('Hi')
        (run_code,) = seen[0].function_tools
        assert run_code.description is not None
        assert 'seconds' not in run_code.description

    async def test_sanitized_tool_names_dispatch_to_the_original_tool(self, kernel: IPythonKernel) -> None:
        def get_weather(city: str) -> str:
            return f'Sunny in {city}'

        assert await _run(
            IPythonCodeMode(kernel=kernel),
            "get_weather(city='Paris')",
            tools=[Tool(get_weather, name='get-weather')],
        ) == ['Sunny in Paris']

    async def test_the_tool_selector_keeps_other_tools_native(self, kernel: IPythonKernel) -> None:
        def inside() -> str:
            return 'inside'  # pragma: no cover

        def outside() -> str:
            return 'outside'  # pragma: no cover

        seen: list[AgentInfo] = []
        agent = Agent(
            _scripted(seen=seen),
            capabilities=[IPythonCodeMode(kernel=kernel, tools=['inside'])],
            tools=[Tool(inside), Tool(outside)],
        )
        await agent.run('Hi')
        assert sorted(tool.name for tool in seen[0].function_tools) == ['outside', 'run_code']
        assert 'def inside()' in (seen[0].function_tools[-1].description or '')

    async def test_a_failed_tool_call_raises_in_the_kernel(self, kernel: IPythonKernel) -> None:
        def flaky() -> str:
            raise ModelRetry('try again later')

        (caught, uncaught) = await _run(
            IPythonCodeMode(kernel=kernel),
            'try:\n    flaky()\nexcept ToolCallError as e:\n    caught = str(e)\ncaught',
            'flaky()',
            tools=[Tool(flaky)],
        )
        assert caught == 'ModelRetry: try again later'
        assert 'ToolCallError: ModelRetry: try again later' in uncaught
        # The generated function's frame is hidden; IPython always shows the frame that raised.
        assert '[... skipping hidden 1 frame]' in uncaught

    async def test_a_tool_the_catalog_dropped_answers_with_an_error(self, kernel: IPythonKernel) -> None:
        (retry,) = await _run(
            IPythonCodeMode(kernel=kernel), "get_ipython().pydantic_ai_define_functions(['dropped'])\ndropped()"
        )
        assert "ToolCallError: 'dropped' is no longer an available function." in retry

    async def test_max_tool_calls_limits_the_calls_per_run_code(self, kernel: IPythonKernel) -> None:
        def ping() -> str:
            return 'pong'

        (retry,) = await _run(
            IPythonCodeMode(kernel=kernel, max_tool_calls=2), 'for _ in range(3):\n    ping()', tools=[Tool(ping)]
        )
        assert 'Code mode allows 2 nested tool calls per `run_code` call' in retry

    async def test_tool_return_content_reaches_the_model_in_call_order(self, kernel: IPythonKernel) -> None:
        def note(text: str) -> ToolReturn[str]:
            return ToolReturn(return_value='noted', content=f'Note: {text}')

        agent = Agent(
            _scripted("note(text='first')\nnote(text='second')"),
            capabilities=[IPythonCodeMode(kernel=kernel)],
            tools=[Tool(note)],
        )
        result = await agent.run('Run the code')
        prompts = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, UserPromptPart) and part.content != 'Run the code'
        ]
        assert prompts == [['Note: first', 'Note: second']]

    async def test_tool_calls_count_toward_usage_limits(self, kernel: IPythonKernel) -> None:
        def ping() -> str:
            return 'pong'

        agent = Agent(
            _scripted('ping()\nping()\nping()'), capabilities=[IPythonCodeMode(kernel=kernel)], tools=[Tool(ping)]
        )
        with pytest.raises(UsageLimitExceeded):
            await agent.run('Run the code', usage_limits=UsageLimits(tool_calls_limit=2))

    async def test_a_tool_named_run_code_is_rejected(self, kernel: IPythonKernel) -> None:
        def run_code() -> str:
            return 'mine'  # pragma: no cover

        agent = Agent(_scripted(), capabilities=[IPythonCodeMode(kernel=kernel, tools=[])], tools=[Tool(run_code)])
        with pytest.raises(UserError, match="Tool name 'run_code' is reserved"):
            await agent.run('Hi')

    async def test_the_tools_get_the_run_context(self, kernel: IPythonKernel) -> None:
        def whoami(ctx: RunContext[str]) -> str:
            return ctx.deps

        agent = Agent(
            _scripted('whoami()'), deps_type=str, capabilities=[IPythonCodeMode(kernel=kernel)], tools=[Tool(whoami)]
        )
        result = await agent.run('Run the code', deps='alice')
        assert _returns(result.all_messages()) == ['alice']

    @pytest.mark.moves_cache_prefix(
        reason='a revealed tool joins the `run_code` description; `IPythonCodeMode` has no `dynamic_catalog`'
    )
    async def test_tools_tool_search_has_not_revealed_stay_off_the_kernel(self, kernel: IPythonKernel) -> None:
        def forecast(city: str) -> str:
            """Get the weather forecast for a city."""
            return f'Sunny in {city}'

        seen: list[AgentInfo] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            turn = sum(isinstance(message, ModelResponse) for message in messages)
            if turn == 0:
                return ModelResponse(parts=[ToolCallPart('search_tools', {'queries': ['forecast']})])
            if turn == 1:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': "forecast(city='Paris')"})])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[IPythonCodeMode(kernel=kernel), ToolSearch(strategy='keywords')],
            tools=[Tool(forecast, defer_loading=True)],
        )
        result = await agent.run('Forecast Paris')
        before, after = (
            next(tool.description or '' for tool in info.function_tools if tool.name == 'run_code') for info in seen[:2]
        )
        assert 'def forecast(' not in before
        assert 'def forecast(' in after
        assert _returns(result.all_messages()) == ['Sunny in Paris']

    async def test_a_tool_that_needs_approval_runs_once_a_handler_approves_it(self, kernel: IPythonKernel) -> None:
        def delete(path: str) -> str:
            return f'deleted {path}'

        decisions: list[bool] = []

        async def decide(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
            approve = len(decisions) == 0
            decisions.append(approve)
            return DeferredToolResults(
                approvals={call.tool_call_id: True if approve else ToolDenied('no') for call in requests.approvals}
            )

        assert await _run(
            IPythonCodeMode(kernel=kernel),
            "first = delete(path='a')\ntry:\n    delete(path='b')\nexcept ToolCallError as e:\n    second = str(e)\n[first, second]",
            tools=[Tool(delete, requires_approval=True)],
            capabilities=[HandleDeferredToolCalls(handler=decide)],
        ) == snapshot([['deleted a', "RuntimeError: Tool 'delete' call denied: no"]])

    async def test_a_tool_that_needs_approval_fails_without_a_handler(self, kernel: IPythonKernel) -> None:
        def delete(path: str) -> str:
            return f'deleted {path}'  # pragma: no cover

        (retry,) = await _run(
            IPythonCodeMode(kernel=kernel),
            "delete(path='a')",
            tools=[Tool(delete, requires_approval=True)],
        )
        assert 'no `HandleDeferredToolCalls` capability resolved it' in retry

    async def test_cancelling_a_run_interrupts_its_cell(self, kernel: IPythonKernel) -> None:
        started = asyncio.Event()

        async def begin() -> None:
            started.set()

        agent = Agent(
            _scripted('begin()\nimport time\ntime.sleep(30)'),
            capabilities=[IPythonCodeMode(kernel=kernel, timeout=None)],
            tools=[Tool(begin)],
        )
        task = asyncio.ensure_future(agent.run('Run the code'))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert await _run(IPythonCodeMode(kernel=kernel), "'free'") == ['free']


class TestIPythonCodeModeConfiguration:
    @pytest.mark.parametrize(
        ('options', 'message'),
        [
            ({'max_tool_calls': 0}, '`max_tool_calls` must be at least 1'),
            ({'timeout': 0}, '`timeout` must be a positive number of seconds'),
        ],
    )
    async def test_invalid_options_fail_when_the_run_starts(self, options: dict[str, Any], message: str) -> None:
        agent = Agent(_scripted(), capabilities=[IPythonCodeMode(**options)])
        with pytest.raises(UserError, match=message):
            await agent.run('Hi')

    async def test_an_unknown_kernelspec_fails_to_start(self) -> None:
        with pytest.raises(KeyError, match='no-such-kernel'):
            await IPythonKernel('no-such-kernel').start()

    async def test_durable_execution_is_refused_before_any_code_runs(self) -> None:
        agent = Agent(_scripted('1'), name='durable', capabilities=[IPythonCodeMode(), RecordingDurability()])
        with pytest.raises(UserError, match='`IPythonCodeMode` is not supported under durable execution'):
            await agent.run('Hi')


@pytest.mark.subprocess(reason='checks the lifecycle of the kernel process the capability launches')
class TestIPythonCodeModeKernelLifecycle:
    async def test_a_run_starts_its_kernel_on_first_use_and_stops_it_at_the_end(self) -> None:
        agent = Agent(_scripted('import os\nos.getpid()', 'import os\nos.getpid()'), capabilities=[IPythonCodeMode()])
        first = _returns((await agent.run('Run the code')).all_messages())
        second = _returns((await agent.run('Run the code')).all_messages())
        assert first[0] == first[1]
        assert second[0] == second[1]
        assert first[0] != second[0]
        _assert_process_gone(first[0])
        _assert_process_gone(second[0])

    async def test_shutdown_closes_the_clients_zmq_context(self) -> None:
        # Held here, so a client that owns its context can't close it from `__del__` before the assertion.
        async with IPythonKernel() as kernel:
            client = kernel._client  # pyright: ignore[reportPrivateUsage]
            assert client is not None
        assert client.context.closed

    async def test_a_cancelled_run_stops_its_kernel(self) -> None:
        pids: list[int] = []
        started = anyio.Event()

        async def begin(pid: int) -> None:
            pids.append(pid)
            started.set()

        agent = Agent(
            _scripted('import os, time\nbegin(pid=os.getpid())\ntime.sleep(30)'),
            capabilities=[IPythonCodeMode(timeout=None)],
            tools=[Tool(begin)],
        )
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tg:
                tg.start_soon(agent.run, 'Run the code')
                await started.wait()
                scope.cancel()
        assert scope.cancelled_caught
        _assert_process_gone(pids[0])

    async def test_a_cancelled_cell_that_ignores_the_interrupt_gets_its_kernel_stopped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.ipython_code_mode._kernel._INTERRUPT_GRACE_SECS', 0.2)
        pids: list[int] = []
        started = anyio.Event()

        async def begin(pid: int) -> None:
            pids.append(pid)
            started.set()

        async with IPythonKernel() as kernel:
            agent = Agent(
                _scripted(
                    'import os, signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nbegin(pid=os.getpid())\n'
                    'while True:\n    pass'
                ),
                capabilities=[IPythonCodeMode(kernel=kernel, timeout=None)],
                tools=[Tool(begin)],
            )
            with anyio.CancelScope() as scope:
                async with anyio.create_task_group() as tg:
                    tg.start_soon(agent.run, 'Run the code')
                    await started.wait()
                    scope.cancel()
            _assert_process_gone(pids[0])
            # The next call is not stuck behind the abandoned cell: it gets a new kernel.
            assert await _run(IPythonCodeMode(kernel=kernel), "'fresh'") == ['fresh']

    async def test_a_shared_kernel_stops_when_its_context_is_cancelled(self) -> None:
        pids: list[int] = []
        with anyio.CancelScope() as scope:
            async with IPythonKernel() as kernel:
                pids.extend(await _run(IPythonCodeMode(kernel=kernel), 'import os\nos.getpid()'))
                scope.cancel()
                await anyio.sleep(0)
        assert scope.cancelled_caught
        _assert_process_gone(pids[0])

    @pytest.mark.parametrize(
        'code',
        [
            pytest.param('time.sleep(30)', id='interrupted'),
            pytest.param('signal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True:\n    pass', id='killed'),
        ],
    )
    async def test_shutdown_stops_a_cell_another_run_is_still_running(
        self, code: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Long enough for an interrupted cell to format its traceback and finish within the first escalation.
        monkeypatch.setattr('pydantic_ai_harness.ipython_code_mode._kernel._SHUTDOWN_WAIT_SECS', 1.0)
        pids: list[int] = []
        started = anyio.Event()

        async def begin(pid: int) -> None:
            pids.append(pid)
            started.set()

        kernel = IPythonKernel()
        agent = Agent(
            _scripted(f'import os, signal, time\nbegin(pid=os.getpid())\n{code}'),
            capabilities=[IPythonCodeMode(kernel=kernel, timeout=None)],
            tools=[Tool(begin)],
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(agent.run, 'Run the code')
            await started.wait()
            await kernel.shutdown()
        _assert_process_gone(pids[0])

    async def test_shutdown_returns_while_another_run_waits_on_a_tool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.ipython_code_mode._kernel._SHUTDOWN_WAIT_SECS', 0.2)
        pids: list[int] = []
        started = anyio.Event()

        async def hang(pid: int) -> None:
            pids.append(pid)
            started.set()
            await anyio.sleep_forever()

        kernel = IPythonKernel()
        agent = Agent(
            _scripted('import os\nhang(pid=os.getpid())'),
            capabilities=[IPythonCodeMode(kernel=kernel, timeout=None)],
            tools=[Tool(hang)],
        )
        async with anyio.create_task_group() as tg:
            tg.start_soon(agent.run, 'Run the code')
            await started.wait()
            await kernel.shutdown()
            _assert_process_gone(pids[0])
            tg.cancel_scope.cancel()
        # The cancelled run closes the dead kernel's channels; a leaked zmq socket would fail this test here.
        gc.collect()

    async def test_restart_clears_the_state(self) -> None:
        async with IPythonKernel() as kernel:
            assert await _run(
                IPythonCodeMode(kernel=kernel),
                "before_restart = 'set'",
                {'code': "'before_restart' in dir()", 'restart': True},
            ) == [{}, False]

    async def test_a_kernel_that_exits_is_replaced_on_the_next_call(self) -> None:
        async with IPythonKernel() as kernel:
            (retry, after) = await _run(IPythonCodeMode(kernel=kernel), 'import os\nos._exit(1)', "'replaced'")
        assert retry == snapshot(
            'The kernel process exited, for example because the code crashed it or called `os._exit()`, so its state is gone. The next call starts a new kernel: re-run any imports and recreate any state you need.'
        )
        assert after == 'replaced'

    async def test_a_kernel_that_exits_between_calls_is_found_before_the_next(self) -> None:
        def ping() -> str:
            return 'pong'  # pragma: no cover

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            turn = sum(isinstance(message, ModelResponse) for message in messages)
            if turn == 0:
                code = 'import os, threading\nthreading.Timer(0.1, os._exit, args=(1,)).start()'
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': code})])
            if turn == 1:
                # Gives the timer time to end the kernel before the next call reaches it.
                await anyio.sleep(1)
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'ping()'})])
            return ModelResponse(parts=[TextPart('done')])

        async with IPythonKernel() as kernel:
            agent = Agent(FunctionModel(respond), capabilities=[IPythonCodeMode(kernel=kernel)], tools=[Tool(ping)])
            (exited, after) = _returns((await agent.run('Run the code')).all_messages())
        assert exited == {}
        assert after.startswith('The kernel process exited')

    async def test_a_run_keeps_its_kernel_when_the_toolset_changes_between_steps(self) -> None:
        def ping() -> int:
            return 1

        agent = Agent(
            _scripted('step_value = 1', 'step_value + ping()'),
            deps_type=type(None),
            capabilities=[IPythonCodeMode()],
            toolsets=[_NewEachStep(FunctionToolset([Tool(ping)]))],
        )
        assert _returns((await agent.run('Run the code')).all_messages()) == [{}, 2]

    async def test_code_that_ignores_the_interrupt_gets_a_new_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.ipython_code_mode._kernel._INTERRUPT_GRACE_SECS', 0.2)
        async with IPythonKernel() as kernel:
            (pid, retry, after) = await _run(
                IPythonCodeMode(kernel=kernel, timeout=0.3),
                "import os\nlost_on_restart = 'set'\nos.getpid()",
                'import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True:\n    pass',
                "'lost_on_restart' in dir()",
            )
            _assert_process_gone(pid)
        assert retry.startswith('The code ran longer than the 0.3-second timeout and did not stop when interrupted')
        assert after is False

    async def test_a_kernel_that_is_not_ipython_fails_to_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = tmp_path / 'kernels' / 'broken'
        spec.mkdir(parents=True)
        # An ipykernel whose IPython lacks the display formatter base class the bootstrap subclasses.
        breaks_bootstrap = 'import IPython.core.formatters as f; del f.BaseFormatter'
        argv = [
            'python',
            '-m',
            'ipykernel_launcher',
            '-f',
            '{connection_file}',
            f'--IPKernelApp.exec_lines={[breaks_bootstrap]!r}',
        ]
        (spec / 'kernel.json').write_text(json.dumps({'argv': argv, 'display_name': 'Broken', 'language': 'python'}))
        monkeypatch.setenv('JUPYTER_PATH', str(tmp_path))
        kernel = IPythonKernel('broken')
        with pytest.raises(RuntimeError, match="Kernel 'broken' could not be set up for code mode"):
            await kernel.start()
        # The failed kernel was stopped, so a retry launches a new one rather than reusing it.
        with pytest.raises(RuntimeError, match="Kernel 'broken' could not be set up for code mode"):
            await kernel.start()

    async def test_the_kernel_runs_in_its_working_directory(self, tmp_path: Path) -> None:
        async with IPythonKernel(cwd=tmp_path) as kernel:
            assert await _run(IPythonCodeMode(kernel=kernel), 'import os\nos.getcwd()') == [str(tmp_path)]
