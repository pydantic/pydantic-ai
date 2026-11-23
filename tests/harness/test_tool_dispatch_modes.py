"""The outcomes of a tool call, whichever way the call reaches the tool.

A call the model makes runs through the run's tool manager. Pydantic AI Harness runs the same tool other ways: code
mode's `run_code` calls it from a sandbox through a nested tool manager, `ToolBridge`'s `tool_call` through another,
`BackgroundTools` runs it in a task whose result reaches the model after the step that made the call has ended, and
`IPythonCodeMode` calls it from a kernel. Each test here takes one outcome of a direct call and checks it in each of
those dispatch modes with the same assertions. Where a mode differs by design, the test's expectations say so and why.

Each of these PRs fixed a mode that had drifted from a direct call:
https://github.com/pydantic/pydantic-ai/pull/10137, https://github.com/pydantic/pydantic-ai/pull/10362,
https://github.com/pydantic/pydantic-ai/pull/10750, https://github.com/pydantic/pydantic-ai/pull/10830,
https://github.com/pydantic/pydantic-ai/pull/11456, https://github.com/pydantic/pydantic-ai/pull/11458,
https://github.com/pydantic/pydantic-ai/pull/12767 and https://github.com/pydantic/pydantic-ai/pull/13076.

The tests drive whole runs with a `FunctionModel`: the outcomes depend on the run loop's retry and usage accounting,
which only a run exercises. No provider is involved.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal

import anyio
import pytest

from pydantic_ai import Agent, RunContext, Tool, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls, ValidatedToolArgs
from pydantic_ai.exceptions import (
    ApprovalRequired,
    ModelRetry,
    ToolFailed,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, DeferredToolResults, ToolDefinition
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import BackgroundTools, CodeMode, ToolBridge
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.conftest import try_import
from tests.harness._rendering import sent_messages

with try_import() as ipython_available:
    from pydantic_ai_harness import IPythonCodeMode
    from pydantic_ai_harness.ipython_code_mode import IPythonKernel

Mode = Literal['direct', 'run_code', 'run_code_background', 'background', 'tool_bridge', 'ipython']

_BACKGROUND_STARTED = 'running in background'
"""What a `BackgroundTools` call's first result says, before its real result arrives."""

_Call = tuple[str, dict[str, object]]
"""A call the model makes: the tool's name and its arguments."""


@pytest.fixture(scope='module')
async def kernel(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[IPythonKernel]:
    """One IPython kernel for the module's `ipython` cases, with its own profile so it shares no history file."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv('IPYTHONDIR', str(tmp_path_factory.mktemp('ipython')))
        async with IPythonKernel() as kernel:
            yield kernel


@dataclass
class _Dispatch:
    """How the scripted model reaches the tools in one mode, and what that mode adds to the agent."""

    mode: Mode
    kernel: IPythonKernel | None = None
    sandbox_retries: int = 3
    """`max_retries` of `run_code` in the sandbox modes, the budget a sandbox failure spends."""
    released: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when the model has been told a background call started, so a held call may finish."""

    def capabilities(self) -> list[AbstractCapability[object]]:
        match self.mode:
            case 'direct':
                return []
            case 'run_code':
                return [CodeMode[object](max_retries=self.sandbox_retries)]
            case 'run_code_background':
                return [CodeMode[object](max_retries=self.sandbox_retries), BackgroundTools[object]()]
            case 'background':
                return [BackgroundTools[object]()]
            case 'tool_bridge':
                # Above the tools' own budgets, so a bridged tool's own `max_retries` is the one that binds, as for a
                # direct call. A bridged retry also spends a `tool_call` retry, by design.
                return [ToolBridge[object](max_retries=10)]
            case 'ipython':
                assert self.kernel is not None
                return [IPythonCodeMode[object](kernel=self.kernel, max_retries=self.sandbox_retries)]

    def tool(self, function: Callable[..., object], *, max_retries: int | None = None) -> Tool[object]:
        background = self.mode in ('background', 'run_code_background')
        return Tool[object](function, max_retries=max_retries, metadata={'background': True} if background else None)

    async def hold(self) -> None:
        """In `background` mode, hold the call until the model has seen that it started, so it ends after its step."""
        if self.mode == 'background':
            await self.released.wait()
            self.released.clear()

    def model(self, turns: Sequence[Sequence[_Call]]) -> FunctionModel:
        """A model that makes each turn's calls in turn, waits out background calls, then answers `done`."""
        made = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal made
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            if last.parts and all(
                isinstance(part, ToolReturnPart) and _BACKGROUND_STARTED in str(part.content) for part in last.parts
            ):
                self.released.set()
                return ModelResponse(parts=[TextPart('waiting')])
            if made == len(turns):
                return ModelResponse(parts=[TextPart('done')])
            made += 1
            return ModelResponse(parts=self._calls(turns[made - 1], made))

        return FunctionModel(respond)

    def _calls(self, calls: Sequence[_Call], turn: int) -> list[ToolCallPart]:
        match self.mode:
            case 'direct' | 'background':
                return [
                    ToolCallPart(name, args, tool_call_id=f'call_{turn}_{i}') for i, (name, args) in enumerate(calls)
                ]
            case 'tool_bridge':
                return [
                    ToolCallPart('tool_call', {'name': name, 'args_json': json.dumps(args)}, f'call_{turn}_{i}')
                    for i, (name, args) in enumerate(calls)
                ]
            case 'run_code' | 'run_code_background' | 'ipython':
                rendered = [f'{name}({", ".join(f"{k}={v!r}" for k, v in args.items())})' for name, args in calls]
                if self.mode == 'ipython':
                    code = rendered[0] if len(rendered) == 1 else f'[{", ".join(rendered)}]'
                elif len(rendered) == 1:
                    code = f'await {rendered[0]}'
                else:
                    code = f'import asyncio\nawait asyncio.gather({", ".join(rendered)})'
                return [ToolCallPart('run_code', {'code': code}, tool_call_id=f'call_{turn}')]


@pytest.fixture
def dispatch(request: pytest.FixtureRequest) -> _Dispatch:
    mode: Mode = request.param
    return _Dispatch(mode, kernel=request.getfixturevalue('kernel') if mode == 'ipython' else None)


def _call(name: str, **args: object) -> _Call:
    """A call the model makes: the tool's name and its arguments."""
    return name, args


def _modes(*modes: Mode) -> list[object]:
    return [
        pytest.param(
            mode,
            id=mode,
            marks=pytest.mark.skipif(not ipython_available(), reason='jupyter-client not installed')
            if mode == 'ipython'
            else (),
        )
        for mode in modes
    ]


def _told(messages: Sequence[ModelMessage]) -> str:
    """Everything the model was sent after the user's prompt, as text: tool results, retries and follow-ups."""
    texts: list[str] = []
    for message in sent_messages(messages)[1:]:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, ToolReturnPart):
                texts.append(part.model_response_str())
            elif isinstance(part, UserPromptPart):
                items = [part.content] if isinstance(part.content, str) else part.content
                texts.extend(
                    item if isinstance(item, str) else item.content
                    for item in items
                    if isinstance(item, str | TextContent)
                )
    return '\n'.join(texts)


@pytest.mark.parametrize(
    'dispatch',
    _modes('direct', 'run_code', 'run_code_background', 'background', 'tool_bridge', 'ipython'),
    indirect=True,
)
async def test_the_result_reaches_the_caller(dispatch: _Dispatch) -> None:
    """The tool's return value reaches whoever made the call.

    A tool `BackgroundTools` runs is first reported as started, by design, unless code mode's sandbox made the call
    and awaits its value: https://github.com/pydantic/pydantic-ai/pull/10750.
    """

    def forecast(city: str) -> str:
        return f'Sunny in {city}'

    agent = Agent(
        dispatch.model([[_call('forecast', city='Paris')]]),
        tools=[dispatch.tool(forecast)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Weather?')

    told = _told(result.all_messages())
    assert result.output == 'done'
    assert 'Sunny in Paris' in told
    assert (_BACKGROUND_STARTED in told) == (dispatch.mode == 'background')


@pytest.mark.parametrize(
    'dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge', 'ipython'), indirect=True
)
async def test_tool_return_content_reaches_the_model(dispatch: _Dispatch) -> None:
    """`ToolReturn.content` reaches the model: https://github.com/pydantic/pydantic-ai/pull/11456."""

    def forecast(city: str) -> ToolReturn[str]:
        return ToolReturn(return_value=f'Sunny in {city}', content='Bring sunglasses.')

    agent = Agent(
        dispatch.model([[_call('forecast', city='Paris')]]),
        tools=[dispatch.tool(forecast)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Weather?')

    assert result.output == 'done'
    assert 'Bring sunglasses.' in _told(result.all_messages())


@pytest.mark.parametrize(
    'dispatch',
    [
        *_modes('direct', 'run_code', 'background', 'tool_bridge'),
        pytest.param(
            'ipython',
            id='ipython',
            marks=[
                pytest.mark.skipif(not ipython_available(), reason='jupyter-client not installed'),
                pytest.mark.xfail(
                    raises=UnexpectedModelBehavior,
                    strict=True,
                    reason='`IPythonCodeMode` reports a nested `ToolFailed` as a `run_code` retry; `CodeMode` returns '
                    'a failed result since https://github.com/pydantic/pydantic-ai/pull/13076',
                ),
            ],
        ),
    ],
    indirect=True,
)
async def test_tool_failed_spends_no_retries(dispatch: _Dispatch) -> None:
    """A `ToolFailed` is a result the model reads, not a retry, so repeating it doesn't end the run.

    `run_code` gets one retry, so a sandbox that counted the failure as a retry would end the run on the second.
    """
    dispatch.sandbox_retries = 1

    def failing() -> str:
        raise ToolFailed('Disk full')

    agent = Agent(
        dispatch.model([[_call('failing')]] * 3),
        tools=[dispatch.tool(failing, max_retries=1)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Save it')

    assert result.output == 'done'
    assert _told(result.all_messages()).count('Disk full') == 3


_RETRIES_RUN_OUT: dict[Mode, tuple[str, list[int]]] = {
    'direct': ("Tool 'flaky' exceeded max retries count of 2", [0, 1, 2]),
    'background': ("Tool 'flaky' exceeded max retries count of 2", [0, 1, 2]),
    'tool_bridge': ("Tool 'flaky' exceeded max retries count of 2", [0, 1, 2]),
    # By design, a nested call's `ModelRetry` is the snippet's error: it spends `run_code`'s retries, not the tool's,
    # so the tool's `ctx.retry` stays at 0 (`ToolManager.handle_call`, `wrap_validation_errors=False`).
    'run_code': ("Tool 'run_code' exceeded max retries count of 3", [0, 0, 0, 0]),
}


@pytest.mark.parametrize('dispatch', _modes(*_RETRIES_RUN_OUT), indirect=True)
async def test_retries_run_out(dispatch: _Dispatch) -> None:
    """A tool that keeps asking for a retry ends the run once its `max_retries` are spent.

    In `background` mode each failure arrives after the step that made the call has ended:
    https://github.com/pydantic/pydantic-ai/pull/11458.
    """
    message, expected_retries = _RETRIES_RUN_OUT[dispatch.mode]
    retries: list[int] = []

    async def flaky(ctx: RunContext[object]) -> str:
        retries.append(ctx.retry)
        await dispatch.hold()
        raise ModelRetry('Try again.')

    agent = Agent(
        dispatch.model([[_call('flaky')]] * 6),
        tools=[dispatch.tool(flaky, max_retries=2)],
        capabilities=dispatch.capabilities(),
    )
    with pytest.raises(UnexpectedModelBehavior, match=message):
        await agent.run('Go')

    assert retries == expected_retries


_SUCCESS_RESETS: dict[Mode, list[int]] = {
    'direct': [0, 1, 0],
    'background': [0, 1, 0],
    'tool_bridge': [0, 1, 0],
    # By design, as in `_RETRIES_RUN_OUT`: the tool's own count never moves.
    'run_code': [0, 0, 0],
}


@pytest.mark.parametrize('dispatch', _modes(*_SUCCESS_RESETS), indirect=True)
async def test_a_success_resets_the_retry_count(dispatch: _Dispatch) -> None:
    """A call that succeeds after a retry resets the tool's count, so a later failure starts again from 0."""
    retries: list[int] = []

    async def research(ctx: RunContext[object]) -> str:
        retries.append(ctx.retry)
        await dispatch.hold()
        if len(retries) == 2:
            return 'Found it.'
        raise ModelRetry('Try again.')

    agent = Agent(
        dispatch.model([[_call('research')]] * 3),
        tools=[dispatch.tool(research, max_retries=1)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Go')

    assert result.output == 'done'
    assert retries == _SUCCESS_RESETS[dispatch.mode]


@dataclass
class _BlockFirstCall(AbstractCapability[AgentDepsT]):
    """A policy hook that refuses the first call to `tool_name` with a `ModelRetry` from `before_tool_execute`."""

    tool_name: str
    refused: bool = False

    async def before_tool_execute(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        if call.tool_name == self.tool_name and not self.refused:
            self.refused = True
            raise ModelRetry('Blocked by policy.')
        return args


@pytest.mark.parametrize('dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge'), indirect=True)
async def test_an_execute_hook_retry_reaches_the_model(dispatch: _Dispatch) -> None:
    """A `ModelRetry` from `before_tool_execute` asks the model to retry: https://github.com/pydantic/pydantic-ai/pull/10830."""

    def forecast(city: str) -> str:
        return f'Sunny in {city}'

    agent = Agent(
        dispatch.model([[_call('forecast', city='Paris')]] * 2),
        tools=[dispatch.tool(forecast)],
        capabilities=[*dispatch.capabilities(), _BlockFirstCall[object]('forecast')],
    )
    result = await agent.run('Weather?')

    told = _told(result.all_messages())
    assert result.output == 'done'
    assert 'Blocked by policy.' in told
    assert 'Sunny in Paris' in told


@pytest.mark.parametrize('dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge'), indirect=True)
async def test_tool_calls_limit_bounds_the_calls_that_run(dispatch: _Dispatch) -> None:
    """`tool_calls_limit` stops a batch of calls before more run than it allows.

    The batch is one model response in `direct`, `background` and `tool_bridge`, and one snippet's `asyncio.gather`
    in `run_code`: https://github.com/pydantic/pydantic-ai/pull/10362.
    """
    ran: list[int] = []

    def probe(n: int) -> int:
        ran.append(n)
        return n

    agent = Agent(
        dispatch.model([[_call('probe', n=n) for n in range(3)]]),
        tools=[dispatch.tool(probe)],
        capabilities=dispatch.capabilities(),
    )
    with pytest.raises(UsageLimitExceeded):
        await agent.run('Go', usage_limits=UsageLimits(tool_calls_limit=2))

    assert len(ran) <= 2


@pytest.mark.parametrize('dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge'), indirect=True)
async def test_parallel_ordered_events_runs_calls_concurrently(dispatch: _Dispatch) -> None:
    """Under `parallel_ordered_events`, calls made together run concurrently.

    Only their result events are ordered: https://github.com/pydantic/pydantic-ai/pull/10137. Each call waits until
    both have started, so the run completes only if they overlap.
    """
    started = 0
    both_started = asyncio.Event()

    async def probe(n: int) -> int:
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        return n

    agent = Agent(
        dispatch.model([[_call('probe', n=1), _call('probe', n=2)]]),
        tools=[dispatch.tool(probe)],
        capabilities=dispatch.capabilities(),
    )
    with anyio.fail_after(READINESS_WAIT_TIMEOUT), ToolManager.parallel_execution_mode('parallel_ordered_events'):
        result = await agent.run('Go')

    assert result.output == 'done'


@pytest.mark.parametrize('dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge'), indirect=True)
async def test_invalid_arguments_ask_the_model_to_retry(dispatch: _Dispatch) -> None:
    """Arguments that fail validation are reported to the model without running the tool, and the run goes on."""
    ran: list[int] = []

    def probe(n: int) -> int:
        ran.append(n)
        return n

    agent = Agent(
        dispatch.model([[_call('probe', n='lots')], [_call('probe', n=1)]]),
        tools=[dispatch.tool(probe)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Go')

    assert result.output == 'done'
    assert ran == [1]
    assert 'lots' in _told(result.all_messages())


@pytest.mark.parametrize('dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge'), indirect=True)
async def test_an_unknown_tool_name_asks_the_model_to_retry(dispatch: _Dispatch) -> None:
    """A call to a tool that doesn't exist is reported to the model by name, and the run goes on."""

    def probe() -> str:
        return 'probed'

    agent = Agent(
        dispatch.model([[_call('nonexistent')], [_call('probe')]]),
        tools=[dispatch.tool(probe)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Go')

    told = _told(result.all_messages())
    assert result.output == 'done'
    assert 'nonexistent' in told
    assert 'probed' in told


@pytest.mark.parametrize(
    'dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge', 'ipython'), indirect=True
)
async def test_a_tool_named_after_a_builtin(dispatch: _Dispatch) -> None:
    """A tool may be named `open` unless code calls it, where the name would mean the builtin.

    Both sandboxes refuse such a tool when the run starts: https://github.com/pydantic/pydantic-ai/pull/12767.
    """

    def open(path: str) -> str:
        return f'opened {path}'

    agent = Agent(
        dispatch.model([[_call('open', path='a.txt')]]),
        tools=[dispatch.tool(open)],
        capabilities=dispatch.capabilities(),
    )
    if dispatch.mode in ('run_code', 'ipython'):
        with pytest.raises(UserError, match="Tool name 'open' is a Python builtin"):
            await agent.run('Go')
        return
    result = await agent.run('Go')

    assert result.output == 'done'
    assert 'opened a.txt' in _told(result.all_messages())


async def _approve_all(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
    return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})


@pytest.mark.parametrize('dispatch', _modes('direct', 'run_code', 'background', 'tool_bridge'), indirect=True)
async def test_a_handler_resolves_an_approval(dispatch: _Dispatch) -> None:
    """A `HandleDeferredToolCalls` handler approves a call the tool asked approval for, and the tool runs.

    By design, a call `BackgroundTools` runs can't be deferred once its task has started: the model is told so instead.
    """

    def delete(ctx: RunContext[object], path: str) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return f'deleted {path}'

    agent = Agent(
        dispatch.model([[_call('delete', path='a.txt')]]),
        tools=[dispatch.tool(delete)],
        capabilities=[*dispatch.capabilities(), HandleDeferredToolCalls(_approve_all)],
    )
    result = await agent.run('Go')

    told = _told(result.all_messages())
    assert result.output == 'done'
    if dispatch.mode == 'background':
        assert 'background tools cannot defer a running task' in told
    else:
        assert 'deleted a.txt' in told


@dataclass
class _StepRecords(AbstractCapability[AgentDepsT]):
    """Copies the run step's failed and successful tools when a call to `tool_name` returns."""

    tool_name: str
    seen: list[tuple[set[str], set[str]]] = field(default_factory=list[tuple[set[str], set[str]]])

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: object,
    ) -> object:
        if call.tool_name == self.tool_name:
            assert ctx.tool_manager is not None
            self.seen.append((set(ctx.tool_manager.failed_tools), set(ctx.tool_manager.succeeded_tools)))
        return result


_RECORDS: dict[Mode, tuple[str, list[tuple[set[str], set[str]]]]] = {
    # A sandbox reports its nested calls through `run_code`, so they leave the step's records alone.
    'run_code': ('run_code', [(set(), set())]),
    'ipython': ('run_code', [(set(), set())]),
    # A bridged call is counted under its own name, as a direct call is.
    'tool_bridge': ('tool_call', [(set(), {'research'})]),
}


@pytest.mark.parametrize('dispatch', _modes(*_RECORDS), indirect=True)
async def test_nested_calls_and_the_step_records(dispatch: _Dispatch) -> None:
    """Whether a nested call's failure and success land in the step's records, which the next step's counts come from.

    The sandboxes run a snippet that catches a failed call and then makes a call that succeeds; the bridge runs one
    call that succeeds.
    """
    calls = 0

    def research() -> str:
        nonlocal calls
        calls += 1
        if calls == 1 and dispatch.mode != 'tool_bridge':
            raise ModelRetry('Try again.')
        return 'Found it.'

    dispatcher, expected = _RECORDS[dispatch.mode]
    records = _StepRecords[object](dispatcher)
    if dispatch.mode == 'tool_bridge':
        model = dispatch.model([[_call('research')]])
    else:
        call = 'research()' if dispatch.mode == 'ipython' else 'await research()'
        code = f'try:\n    {call}\nexcept Exception:\n    pass\n{call}'
        model = FunctionModel(
            lambda messages, info: ModelResponse(
                parts=[ToolCallPart('run_code', {'code': code}, tool_call_id='snippet')]
                if len(messages) == 1
                else [TextPart('done')]
            )
        )
    agent = Agent(model, tools=[dispatch.tool(research)], capabilities=[*dispatch.capabilities(), records])
    result = await agent.run('Go')

    assert result.output == 'done'
    assert records.seen == expected


@pytest.mark.parametrize('dispatch', _modes('tool_bridge'), indirect=True)
async def test_a_bridged_tool_that_is_not_available_yet(dispatch: _Dispatch) -> None:
    """Each `tool_call` gets its own free refusal of a bridged tool that isn't available yet.

    Every refusal still spends a `tool_call` retry, so a model that never takes the advice ends the run on
    `tool_call`'s budget rather than the bridged tool's.
    """
    ran: list[str] = []

    async def mark_searchable(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, with_native='tool_search')

    def hidden() -> str:
        ran.append('hidden')  # pragma: no cover - refused before it runs
        return 'found'  # pragma: no cover

    agent = Agent(
        dispatch.model([[_call('hidden')]] * 4),
        tools=[Tool[object](hidden, prepare=mark_searchable)],
        capabilities=dispatch.capabilities(),
    )
    result = await agent.run('Go')

    assert result.output == 'done'
    assert ran == []
    assert _told(result.all_messages()).count("Tool 'hidden' is not available yet") == 4
