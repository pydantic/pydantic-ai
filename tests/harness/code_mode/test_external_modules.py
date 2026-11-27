"""`CodeMode(external_modules=...)`: sandbox code imports modules whose functions run on the host."""

from __future__ import annotations

import asyncio
import re

import pytest
from pydantic_monty import ExternalModule

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import CodeMode
from tests._timeouts import READINESS_WAIT_TIMEOUT
from tests.harness.code_mode.grpc_relay import MontyRelay

ROWS = {'eu': 160.0, 'us': 180.0}


def total(region: str) -> float:
    if region not in ROWS:
        raise KeyError(f'no region {region!r}')
    return ROWS[region]


async def fx_rate(currency: str) -> float:
    # Waits for the `echo` tool, so the snippet only finishes when the gathered calls overlap.
    await asyncio.wait_for(tool_started.wait(), timeout=READINESS_WAIT_TIMEOUT)
    return {'EUR': 0.9}[currency]


def doubled(n: int) -> int:
    return n * 2


tool_started = asyncio.Event()

WAREHOUSE = ExternalModule(
    {
        'total': total,
        'fx_rate': fx_rate,
        'CURRENCY': 'USD',
        # Hidden by Monty, and never dispatched even when a remote worker names it.
        '_private': total,
        'util': ExternalModule({'doubled': doubled}, stubs='def doubled(n: int) -> int: ...'),
    },
    stubs='def total(region: str) -> float: ...\nasync def fx_rate(currency: str) -> float: ...\nCURRENCY: str',
)


async def echo(value: str) -> str:
    tool_started.set()
    return value


def _code_model(*snippets: str) -> FunctionModel:
    """Runs each snippet as one `run_code` call, then answers with the last result."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        results = [p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if len(results) < len(snippets):
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': snippets[len(results)]})])
        return ModelResponse(parts=[TextPart(repr(results[-1]))])

    return FunctionModel(model)


def _agent(*snippets: str, monty_sandbox: str | None = None, tools: bool = True) -> Agent[None, str]:
    return Agent(
        _code_model(*snippets),
        tools=[echo] if tools else [],
        capabilities=[CodeMode(external_modules={'warehouse': WAREHOUSE}, monty_sandbox=monty_sandbox)],
    )


@pytest.fixture(autouse=True)
def _fresh_tool_started() -> None:
    # An `asyncio.Event` binds to the loop that first waits on it, and each test has its own loop.
    global tool_started
    tool_started = asyncio.Event()


MODULE_CALLS = (
    'import asyncio\nimport warehouse as w\nimport warehouse.util\nfrom warehouse import total\n'
    'rate, tool = await asyncio.gather(w.fx_rate(currency="EUR"), echo(value="tool"))\n'
    '[total(region="us") * rate, tool, warehouse.util.doubled(21), w.CURRENCY]'
)


async def _assert_module_calls(monty_sandbox: str | None) -> None:
    result = await _agent(MODULE_CALLS, monty_sandbox=monty_sandbox).run('go')
    assert result.output == "[162.0, 'tool', 42, 'USD']"
    # Module calls are not tool calls: the history holds the `run_code` call and nothing for `total` or `fx_rate`.
    calls = [p.tool_name for m in result.all_messages() for p in m.parts if isinstance(p, ToolCallPart)]
    assert calls == ['run_code']


async def test_module_calls_run_on_the_host_alongside_tools() -> None:
    """Sync and async module calls, from any import form, run on the host; gathered ones overlap with tools."""
    await _assert_module_calls(None)


async def test_module_calls_from_a_remote_worker(monty_relay: MontyRelay) -> None:
    """A remote worker sends module calls back to the host, as it does tool calls."""
    await _assert_module_calls(monty_relay.url)


async def test_module_exception_is_raised_at_the_call_site() -> None:
    code = 'import warehouse\ntry:\n    warehouse.total(region="apac")\nexcept KeyError as e:\n    r = e.args[0]\nr'
    result = await _agent(code).run('go')
    assert result.output == '"no region \'apac\'"'


async def test_wrong_arguments_to_an_async_function_are_raised_at_the_call_site() -> None:
    """An async function rejects bad arguments when called, before there is a coroutine to defer.

    Only the first snippet is type-checked, so the second reaches the function with the wrong arguments.
    """
    code = (
        'kwargs = {"unknown": 1}\ntry:\n    await warehouse.fx_rate(**kwargs)\n'
        '    r = "called"\nexcept TypeError:\n    r = "caught"\nr'
    )
    assert (await _agent('import warehouse', code).run('go')).output == "'caught'"


async def test_stubs_type_check_the_snippet_and_describe_the_module() -> None:
    # No tools: the module's stubs alone switch the type check on.
    result = await _agent('import warehouse\nwarehouse.total(region=1)', tools=False).run('go')
    (returned,) = [p.content for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert isinstance(returned, str) and 'Type error in code' in returned and 'invalid-argument-type' in returned

    info: list[AgentInfo] = []

    def capture(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
        info.append(agent_info)
        return ModelResponse(parts=[TextPart('done')])

    await _agent().run('go', model=FunctionModel(capture))
    (run_code,) = info[0].function_tools
    assert run_code.description is not None
    assert (
        '- **Host modules**: these modules can also be imported, and their functions run on the host, '
        'outside the sandbox:\n\n```python\n'
        '# warehouse\ndef total(region: str) -> float: ...\nasync def fx_rate(currency: str) -> float: ...\n'
        'CURRENCY: str\n\n# warehouse.util\ndef doubled(n: int) -> int: ...\n```'
    ) in run_code.description


@pytest.mark.parametrize(
    ('modules', 'message'),
    [
        pytest.param(
            {'warehouse.util': WAREHOUSE},
            "`external_modules` key 'warehouse.util' is not a module name.",
            id='dotted_name',
        ),
        pytest.param(
            {'warehouse': ExternalModule({'total': total}), 'other': WAREHOUSE},
            "Missing for: 'warehouse'.",
            id='no_stubs',
        ),
        pytest.param(
            {'warehouse': ExternalModule({'util': ExternalModule({'doubled': doubled})}, stubs='')},
            "Missing for: 'warehouse', 'warehouse.util'.",
            id='no_submodule_stubs',
        ),
    ],
)
def test_modules_are_checked_when_the_agent_is_built(modules: dict[str, ExternalModule], message: str) -> None:
    with pytest.raises(UserError, match=re.escape(message)):
        CodeMode(external_modules=modules)
