"""`CodeMode(os_policy=...)`: the sandbox's clock and zone, merged over the harness defaults."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic_monty import NOT_HANDLED, MountDir, OsFunction

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import CodeMode
from pydantic_ai_harness.code_mode import CodeModeOSPolicy, CodeModeToolset
from pydantic_ai_harness.code_mode._eager import EagerCodeModeToolset

PARIS_NOON = datetime(2026, 1, 15, 12, 0, tzinfo=ZoneInfo('Europe/Paris'))
CLOCK_CODE = """\
import datetime, time
now = datetime.datetime.now()
clock = [now.isoformat(), now.astimezone().isoformat(), now.astimezone().strftime('%Z')]
clock"""
POLICY_CLOCK = 'read the clock configured for this sandbox'
CLOCK_UNAVAILABLE = '`time.time()` are unavailable here'


def paris_clock(*, name: OsFunction, args: tuple[Any, ...], kwargs: dict[str, Any], **_: Any) -> Any:
    """An `os_access` clock answering in Paris wall time, the natural handler from the issue."""
    if name == 'datetime.now':
        return PARIS_NOON.replace(tzinfo=None)
    return NOT_HANDLED  # pragma: no cover


def _wrapper(code_mode: CodeMode[None]) -> CodeModeToolset[None]:
    wrapper = code_mode.get_wrapper_toolset(FunctionToolset[None]())
    assert isinstance(wrapper, CodeModeToolset)
    return wrapper


async def _run(code_mode: CodeMode[None], code: str) -> Any:
    wrapper = _wrapper(code_mode)
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    async with wrapper:
        ctx.tool_manager = await ToolManager(toolset=wrapper).for_run_step(ctx)
        tools = await wrapper.get_tools(ctx)
        return (await wrapper.call_tool('run_code', {'code': code}, ctx, tools['run_code'])).return_value


async def _description(code_mode: CodeMode[None]) -> str:
    wrapper = _wrapper(code_mode)
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    description = (await wrapper.get_tools(ctx))['run_code'].tool_def.description
    assert description is not None
    return description


async def test_policy_clock_and_zone_agree_without_os_access() -> None:
    code_mode = CodeMode[None](os_policy={'datetime': PARIS_NOON, 'timezone': 'Europe/Paris'})
    code = f'{CLOCK_CODE} + [list(time.tzname)]'
    assert await _run(code_mode, code) == [
        '2026-01-15T12:00:00',
        '2026-01-15T12:00:00+01:00',
        'CET',
        ['CET', 'CEST'],
    ]


async def test_zone_alone_keeps_the_clock_on_os_access() -> None:
    code_mode = CodeMode[None](os_access=paris_clock, os_policy={'timezone': 'Europe/Paris'})
    assert await _run(code_mode, CLOCK_CODE) == ['2026-01-15T12:00:00', '2026-01-15T12:00:00+01:00', 'CET']


async def test_without_policy_the_zone_stays_utc() -> None:
    """The default is unchanged: an `os_access` clock is read as UTC wall time."""
    code_mode = CodeMode[None](os_access=paris_clock)
    assert await _run(code_mode, CLOCK_CODE) == ['2026-01-15T12:00:00', '2026-01-15T12:00:00+00:00', 'UTC']


@pytest.mark.parametrize(
    ('code', 'unsupported'),
    [
        pytest.param('import datetime\ndatetime.datetime.now()', 'datetime.now', id='clock'),
        pytest.param('import random\nrandom.random()', 'os.urandom', id='random'),
    ],
)
async def test_keys_the_policy_leaves_out_keep_their_defaults(code: str, unsupported: str) -> None:
    code_mode = CodeMode[None](os_policy={'timezone': 'Europe/Paris'})
    with pytest.raises(ModelRetry, match=rf"'{unsupported}' is not supported in this environment"):
        await _run(code_mode, code)


async def test_description_advertises_a_policy_clock() -> None:
    description = await _description(CodeMode[None](os_policy={'datetime': 'system'}))
    assert POLICY_CLOCK in description
    assert '- **No filesystem or environment**' in description
    assert CLOCK_UNAVAILABLE not in description


async def test_description_advertises_a_policy_clock_with_a_mount(tmp_path: Path) -> None:
    mount = MountDir(virtual_path='/work', host_path=str(tmp_path))
    description = await _description(CodeMode[None](mount=mount, os_policy={'datetime': PARIS_NOON}))
    assert POLICY_CLOCK in description
    assert '`os.getenv`/`os.environ` remain unavailable.' in description
    assert '`time.time()` remain unavailable' not in description


async def test_description_advertises_a_policy_clock_with_os_access() -> None:
    description = await _description(CodeMode[None](os_access=paris_clock, os_policy={'datetime': 'system'}))
    assert POLICY_CLOCK in description
    assert '`pathlib.Path` operations and `os.getenv`/`os.environ` are routed to the OS handler' in description
    assert '`time.time()` are routed' not in description


@pytest.mark.parametrize(
    'os_policy',
    [pytest.param(None, id='default'), pytest.param({'timezone': 'Europe/Paris'}, id='zone-only')],
)
async def test_description_without_a_policy_clock_is_unchanged(os_policy: CodeModeOSPolicy | None) -> None:
    description = await _description(CodeMode[None](os_policy=os_policy))
    assert CLOCK_UNAVAILABLE in description
    assert POLICY_CLOCK not in description


@pytest.mark.parametrize('eager', [False, True])
def test_os_policy_reaches_the_toolset(eager: bool) -> None:
    os_policy: CodeModeOSPolicy = {'datetime': 'system', 'timezone': 'Europe/Paris'}
    wrapper = _wrapper(CodeMode[None](os_policy=os_policy, eager=eager))
    assert isinstance(wrapper, EagerCodeModeToolset) is eager
    assert wrapper.os_policy == os_policy
