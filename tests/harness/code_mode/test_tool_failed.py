from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRetry, RunContext, ToolDefinition, ToolFailed
from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import CodeMode


class AccountNotConnected(ToolFailed):
    """A `ToolFailed` subclass that carries data, as an application's own failure type might."""

    def __init__(self, account: str):
        super().__init__(f'Connect your {account} account first.')
        self.account = account


@dataclass
class RecordRunCodeErrors(AbstractCapability[Any]):
    """Records the exception each `run_code` execution ends with, as an error-reporting hook would."""

    errors: list[Exception] = field(default_factory=list[Exception])

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        try:
            return await handler(args)
        except Exception as error:
            if call.tool_name == 'run_code':
                self.errors.append(error)
            raise


async def _run_code_returns(*snippets: str, recorder: RecordRunCodeErrors | None = None) -> Sequence[ToolReturnPart]:
    """Run each snippet as one `run_code` call, then finish, and return the `run_code` results."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(snippets):
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': snippets[step]})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), capabilities=[CodeMode(), recorder or RecordRunCodeErrors()], retries=1)
    calendar_failed = anyio.Event()

    @agent.tool_plain
    def read_calendar() -> str:
        calendar_failed.set()
        raise ToolFailed('No calendar is connected.')

    @agent.tool_plain
    async def ping_after_calendar() -> str:
        await calendar_failed.wait()
        return 'pong'

    @agent.tool_plain
    def read_inbox() -> str:
        raise AccountNotConnected('email')

    @agent.tool_plain
    def check_input() -> str:
        raise ModelRetry('Pass a date.')

    @agent.tool_plain
    def check_calendar() -> str:
        raise ModelRetry('No calendar is connected.')

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('run the code')
    return [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]


class TestCodeModeToolFailed:
    async def test_uncaught_tool_failed_is_a_failed_result_that_spends_no_retry(self) -> None:
        # Four failures in a row with `retries=1`: a retry each would end the run after the second.
        returns = await _run_code_returns(*['await read_calendar()'] * 4)
        assert [part.outcome for part in returns] == ['failed'] * 4
        assert returns[0].content == snapshot('No calendar is connected.')

    async def test_uncaught_model_retry_is_still_a_retry(self) -> None:
        (part,) = await _run_code_returns('await check_input()')
        assert part.outcome == 'retried'
        assert part.content == snapshot("""\
Runtime error:
Traceback (most recent call last):
  File "<python-input-0>", line 1, in <module>
    await check_input()
    ~~~~~~~~~~~~~~~~~~~
Exception: Pass a date.\
""")

    async def test_tool_failed_reraised_after_a_later_call_is_a_failed_result(self) -> None:
        code = """\
print('checking')
try:
    await check_input()
except Exception:
    pass
try:
    await read_calendar()
except Exception as e:
    error = e
await ping()
raise error
"""
        (part,) = await _run_code_returns(code)
        assert part.outcome == 'failed'
        assert part.content == snapshot("""\
[stdout before error]
checking
[/stdout before error]
No calendar is connected.

3 nested tool calls started before execution stopped:
- check_input({}) did not finish, so it may have applied a partial change
- read_calendar({}) failed: 'No calendar is connected.'
- ping({}) returned 'pong'
Account for all 3 before retrying; repeating a call repeats whatever it already did.\
""")

    async def test_caught_tool_failed_then_another_error_is_a_retry(self) -> None:
        code = """\
try:
    await read_calendar()
except Exception:
    pass
raise Exception('Pick another day.')
"""
        (part,) = await _run_code_returns(code)
        assert part.outcome == 'retried'
        assert part.content == snapshot("""\
Runtime error:
Traceback (most recent call last):
  File "<python-input-0>", line 5, in <module>
    raise Exception('Pick another day.')
Exception: Pick another day.\
""")

    async def test_later_model_retry_with_the_same_message_is_a_retry(self) -> None:
        code = """\
try:
    await read_calendar()
except Exception:
    pass
await check_calendar()
"""
        (part,) = await _run_code_returns(code)
        assert part.outcome == 'retried'
        assert part.content == snapshot("""\
Runtime error:
Traceback (most recent call last):
  File "<python-input-0>", line 5, in <module>
    await check_calendar()
    ~~~~~~~~~~~~~~~~~~~~~~
Exception: No calendar is connected.\
""")

    async def test_snippet_error_with_the_same_message_is_a_retry(self) -> None:
        code = """\
try:
    await read_calendar()
except Exception:
    pass
{}['No calendar is connected.']
"""
        (part,) = await _run_code_returns(code)
        assert part.outcome == 'retried'
        assert part.content == snapshot("""\
Runtime error:
Traceback (most recent call last):
  File "<python-input-0>", line 5, in <module>
    {}['No calendar is connected.']
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
KeyError: No calendar is connected.\
""")

    async def test_tool_failed_beside_a_later_success_is_a_failed_result(self) -> None:
        code = """\
import asyncio
await asyncio.gather(read_calendar(), ping_after_calendar())
"""
        (part,) = await _run_code_returns(code)
        assert part.outcome == 'failed'
        assert part.content == snapshot("""\
No calendar is connected.

2 nested tool calls started before execution stopped:
- read_calendar({}) failed: 'No calendar is connected.'
- ping_after_calendar({}) returned 'pong'
Account for all 2 before retrying; repeating a call repeats whatever it already did.\
""")

    async def test_tool_failed_subclass_reaches_run_code_hooks_as_the_cause(self) -> None:
        recorder = RecordRunCodeErrors()
        (part,) = await _run_code_returns('print("checking")\nawait read_inbox()', recorder=recorder)
        assert part.outcome == 'failed'
        (error,) = recorder.errors
        causes: list[BaseException] = []
        cause: BaseException | None = error
        while cause is not None:
            causes.append(cause)
            cause = cause.__cause__
        # Core wraps the `ToolFailed` `run_code` raises, which carries the printed output, and that
        # one is caused by the instance the nested tool raised.
        assert [type(cause).__name__ for cause in causes] == snapshot(
            ['ToolFailedError', 'ToolFailed', 'AccountNotConnected']
        )
        assert isinstance(causes[-1], AccountNotConnected)
        assert causes[-1].account == 'email'
