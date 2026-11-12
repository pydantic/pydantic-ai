from __future__ import annotations

from collections.abc import Sequence

from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRetry, ToolFailed
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import CodeMode


def _run_code_returns(*snippets: str) -> Sequence[ToolReturnPart]:
    """Run each snippet as one `run_code` call, then finish, and return the `run_code` results."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(snippets):
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': snippets[step]})])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(respond), capabilities=[CodeMode()], retries=1)

    @agent.tool_plain
    def read_calendar() -> str:
        raise ToolFailed('No calendar is connected.')

    @agent.tool_plain
    def check_input() -> str:
        raise ModelRetry('Pass a date.')

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = agent.run_sync('run the code')
    return [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]


class TestCodeModeToolFailed:
    def test_uncaught_tool_failed_is_a_failed_result_that_spends_no_retry(self) -> None:
        # Four failures in a row with `retries=1`: a retry each would end the run after the second.
        returns = _run_code_returns(*['await read_calendar()'] * 4)
        assert [part.outcome for part in returns] == ['failed'] * 4
        assert returns[0].content == snapshot("""\
No calendar is connected.

1 nested tool calls started before execution stopped:
- read_calendar({}) did not finish, so it may have applied a partial change
Account for all 1 before retrying; repeating a call repeats whatever it already did.\
""")

    def test_uncaught_model_retry_is_still_a_retry(self) -> None:
        (part,) = _run_code_returns('await check_input()')
        assert part.outcome == 'retried'
        assert 'Pass a date.' in str(part.content)

    def test_tool_failed_reraised_after_a_later_call_is_a_failed_result(self) -> None:
        code = """\
print('checking')
try:
    await read_calendar()
except Exception as e:
    error = e
await ping()
raise error
"""
        (part,) = _run_code_returns(code)
        assert part.outcome == 'failed'
        assert part.content == snapshot("""\
[stdout before error]
checking
[/stdout before error]
No calendar is connected.

2 nested tool calls started before execution stopped:
- read_calendar({}) did not finish, so it may have applied a partial change
- ping({}) returned 'pong'
Account for all 2 before retrying; repeating a call repeats whatever it already did.\
""")

    def test_caught_tool_failed_then_another_error_is_a_retry(self) -> None:
        code = """\
try:
    await read_calendar()
except Exception:
    pass
undefined_name
"""
        (part,) = _run_code_returns(code)
        assert part.outcome == 'retried'
        assert 'undefined_name' in str(part.content)
