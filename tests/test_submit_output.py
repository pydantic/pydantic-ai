"""Tests for `RunContext.submit_output`, which lets a tool submit the run's output.

These exercise the agent loop's own handling of a submission (validation, retries, `end_strategy`),
which doesn't depend on any provider's behavior, so they drive the loop with `FunctionModel`.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelRetry,
    NativeOutput,
    PromptedOutput,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolOutput,
    ToolReturnPart,
)
from pydantic_ai.capabilities import PrepareOutputTools
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from ._inline_snapshot import snapshot

pytestmark = pytest.mark.anyio


class Invoice(BaseModel):
    order_id: str
    total: int


class Receipt(BaseModel):
    payment_id: str


def call_tool_then_reply(tool_name: str, args: dict[str, Any] | None = None):
    """A model that calls `tool_name` on its first turn and replies with text after that."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, args or {}, tool_call_id='call-1')])
        return ModelResponse(parts=[TextPart('model turn after the tool call')])

    return FunctionModel(model)


def tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


async def test_tool_output_submission_ends_run_without_another_model_request():
    agent = Agent(
        call_tool_then_reply('assemble'),
        output_type=[ToolOutput(Invoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt')],
    )
    seen_defs: list[ToolDefinition] = []

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        seen_defs.extend(ctx.output_tool_defs)
        await ctx.submit_output('submit_invoice', {'order_id': 'ord_1', 'total': 42})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='ord_1', total=42)
    assert [d.name for d in seen_defs] == ['submit_invoice', 'submit_receipt']
    assert [m.kind for m in result.all_messages()] == ['request', 'response', 'request']
    [tool_return] = tool_returns(result.all_messages())
    assert (tool_return.tool_name, tool_return.content, tool_return.outcome) == ('assemble', 'assembled', 'success')


@pytest.mark.parametrize(
    'output_type',
    [
        pytest.param(NativeOutput([Invoice, Receipt]), id='native'),
        pytest.param(PromptedOutput([Invoice, Receipt]), id='prompted'),
        pytest.param([Invoice, Receipt], id='auto'),
    ],
)
async def test_union_submission_in_every_structured_mode(output_type: Any):
    agent = Agent(call_tool_then_reply('assemble'), output_type=output_type)
    seen_names: list[str] = []

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        seen_names.extend(d.name for d in ctx.output_tool_defs)
        await ctx.submit_output('final_result_Receipt', {'payment_id': 'pay_1'})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Receipt(payment_id='pay_1')
    assert seen_names == ['final_result_Invoice', 'final_result_Receipt']


async def test_native_single_output_and_text_output():
    native = Agent(call_tool_then_reply('assemble'), output_type=NativeOutput(int))

    @native.tool
    async def assemble(ctx: RunContext[None]) -> str:
        [tool_def] = ctx.output_tool_defs
        assert tool_def.name == 'final_result'
        await ctx.submit_output('final_result', {'response': 7})
        return 'assembled'

    assert (await native.run('go')).output == 7

    text = Agent(call_tool_then_reply('write'), output_type=str)

    @text.tool
    async def write(ctx: RunContext[None]) -> str:
        assert [d.name for d in ctx.output_tool_defs] == ['final_result']
        await ctx.submit_output('final_result', {'response': 'the report'})
        return 'written'

    assert (await text.run('go')).output == 'the report'


async def test_text_output_alongside_output_tools():
    agent = Agent(call_tool_then_reply('write'), output_type=[Invoice, str])

    @agent.tool
    async def write(ctx: RunContext[None]) -> str:
        assert [d.name for d in ctx.output_tool_defs] == ['final_result', 'final_result_text']
        await ctx.submit_output('final_result_text', {'response': 'plain text'})
        return 'written'

    assert (await agent.run('go')).output == 'plain text'


async def test_rejected_submission_answers_the_tool_call_and_uses_an_output_retry():
    attempts: list[dict[str, Any]] = [{'order_id': 'ord_1', 'total': 'lots'}, {'order_id': 'ord_1', 'total': 3}]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id=f'call-{len(messages)}')])

    agent = Agent(FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'), retries={'output': 1})

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', attempts.pop(0))
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='ord_1', total=3)
    rejected, accepted = tool_returns(result.all_messages())
    assert (rejected.tool_name, rejected.outcome) == ('assemble', 'retried')
    assert rejected.content == snapshot(
        [
            {
                'type': 'int_parsing',
                'loc': ['total'],
                'msg': 'Input should be a valid integer, unable to parse string as an integer',
                'input': 'lots',
            }
        ]
    )
    assert (accepted.outcome, accepted.content) == ('success', 'assembled')


async def test_rejected_submissions_run_out_of_output_retries():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id=f'call-{len(messages)}')])

    agent = Agent(FunctionModel(model), output_type=NativeOutput(Invoice), retries={'output': 1})

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('final_result', {'order_id': 'ord_1'})
        return 'assembled'  # pragma: no cover

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(1\)'):
        await agent.run('go')


async def test_output_functions_and_validators_run_on_submissions():
    def invoice_total(invoice: Invoice) -> int:
        return invoice.total * 100

    agent = Agent(call_tool_then_reply('assemble'), output_type=ToolOutput(invoice_total, name='submit_invoice'))

    @agent.output_validator
    def no_zero_totals(output: int) -> int:
        if output == 0:
            raise ModelRetry('The total cannot be zero.')
        return output

    totals = [0, 5]

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'ord_1', 'total': totals.pop(0)})
        return 'assembled'

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id=f'call-{len(messages)}')])

    with agent.override(model=FunctionModel(model)):
        result = await agent.run('go')

    assert result.output == 500
    rejected, _ = tool_returns(result.all_messages())
    assert (rejected.outcome, rejected.content) == ('retried', 'The total cannot be zero.')


async def test_submission_dropped_when_tool_then_asks_for_retry():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) < 4:
            return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id=f'call-{len(messages)}')])
        return ModelResponse(parts=[ToolCallPart('submit_invoice', {'order_id': 'from_model', 'total': 1})])

    agent = Agent(FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'), retries=5)

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 1})
        raise ModelRetry('Something else went wrong.')

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_model', total=1)


async def test_earlier_output_tool_call_wins_over_a_later_submission():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('submit_invoice', {'order_id': 'from_model', 'total': 1}, tool_call_id='output-call'),
                ToolCallPart('assemble', {}, tool_call_id='tool-call'),
            ]
        )

    agent = Agent(
        FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'), end_strategy='exhaustive'
    )

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 2})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_model', total=1)


@pytest.mark.parametrize('end_strategy', ['graceful', 'exhaustive'])
async def test_submission_wins_over_a_later_output_tool_call(end_strategy: Any):
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('assemble', {}, tool_call_id='tool-call'),
                ToolCallPart('submit_invoice', {'order_id': 'from_model', 'total': 1}, tool_call_id='output-call'),
            ]
        )

    agent = Agent(
        FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'), end_strategy=end_strategy
    )

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 2})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_tool', total=2)


async def test_retry_from_sibling_tool_suppresses_submission():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('assemble', {}, tool_call_id='assemble-call'),
                    ToolCallPart('flaky', {}, tool_call_id='flaky-call'),
                ]
            )
        return ModelResponse(parts=[ToolCallPart('submit_invoice', {'order_id': 'from_model', 'total': 1})])

    agent = Agent(FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'))

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 2})
        return 'assembled'

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry('Try again.')

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_model', total=1)
    suppressed = next(part for part in tool_returns(result.all_messages()) if part.tool_name == 'assemble')
    assert suppressed.content == 'Output not used as the final result - addressing tool retries from this round first.'


async def hide_receipts(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return [d for d in tool_defs if d.name != 'submit_receipt']


async def test_unknown_and_unavailable_output_tools_are_rejected():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) < 5:
            return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id=f'call-{len(messages)}')])
        return ModelResponse(parts=[ToolCallPart('submit_invoice', {'order_id': 'from_model', 'total': 1})])

    agent = Agent(
        FunctionModel(model),
        output_type=[ToolOutput(Invoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt')],
        retries={'output': 3},
        capabilities=[PrepareOutputTools(hide_receipts)],
    )

    names = ['nope', 'submit_receipt']

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output(names.pop(0), {'payment_id': 'pay_1'})
        return 'assembled'  # pragma: no cover

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_model', total=1)
    assert [part.content for part in tool_returns(result.all_messages()) if part.tool_name == 'assemble'] == [
        "Unknown output tool 'nope'.",
        "Output tool 'submit_receipt' is not available right now.",
    ]


async def test_submit_output_misuse():
    agent = Agent(call_tool_then_reply('assemble'), output_type=Invoice)

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('final_result', {'order_id': 'ord_1', 'total': 1})
        await ctx.submit_output('final_result', {'order_id': 'ord_2', 'total': 2})
        return 'assembled'  # pragma: no cover

    with pytest.raises(UserError, match="Tool call 'call-1' already submitted an output"):
        await agent.run('go')

    ctx = RunContext(deps=None, model=FunctionModel(lambda m, i: ModelResponse(parts=[])), usage=None)  # type: ignore[arg-type]
    assert ctx.output_tool_defs == []
    with pytest.raises(UserError, match='can only be called from a tool call during an agent run'):
        await ctx.submit_output('final_result', {})


async def test_message_history_after_submission():
    agent = Agent(call_tool_then_reply('assemble'), output_type=NativeOutput(Invoice))

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> dict[str, str]:
        await ctx.submit_output('final_result', {'order_id': 'ord_1', 'total': 1})
        return {'status': 'submitted'}

    result = await agent.run('go')

    [request, response, tool_request] = result.all_messages()
    assert isinstance(request, ModelRequest)
    assert isinstance(response, ModelResponse)
    assert isinstance(tool_request, ModelRequest)
    [tool_return] = tool_request.parts
    assert isinstance(tool_return, ToolReturnPart)
    assert (tool_return.tool_call_id, tool_return.content) == ('call-1', {'status': 'submitted'})


async def test_rejected_output_tool_submissions_run_out_of_the_tool_retries():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id=f'call-{len(messages)}')])

    agent = Agent(FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'), retries={'output': 1})

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'ord_1'})
        return 'assembled'  # pragma: no cover

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(1\)'):
        await agent.run('go')


async def test_text_submission_with_invalid_arguments_is_rejected():
    texts: list[dict[str, Any]] = [{'text': 'wrong key'}, {'response': 'the report'}]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('write', {}, tool_call_id=f'call-{len(messages)}')])

    agent = Agent(FunctionModel(model), output_type=str)

    @agent.tool
    async def write(ctx: RunContext[None]) -> str:
        await ctx.submit_output('final_result', texts.pop(0))
        return 'written'

    result = await agent.run('go')

    assert result.output == 'the report'
    rejected, _ = tool_returns(result.all_messages())
    assert rejected.outcome == 'retried'
    assert rejected.content == snapshot(
        [{'type': 'missing', 'loc': ['response'], 'msg': 'Field required', 'input': {'text': 'wrong key'}}]
    )


class Refund(BaseModel):
    """A refund of a previous payment."""

    payment_id: str


async def test_output_tool_defs_names_and_descriptions():
    agent = Agent(
        call_tool_then_reply('inspect'),
        output_type=[ToolOutput(Invoice, name='final_result_text'), str],
    )
    native = Agent(call_tool_then_reply('inspect'), output_type=NativeOutput([Invoice, Refund]))
    seen: list[list[tuple[str, str | None]]] = []

    async def inspect(ctx: RunContext[None]) -> str:
        defs = ctx.output_tool_defs
        seen.append([(d.name, d.description) for d in defs])
        last = defs[-1]
        await ctx.submit_output(
            last.name, {'payment_id': 'pay_1'} if last.name.endswith('Refund') else {'response': 'x'}
        )
        return 'done'

    agent.tool(inspect)
    native.tool(inspect)
    assert (await agent.run('go')).output == 'x'
    assert (await native.run('go')).output == Refund(payment_id='pay_1')

    assert seen == snapshot(
        [
            [
                ('final_result_text', 'The final response which ends this conversation'),
                ('final_result_text_2', 'The final response which ends this conversation'),
            ],
            [
                ('final_result_Invoice', 'Invoice: The final response which ends this conversation'),
                ('final_result_Refund', 'A refund of a previous payment.'),
            ],
        ]
    )


async def test_exhaustive_submission_wins_after_a_failed_output_tool_call():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('submit_invoice', {'order_id': 'from_model'}, tool_call_id='output-call'),
                ToolCallPart('lookup', {}, tool_call_id='lookup-call'),
                ToolCallPart('assemble', {}, tool_call_id='tool-call'),
            ]
        )

    agent = Agent(
        FunctionModel(model), output_type=ToolOutput(Invoice, name='submit_invoice'), end_strategy='exhaustive'
    )

    @agent.tool_plain
    def lookup() -> str:
        return 'found'

    @agent.tool
    async def assemble(ctx: RunContext[None]) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 2})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_tool', total=2)
