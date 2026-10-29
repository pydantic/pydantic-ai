"""Tests for `RunContext.submit_output`, which lets a tool submit the run's output.

These exercise the agent loop's own handling of a submission (validation, retries, `end_strategy`),
which doesn't depend on any provider's behavior, so they drive the loop with `FunctionModel`.
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterable, AsyncIterator
from datetime import datetime
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
from pydantic_ai._agent_graph import CallToolsNode
from pydantic_ai.agent import EndStrategy
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    HandleDeferredToolCalls,
    NodeResult,
    PrepareOutputTools,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.durable_exec import durable_unit_scope
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import AgentStreamEvent
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_graph import End

from ._inline_snapshot import snapshot


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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
        seen_names.extend(d.name for d in ctx.output_tool_defs)
        await ctx.submit_output('final_result_Receipt', {'payment_id': 'pay_1'})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Receipt(payment_id='pay_1')
    assert seen_names == ['final_result_Invoice', 'final_result_Receipt']


async def test_native_single_output_and_text_output():
    native = Agent(call_tool_then_reply('assemble'), output_type=NativeOutput(int))

    @native.tool
    async def assemble(ctx: RunContext) -> str:
        [tool_def] = ctx.output_tool_defs
        assert tool_def.name == 'final_result'
        await ctx.submit_output('final_result', {'response': 7})
        return 'assembled'

    assert (await native.run('go')).output == 7

    text = Agent(call_tool_then_reply('write'), output_type=str)

    @text.tool
    async def write(ctx: RunContext) -> str:
        assert [d.name for d in ctx.output_tool_defs] == ['final_result']
        await ctx.submit_output('final_result', {'response': 'the report'})
        return 'written'

    assert (await text.run('go')).output == 'the report'


async def test_text_output_alongside_output_tools():
    agent = Agent(call_tool_then_reply('write'), output_type=[Invoice, str])

    @agent.tool
    async def write(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 2})
        return 'assembled'

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry('Try again.')

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_model', total=1)
    suppressed = next(part for part in tool_returns(result.all_messages()) if part.tool_name == 'assemble')
    assert suppressed.content == 'Output not used as the final result - addressing tool retries from this round first.'


async def hide_receipts(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
        await ctx.submit_output('final_result', {'order_id': 'ord_1', 'total': 1})
        await ctx.submit_output('final_result', {'order_id': 'ord_2', 'total': 2})
        return 'assembled'  # pragma: no cover

    with pytest.raises(UserError, match="Tool call 'call-1' already submitted an output"):
        await agent.run('go')

    ctx = RunContext(deps=None, model=FunctionModel(lambda m, i: ModelResponse(parts=[])), usage=None)  # type: ignore[arg-type]
    assert ctx.output_tool_defs == []
    with pytest.raises(UserError, match='can only be called during an agent run'):
        await ctx.submit_output('final_result', {})


async def test_message_history_after_submission():
    agent = Agent(call_tool_then_reply('assemble'), output_type=NativeOutput(Invoice))

    @agent.tool
    async def assemble(ctx: RunContext) -> dict[str, str]:
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
    async def assemble(ctx: RunContext) -> str:
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
    async def write(ctx: RunContext) -> str:
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

    async def inspect(ctx: RunContext) -> str:
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
    async def assemble(ctx: RunContext) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 2})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_tool', total=2)


def two_tools_in_one_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart('submit_good', {}, tool_call_id='good'),
                ToolCallPart('submit_bad', {}, tool_call_id='bad'),
            ]
        )
    return ModelResponse(parts=[ToolCallPart('final_result', {'response': 99})])  # pragma: no cover


@pytest.mark.parametrize('end_strategy', ['graceful', 'exhaustive'])
@pytest.mark.parametrize('output_retries', [0, 1])
async def test_failed_submission_does_not_keep_a_sibling_submission_from_winning(
    end_strategy: Any, output_retries: int
):
    """A rejected submission, even one out of retries, is a failed output, like a failed output tool call."""
    agent = Agent(
        FunctionModel(two_tools_in_one_response),
        output_type=ToolOutput(int),
        end_strategy=end_strategy,
        retries={'output': output_retries},
    )

    @agent.tool
    async def submit_good(ctx: RunContext) -> str:
        await ctx.submit_output('final_result', {'response': 42})
        return 'submitted'

    @agent.tool
    async def submit_bad(ctx: RunContext) -> str:
        await ctx.submit_output('final_result', {'response': 'not a number'})
        return 'submitted'  # pragma: no cover

    result = await agent.run('go')

    assert result.output == 42


async def test_concurrent_submissions_from_one_call_are_refused():
    agent = Agent(call_tool_then_reply('assemble'), output_type=int)
    validated: list[int] = []
    outcomes: list[BaseException | None] = []

    @agent.output_validator
    def record(output: int) -> int:
        validated.append(output)
        return output

    @agent.tool
    async def assemble(ctx: RunContext) -> str:
        outcomes.extend(
            await asyncio.gather(
                ctx.submit_output('final_result', {'response': 1}),
                ctx.submit_output('final_result', {'response': 2}),
                return_exceptions=True,
            )
        )
        return 'assembled'

    result = await agent.run('go')

    assert result.output == 1
    assert validated == [1]
    accepted, refused = outcomes
    assert accepted is None
    assert isinstance(refused, UserError)
    assert str(refused) == snapshot("Tool call 'call-1' already submitted an output with `submit_output`.")


async def test_submit_output_refused_inside_a_durable_unit():
    """Unit test: a DBOS step or Prefect task records the tool's return value, so replay can't resubmit.

    Not run under a real engine, since what's under test is the refusal at the call, which the engines
    reach through the same `durable_unit_scope`.
    """
    agent = Agent(call_tool_then_reply('assemble'), output_type=int)

    @agent.tool
    async def assemble(ctx: RunContext) -> None:
        with durable_unit_scope():
            await ctx.submit_output('final_result', {'response': 1})

    with pytest.raises(UserError, match='cannot be called from a tool that runs as a durable step or task'):
        await agent.run('go')


class Meeting(BaseModel):
    starts_at: datetime


async def test_native_submission_with_values_json_cannot_encode():
    """Arguments from a Python tool can hold values like `datetime` that plain `json.dumps` can't encode."""
    agent = Agent(call_tool_then_reply('schedule'), output_type=NativeOutput(Meeting))

    @agent.tool
    async def schedule(ctx: RunContext) -> str:
        await ctx.submit_output('final_result', {'starts_at': datetime(2026, 10, 27, 9, 30)})
        return 'scheduled'

    result = await agent.run('go')

    assert result.output == Meeting(starts_at=datetime(2026, 10, 27, 9, 30))


async def test_inline_approved_tool_runs_out_of_output_retries():
    """A tool approved by a deferred-call handler runs after the first exhaustion check, and is still caught."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('assemble', {}, tool_call_id='call-1')])
        return ModelResponse(  # pragma: no cover
            parts=[ToolCallPart('submit_invoice', {'order_id': 'ord_1', 'total': 1}, tool_call_id='call-2')]
        )

    agent = Agent(
        FunctionModel(model),
        output_type=ToolOutput(Invoice, name='submit_invoice'),
        retries={'output': 0},
        capabilities=[HandleDeferredToolCalls(handler=lambda ctx, requests: requests.build_results(approve_all=True))],
    )

    @agent.tool(requires_approval=True)
    async def assemble(ctx: RunContext) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'ord_1', 'total': 'lots'})
        return 'assembled'  # pragma: no cover

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(0\)'):
        await agent.run('go')


async def test_context_retained_past_the_end_of_its_run():
    retained: list[RunContext] = []
    agent = Agent(call_tool_then_reply('keep'))

    @agent.tool
    async def keep(ctx: RunContext) -> str:
        retained.append(ctx)
        return 'kept'

    await agent.run('go')
    gc.collect()
    [ctx] = retained

    assert ctx.output_tool_defs == []
    with pytest.raises(UserError, match='The run this context belongs to has ended'):
        await ctx.submit_output('final_result', {'response': 'late'})


async def test_output_tool_defs_reflect_prepare_output_tools():
    seen: list[list[str]] = []
    agent = Agent(
        call_tool_then_reply('inspect'),
        output_type=[ToolOutput(Invoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt'), str],
        capabilities=[PrepareOutputTools(hide_receipts)],
    )

    @agent.tool
    async def inspect(ctx: RunContext) -> str:
        seen.append([tool_def.name for tool_def in ctx.output_tool_defs])
        return 'inspected'

    await agent.run('go')

    assert seen == [['submit_invoice', 'final_result_text']]


async def test_retry_from_a_sibling_suppresses_an_inline_approved_submission():
    """A tool approved inline runs after retry-wins was first applied, and a sibling's retry still wins."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('flaky', {}, tool_call_id='call-flaky'),
                    ToolCallPart('assemble', {}, tool_call_id='call-assemble'),
                ]
            )
        return ModelResponse(parts=[ToolCallPart('submit_invoice', {'order_id': 'from_model', 'total': 2})])

    agent = Agent(
        FunctionModel(model),
        output_type=ToolOutput(Invoice, name='submit_invoice'),
        end_strategy='graceful',
        capabilities=[HandleDeferredToolCalls(handler=lambda ctx, requests: requests.build_results(approve_all=True))],
    )

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry('Try again.')

    @agent.tool(requires_approval=True)
    async def assemble(ctx: RunContext) -> str:
        await ctx.submit_output('submit_invoice', {'order_id': 'from_tool', 'total': 1})
        return 'assembled'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_model', total=2)


@pytest.mark.parametrize('hook', ['before_tool_execute', 'after_tool_execute', 'wrap_tool_execute'])
async def test_submission_from_a_tool_execution_hook(hook: str):
    """A tool's hooks run with its call's context, so their submission counts as the tool call's, like the tool's own."""

    class SubmitForTheTool(AbstractCapability[None]):
        async def before_tool_execute(
            self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
        ) -> ValidatedToolArgs:
            if hook == 'before_tool_execute':
                await ctx.submit_output('final_result', {'order_id': 'before', 'total': 1})
            return args

        async def after_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: ValidatedToolArgs,
            result: Any,
        ) -> Any:
            if hook == 'after_tool_execute':
                await ctx.submit_output('final_result', {'order_id': 'after', 'total': 2})
            return result

        async def wrap_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: ValidatedToolArgs,
            handler: WrapToolExecuteHandler,
        ) -> Any:
            result = await handler(args)
            if hook == 'wrap_tool_execute':
                await ctx.submit_output('final_result', {'order_id': 'wrap', 'total': 3})
            return result

    agent = Agent(call_tool_then_reply('assemble'), output_type=Invoice, capabilities=[SubmitForTheTool()])

    @agent.tool_plain
    def assemble() -> str:
        return 'assembled'

    result = await agent.run('go')

    assert result.output.order_id == hook.removesuffix('_tool_execute')
    assert result.usage.requests == 1
    [tool_return] = tool_returns(result.all_messages())
    assert (tool_return.content, tool_return.outcome) == ('assembled', 'success')


class SubmitBeforeModelRequest(AbstractCapability[None]):
    """Submits an output once the step it's set for starts, before its request is sent."""

    def __init__(self, step: int, *, in_prepare: bool = False):
        super().__init__()
        self.step = step
        self.in_prepare = in_prepare

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if not self.in_prepare and ctx.run_step == self.step:
            await ctx.submit_output('final_result', {'order_id': 'from_hook', 'total': 1})
        return request_context

    async def prepare_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if self.in_prepare and ctx.run_step == self.step:
            await ctx.submit_output('final_result', {'order_id': 'from_hook', 'total': 1})
        return request_context


@pytest.mark.parametrize('in_prepare', [False, True], ids=['before_model_request', 'prepare_model_request'])
@pytest.mark.parametrize('mode', ['run', 'event_stream_handler', 'run_stream'])
async def test_submission_before_a_model_request_ends_the_run_without_sending_it(in_prepare: bool, mode: str):
    """The request goes unsent, but stays in history, so the tool return it carries still answers its call."""
    requests: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append(len(messages))
        return ModelResponse(parts=[ToolCallPart('look_up', {}, tool_call_id='call-1')])

    async def stream_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        requests.append(len(messages))
        yield {0: DeltaToolCall(name='look_up', json_args='{}', tool_call_id='call-1')}

    agent = Agent(
        FunctionModel(model, stream_function=stream_model),
        output_type=Invoice,
        capabilities=[SubmitBeforeModelRequest(2, in_prepare=in_prepare)],
    )

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    async def handle_events(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in events:
            pass

    if mode == 'run_stream':
        async with agent.run_stream('go') as stream:
            output = await stream.get_output()
            messages = stream.all_messages()
            usage = stream.usage
    else:
        result = await agent.run('go', event_stream_handler=handle_events if mode == 'event_stream_handler' else None)
        output, messages, usage = result.output, result.all_messages(), result.usage

    assert output == Invoice(order_id='from_hook', total=1)
    assert requests == [1]
    assert usage.requests == 1
    assert [m.kind for m in messages] == ['request', 'response', 'request']
    [tool_return] = tool_returns(messages)
    assert (tool_return.tool_name, tool_return.content) == ('look_up', 'found')


async def test_submission_before_the_first_request_ends_the_run_without_one():
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('no request should be sent')  # pragma: no cover

    agent = Agent(FunctionModel(model), output_type=Invoice, capabilities=[SubmitBeforeModelRequest(1)])

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_hook', total=1)
    assert result.usage.requests == 0
    assert [m.kind for m in result.all_messages()] == ['request']


@pytest.mark.parametrize(
    ('end_strategy', 'executed'), [('early', []), ('exhaustive', ['look_up'])], ids=['early', 'exhaustive']
)
async def test_submission_after_a_model_request_is_the_responses_first_output(
    end_strategy: EndStrategy, executed: list[str]
):
    """The response's tool calls are handled as they'd be after an output tool call at its start."""
    ran: list[str] = []

    class SubmitAfterResponse(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            await ctx.submit_output('final_result', {'order_id': 'from_hook', 'total': 1})
            return response

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('look_up', {}, tool_call_id='call-1'),
                ToolCallPart('final_result', {'order_id': 'from_model', 'total': 2}, tool_call_id='call-2'),
            ]
        )

    agent = Agent(
        FunctionModel(model), output_type=Invoice, end_strategy=end_strategy, capabilities=[SubmitAfterResponse()]
    )

    @agent.tool_plain
    def look_up() -> str:
        ran.append('look_up')
        return 'found'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_hook', total=1)
    assert ran == executed
    assert result.usage.requests == 1
    # Every tool call in the response is answered.
    assert {part.tool_call_id for part in tool_returns(result.all_messages())} == {'call-1', 'call-2'}


async def test_submission_after_a_text_response_replaces_its_output():
    class SubmitAfterResponse(AbstractCapability[None]):
        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            await ctx.submit_output('final_result', {'order_id': 'from_hook', 'total': 1})
            return response

    agent = Agent(
        FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('not json')])),
        output_type=NativeOutput(Invoice),
        capabilities=[SubmitAfterResponse()],
    )

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_hook', total=1)
    assert [m.kind for m in result.all_messages()] == ['request', 'response']


async def test_submission_from_a_node_hook_ends_the_run_before_the_next_request():
    class SubmitAfterTools(AbstractCapability[None]):
        async def after_node_run(
            self, ctx: RunContext[None], *, node: AgentNode[None], result: NodeResult[None]
        ) -> NodeResult[None]:
            if isinstance(node, CallToolsNode):
                await ctx.submit_output('final_result', {'order_id': 'from_hook', 'total': 1})
            return result

    agent = Agent(call_tool_then_reply('look_up'), output_type=Invoice, capabilities=[SubmitAfterTools()])

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    result = await agent.run('go')

    assert result.output == Invoice(order_id='from_hook', total=1)
    assert result.usage.requests == 1
    assert [m.kind for m in result.all_messages()] == ['request', 'response', 'request']


@pytest.mark.parametrize(
    ('output_type', 'args', 'feedback'),
    [
        pytest.param(
            Invoice,
            {'order_id': 'ord_1'},
            snapshot('[{"type":"missing","loc":["total"],"msg":"Field required","input":{"order_id":"ord_1"}}]'),
            id='tool',
        ),
        pytest.param(
            NativeOutput(Invoice),
            {'order_id': 'ord_1'},
            snapshot('[{"type":"missing","loc":["total"],"msg":"Field required","input":{"order_id":"ord_1"}}]'),
            id='native',
        ),
        pytest.param(Invoice, None, "Unknown output tool 'nope'.", id='unknown'),
    ],
)
async def test_rejected_submission_outside_a_tool_call_raises_model_retry(
    output_type: Any, args: dict[str, Any] | None, feedback: str
):
    """There's no tool call to answer, so the caller gets the feedback, and no output retry is used up."""
    raised: list[str] = []
    retries: list[int] = []

    class SubmitFirst(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            try:
                await ctx.submit_output('nope' if args is None else 'final_result', args or {})
            except ModelRetry as e:
                raised.append(e.message)
            retries.append(ctx.retry)
            # A rejected submission can be followed by another.
            await ctx.submit_output('final_result', {'order_id': 'fixed', 'total': 1})
            return request_context

    agent = Agent(
        FunctionModel(lambda m, i: ModelResponse(parts=[])), output_type=output_type, capabilities=[SubmitFirst()]
    )

    result = await agent.run('go')

    assert result.output == Invoice(order_id='fixed', total=1)
    assert raised == [feedback]
    assert retries == [0]


async def test_run_level_submission_misuse():
    class SubmitTwice(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await ctx.submit_output('final_result', {'order_id': 'one', 'total': 1})
            await ctx.submit_output('final_result', {'order_id': 'two', 'total': 2})
            return request_context  # pragma: no cover

    agent = Agent(
        FunctionModel(lambda m, i: ModelResponse(parts=[])), output_type=Invoice, capabilities=[SubmitTwice()]
    )
    with pytest.raises(UserError, match='already submitted an output with `submit_output` outside a tool call'):
        await agent.run('go')

    class SubmitAfterEnd(AbstractCapability[None]):
        async def after_node_run(
            self, ctx: RunContext[None], *, node: AgentNode[None], result: NodeResult[None]
        ) -> NodeResult[None]:
            if isinstance(result, End):
                await ctx.submit_output('final_result', {'order_id': 'late', 'total': 1})
            return result

    agent = Agent(
        FunctionModel(lambda m, i: ModelResponse(parts=[ToolCallPart('final_result', {'order_id': 'a', 'total': 1})])),
        output_type=Invoice,
        capabilities=[SubmitAfterEnd()],
    )
    with pytest.raises(UserError, match='after the run already had its final result'):
        await agent.run('go')


async def test_submission_is_refused_while_output_is_processed():
    async def finish(ctx: RunContext[None], invoice: Invoice) -> Invoice:
        await ctx.submit_output('final_result', {'order_id': 'nested', 'total': 1})
        return invoice  # pragma: no cover

    agent = Agent(
        FunctionModel(lambda m, i: ModelResponse(parts=[ToolCallPart('final_result', {'order_id': 'a', 'total': 1})])),
        output_type=ToolOutput(finish),
    )
    with pytest.raises(UserError, match='while output is being validated or processed'):
        await agent.run('go')

    agent = Agent(
        FunctionModel(lambda m, i: ModelResponse(parts=[TextPart('{"order_id": "a", "total": 1}')])),
        output_type=NativeOutput(Invoice),
    )

    @agent.output_validator
    async def validate(ctx: RunContext[None], output: Invoice) -> Invoice:
        await ctx.submit_output('final_result', {'order_id': 'nested', 'total': 1})
        return output  # pragma: no cover

    with pytest.raises(UserError, match='while output is being validated or processed'):
        await agent.run('go')


async def test_run_level_submission_to_a_hidden_output_tool_is_rejected():
    raised: list[str] = []

    class SubmitReceipt(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            try:
                await ctx.submit_output('submit_receipt', {'payment_id': 'pay_1'})
            except ModelRetry as e:
                raised.append(e.message)
            return request_context

    agent = Agent(
        FunctionModel(
            lambda m, i: ModelResponse(parts=[ToolCallPart('submit_invoice', {'order_id': 'ord_1', 'total': 1})])
        ),
        output_type=[ToolOutput(Invoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt')],
        capabilities=[PrepareOutputTools(hide_receipts), SubmitReceipt()],
    )

    result = await agent.run('go')

    assert result.output == Invoice(order_id='ord_1', total=1)
    assert raised == ["Output tool 'submit_receipt' is not available right now."]
