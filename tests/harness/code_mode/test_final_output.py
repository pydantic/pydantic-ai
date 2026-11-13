"""Tests for `CodeMode(allow_final_output=True)`: submitting the run's output from sandboxed code.

These drive the agent loop with `FunctionModel`, since what's under test is how a `run_code` call
that submits an output is executed and recorded, not anything a provider decides.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, NativeOutput, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, RawOutput
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import CodeMode, GuardrailResult, InputGuardrail, OutputBlocked, OutputGuardrail
from pydantic_ai_harness.guardrails._capability import input_guardrails_allow


class Invoice(BaseModel):
    order_id: str
    total: int


class Receipt(BaseModel):
    payment_id: str


ORDERS = {'ord_8Kq2ZpVxL0mN3rT6yB1c': {'lines': [1250, 4999]}}


# `Any` on purpose: with a precise return type, Monty's type checker would reject `total=order['id']` before the code
# runs, and `test_rejected_output_keeps_the_repl_for_a_fix` needs a value that only output validation rejects, at run time.
def load_order(order_id: str) -> dict[str, Any]:
    """Load an order by its ID."""
    return {'id': order_id, **ORDERS[order_id]}


def run_code_model(*snippets: str) -> FunctionModel:
    """A model that calls `run_code` with each snippet in turn, then replies with text."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(m, ModelResponse) for m in messages)
        if step < len(snippets):
            return ModelResponse(
                parts=[ToolCallPart('run_code', {'code': snippets[step]}, tool_call_id=f'call-{step}')]
            )
        return ModelResponse(parts=[TextPart('a turn after the code')])  # pragma: no cover

    return FunctionModel(model)


def run_code_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [
        part
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]


def catalog(info: AgentInfo) -> str:
    """The sandbox's function catalog: `run_code`'s description, or with `dynamic_catalog` the instructions."""
    [run_code] = info.function_tools
    return f'{run_code.description}\n{info.instructions}'


SUBMIT_INVOICE = """
order = await load_order(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c')
print('loaded')
submit_invoice(order_id=order['id'], total=sum(order['lines']))
print('not reached')
"""


NATIVE_SUBMIT = """
order = await load_order(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c')
final_result_Receipt(payment_id='pay_' + order['id'])
"""


REJECTED = """
order = await load_order(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c')
print('submitting')
submit_invoice(order_id=order['id'], total=order['id'])
"""


FIXED = """
submit_invoice(order_id=order['id'], total=sum(order['lines']))
"""


CAUGHT = """
try:
    submit_invoice(order_id='ord_1', total=1)
except Exception as e:
    print(f'caught: {e}')
try:
    submit_invoice(order_id='ord_2', total=2)
except Exception as e:
    print(f'caught: {e}')
await load_order(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c')
"""


EAGER = """
order = await load_order(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c')
submit_invoice(order_id=order['id'], total=sum(order['lines']))
print('not reached')
"""


def eager_stream(code: str) -> Callable[[list[ModelMessage], AgentInfo], AsyncIterator[DeltaToolCalls]]:
    """A streaming model that sends a `run_code` call whose code arrives a few characters at a time."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        args = json.dumps({'code': code})
        yield {0: DeltaToolCall('run_code', tool_call_id='call-0')}
        for i in range(0, len(args), 20):
            yield {0: DeltaToolCall(json_args=args[i : i + 20])}

    return stream


LIMITED = """
try:
    await load_order(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c')
except Exception:
    pass
submit_invoice(order_id='ord_1', total=1)
"""


class TestCodeModeFinalOutput:
    async def test_output_function_submits_and_stops_the_code(self):
        agent = Agent(
            run_code_model(SUBMIT_INVOICE),
            tools=[load_order],
            output_type=[ToolOutput(Invoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt')],
            capabilities=[CodeMode(allow_final_output=True)],
        )

        result = await agent.run('Invoice the order.')

        assert result.output == Invoice(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c', total=6249)
        assert result.usage.requests == 1
        [run_code] = run_code_returns(result.all_messages())
        assert run_code.outcome == 'success'
        assert run_code.content == {'output': 'loaded\n'}
        assert [
            (c.tool_name, r.content)
            for c, r in zip(run_code.metadata['tool_calls'].values(), run_code.metadata['tool_returns'].values())
        ] == snapshot(
            [
                ('load_order', {'id': 'ord_8Kq2ZpVxL0mN3rT6yB1c', 'lines': [1250, 4999]}),
                ('submit_invoice', 'Final output submitted.'),
            ]
        )

    async def test_output_functions_are_in_the_catalog(self):
        catalogs: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            catalogs.append(catalog(info))
            if len(messages) == 1 and 'final_result_Receipt' in catalogs[-1]:
                return ModelResponse(parts=[ToolCallPart('run_code', {'code': NATIVE_SUBMIT})])
            return ModelResponse(parts=[TextPart('done')])

        with_outputs = Agent(
            FunctionModel(model),
            tools=[load_order],
            output_type=NativeOutput([Invoice, Receipt]),
            capabilities=[CodeMode(allow_final_output=True)],
        )
        without_outputs = Agent(FunctionModel(model), tools=[load_order], capabilities=[CodeMode()])

        await with_outputs.run('Pay the order.')
        await without_outputs.run('Pay the order.')

        assert 'def final_result_Invoice(*, order_id: str, total: int) -> None:' in catalogs[0]
        assert 'Calling this submits the final output of the agent run and stops the code.' in catalogs[0]
        assert 'final_result' not in catalogs[-1]

    async def test_output_functions_in_native_output_mode(self):
        agent = Agent(
            run_code_model(NATIVE_SUBMIT),
            tools=[load_order],
            output_type=NativeOutput([Invoice, Receipt]),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        result = await agent.run('Pay the order.')

        assert result.output == Receipt(payment_id='pay_ord_8Kq2ZpVxL0mN3rT6yB1c')

    async def test_output_function_in_text_output_mode(self):
        """Plain text output gets one output function, which takes the text as `response`."""
        code = 'order = await load_order(order_id="ord_8Kq2ZpVxL0mN3rT6yB1c")\nfinal_result(response=f"Order {order[\'id\']} is ready.")'
        agent = Agent(run_code_model(code), tools=[load_order], capabilities=[CodeMode(allow_final_output=True)])

        result = await agent.run('Summarize the order.')

        assert result.output == 'Order ord_8Kq2ZpVxL0mN3rT6yB1c is ready.'

    async def test_rejected_output_keeps_the_repl_for_a_fix(self):
        """A value only known at run time is rejected by validation, and the REPL keeps `order` for the retry."""
        agent = Agent(
            run_code_model(REJECTED, FIXED),
            tools=[load_order],
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        result = await agent.run('Invoice the order.')

        assert result.output == Invoice(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c', total=6249)
        rejected, accepted = run_code_returns(result.all_messages())
        assert rejected.outcome == 'retried'
        assert [call.tool_name for call in rejected.metadata['tool_calls'].values()] == ['load_order', 'submit_invoice']
        assert rejected.content == snapshot("""\
[stdout before error]
submitting
[/stdout before error]
The output was rejected:
[
  {
    "type": "int_parsing",
    "loc": [
      "total"
    ],
    "msg": "Input should be a valid integer, unable to parse string as an integer"
  }
]\
""")
        assert accepted.outcome == 'success'

    async def test_type_checker_rejects_a_mistyped_output_call(self):
        """On a fresh REPL, Monty checks the code against the output function's signature before running it."""
        agent = Agent(
            run_code_model(
                "submit_invoice(order_id='ord_1', total='lots')", "submit_invoice(order_id='ord_1', total=1)"
            ),
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        result = await agent.run('Invoice the order.')

        assert result.output == Invoice(order_id='ord_1', total=1)
        type_error, _ = run_code_returns(result.all_messages())
        assert isinstance(type_error.content, str)
        assert type_error.content.startswith('Type error in code:')

    async def test_code_that_catches_the_stop_cannot_submit_again(self):
        agent = Agent(
            run_code_model(CAUGHT),
            tools=[load_order],
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        result = await agent.run('Invoice the order.')

        assert result.output == Invoice(order_id='ord_1', total=1)
        [run_code] = run_code_returns(result.all_messages())
        assert run_code.content == snapshot(
            {
                'output': """\
caught: The final output was submitted, so the code stopped here.
caught: This code already submitted the final output, or had a submission blocked, so it cannot call more functions.
"""
            }
        )
        assert [call.tool_name for call in run_code.metadata['tool_calls'].values()] == ['submit_invoice']

    async def test_output_function_name_conflict(self):
        def final_result(x: int) -> int:
            return x  # pragma: no cover

        agent = Agent(
            run_code_model('1'),
            tools=[final_result],
            output_type=NativeOutput(Invoice),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        with pytest.raises(UserError, match="Output tool 'final_result' conflicts with a function already available"):
            await agent.run('go')

    async def test_eager_submission(self):
        agent = Agent(
            FunctionModel(stream_function=eager_stream(EAGER)),
            tools=[load_order],
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True, eager=True)],
        )

        result = await agent.run('Invoice the order.')

        assert result.output == Invoice(order_id='ord_8Kq2ZpVxL0mN3rT6yB1c', total=6249)
        [run_code] = run_code_returns(result.all_messages())
        assert run_code.content == {}

    async def test_rejection_from_an_output_validator(self):
        agent = Agent(
            run_code_model("submit_invoice(order_id='ord_1', total=0)", "submit_invoice(order_id='ord_1', total=5)"),
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        @agent.output_validator
        def positive_total(invoice: Invoice) -> Invoice:
            if invoice.total <= 0:
                raise ModelRetry('The total must be positive.')
            return invoice

        result = await agent.run('Invoice the order.')

        assert result.output == Invoice(order_id='ord_1', total=5)
        rejected, _ = run_code_returns(result.all_messages())
        assert (rejected.outcome, rejected.content) == ('retried', 'The total must be positive.')

    async def test_usage_limit_still_ends_the_run_after_a_submission(self):
        agent = Agent(
            run_code_model(LIMITED),
            tools=[load_order],
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        with pytest.raises(UsageLimitExceeded):
            await agent.run('Invoice the order.', usage_limits=UsageLimits(tool_calls_limit=1))

    async def test_usage_limit_from_an_output_function_ends_the_run_even_if_caught(self):
        def submit(order_id: str, total: int) -> Invoice:
            raise UsageLimitExceeded('The output function went over its limit.')

        code = 'try:\n    submit_invoice(order_id="ord_1", total=1)\nexcept Exception:\n    pass\n"carried on"'
        agent = Agent(
            run_code_model(code),
            output_type=ToolOutput(submit, name='submit_invoice'),
            capabilities=[CodeMode(allow_final_output=True)],
        )

        with pytest.raises(UsageLimitExceeded, match='went over its limit'):
            await agent.run('Invoice the order.')

    async def test_output_guardrail_block_ends_the_run_even_if_caught(self):
        code = (
            'try:\n    submit_invoice(order_id="ord_1", total=1)\nexcept Exception:\n    pass\n'
            'await load_order(order_id="ord_8Kq2ZpVxL0mN3rT6yB1c")'
        )
        agent = Agent(
            run_code_model(code),
            tools=[load_order],
            output_type=ToolOutput(Invoice, name='submit_invoice'),
            capabilities=[
                CodeMode(allow_final_output=True),
                OutputGuardrail(guard=lambda output: GuardrailResult.block('Invoices need a review.')),
            ],
        )

        with pytest.raises(OutputBlocked, match='Invoices need a review'):
            await agent.run('Invoice the order.')

    async def test_eager_submission_waits_for_a_parallel_input_guardrail(self, monkeypatch: pytest.MonkeyPatch):
        """An output function can have side effects, so an eager submission waits for the guard's verdict too."""
        submitted: list[Invoice] = []
        submission_reached = asyncio.Event()
        """Set once the submission either waits for the verdict or runs, so the guard decides then."""

        async def allow(ctx: RunContext[object]) -> bool:
            submission_reached.set()
            return await input_guardrails_allow(ctx)

        monkeypatch.setattr('pydantic_ai_harness.code_mode._toolset.input_guardrails_allow', allow)

        def record_invoice(order_id: str, total: int) -> Invoice:
            submitted.append(invoice := Invoice(order_id=order_id, total=total))  # pragma: no cover
            submission_reached.set()  # pragma: no cover
            return invoice  # pragma: no cover

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            args = json.dumps({'code': "submit_invoice(order_id='ord_1', total=1)\nprint('done')\n'end'"})
            # The submission's statement is closed once the next one is, so eager execution runs it now.
            yield {0: DeltaToolCall('run_code', json_args=args[: args.index("'end'")], tool_call_id='call-0')}
            await asyncio.Event().wait()

        async def guard(prompt: str) -> GuardrailResult:
            await submission_reached.wait()
            return GuardrailResult.block('blocked')

        agent = Agent(
            FunctionModel(stream_function=stream),
            output_type=[ToolOutput(record_invoice, name='submit_invoice'), str],
            capabilities=[CodeMode(allow_final_output=True, eager=True), InputGuardrail(guard=guard, parallel=True)],
        )

        result = await agent.run('Invoice the order.')

        assert result.output == 'blocked'
        assert submitted == []


class AmountInvoice(BaseModel):
    amount: int


class TestCodeModePreparedOutput:
    """Prepared output tools have the same contract inside and outside Code Mode.

    `FunctionModel` makes the same output call through both routes, so these tests isolate catalog preparation and
    sandbox validation without depending on a provider's choices.
    """

    @pytest.mark.parametrize('code_mode', [False, True], ids=['direct', 'code-mode'])
    async def test_hidden_output_is_absent_from_catalog(self, code_mode: bool) -> None:
        catalogs: list[str] = []

        class HideInvoice(AbstractCapability[None]):
            async def prepare_output_tools(
                self, ctx: RunContext[None], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [tool for tool in tool_defs if tool.name != 'submit_invoice']

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert [tool.name for tool in info.output_tools] == ['submit_receipt']
            if code_mode:
                [run_code] = info.function_tools
                # With `dynamic_catalog`, the catalog is in the instructions instead of the description.
                catalogs.append(f'{run_code.description}\n{info.instructions}')
                call = ToolCallPart('run_code', {'code': "submit_receipt(payment_id='pay_1')"})
            else:
                call = ToolCallPart('submit_receipt', {'payment_id': 'pay_1'})
            return ModelResponse(parts=[call])

        agent = Agent(
            FunctionModel(model),
            deps_type=type(None),
            output_type=[ToolOutput(AmountInvoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt')],
            capabilities=[HideInvoice(), CodeMode(allow_final_output=True)] if code_mode else [HideInvoice()],
        )

        result = await agent.run('Return the receipt.')

        assert result.output == Receipt(payment_id='pay_1')
        assert result.usage.requests == 1
        if code_mode:
            [catalog] = catalogs
            assert 'def submit_receipt(' in catalog
            assert 'def submit_invoice(' not in catalog

    @pytest.mark.parametrize('code_mode', [False, True], ids=['direct', 'code-mode'])
    async def test_added_argument_reaches_output_hooks(self, code_mode: bool) -> None:
        sources: list[str] = []
        executed: list[int] = []

        class RequireSource(AbstractCapability[str]):
            async def prepare_output_tools(
                self, ctx: RunContext[str], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                [tool] = tool_defs
                schema = deepcopy(tool.parameters_json_schema)
                schema['properties']['source_id'] = {'type': 'string'}
                schema['required'] = [*schema.get('required', []), 'source_id']
                return [replace(tool, parameters_json_schema=schema)]

            async def before_output_validate(
                self, ctx: RunContext[str], *, output_context: OutputContext, output: RawOutput
            ) -> RawOutput:
                assert output_context.tool_call is not None
                assert output_context.tool_call.tool_name == 'submit_invoice'
                assert isinstance(output, dict)
                args = dict(output)
                source = args.pop('source_id')
                assert isinstance(source, str)
                sources.append(source)
                return args

        async def finish(ctx: RunContext[str], amount: int) -> str:
            executed.append(amount)
            return f'{ctx.deps}:{amount}'

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) > 1:  # pragma: no cover
                feedback = [
                    part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)
                ]
                pytest.fail(f'The prepared output call was rejected: {feedback}')
            [output_tool] = info.output_tools
            assert 'source_id' in output_tool.parameters_json_schema['required']
            if code_mode:
                call = ToolCallPart('run_code', {'code': "submit_invoice(amount=42, source_id='ledger')"})
            else:
                call = ToolCallPart('submit_invoice', {'amount': 42, 'source_id': 'ledger'})
            return ModelResponse(parts=[call])

        agent = Agent(
            FunctionModel(model),
            deps_type=str,
            output_type=ToolOutput(finish, name='submit_invoice'),
            capabilities=[RequireSource(), CodeMode(allow_final_output=True)] if code_mode else [RequireSource()],
        )

        result = await agent.run('Submit the sourced invoice.', deps='accepted')

        assert result.output == 'accepted:42'
        assert result.usage.requests == 1
        assert sources == ['ledger']
        assert executed == [42]
