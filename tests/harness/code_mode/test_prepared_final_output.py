"""Prepared output tools should have the same contract inside and outside Code Mode.

FunctionModel makes the same output call through both routes, so these tests isolate
catalog preparation and sandbox validation without depending on a provider's choices.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ToolOutput
from pydantic_ai.capabilities import AbstractCapability, RawOutput
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputContext
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness import CodeMode


class Invoice(BaseModel):
    amount: int


class Receipt(BaseModel):
    payment_id: str


class TestCodeModePreparedOutput:
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
            output_type=[ToolOutput(Invoice, name='submit_invoice'), ToolOutput(Receipt, name='submit_receipt')],
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
