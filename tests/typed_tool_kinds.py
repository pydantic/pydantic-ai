"""Pins how type checkers see a `ToolKind`: its typed parts, and the tool functions it accepts.

`reportUnnecessaryTypeIgnoreComment` is on, so an ignore below that stops being needed fails the type check.
Nothing here is executed: the refused definitions would also raise at run time.
"""

from typing import Any, NotRequired, assert_type

from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, ToolKind
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.toolsets import FunctionToolset


class StockArgs(TypedDict):
    sku: str
    quantity: NotRequired[int]


class StockResult(TypedDict):
    in_stock: bool


def typed_parts() -> None:
    stock = ToolKind('typed', 'stock', args=StockArgs, content=StockResult)
    assert_type(stock, ToolKind[StockArgs, StockResult])
    assert_type(ToolKind('typed', 'audit'), ToolKind[Any, Any])

    class StockCallPart(ToolCallPart[StockArgs], kind=stock):
        pass

    class StockReturnPart(ToolReturnPart[StockResult], kind=stock):
        pass

    # A bare base takes any kind, but its typed view stays untyped.
    class UntypedStockCallPart(ToolCallPart, kind=stock):  # pyright: ignore[reportUnusedClass]
        pass

    # The parameterized base must agree with the kind.
    class MismatchedCallPart(ToolCallPart[StockResult], kind=stock):  # pyright: ignore[reportUnusedClass,reportGeneralTypeIssues,reportArgumentType]
        pass

    call = StockCallPart('stock', {'sku': 'A-1'})
    assert_type(call.typed_args, StockArgs | None)
    assert_type(StockReturnPart('stock', {'in_stock': True}).typed_content, StockResult | None)
    assert_type(ToolCallPart('t').typed_args, Any | None)

    # A typed part is still a `ToolCallPart` wherever one is expected.
    parts: list[ToolCallPart] = [call]
    assert_type(parts, list[ToolCallPart])


def kinded_tools(agent: Agent[int, str], toolset: FunctionToolset[int]) -> None:
    stock = ToolKind('typed', 'stock', args=StockArgs, content=StockResult)

    @agent.tool_plain(tool_kind=stock)
    def check(args: StockArgs) -> StockResult:
        return {'in_stock': args['sku'] == 'A-1'}

    @agent.tool_plain(tool_kind=stock)
    async def check_async(args: StockArgs) -> StockResult:
        return {'in_stock': args['sku'] == 'A-1'}

    @agent.tool(tool_kind=stock)
    def check_with_ctx(ctx: RunContext[int], args: StockArgs) -> StockResult:
        return {'in_stock': ctx.deps > 0}

    # The arguments' fields as separate parameters type-check too; they're checked when tools are collected.
    @agent.tool_plain(tool_kind=stock)
    def check_fields(sku: str, quantity: int = 1) -> StockResult:
        return {'in_stock': quantity > 0}

    @toolset.tool(tool_kind=stock)
    async def check_in_toolset(ctx: RunContext[int], args: StockArgs) -> StockResult:
        return {'in_stock': True}

    @toolset.tool_plain(tool_kind='typed.stock')
    def check_by_id(sku: str) -> str:
        return sku

    # The result must be the kind's content shape.
    @agent.tool_plain(tool_kind=stock)  # pyright: ignore[reportCallIssue,reportArgumentType,reportUntypedFunctionDecorator]
    def wrong_result(args: StockArgs) -> str:
        return args['sku']

    @agent.tool(tool_kind=stock)  # pyright: ignore[reportCallIssue,reportArgumentType,reportUntypedFunctionDecorator]
    def wrong_result_with_ctx(ctx: RunContext[int], args: StockArgs) -> int:
        return ctx.deps
