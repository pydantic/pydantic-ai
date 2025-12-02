"""Example of using Grok's server-side web_search tool.

This agent:
1. Uses web_search to find the hottest performing stock yesterday
2. Provides buy analysis for the user
"""

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BuiltinToolCallPart,
    RunContext,
    WebSearchTool,
)
from pydantic_ai.models.xai import XaiModel

logfire.configure()
logfire.instrument_pydantic_ai()

# Configure for xAI API - XAI_API_KEY environment variable is required
# The model will automatically use XaiProvider with the API key from the environment

# Create the model using XaiModel with server-side tools
model = XaiModel('grok-4-1-fast-non-reasoning')


class StockAnalysis(BaseModel):
    """Analysis of top performing stock."""

    stock_symbol: str = Field(description='Stock ticker symbol')
    performance_change: float = Field(
        description="Yesterday's performance change percentage"
    )
    current_price: float = Field(description='Current stock price')
    buy_analysis: str = Field(description='Brief analysis for whether to buy the stock')


# This agent uses server-side web search to research stocks
stock_analysis_agent = Agent[None, StockAnalysis](
    model=model,
    output_type=StockAnalysis,
    builtin_tools=[WebSearchTool()],
    system_prompt=(
        'You are a stock analysis assistant. '
        'First, use web_search to find the hottest performing stock from yesterday on NASDAQ, including its symbol and performance change. '
        'Then, use the local get_current_price tool to get its current price. '
        'Finally, provide a brief buy analysis explaining whether this is a good buy.'
    ),
)


@stock_analysis_agent.tool
def get_current_price(ctx: RunContext, symbol: str) -> float:
    """Get the current price for a stock symbol from local database."""
    # Simulate local database lookup
    price_db = {
        'ATGL': 20.00,
        'EXAMPLE': 150.00,
        # Add more as needed
    }
    return price_db.get(symbol, 0.0)  # Return 0 if not found


async def main():
    """Run the stock analysis agent."""
    query = 'What was the hottest performing stock on NASDAQ yesterday?'

    print('🔍 Starting stock analysis...\n')
    print(f'Query: {query}\n')

    async with stock_analysis_agent.run_stream(query) as result:
        # Stream responses as they happen
        async for message, _is_last in result.stream_responses():
            for part in message.parts:
                if isinstance(part, BuiltinToolCallPart):
                    print(f'🔧 Server-side tool: {part.tool_name}')

    # Access output after streaming is complete
    output = await result.get_output()

    print('\n✅ Analysis complete!\n')

    print(f'📊 Top Stock: {output.stock_symbol}')
    print(f'📈 Performance Change: {output.performance_change}%')
    print(f'💰 Current Price: ${output.current_price:.2f}')
    print(f'\n📊 Buy Analysis:\n{output.buy_analysis}')

    # Show usage statistics
    usage = result.usage()
    print('\n📊 Usage Statistics:')
    print(f'   Requests: {usage.requests}')
    print(f'   Input Tokens: {usage.input_tokens}')
    print(f'   Output Tokens: {usage.output_tokens}')
    print(f'   Total Tokens: {usage.total_tokens}')

    # Show server-side tools usage if available
    if usage.details and 'server_side_tools_used' in usage.details:
        print(f'   Server-Side Tools: {usage.details["server_side_tools_used"]}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
