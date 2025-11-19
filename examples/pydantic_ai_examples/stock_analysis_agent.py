"""Example of using Grok's server-side web_search tool.

This agent:
1. Uses web_search to find the hottest performing stock yesterday
2. Provides buy analysis for the user
"""

import os

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BuiltinToolCallPart,
    WebSearchTool,
)
from pydantic_ai.models.grok import GrokModel

logfire.configure()
logfire.instrument_pydantic_ai()

# Configure for xAI API
xai_api_key = os.getenv('XAI_API_KEY')
if not xai_api_key:
    raise ValueError('XAI_API_KEY environment variable is required')


# Create the model using GrokModel with server-side tools
model = GrokModel('grok-4-fast', api_key=xai_api_key)


class StockAnalysis(BaseModel):
    """Analysis of top performing stock."""

    stock_symbol: str = Field(description='Stock ticker symbol')
    current_price: float = Field(description='Current stock price')
    buy_analysis: str = Field(description='Brief analysis for whether to buy the stock')


# This agent uses server-side web search to research stocks
stock_analysis_agent = Agent[None, StockAnalysis](
    model=model,
    output_type=StockAnalysis,
    builtin_tools=[WebSearchTool()],
    system_prompt=(
        'You are a stock analysis assistant. '
        'Use web_search to find the hottest performing stock from yesterday on NASDAQ. '
        'Provide the current price and a brief buy analysis explaining whether this is a good buy.'
    ),
)


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
                    print(f'🔧 Server-side tool: {part.tool_name}\n')

    # Access output after streaming is complete
    output = await result.get_output()

    print('\n✅ Analysis complete!\n')

    print(f'📊 Top Stock: {output.stock_symbol}')
    print(f'💰 Current Price: ${output.current_price:.2f}')
    print(f'\n📈 Buy Analysis:\n{output.buy_analysis}')

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
