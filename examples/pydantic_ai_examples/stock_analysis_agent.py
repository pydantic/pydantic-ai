"""Example of using Grok's server-side tools (web_search, code_execution) with a local function.

This agent:
1. Uses web_search to find the best performing NASDAQ stock over the last week
2. Uses code_execution to project the price using linear regression
3. Calls a local function project_price with the results
"""

import os
from datetime import datetime

import logfire
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    BuiltinToolCallPart,
    CodeExecutionTool,
    ModelResponse,
    RunContext,
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


class StockProjection(BaseModel):
    """Projection of stock price at year end."""

    stock_symbol: str = Field(description='Stock ticker symbol')
    current_price: float = Field(description='Current stock price')
    projected_price: float = Field(description='Projected price at end of year')
    analysis: str = Field(description='Brief analysis of the projection')


# This agent uses server-side tools to research and analyze stocks
stock_analysis_agent = Agent[None, StockProjection](
    model=model,
    output_type=StockProjection,
    builtin_tools=[
        WebSearchTool(),  # Server-side web search
        CodeExecutionTool(),  # Server-side code execution
    ],
    system_prompt=(
        'You are a stock analysis assistant. '
        'Use web_search to find recent stock performance data on NASDAQ. '
        'Use code_execution to perform linear regression for price projection. '
        'After analysis, call project_price with your findings.'
    ),
)


@stock_analysis_agent.tool
def project_price(ctx: RunContext[None], stock: str, price: float) -> str:
    """Record the projected stock price.

    This is a local/client-side function that gets called with the analysis results.

    Args:
        ctx: The run context (not used in this function)
        stock: Stock ticker symbol
        price: Projected price at end of year
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logfire.info(
        'Stock projection recorded',
        stock=stock,
        projected_price=price,
        timestamp=timestamp,
    )
    print('\n📊 PROJECTION RECORDED:')
    print(f'   Stock: {stock}')
    print(f'   Projected End-of-Year Price: ${price:.2f}')
    print(f'   Timestamp: {timestamp}\n')

    return f'Projection for {stock} at ${price:.2f} has been recorded successfully.'


async def main():
    """Run the stock analysis agent."""
    query = (
        'Can you find me the best performing stock on the NASDAQ over the last week, '
        'and return the price project for the end of the year using a simple linear regression. '
    )

    print('🔍 Starting stock analysis...\n')
    print(f'Query: {query}\n')

    result = await stock_analysis_agent.run(query)

    # Track which builtin tools were used
    web_search_count = 0
    code_execution_count = 0

    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, BuiltinToolCallPart):
                    if 'web_search' in part.tool_name or 'browse' in part.tool_name:
                        web_search_count += 1
                        logfire.info(
                            'Server-side web_search tool called',
                            tool_name=part.tool_name,
                            tool_call_id=part.tool_call_id,
                        )
                    elif 'code_execution' in part.tool_name:
                        code_execution_count += 1
                        logfire.info(
                            'Server-side code_execution tool called',
                            tool_name=part.tool_name,
                            tool_call_id=part.tool_call_id,
                            code=part.args_as_dict().get('code', 'N/A')
                            if part.args
                            else 'N/A',
                        )

    print('\n✅ Analysis complete!')
    print('\n🔧 Server-Side Tools Used:')
    print(f'   Web Search calls: {web_search_count}')
    print(f'   Code Execution calls: {code_execution_count}')

    print(f'\nStock: {result.output.stock_symbol}')
    print(f'Current Price: ${result.output.current_price:.2f}')
    print(f'Projected Year-End Price: ${result.output.projected_price:.2f}')
    print(f'\nAnalysis: {result.output.analysis}')

    # Get the final response message for metadata
    final_message = result.all_messages()[-1]
    if isinstance(final_message, ModelResponse):
        print('\n🆔 Response Metadata:')
        if final_message.provider_response_id:
            print(f'   Response ID: {final_message.provider_response_id}')
        if final_message.model_name:
            print(f'   Model: {final_message.model_name}')
        if final_message.timestamp:
            print(f'   Timestamp: {final_message.timestamp}')

    # Show usage statistics
    usage = result.usage()
    print('\n📈 Usage Statistics:')
    print(f'   Requests: {usage.requests}')
    print(f'   Input Tokens: {usage.input_tokens}')
    print(f'   Output Tokens: {usage.output_tokens}')
    print(f'   Total Tokens: {usage.total_tokens}')


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
