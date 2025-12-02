Example of using xAI's server-side web_search tool for stock analysis.

This example demonstrates:

- Using server-side tools with `XaiModel`
- Structured output with Agent
- Streaming responses
- Usage tracking with logfire
- Using built-in and local tools to perform a stock analysis workflow

In this scenario, the agent uses built-in server-side tools like web_search along with local tools to find the hottest performing stock from yesterday on NASDAQ, retrieve its current price, and provide a brief buy analysis.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
uv run python -m pydantic_ai_examples.stock_analysis
```

Note: Requires `XAI_API_KEY` environment variable for xAI API access.

## Example Output

```
🔍 Starting stock analysis...

Query: What was the hottest performing stock on NASDAQ yesterday?

🔧 Server-side tool: web_search

🔧 Server-side tool: browse_page

✅ Analysis complete!

📊 Top Stock: ATGL

💰 Current Price: $20.00

📈 Buy Analysis:
ATGL surged 139.84% on November 18, 2025, making it the hottest performer on NASDAQ, but pulled back 7.45% today to $20.00. The extreme volatility suggests high risk; without clear fundamental drivers, it's not recommended as a buy for most investors. Suitable only for speculative short-term trades.

📊 Usage Statistics:
   Requests: 1
   Input Tokens: 19150
   Output Tokens: 526
   Total Tokens: 19676
   Server-Side Tools: 7
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/stock_analysis.py"}```
