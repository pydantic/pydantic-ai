# Toolsets

PydanticAI ships with native tools and toolsets (collections of tools) that can be used to enhance your agent's capabilities.

## DuckDuckGo Search Tool

The DuckDuckGo search tool allows you to search the web for information. It is built on top of the
[DuckDuckGo API](https://github.com/deedy5/duckduckgo_search).

```py {title="main.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    tools=[duckduckgo_search_tool()],
    system_prompt='Search DuckDuckGo for the given query and return the results.',
)
result = agent.run_sync('Does Pydantic have a company, and investors?')
print(result.data)
# Based on the search results, I can provide comprehensive information about Pydantic's company and investors:

# Yes, Pydantic does have a company:
# 1. Company Details:
# - Official name: Pydantic Services Inc.
# - Founded: The library was created in 2017, but the commercial company was launched in 2022
# - Founder: Samuel Colvin (London-based developer)

# 2. Funding and Investors:
# The company has raised significant funding across multiple rounds:
# - Seed Round (February 2023): $4.7 million
#   - Led by Sequoia Capital
#   - Other investors included ParTech and Irregular Expressions
# - Series A Round (October 2024): $12.5 million
#   - Also led by Sequoia Capital
#   - Total funding to date: $17.2 million from 7 investors

# The company has transformed from being just an open-source library to a commercial entity, with backing
# from prominent venture capital firms. It's notable that major tech companies like Meta, Microsoft, Amazon,
# Apple, and Google use Pydantic's technology. The company is now expanding beyond its original data-validation
# framework and has launched commercial products like Logfire, an observability platform.
```

## Third Party Tools

If you've built a third party tool, and want to have it listed here, feel free to submit a pull request!
