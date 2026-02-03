# Common Tools Reference

Source: `pydantic_ai_slim/pydantic_ai/common_tools/`

Common tools are locally-executed search integrations shipped with PydanticAI.

## DuckDuckGo Search

Free web search via DuckDuckGo API.

### Installation

```bash
pip/uv-add "pydantic-ai-slim[duckduckgo]"
```

### Usage

```py {title="duckduckgo_search.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

agent = Agent(
    'openai:o3-mini',
    tools=[duckduckgo_search_tool()],
    instructions='Search DuckDuckGo for the given query and return the results.',
)

result = agent.run_sync(
    'Can you list the top five highest-grossing animated films of 2025?'
)
print(result.output)
"""
I looked into several sources on animated box‐office performance in 2025, and while detailed
rankings can shift as more money is tallied, multiple independent reports have already
highlighted a couple of record‐breaking shows. For example:

• Ne Zha 2 – News outlets (Variety, Wikipedia's "List of animated feature films of 2025", and others)
    have reported that this Chinese title not only became the highest‑grossing animated film of 2025
    but also broke records as the highest‑grossing non‑English animated film ever. One article noted
    its run exceeded US$1.7 billion.
• Inside Out 2 – According to data shared on Statista and in industry news, this Pixar sequel has been
    on pace to set new records (with some sources even noting it as the highest‑grossing animated film
    ever, as of January 2025).

Beyond those two, some entertainment trade sites (for example, a Just Jared article titled
"Top 10 Highest-Earning Animated Films at the Box Office Revealed") have begun listing a broader
top‑10. Although full consolidated figures can sometimes differ by source and are updated daily during
a box‑office run, many of the industry trackers have begun to single out five films as the biggest
earners so far in 2025.

Unfortunately, although multiple articles discuss the "top animated films" of 2025, there isn't yet a
single, universally accepted list with final numbers that names the complete top five. (Box‑office
rankings, especially mid‑year, can be fluid as films continue to add to their totals.)

Based on what several sources note so far, the two undisputed leaders are:
1. Ne Zha 2
2. Inside Out 2

The remaining top spots (3–5) are reported by some outlets in their "Top‑10 Animated Films"
lists for 2025 but the titles and order can vary depending on the source and the exact cut‑off
date of the data. For the most up‑to‑date and detailed ranking (including the 3rd, 4th, and 5th
highest‑grossing films), I recommend checking resources like:
• Wikipedia's "List of animated feature films of 2025" page
• Box‑office tracking sites (such as Box Office Mojo or The Numbers)
• Trade articles like the one on Just Jared

To summarize with what is clear from the current reporting:
1. Ne Zha 2
2. Inside Out 2
3–5. Other animated films (yet to be definitively finalized across all reporting outlets)

If you're looking for a final, consensus list of the top five, it may be best to wait until
the 2025 year‑end box‑office tallies are in or to consult a regularly updated entertainment industry source.

Would you like help finding a current source or additional details on where to look for the complete updated list?
"""
```

## Tavily Search

Paid web search with free credits for exploration.

### Installation

```bash
pip/uv-add "pydantic-ai-slim[tavily]"
```

### Configuration

Get API key from [app.tavily.com/home](https://app.tavily.com/home)

### Usage

```py {title="tavily_search.py" test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.common_tools.tavily import tavily_search_tool

api_key = os.getenv('TAVILY_API_KEY')
assert api_key is not None

agent = Agent(
    'openai:o3-mini',
    tools=[tavily_search_tool(api_key)],
    instructions='Search Tavily for the given query and return the results.',
)

result = agent.run_sync('Tell me the top news in the GenAI world, give me links.')
print(result.output)
"""
Here are some of the top recent news articles related to GenAI:

1. How CLEAR users can improve risk analysis with GenAI – Thomson Reuters
   Read more: https://legal.thomsonreuters.com/blog/how-clear-users-can-improve-risk-analysis-with-genai/
   (This article discusses how CLEAR's new GenAI-powered tool streamlines risk analysis by quickly summarizing key information from various public data sources.)

2. TELUS Digital Survey Reveals Enterprise Employees Are Entering Sensitive Data Into AI Assistants More Than You Think – FT.com
   Read more: https://markets.ft.com/data/announce/detail?dockey=600-202502260645BIZWIRE_USPRX____20250226_BW490609-1
   (This news piece highlights findings from a TELUS Digital survey showing that many enterprise employees use public GenAI tools and sometimes even enter sensitive data.)

3. The Essential Guide to Generative AI – Virtualization Review
   Read more: https://virtualizationreview.com/Whitepapers/2025/02/SNOWFLAKE-The-Essential-Guide-to-Generative-AI.aspx
   (This guide provides insights into how GenAI is revolutionizing enterprise strategies and productivity, with input from industry leaders.)

Feel free to click on the links to dive deeper into each story!
"""
```

## Exa Search

Neural search engine with AI-powered answers. Paid with free credits.

### Installation

```bash
pip/uv-add "pydantic-ai-slim[exa]"
```

### Configuration

Get API key from [dashboard.exa.ai](https://dashboard.exa.ai)

### Individual Tools

```py {title="exa_search.py" test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import exa_search_tool

api_key = os.getenv('EXA_API_KEY')
assert api_key is not None

agent = Agent(
    'openai:gpt-4o',
    tools=[exa_search_tool(api_key, num_results=5, max_characters=1000)],
    system_prompt='Search the web for information using Exa.',
)

result = agent.run_sync('What are the latest developments in quantum computing?')
print(result.output)
```

Available tools:
- `exa_search_tool()` — Web search (auto/keyword/neural/fast/deep)
- `exa_find_similar_tool()` — Find similar pages to a URL
- `exa_get_contents_tool()` — Get full text from URLs
- `exa_answer_tool()` — AI-powered answers with citations

### ExaToolset

For multiple Exa tools with shared client:

```py {title="exa_toolset.py" test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.common_tools.exa import ExaToolset

api_key = os.getenv('EXA_API_KEY')
assert api_key is not None

toolset = ExaToolset(
    api_key,
    num_results=5,
    max_characters=1000,  # Limit text content to control token usage
    include_search=True,  # Include the search tool (default: True)
    include_find_similar=True,  # Include the find_similar tool (default: True)
    include_get_contents=False,  # Exclude the get_contents tool
    include_answer=True,  # Include the answer tool (default: True)
)

agent = Agent(
    'openai:gpt-4o',
    toolsets=[toolset],
    system_prompt='You have access to Exa search tools to find information on the web.',
)

result = agent.run_sync('Find recent AI research papers and summarize the key findings.')
print(result.output)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `duckduckgo_search_tool` | `pydantic_ai.common_tools.duckduckgo` | DuckDuckGo search |
| `tavily_search_tool` | `pydantic_ai.common_tools.tavily` | Tavily search |
| `exa_search_tool` | `pydantic_ai.common_tools.exa` | Exa web search |
| `exa_find_similar_tool` | `pydantic_ai.common_tools.exa` | Find similar pages |
| `exa_get_contents_tool` | `pydantic_ai.common_tools.exa` | Get URL contents |
| `exa_answer_tool` | `pydantic_ai.common_tools.exa` | AI-powered answers |
| `ExaToolset` | `pydantic_ai.common_tools.exa` | Combined Exa toolset |
