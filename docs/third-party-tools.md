# Third-Party Tools

Pydantic AI supports integration with various third-party tool libraries, allowing you to leverage existing tool ecosystems in your agents. Third-party tools are also available as [capabilities](capabilities.md#third-party-capabilities) — see [Extensibility](extensibility.md) for the full ecosystem.

## MCP Tools {#mcp-tools}

See the [MCP Client](./mcp/client.md) documentation for how to use MCP servers with Pydantic AI as [toolsets](toolsets.md).

## LangChain Tools {#langchain-tools}

If you'd like to use a tool from LangChain's [community tool library](https://python.langchain.com/docs/integrations/tools/) with Pydantic AI, you can use the [`tool_from_langchain`][pydantic_ai.ext.langchain.tool_from_langchain] convenience method. Note that Pydantic AI will not validate the arguments in this case -- it's up to the model to provide arguments matching the schema specified by the LangChain tool, and up to the LangChain tool to raise an error if the arguments are invalid.

You will need to install the `langchain-community` package and any others required by the tool in question.

Here is how you can use the LangChain `DuckDuckGoSearchRun` tool, which requires the `ddgs` package:

```python {test="skip"}
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import tool_from_langchain

search = DuckDuckGoSearchRun()
search_tool = tool_from_langchain(search)

agent = Agent(
    'google-gla:gemini-3-flash-preview',
    tools=[search_tool],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')  # (1)!
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

1. The release date of this game is the 30th of May 2025, which is after the knowledge cutoff for Gemini 2.0 (August 2024).

If you'd like to use multiple LangChain tools or a LangChain [toolkit](https://python.langchain.com/docs/concepts/tools/#toolkits), you can use the [`LangChainToolset`][pydantic_ai.ext.langchain.LangChainToolset] [toolset](toolsets.md) which takes a list of LangChain tools:

```python {test="skip"}
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset

toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-5.2', toolsets=[toolset])
# ...
```

## ACI.dev Tools {#aci-tools}

If you'd like to use a tool from the [ACI.dev tool library](https://www.aci.dev/tools) with Pydantic AI, you can use the [`tool_from_aci`][pydantic_ai.ext.aci.tool_from_aci] convenience method. Note that Pydantic AI will not validate the arguments in this case -- it's up to the model to provide arguments matching the schema specified by the ACI tool, and up to the ACI tool to raise an error if the arguments are invalid.

You will need to install the `aci-sdk` package, set your ACI API key in the `ACI_API_KEY` environment variable, and pass your ACI "linked account owner ID" to the function.

Here is how you can use the ACI.dev `TAVILY__SEARCH` tool:

```python {test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import tool_from_aci

tavily_search = tool_from_aci(
    'TAVILY__SEARCH',
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent(
    'google-gla:gemini-3-flash-preview',
    tools=[tavily_search],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')  # (1)!
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

1. The release date of this game is the 30th of May 2025, which is after the knowledge cutoff for Gemini 2.0 (August 2024).

If you'd like to use multiple ACI.dev tools, you can use the [`ACIToolset`][pydantic_ai.ext.aci.ACIToolset] [toolset](toolsets.md) which takes a list of ACI tool names as well as the `linked_account_owner_id`:

```python {test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import ACIToolset

toolset = ACIToolset(
    [
        'OPEN_WEATHER_MAP__CURRENT_WEATHER',
        'OPEN_WEATHER_MAP__FORECAST',
    ],
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent('openai:gpt-5.2', toolsets=[toolset])
```

## StackOne Tools {#stackone-tools}

[StackOne](https://www.stackone.com) is Integration Infrastructure for AI Agents, providing tools for 200+ enterprise applications. Use the [`StackOneToolset`][pydantic_ai.ext.stackone.StackOneToolset] [toolset](toolsets.md#stackone-tools) to give your agent access to every tool for your connected account(s). Requires the `stackone-ai` package (Python 3.10+), a StackOne API key (`STACKONE_API_KEY`), and at least one connected account ID. Set `STACKONE_ACCOUNT_ID` in your environment, or pass `account_ids=` (a string or list of strings) directly.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.ext.stackone import StackOneToolset

toolset = StackOneToolset()

agent = Agent('openai:gpt-5.4', toolsets=[toolset])

result = agent.run_sync('List the first 5 workers')
print(result.output)
```

!!! note
    Bare `StackOneToolset()` loads every tool for your connected account(s). Large catalogs may exceed the model's per-request tool-count limit. Use `mode='search_and_execute'` for on-demand discovery (see the Search & Execute tab below), or narrow the toolset with `tools=` or `filter_pattern=` (see the Tools tab below).

For more control, discover tools on demand, narrow the toolset explicitly, or connect multiple accounts:

=== "Search & Execute"

    Instead of exposing every tool to the agent, `mode='search_and_execute'` exposes only two tools, `tool_search` and `tool_execute`. The model decides the search query at runtime, picks a result, and calls `tool_execute` with the tool name and parameters. Prompt size stays constant regardless of catalog size. Requires `stackone-ai >= 2.5.0`.

    `search_config['method']` picks the search strategy:

    - `auto` (default): tries `semantic` first, falls back to `local` if the StackOne search API is unavailable
    - `semantic`: calls the StackOne semantic search API, which uses server-side model embeddings to rank tools by relevance
    - `local`: runs a BM25 + TF-IDF hybrid keyword search on-device, with no call to the StackOne API

    ```python {test="skip"}
    # Default: the SDK picks the search strategy
    toolset = StackOneToolset(mode='search_and_execute')

    # Or force a strategy explicitly
    toolset = StackOneToolset(
        mode='search_and_execute',
        search_config={'method': 'semantic'},  # or 'auto', 'local'
    )

    agent = Agent('openai:gpt-5.4', toolsets=[toolset])
    ```

=== "Tools"

    Narrow the toolset with an explicit list of tool names or a glob pattern. You can also wrap a single tool directly with [`tool_from_stackone`][pydantic_ai.ext.stackone.tool_from_stackone], and a toolset can be combined with one or more standalone tools on the same agent. Examples use Workday tool names; any connected vendor works identically.

    ```python {test="skip"}
    from pydantic_ai.ext.stackone import tool_from_stackone

    # An explicit list of tools
    toolset = StackOneToolset(tools=['workday_list_workers', 'workday_get_worker'])

    # Or a glob pattern
    toolset = StackOneToolset(filter_pattern='workday_list_worker*')

    # Or a single tool without a toolset
    worker_tool = tool_from_stackone('workday_list_workers')

    # A toolset and standalone tools can be combined on the same agent
    agent = Agent('openai:gpt-5.4', toolsets=[toolset], tools=[worker_tool])
    ```

    For large catalogs, prefer `tools=` or `filter_pattern=` over loading every tool. Unfiltered loading can exceed model tool-count limits.

=== "Multiple accounts"

    Pass `account_ids` as a list to fetch tools from several connected accounts. The SDK fans out in parallel and merges the results.

    ```python {test="skip"}
    toolset = StackOneToolset(
        filter_pattern='workday_list_worker*',
        account_ids=['workday-acct-1', 'bamboohr-acct-2'],
    )
    agent = Agent('openai:gpt-5.4', toolsets=[toolset])
    ```

## See Also

- [Function Tools](tools.md) - Basic tool concepts and registration
- [Toolsets](toolsets.md) - Managing collections of tools
- [MCP Client](mcp/client.md) - Using MCP servers with Pydantic AI
- [LangChain Toolsets](toolsets.md#langchain-tools) - Using LangChain toolsets
- [ACI.dev Toolsets](toolsets.md#aci-tools) - Using ACI.dev toolsets
- [StackOne Toolsets](toolsets.md#stackone-tools) - Using StackOne toolsets
