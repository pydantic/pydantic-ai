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

## Semantix — Semantic Output Validation {#semantix}

[Semantix](https://github.com/labrat-akhona/semantix-ai) adds semantic validation to Pydantic AI agents — validating that outputs match a **natural language intent**, not just a structural schema. It uses local NLI (Natural Language Inference) models for fast, offline evaluation and integrates with Pydantic AI's [`output_validator`][pydantic_ai.agent.Agent.output_validator] and [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] mechanism.

Install with:

```bash
pip install 'semantix-ai[pydantic-ai]'
```

Define an intent (what the output should *mean*) and attach it as an [output validator](output.md#output-validator-functions):

```python {test="skip"}
from pydantic_ai import Agent
from semantix import Intent
from semantix.integrations.pydantic_ai import semantix_validator

class Polite(Intent):
    """The text must be polite, professional, and free of aggressive language."""

agent = Agent('openai:gpt-4o', output_type=str)

@agent.output_validator
async def validate_polite(ctx, output):
    return await semantix_validator(Polite)(ctx, output)
```

When the output fails semantic validation, the adapter raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] with a detailed explanation, which Pydantic AI feeds back to the model for self-correction. The NLI judge runs locally (~15ms per evaluation), so no additional API calls are needed for validation.

Semantix also supports composite intents (`AllOf`, `AnyOf`), custom score thresholds, and pluggable judge backends (NLI, LLM, embedding-based).

## See Also

- [Function Tools](tools.md) - Basic tool concepts and registration
- [Toolsets](toolsets.md) - Managing collections of tools
- [MCP Client](mcp/client.md) - Using MCP servers with Pydantic AI
- [LangChain Toolsets](toolsets.md#langchain-tools) - Using LangChain toolsets
- [ACI.dev Toolsets](toolsets.md#aci-tools) - Using ACI.dev toolsets
