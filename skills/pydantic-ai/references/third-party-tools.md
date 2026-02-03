# Third-Party Tools Reference

Source: `pydantic_ai_slim/pydantic_ai/ext/`

## Overview

PydanticAI integrates with external tool libraries: LangChain and ACI.dev. These tools are not validated by PydanticAI — argument validation is handled by the third-party libraries.

## LangChain Tools

### Single Tool

```python {title="langchain_tool.py" test="skip"}
from langchain_community.tools import DuckDuckGoSearchRun

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import tool_from_langchain

search = DuckDuckGoSearchRun()
search_tool = tool_from_langchain(search)

agent = Agent(
    'google-gla:gemini-2.5-flash',
    tools=[search_tool],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
#> Elden Ring Nightreign is planned to be released on May 30, 2025.
```

Requires: `langchain-community` and tool-specific packages (e.g., `ddgs` for DuckDuckGo).

### Multiple Tools / Toolkit

```python {title="langchain_toolkit.py" test="skip"}
from langchain_community.agent_toolkits import SlackToolkit

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset

toolkit = SlackToolkit()
toolset = LangChainToolset(toolkit.get_tools())

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

## ACI.dev Tools

### Single Tool

```python {title="aci_tool.py" test="skip"}
import os

from pydantic_ai import Agent
from pydantic_ai.ext.aci import tool_from_aci

tavily_search = tool_from_aci(
    'TAVILY__SEARCH',
    linked_account_owner_id=os.getenv('LINKED_ACCOUNT_OWNER_ID'),
)

agent = Agent(
    'google-gla:gemini-2.5-flash',
    tools=[tavily_search],
)

result = agent.run_sync('What is the release date of Elden Ring Nightreign?')
print(result.output)
```

Requires: `aci-sdk` package and `ACI_API_KEY` environment variable.

### Multiple Tools

```python {title="aci_toolset.py" test="skip"}
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

agent = Agent('openai:gpt-5', toolsets=[toolset])
```

## MCP Tools

See [mcp.md](mcp.md) for Model Context Protocol server integration.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `tool_from_langchain` | `pydantic_ai.ext.langchain.tool_from_langchain` | Convert single LangChain tool |
| `LangChainToolset` | `pydantic_ai.ext.langchain.LangChainToolset` | Toolset from LangChain tools |
| `tool_from_aci` | `pydantic_ai.ext.aci.tool_from_aci` | Convert single ACI.dev tool |
| `ACIToolset` | `pydantic_ai.ext.aci.ACIToolset` | Toolset from ACI.dev tools |

## Installation

```bash
# LangChain tools
pip install langchain-community

# ACI.dev tools
pip install aci-sdk
```

## Important Notes

- PydanticAI does **not** validate arguments for third-party tools
- Models provide arguments based on tool schemas
- Third-party tools handle their own error cases
- Check tool-specific documentation for required packages

## See Also

- [tools.md](tools.md) — Native PydanticAI tools
- [toolsets.md](toolsets.md) — Toolset patterns
- [mcp.md](mcp.md) — MCP server integration
- [observability.md](observability.md) — Logfire debugging
- [LangChain Tools](https://python.langchain.com/docs/integrations/tools/) — LangChain tool library
- [ACI.dev Tools](https://www.aci.dev/tools) — ACI.dev tool library
