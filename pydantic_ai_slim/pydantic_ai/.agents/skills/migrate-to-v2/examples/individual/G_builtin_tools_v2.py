"""v2: register native tools via capabilities=[NativeTool(...)] (or WebSearch / WebFetch / MCP)."""
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch


def trigger():
    # Provider-adaptive form — falls back to local impl on providers without native web search.
    Agent('test', capabilities=[WebSearch()])


if __name__ == '__main__':
    trigger()
