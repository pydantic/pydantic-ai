"""Synthetic production-like snippet for denylist tests."""

from pydantic_ai.ag_ui import something  # noqa: F401
from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio, MCPServerStreamableHTTP
from pydantic_ai.models import cached_async_http_client
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.output import DeferredToolCalls
from pydantic_ai.usage import RunUsage, Usage

_REMOVED_IN_V2 = {'MCPServerHTTP': 'use `MCPToolset`.'}

def use_removed():
    load_mcp_servers()  # forbidden
    x = BuiltinToolCallPart()
    return x
