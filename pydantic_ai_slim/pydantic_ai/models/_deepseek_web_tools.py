"""Shared names and billing fields for DeepSeek's hosted web tools."""

from typing import Literal

from pydantic import BaseModel

from ..native_tools import WebFetchTool, WebSearchTool
from ..usage import RequestUsage

DeepSeekWebToolName = Literal['deepseek_web_search', 'deepseek_read_url']
DeepSeekWebToolKind = Literal['search', 'deep_search', 'image_search', 'deep_image_search', 'read_link']


def web_tool_name(tool: WebSearchTool | WebFetchTool) -> DeepSeekWebToolName:
    return 'deepseek_read_url' if isinstance(tool, WebFetchTool) else 'deepseek_web_search'


def native_web_tool_name(kind: DeepSeekWebToolKind) -> str:
    return WebFetchTool.kind if kind == 'read_link' else WebSearchTool.kind


class DeepSeekWebUsage(BaseModel):
    tool_search: int = 0
    tool_fetch: int = 0
    tool_image_search: int = 0

    def apply(self, usage: RequestUsage) -> None:
        usage.details.update(self.model_dump())
        # RequestUsage stores provider-specific billing units as extra attributes.
        setattr(usage, 'web_searches', self.tool_search)
