"""Firecrawl capability: web search with highlights or full pages, and page scraping for agents."""

from pydantic_ai_harness.firecrawl._capability import FirecrawlSearch
from pydantic_ai_harness.firecrawl._toolset import (
    FirecrawlClient,
    FirecrawlExtractionMode,
    FirecrawlSearchToolset,
    FirecrawlSource,
    HttpFirecrawlClient,
)

__all__ = [
    'FirecrawlClient',
    'FirecrawlExtractionMode',
    'FirecrawlSearch',
    'FirecrawlSearchToolset',
    'FirecrawlSource',
    'HttpFirecrawlClient',
]
