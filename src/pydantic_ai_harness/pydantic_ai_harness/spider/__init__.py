"""Spider web crawl, search, and scrape capability."""

from ._capability import Spider
from ._toolset import HttpSpiderClient, SpiderClient, SpiderToolset

__all__ = ['HttpSpiderClient', 'Spider', 'SpiderClient', 'SpiderToolset']
