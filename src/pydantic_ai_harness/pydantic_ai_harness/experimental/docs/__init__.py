"""Docs capability: an on-demand tool that locates Pydantic AI documentation."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.docs._capability import PyaiDocs
from pydantic_ai_harness.experimental.docs._toolset import PyaiDocsToolset, PyaiDocsTopic

warn_experimental('docs')

__all__ = ['PyaiDocs', 'PyaiDocsToolset', 'PyaiDocsTopic']
