"""Docs capability: an on-demand tool that locates Pydantic AI documentation."""

from pydantic_ai_harness.docs._capability import PyaiDocs
from pydantic_ai_harness.docs._toolset import PyaiDocsToolset, PyaiDocsTopic

__all__ = ['PyaiDocs', 'PyaiDocsToolset', 'PyaiDocsTopic']
