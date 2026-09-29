"""Deprecated import location for `pydantic_ai_harness.pydantic_ai_docs`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.pydantic_ai_docs` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.pydantic_ai_docs import (
    PydanticAIDocsToolset,
    PydanticAIDocsTopic,
)
from pydantic_ai_harness.pydantic_ai_docs._deprecated import PyaiDocs

PyaiDocsToolset = PydanticAIDocsToolset
PyaiDocsTopic = PydanticAIDocsTopic

warn_moved('docs', 'pydantic_ai_docs')

__all__ = [
    'PyaiDocs',
    'PyaiDocsToolset',
    'PyaiDocsTopic',
]
