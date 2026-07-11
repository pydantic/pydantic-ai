"""Deprecated import location for `pydantic_ai_harness.docs`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.docs` instead.
"""

from pydantic_ai_harness.docs import (
    PyaiDocs,
    PyaiDocsToolset,
    PyaiDocsTopic,
)
from pydantic_ai_harness.experimental._warn import warn_moved

warn_moved('docs', 'docs')

__all__ = [
    'PyaiDocs',
    'PyaiDocsToolset',
    'PyaiDocsTopic',
]
