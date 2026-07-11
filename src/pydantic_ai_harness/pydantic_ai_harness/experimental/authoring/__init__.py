"""Deprecated import location for `pydantic_ai_harness.runtime_authoring`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.runtime_authoring` instead.
"""

from pydantic_ai_harness.experimental._warn import warn_moved
from pydantic_ai_harness.runtime_authoring import (
    AuthoredCapability,
    AuthoringToolset,
    CapabilityStore,
    CapabilityValidationError,
    RuntimeAuthoring,
    load_capability_instance,
    validate_capability_file,
)

warn_moved('authoring', 'runtime_authoring')

__all__ = [
    'AuthoredCapability',
    'AuthoringToolset',
    'CapabilityStore',
    'CapabilityValidationError',
    'RuntimeAuthoring',
    'load_capability_instance',
    'validate_capability_file',
]
