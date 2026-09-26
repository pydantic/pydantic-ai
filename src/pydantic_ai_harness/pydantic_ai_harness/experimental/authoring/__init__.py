"""Deprecated import location for `pydantic_ai_harness.capability_creation`.

This capability graduated out of `experimental`; importing from here still works but
emits a `DeprecationWarning`. Import from `pydantic_ai_harness.capability_creation` instead.
"""

from pydantic_ai_harness.capability_creation import (
    AuthoredCapability,
    CapabilityCreation,
    CapabilityCreationToolset,
    CapabilityStore,
    CapabilityValidationError,
    load_capability_instance,
    validate_capability_file,
)
from pydantic_ai_harness.experimental._warn import warn_moved

RuntimeAuthoring = CapabilityCreation
AuthoringToolset = CapabilityCreationToolset

warn_moved('authoring', 'capability_creation')

__all__ = [
    'AuthoredCapability',
    'AuthoringToolset',
    'CapabilityStore',
    'CapabilityValidationError',
    'RuntimeAuthoring',
    'load_capability_instance',
    'validate_capability_file',
]
