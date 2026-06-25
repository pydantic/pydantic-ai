"""Runtime capability authoring: let an agent write, validate, and register real capabilities."""

from pydantic_ai_harness.experimental._warn import warn_experimental
from pydantic_ai_harness.experimental.authoring._capability import RuntimeAuthoring
from pydantic_ai_harness.experimental.authoring._store import AuthoredCapability, CapabilityStore
from pydantic_ai_harness.experimental.authoring._toolset import AuthoringToolset
from pydantic_ai_harness.experimental.authoring._validate import (
    CapabilityValidationError,
    load_capability_instance,
    validate_capability_file,
)

warn_experimental('authoring')

__all__ = [
    'AuthoredCapability',
    'AuthoringToolset',
    'CapabilityStore',
    'CapabilityValidationError',
    'RuntimeAuthoring',
    'load_capability_instance',
    'validate_capability_file',
]
