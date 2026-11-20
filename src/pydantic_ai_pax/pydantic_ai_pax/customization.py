"""The former home of Pax's on-demand docs, kept so existing imports work."""

from pydantic_ai.capabilities import Capability
from pydantic_ai_pax.clai_docs import CAPABILITY_ID as CAPABILITY_ID, ClaiDocs


def customization_guide() -> Capability[None]:
    """Return the legacy `clai_customization` capability for existing custom agents."""
    return ClaiDocs()
