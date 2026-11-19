"""The former home of Pax's on-demand docs, kept so existing imports work."""

from pydantic_ai.capabilities import Capability
from pydantic_ai_pax.clai_docs import CAPABILITY_ID as CAPABILITY_ID, ClaiDocs


def customization_guide() -> Capability[None]:
    """Pax's docs tools, `ClaiDocs`, under the name custom agents have been adding them by."""
    return ClaiDocs()
