"""The former home of CLAI's on-demand docs, kept so existing imports work."""

from pydantic_ai.capabilities import Capability
from pydantic_clai2.clai_docs import CAPABILITY_ID as CAPABILITY_ID, ClaiDocs


def customization_guide() -> Capability[None]:
    """CLAI's docs tools, `ClaiDocs`, under the name custom agents have been adding them by."""
    return ClaiDocs()
