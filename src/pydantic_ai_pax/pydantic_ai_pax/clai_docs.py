"""Compatibility names for agents and saved sessions using the original docs tools."""

from pydantic_ai import Tool
from pydantic_ai.capabilities import Capability
from pydantic_ai_pax.pax_docs import _read_pax_docs, _search_pax_docs

CAPABILITY_ID = 'clai_customization'


class ClaiDocs(Capability[None]):
    """The former docs capability. New agents should use `PaxDocs`."""

    def __init__(self) -> None:
        super().__init__(
            id=CAPABILITY_ID,
            description='Compatibility name for pax_customization. New sessions should load pax_customization.',
            defer_loading=True,
            instructions='Use search_clai_docs and read_clai_docs for Pax documentation.',
            tools=[Tool(_search_pax_docs, name='search_clai_docs'), Tool(_read_pax_docs, name='read_clai_docs')],
        )
