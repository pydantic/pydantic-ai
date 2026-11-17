"""In-memory REPL storage shared by CodeMode persistence and remote error-routing tests.

`MemoryReplStore` records loads and saves by conversation and can simulate storage failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai import RunContext


@dataclass
class MemoryReplStore:
    """REPL states keyed by conversation, recording every load and save."""

    states: dict[str, bytes] = field(default_factory=dict[str, bytes])
    loads: list[str] = field(default_factory=list[str])
    saves: list[tuple[str, bool]] = field(default_factory=list[tuple[str, bool]])
    load_error: Exception | None = None
    """Raised by `load` while set, as a store whose storage cannot be read."""
    save_error: Exception | None = None
    """Raised by `save` while set, as a store whose storage cannot be written."""

    async def load(self, ctx: RunContext[None]) -> bytes | None:
        assert ctx.conversation_id is not None
        self.loads.append(ctx.conversation_id)
        if self.load_error is not None:
            raise self.load_error
        return self.states.get(ctx.conversation_id)

    async def save(self, ctx: RunContext[None], state: bytes | None) -> None:
        assert ctx.conversation_id is not None
        if self.save_error is not None:
            raise self.save_error
        self.saves.append((ctx.conversation_id, state is not None))
        if state is None:
            self.states.pop(ctx.conversation_id, None)
        else:
            self.states[ctx.conversation_id] = state
