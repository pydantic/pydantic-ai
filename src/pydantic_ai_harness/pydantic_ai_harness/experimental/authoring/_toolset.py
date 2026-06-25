"""Authoring toolset: tools for the model to author, list, and disable capabilities."""

from __future__ import annotations

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

from pydantic_ai_harness.experimental.authoring._store import CapabilityStore


class AuthoringToolset(FunctionToolset[AgentDepsT]):
    """Exposes `author_capability`, `list_authored_capabilities`, and `disable_authored_capability`."""

    def __init__(self, store: CapabilityStore) -> None:
        super().__init__()
        self._store = store
        self.add_function(self.author_capability, name='author_capability')
        self.add_function(self.list_authored_capabilities, name='list_authored_capabilities')
        self.add_function(self.disable_authored_capability, name='disable_authored_capability')

    async def author_capability(self, name: str, code: str) -> str:
        """Author a new pydantic-ai capability from Python source.

        `code` must define exactly one `pydantic_ai.capabilities.AbstractCapability`
        subclass that constructs with no arguments. The capability is written to
        disk, imported, and validated immediately. It becomes usable on the next
        agent run (the next orchestrator loop iteration), not the current run.

        Args:
            name: Identifier for the capability. Lowercase letters, digits, and
                underscores, starting with a letter. Reusing a name replaces the
                previous capability of that name.
            code: Complete Python source defining one `AbstractCapability` subclass.
        """
        try:
            record = self._store.write(name, code)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        if record.last_error is not None:
            return (
                f'Capability {name!r} was written but failed validation: {record.last_error}\n'
                f'Fix the code and call author_capability again with the same name.'
            )
        return (
            f'Capability {name!r} ({record.class_name}) authored and validated. It becomes active on '
            f'the next agent run (the next orchestrator loop iteration), not the current run.'
        )

    async def list_authored_capabilities(self) -> str:
        """List the capabilities authored so far, with their status and any validation error."""
        records = self._store.list_all()
        if not records:
            return 'No capabilities authored yet.'
        lines: list[str] = []
        for record in records:
            suffix = f' -- ERROR: {record.last_error}' if record.last_error is not None else ''
            class_name = record.class_name or '?'
            lines.append(f'- {record.name} [{record.status}] {class_name}{suffix}')
        return '\n'.join(lines)

    async def disable_authored_capability(self, name: str) -> str:
        """Disable an authored capability so it is no longer injected on the next run.

        Args:
            name: Name of the capability to disable.
        """
        if self._store.disable(name):
            return f'Capability {name!r} disabled; it will not be injected on the next run.'
        return f'No authored capability named {name!r}.'
