"""Capability creation toolset: tools for the model to author, list, and disable capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio.to_thread

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness._durable import RetryRequest, raise_retry
from pydantic_ai_harness.capability_creation._store import AuthoredCapability, CapabilityStore, rejected_module_file


@dataclass(frozen=True)
class CapabilityCreationOperations:
    """The durable operations a `CapabilityCreation` capability runs its toolset's store reads and writes through."""

    write: Callable[[str, str], Awaitable[AuthoredCapability | RetryRequest]]
    list_all: Callable[[], Awaitable[list[AuthoredCapability]]]
    disable: Callable[[str], Awaitable[bool]]


class CapabilityCreationToolset(FunctionToolset[AgentDepsT]):
    """Exposes `author_capability`, `list_authored_capabilities`, and `disable_authored_capability`.

    `CapabilityCreation` passes `operations` so that each store read and write
    runs as one of its durable operations, whose result durable execution
    records instead of writing the authored module again on recovery.
    """

    # The store is synchronous disk I/O (and imports the authored module), so each tool runs it
    # in a worker thread rather than on the event loop. The store serializes its own updates, so
    # a cancelled call stops waiting for the thread instead of blocking until the authored code
    # returns; the thread finishes the update on its own.
    def __init__(
        self,
        store: CapabilityStore,
        *,
        id: str | None = None,
        operations: CapabilityCreationOperations | None = None,
    ) -> None:
        super().__init__(id=id)
        self._store = store
        self._operations = operations
        self.add_function(
            self.author_capability,
            name='author_capability',
            metadata={'code_arg_name': 'code', 'code_arg_language': 'python'},
        )
        self.add_function(self.list_authored_capabilities, name='list_authored_capabilities')
        self.add_function(self.disable_authored_capability, name='disable_authored_capability')

    async def author_capability(self, name: str, code: str) -> str:
        """Author a new pydantic-ai capability from Python source.

        `code` must define exactly one `pydantic_ai.capabilities.AbstractCapability`
        subclass that constructs with no arguments. The capability is written to
        disk, imported, and validated immediately. It does not take effect in this
        run; whether a later run loads it depends on how this agent is set up.

        Args:
            name: Identifier for the capability. At most 64 lowercase letters, digits,
                and underscores, starting with a letter. Reusing a name replaces the
                previous capability of that name once the new code passes validation.
            code: Complete Python source defining one `AbstractCapability` subclass.
        """
        try:
            if self._operations is not None:
                record = raise_retry(await self._operations.write(name, code))
            else:
                record = await anyio.to_thread.run_sync(self._store.write, name, code, abandon_on_cancel=True)
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        if record.last_error is not None:
            # `CapabilityStore.write` saves rejected code elsewhere when it keeps a previous version.
            kept = (
                f'The previous version of {name!r} is unchanged.\n'
                if record.module_file == rejected_module_file(name)
                else ''
            )
            return (
                f'Capability {name!r} failed validation: {record.last_error}\n{kept}'
                'Fix the code and call author_capability again with the same name.'
            )
        return (
            f'Capability {name!r} ({record.class_name}) authored, validated and saved. It does not take effect '
            'in this run; whether a later run loads it depends on how this agent is set up.'
        )

    async def list_authored_capabilities(self) -> str:
        """List the capabilities authored so far, with their status and any validation error."""
        if self._operations is not None:
            records = await self._operations.list_all()
        else:
            records = await anyio.to_thread.run_sync(self._store.list_all, abandon_on_cancel=True)
        if not records:
            return 'No capabilities authored yet.'
        lines: list[str] = []
        for record in records:
            suffix = f' -- ERROR: {record.last_error}' if record.last_error is not None else ''
            class_name = record.class_name or '?'
            lines.append(f'- {record.name} [{record.status}] {class_name}{suffix}')
        return '\n'.join(lines)

    async def disable_authored_capability(self, name: str) -> str:
        """Disable an authored capability so it is skipped when this agent loads its authored capabilities.

        Args:
            name: Name of the capability to disable.
        """
        if self._operations is not None:
            found = await self._operations.disable(name)
        else:
            found = await anyio.to_thread.run_sync(self._store.disable, name, abandon_on_cancel=True)
        if found:
            return f'Capability {name!r} disabled; it is skipped when this agent loads its authored capabilities.'
        return f'No authored capability named {name!r}.'
