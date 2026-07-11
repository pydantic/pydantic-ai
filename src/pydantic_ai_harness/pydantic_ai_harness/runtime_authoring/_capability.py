"""Runtime authoring capability: let an agent write and register real pydantic-ai capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.runtime_authoring._store import CapabilityStore
from pydantic_ai_harness.runtime_authoring._toolset import AuthoringToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

_DEFAULT_GUIDANCE = (
    'You can author new pydantic-ai capabilities at runtime with `author_capability(name, code)`. '
    'A capability is a subclass of `pydantic_ai.capabilities.AbstractCapability` that constructs with '
    'no arguments and overrides one or more lifecycle hooks (a single overridden hook is a valid '
    'capability). Authored capabilities are validated immediately but become active on the next agent '
    'run, not the current one. Use `list_authored_capabilities` and `disable_authored_capability` to '
    'manage them.'
)


@dataclass
class RuntimeAuthoring(AbstractCapability[AgentDepsT]):
    """Let an agent author, validate, and persist real pydantic-ai capabilities at runtime.

    Exposes `author_capability(name, code)`, `list_authored_capabilities`, and
    `disable_authored_capability`. Authoring writes a real `.py` to `directory`,
    imports it, and validates it (exactly one `AbstractCapability` subclass that
    constructs with no arguments and whose static getters run). Authored
    capabilities hold live code, so they are not spec-serializable and are
    persisted as source rather than as a spec.

    Activation boundary: a capability cannot be added to a live, already-executing
    run -- pydantic-ai resolves the capability set once at the start of each run.
    The authored capability becomes usable on the next `agent.run(...)`. The
    integration contract is one line on the orchestrator side: thread the store's
    active capabilities into the next run.

    ```python
    from pathlib import Path

    from pydantic_ai import Agent
    from pydantic_ai_harness.runtime_authoring import RuntimeAuthoring

    authoring = RuntimeAuthoring(directory=Path('.authored'))
    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[authoring])

    # Loop: each iteration injects whatever the agent has authored so far.
    result = await agent.run('build a logging capability', capabilities=authoring.store.load_active())
    ```

    This executes authored Python in-process -- the same trust boundary an agent
    that already runs shell commands and edits files operates under. The dormant
    `pa` Monty hook-slot registration system is the sandboxed alternative; see the
    capability README.
    """

    directory: Path
    """Directory holding the authored `<name>.py` files and the `manifest.json` index."""

    guidance: str | None = None
    """Static system-prompt guidance on authoring. Cache-stable. Leave `None` for the
    default, or set `''` to omit guidance entirely."""

    @property
    def store(self) -> CapabilityStore:
        """The disk-backed store. Call `store.load_active()` to inject authored capabilities into the next run."""
        return CapabilityStore(self.directory)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable guidance on the authoring tools."""
        guidance = _DEFAULT_GUIDANCE if self.guidance is None else self.guidance
        return guidance or None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the authoring tools over this capability's store."""
        return AuthoringToolset[AgentDepsT](self.store)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: the capability holds a live, disk-backed store."""
        return None
