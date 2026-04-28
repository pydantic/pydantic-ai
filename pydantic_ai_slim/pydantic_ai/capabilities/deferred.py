from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.tools import AgentDepsT

from .wrapper import WrapperCapability


@dataclass
class DeferredCapability(WrapperCapability[AgentDepsT]):
    """A wrapper that suppresses a capability's contributions until explicitly loaded.

    When a capability has ``defer_loading=True``, the agent automatically wraps it
    in a ``DeferredCapability``. Before :meth:`load` is called, all ``get_*`` methods
    return empty values (``None`` / ``[]``), hiding the capability's instructions,
    tools, and settings from the model. The capability's ``id`` and ``description``
    remain visible for catalog rendering so the model can discover and request it
    via ``load_capability(id)``.

    Hooks are always forwarded regardless of loaded state, allowing the wrapped
    capability to observe lifecycle events and decide how to act based on
    whether it has been loaded.
    """

    _loaded: bool = field(default=False, init=False, repr=False)

    def load(self) -> None:
        self._loaded = True
