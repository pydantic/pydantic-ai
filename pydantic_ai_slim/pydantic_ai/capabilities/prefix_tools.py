from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset, AgentToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset

from .wrapper import WrapperCapability


@dataclass
class PrefixTools(WrapperCapability[AgentDepsT]):
    """A capability that wraps another capability and prefixes its tool names.

    Only the wrapped capability's tools are prefixed; other agent tools are unaffected.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import PrefixTools, Toolset
    from pydantic_ai.toolsets import FunctionToolset

    toolset = FunctionToolset()

    agent = Agent(
        'openai:gpt-5',
        capabilities=[
            PrefixTools(
                wrapped=Toolset(toolset),
                prefix='ns',
            ),
        ],
    )
    ```
    """

    prefix: str

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'PrefixTools'

    @classmethod
    def from_spec(cls, *, prefix: str, capability: dict[str, Any] | str) -> PrefixTools[Any]:
        """Create from spec with a nested capability specification.

        Args:
            prefix: The prefix to add to tool names (e.g. ``'mcp'`` turns ``'search'`` into ``'mcp_search'``).
            capability: A capability spec (same format as entries in the ``capabilities`` list).
        """
        from pydantic_ai._spec import NamedSpec, load_from_registry
        from pydantic_ai.agent.spec import get_capability_registry

        registry = get_capability_registry()
        cap_spec = NamedSpec.model_validate(capability)
        wrapped = load_from_registry(
            registry,
            cap_spec,
            label='capability',
            custom_types_param='custom_capability_types',
            instantiate=lambda cap_cls, args, kwargs: cap_cls.from_spec(*args, **kwargs),
        )
        return cls(wrapped=wrapped, prefix=prefix)

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolset = super().get_toolset()
        if toolset is None:
            return None
        if isinstance(toolset, AbstractToolset):
            # Pyright can't narrow Callable type aliases out of unions after isinstance check
            return PrefixedToolset(toolset, prefix=self.prefix)  # pyright: ignore[reportUnknownArgumentType]
        # ToolsetFunc callable — wrap in DynamicToolset so PrefixedToolset can delegate
        from pydantic_ai.toolsets._dynamic import DynamicToolset

        return PrefixedToolset(DynamicToolset[AgentDepsT](toolset_func=toolset), prefix=self.prefix)
