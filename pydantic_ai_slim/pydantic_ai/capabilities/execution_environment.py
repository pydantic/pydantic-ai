from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, overload

from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.environments import EnvToolName, ExecutionEnvironment as BaseExecutionEnvironment
from pydantic_ai.environments.local import LocalEnvironment
from pydantic_ai.environments.memory import MemoryEnvironment
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets.abstract import AbstractToolset
from pydantic_ai.toolsets.execution_environment import ExecutionEnvironmentToolset


@dataclass(init=False)
class ExecutionEnvironment(AbstractCapability[AgentDepsT]):
    """A capability that provides an execution environment for running code."""

    toolset: ExecutionEnvironmentToolset = field(default_factory=ExecutionEnvironmentToolset)

    @overload
    def __init__(
        self,
        *,
        environment: Literal['local', 'memory'] | BaseExecutionEnvironment,
        include: list[EnvToolName] | None = None,
        exclude: list[EnvToolName] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        toolset: ExecutionEnvironmentToolset,
    ) -> None: ...

    def __init__(
        self,
        *,
        environment: Literal['local', 'memory'] | BaseExecutionEnvironment | None = None,
        toolset: ExecutionEnvironmentToolset | None = None,
        include: list[EnvToolName] | None = None,
        exclude: list[EnvToolName] | None = None,
    ):
        if toolset:
            self.toolset = toolset
        else:
            if isinstance(environment, str):
                if environment == 'local':
                    environment = LocalEnvironment()
                elif environment == 'memory':
                    environment = MemoryEnvironment()
                else:
                    raise ValueError(f'Invalid environment: {environment}')

            self.toolset = ExecutionEnvironmentToolset(environment, include=include, exclude=exclude)

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        return self.toolset

    @classmethod
    def from_spec(
        cls,
        *args: Any,
        environment: Literal['local', 'memory'] | None = None,
        include: list[EnvToolName] | None = None,
        exclude: list[EnvToolName] | None = None,
        **kwargs: Any,
    ) -> ExecutionEnvironment[Any]:
        """Create from spec. Accepts string environment names ('local', 'memory')."""
        if args:
            return cls(environment=args[0], **kwargs)
        return cls(environment=environment, include=include, exclude=exclude, **kwargs)  # type: ignore[arg-type]
