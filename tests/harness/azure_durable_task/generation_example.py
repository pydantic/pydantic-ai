from __future__ import annotations

import re
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType
from unittest.mock import patch

from durabletask import task
from pydantic import InstanceOf, JsonValue, TypeAdapter

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model

README = (
    Path(__file__).parents[3]
    / 'src'
    / 'pydantic_ai_harness'
    / 'pydantic_ai_harness'
    / 'azure_durable_task'
    / 'README.md'
)


@dataclass
class GenerationExample:
    agent: InstanceOf[Agent[None, str]]
    orchestrator: Callable[[task.OrchestrationContext, object], Generator[task.Task[object], object, JsonValue]]
    main: Callable[[], None]


def python_examples() -> list[str]:
    return re.findall(r'```python[^\n]*\n(.*?)\n```', README.read_text(encoding='utf-8'), flags=re.DOTALL)


def load_generation_example(model: Model, *, agent_name: str) -> GenerationExample:
    """Execute the published continuation example with only model/name substitution."""

    def make_agent(model_name: str, *, name: str, capabilities: Sequence[AbstractCapability[None]]) -> Agent[None, str]:
        assert model_name == 'openai:gpt-6.1-sol'
        assert name == 'investigator'
        return Agent(model, name=agent_name, deps_type=type(None), capabilities=capabilities)

    source = next(source for source in python_examples() if source.startswith('# generations.py\n'))
    namespace: dict[str, object] = {'__name__': __name__}
    with patch('pydantic_ai.Agent', make_agent):
        exec(compile(source, f'{README}:generations', 'exec', dont_inherit=True), namespace)
    workflow = namespace['investigation']
    assert isinstance(workflow, FunctionType)
    workflow.__name__ = f'{agent_name}_workflow'
    return TypeAdapter(GenerationExample).validate_python(
        {'agent': namespace['agent'], 'orchestrator': workflow, 'main': namespace['main']}
    )
