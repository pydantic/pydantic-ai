import subprocess
import sys
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

import pydantic_ai_harness.coder
from pydantic_ai_harness.coder import Coder, coder_agent
from pydantic_ai_harness.repo_context import RepoContext


def test_coder_agent_is_model_less_and_composed() -> None:
    assert isinstance(coder_agent, Agent)
    assert coder_agent.model is None
    assert coder_agent.name == 'coder'


def test_coder_agent_export_is_lazy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sys; import pydantic_ai_harness.coder; '
            "assert 'pydantic_ai_harness.coder._agent' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_coder_unknown_export() -> None:
    with pytest.raises(AttributeError, match='has no attribute'):
        pydantic_ai_harness.coder.__getattr__('missing')


def test_coder_members_and_parameters(tmp_path: Path) -> None:
    coder = Coder(tmp_path, instructions='Custom instructions')
    assert [type(capability).__name__ for capability in coder.capabilities] == [
        '_RepairToolArguments',
        'Capability',
        'RepoContext',
        'ClearToolResults',
        'WarnNearLimits',
        '_BoundToolOutputs',
    ]
    context = next(item for item in coder.capabilities if isinstance(item, RepoContext))
    assert context.workspace_dir == tmp_path
    guidance = next(item for item in coder.capabilities if isinstance(item, Capability))
    instructions = str(guidance.get_instructions())
    for text in ('Custom instructions', 'DRY', 'YAGNI', 'SOLID', '600', 'sleep 60', '270'):
        assert text in instructions
    assert coder.capabilities[-1].id is None
    assert isinstance(coder.for_agent(Agent(TestModel())), Coder)
