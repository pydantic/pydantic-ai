"""Workspace-independent researcher configuration."""

from pathlib import Path

import pytest

# `Researcher` builds `WebFetch(local=True)`, which needs the `web-fetch` extra.
pytest.importorskip('markdownify')

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.researcher import Researcher
from pydantic_ai_harness.tool_output_limits import LocalFileStore


async def test_researcher_with_local_store_needs_no_workspace(tmp_path: Path) -> None:
    agent = Agent(
        TestModel(call_tools=[], custom_output_text='done'), capabilities=[Researcher(store=LocalFileStore(tmp_path))]
    )
    result = await agent.run('research')
    assert result.output == 'done'
