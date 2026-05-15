"""Card 46: fasta2a 1.x deprecation surfaces.

In 1.x, calling `Agent.to_a2a()` emits a `PydanticAIDeprecationWarning` pointing users at
[datalayer/fasta2a](https://github.com/datalayer/fasta2a), the new upstream home of the
package. The method itself remains functional until the v2 cut.
"""

from __future__ import annotations

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from ..conftest import try_import

with try_import() as imports_successful:
    import fasta2a  # noqa: F401  # pyright: ignore[reportUnusedImport]

    from pydantic_ai.agent import Agent
    from pydantic_ai.models.test import TestModel


pytestmark = pytest.mark.skipif(not imports_successful(), reason='fasta2a not installed')


def test_agent_to_a2a_emits_deprecation_warning():
    """`Agent.to_a2a(...)` emits a `PydanticAIDeprecationWarning` pointing at the upstream bridge."""
    agent = Agent(TestModel())
    with pytest.warns(PydanticAIDeprecationWarning, match=r'fasta2a\.pydantic_ai import agent_to_a2a'):
        agent.to_a2a()  # pyright: ignore[reportDeprecated]
