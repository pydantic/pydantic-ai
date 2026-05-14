"""Cards 15 + 19: AG-UI 1.x deprecation surfaces.

In 1.x, importing `pydantic_ai.ag_ui`, calling `Agent.to_ag_ui()`, and constructing
`AGUIApp` each emit a `PydanticAIDeprecationWarning` pointing users at `pydantic_ai.ui.ag_ui`
and direct `AGUIAdapter` composition. The module/symbols themselves remain functional
until the v2 cut.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.agent import Agent
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.ui.ag_ui.app import AGUIApp


pytestmark = pytest.mark.skipif(not imports_successful(), reason='ag-ui-protocol not installed')


def test_pydantic_ai_ag_ui_module_import_emits_deprecation_warning():
    """`import pydantic_ai.ag_ui` emits a `PydanticAIDeprecationWarning` pointing at `pydantic_ai.ui.ag_ui`."""
    sys.modules.pop('pydantic_ai.ag_ui', None)
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`pydantic_ai\.ag_ui` module is deprecated'):
        importlib.import_module('pydantic_ai.ag_ui')


def test_agent_to_ag_ui_emits_deprecation_warning():
    """`Agent.to_ag_ui(...)` emits a `PydanticAIDeprecationWarning` pointing at composing `AGUIAdapter` directly."""
    agent = Agent(TestModel())
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`Agent\.to_ag_ui\(\)` is deprecated'):
        agent.to_ag_ui()  # pyright: ignore[reportDeprecated]


def test_agui_app_construction_emits_deprecation_warning():
    """`AGUIApp(...)` emits a `PydanticAIDeprecationWarning` pointing at composing `AGUIAdapter` directly."""
    agent = Agent(TestModel())
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`AGUIApp` is deprecated'):
        AGUIApp(agent)  # pyright: ignore[reportDeprecated]
