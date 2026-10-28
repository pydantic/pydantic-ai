"""`json-repair` ships in the `coder` extra, not the base install.

Only metadata checks belong here: this file must not import `pydantic_ai_harness.repair_tool_arguments`.
"""

from __future__ import annotations

import importlib.metadata


def test_json_repair_is_a_coder_extra_dependency() -> None:
    requirements = importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []
    json_repair = [req for req in requirements if req.startswith('json-repair')]
    assert json_repair, 'json-repair must be declared'
    assert all('extra == "coder"' in req or "extra == 'coder'" in req for req in json_repair)
