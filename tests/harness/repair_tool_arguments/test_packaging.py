"""`json-repair` ships in the `coder` and `repair-tool-arguments` extras, not the base install.

Only metadata checks belong here: this file must not import `pydantic_ai_harness.repair_tool_arguments`.
"""

from __future__ import annotations

import importlib.metadata
import re


def test_json_repair_is_only_an_extra_dependency() -> None:
    requirements = importlib.metadata.metadata('pydantic-ai-harness').get_all('Requires-Dist') or []
    json_repair = [req for req in requirements if req.startswith('json-repair')]
    extras = {match.group(1) for req in json_repair if (match := re.search(r"""extra == ['"]([\w-]+)['"]""", req))}
    assert len(extras) == len(json_repair), 'json-repair must not be a base dependency'
    assert extras == {'coder', 'repair-tool-arguments'}
