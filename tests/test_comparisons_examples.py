"""Comparison-page snippets: run offline in CI and must pass their own assertions.

The code embedded in `docs/comparisons/*.md` via the `snippet` directive is
the source of truth; executing it here keeps the printed outputs on the pages
fresh. All snippets are deterministic and need no API keys.
"""

from __future__ import annotations as _annotations

import asyncio
import importlib
from typing import Any, Callable

import pytest

_SNIPPETS = (
    'deps_boundary',
    'deferred_capability',
    'cancel_from_tool',
    'cancel_token_thread',
    'usage_limits_atomic',
    'history_repair',
    'spec_validation',
    'event_stream',
    'evals_ci',
    'durability_wrap',
    'graph_is_a_value',
)


@pytest.mark.parametrize('module_name', _SNIPPETS)
def test_comparison_snippet(module_name: str) -> None:
    mod = importlib.import_module(f'pydantic_ai_examples.comparisons.{module_name}')
    main: Callable[..., Any] = getattr(mod, 'main', None)
    assert callable(main), f'{module_name} must define main()'
    result = main()
    if asyncio.iscoroutine(result):
        asyncio.run(result)
