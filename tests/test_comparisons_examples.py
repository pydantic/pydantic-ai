"""Comparison-page snippets: run offline in CI and must pass their own assertions.

The code embedded in `docs/comparisons/*.md` via the `snippet` directive is the
source of truth; executing it here keeps the printed outputs on the pages fresh.
All snippets are deterministic and need no API keys. Each comparison keeps its
snippets next to its page: `examples/pydantic_ai_examples/comparisons/<topic>/`.
"""

from __future__ import annotations as _annotations

import asyncio
import importlib
from typing import Any, Callable

import pytest

_SNIPPETS = (
    # topic, module
    ('production_agent', 'deps_boundary'),
    ('production_agent', 'deferred_capability'),
    ('production_agent', 'cancel_from_tool'),
    ('production_agent', 'cancel_token_thread'),
    ('production_agent', 'usage_limits_atomic'),
    ('production_agent', 'history_repair'),
    ('production_agent', 'spec_validation'),
    ('production_agent', 'event_stream'),
    ('production_agent', 'evals_ci'),
    ('production_agent', 'durability_wrap'),
    ('vs_langchain_langgraph', 'graph_is_a_value'),
)


@pytest.mark.parametrize('topic,module_name', _SNIPPETS)
def test_comparison_snippet(topic: str, module_name: str) -> None:
    mod = importlib.import_module(f'pydantic_ai_examples.comparisons.{topic}.{module_name}')
    main: Callable[..., Any] = getattr(mod, 'main', None)
    assert callable(main), f'{topic}/{module_name} must define main()'
    result = main()
    if asyncio.iscoroutine(result):
        asyncio.run(result)
