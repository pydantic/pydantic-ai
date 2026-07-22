"""Tests that the deprecated Exa common tools emit `PydanticAIDeprecationWarning`.

Not a VCR test: the deprecation fires at construction time via `@deprecated`, before any Exa API
call is made, so there is no request to record. The `ExaToolset` case also pins the suppression of
the redundant per-factory warnings it triggers internally.
"""

from __future__ import annotations

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.common_tools.exa import (
    ExaToolset,  # pyright: ignore[reportDeprecated]
    exa_answer_tool,  # pyright: ignore[reportDeprecated]
    exa_find_similar_tool,  # pyright: ignore[reportDeprecated]
    exa_get_contents_tool,  # pyright: ignore[reportDeprecated]
    exa_search_tool,  # pyright: ignore[reportDeprecated]
)


def test_exa_factory_tools_deprecated():
    with pytest.warns(PydanticAIDeprecationWarning, match='`exa_search_tool` is deprecated'):
        exa_search_tool(api_key='x')  # pyright: ignore[reportDeprecated]
    with pytest.warns(PydanticAIDeprecationWarning, match='`exa_find_similar_tool` is deprecated'):
        exa_find_similar_tool(api_key='x')  # pyright: ignore[reportDeprecated]
    with pytest.warns(PydanticAIDeprecationWarning, match='`exa_get_contents_tool` is deprecated'):
        exa_get_contents_tool(api_key='x')  # pyright: ignore[reportDeprecated]
    with pytest.warns(PydanticAIDeprecationWarning, match='`exa_answer_tool` is deprecated'):
        exa_answer_tool(api_key='x')  # pyright: ignore[reportDeprecated]


def test_exa_toolset_deprecated_emits_single_warning():
    """`ExaToolset` warns once; the per-factory warnings it triggers internally are suppressed."""
    with pytest.warns(PydanticAIDeprecationWarning, match='`ExaToolset` is deprecated') as records:
        toolset = ExaToolset(api_key='x')  # pyright: ignore[reportDeprecated]

    exa_warnings = [r for r in records if issubclass(r.category, PydanticAIDeprecationWarning)]
    assert len(exa_warnings) == 1
    assert set(toolset.tools) == {'exa_search', 'exa_find_similar', 'exa_get_contents', 'exa_answer'}
