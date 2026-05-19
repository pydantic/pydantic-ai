"""ACI.dev integration 1.x deprecation surfaces.

In 1.x, calling `tool_from_aci(...)` or constructing `ACIToolset(...)` emits a
`PydanticAIDeprecationWarning` pointing users at wrapping ACI tools directly with
`Tool.from_schema`. The functions remain functional until the v2 cut, which removes the
`pydantic_ai.ext.aci` module entirely (see #5467).
"""

from __future__ import annotations

import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning

from ..conftest import try_import

with try_import() as imports_successful:
    import aci  # noqa: F401  # pyright: ignore[reportUnusedImport]

    from pydantic_ai.ext.aci import ACIToolset, tool_from_aci  # pyright: ignore[reportDeprecated]


pytestmark = pytest.mark.skipif(not imports_successful(), reason='aci-sdk not installed')


def test_tool_from_aci_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch):
    """`tool_from_aci(...)` emits a `PydanticAIDeprecationWarning` before reaching the SDK."""
    monkeypatch.setenv('ACI_API_KEY', 'dummy')
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`pydantic_ai\.ext\.aci` is deprecated'):
        # The call raises once it tries to hit the network; the warning fires first.
        with pytest.raises(Exception):
            tool_from_aci('TAVILY__SEARCH', linked_account_owner_id='dummy')  # pyright: ignore[reportDeprecated]


def test_aci_toolset_emits_deprecation_warning(monkeypatch: pytest.MonkeyPatch):
    """`ACIToolset(...)` emits a `PydanticAIDeprecationWarning` on construction."""
    monkeypatch.setenv('ACI_API_KEY', 'dummy')
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`pydantic_ai\.ext\.aci` is deprecated'):
        ACIToolset([], linked_account_owner_id='dummy')  # pyright: ignore[reportDeprecated]
