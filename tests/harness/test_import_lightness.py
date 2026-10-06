"""Import-lightness regressions for the `pydantic_ai_harness` entry points.

Importing any submodule of `pydantic_ai_harness` executes the root package first, so
one eager `pydantic_ai` import there made even `import pydantic_ai_harness.media` --
the path storage users hit on cold start -- load all of Pydantic AI plus the MCP
stack. The checks run in fresh subprocesses so imports made elsewhere in this test
session cannot hide a regression (#9916).
"""

import subprocess
import sys

_FORBIDDEN_MODULES = ('fastmcp', 'mcp', 'pydantic_ai')


def _assert_no_modules_loaded(entry_point: str, forbidden: tuple[str, ...] = _FORBIDDEN_MODULES) -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            f'import sys; import {entry_point}; print("\\n".join(sys.modules))',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = [module for module in result.stdout.splitlines() if module]
    unexpected = sorted(
        module for module in loaded if any(module == name or module.startswith(f'{name}.') for name in forbidden)
    )
    assert not unexpected, f'importing {entry_point} unexpectedly loaded {unexpected}'


def test_root_import_is_light() -> None:
    _assert_no_modules_loaded('pydantic_ai_harness')


def test_media_import_is_light() -> None:
    _assert_no_modules_loaded('pydantic_ai_harness.media')


def test_step_persistence_import_is_light() -> None:
    # `StepPersistence` legitimately imports `pydantic_ai` (it hooks the agent
    # graph), and core's own eager MCP chain is tracked separately (#9770). What
    # this entry point must not load is the harness MCP wiring that the root
    # package used to import eagerly (#9916).
    _assert_no_modules_loaded('pydantic_ai_harness.step_persistence', forbidden=('pydantic_ai_harness._mcp',))


def test_mcp_warning_export_survives_relocation() -> None:
    import pydantic_ai_harness
    import pydantic_ai_harness._mcp
    import pydantic_ai_harness._warn

    warning = pydantic_ai_harness.MCPReadOnlyNoToolsWarning
    assert warning is pydantic_ai_harness._warn.MCPReadOnlyNoToolsWarning
    assert warning is pydantic_ai_harness._mcp.MCPReadOnlyNoToolsWarning
    assert issubclass(warning, UserWarning)
