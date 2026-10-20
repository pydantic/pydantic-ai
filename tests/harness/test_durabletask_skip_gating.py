from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.subprocess(reason='checks optional installs and pytest conftest startup in a fresh interpreter')
@pytest.mark.parametrize(
    ('missing', 'target'),
    [
        ('durabletask', r'tests\harness\durabletask'),
        ('durabletask', r'tests\harness\durabletask\test_durabletask.py::TestDurableTaskDurability::test_happy_path'),
        ('durabletask.azuremanaged', r'tests\harness\durabletask\test_documentation.py'),
        ('mcp', r'tests\harness\durabletask\test_transports.py::test_mcp_discovery_instructions_and_calls'),
    ],
)
def test_durabletask_optional_install_gates(missing: str, target: str) -> None:
    args = [target, '-q', '-rs', '-p', 'no:pretty', '-p', 'no:cacheprovider']
    code = f"""
import ast
import importlib.util
import sys
from pathlib import Path

missing = {missing!r}
class Missing:
    def find_spec(self, name, path=None, target=None):
        if name == missing or name.startswith(missing + '.'):
            raise ModuleNotFoundError('No module named ' + repr(name), name=name)
        return None

original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: (
    None if name == missing or name.startswith(missing + '.') else original_find_spec(name, *args, **kwargs)
)
sys.meta_path.insert(0, Missing())
import pydantic_ai_harness
assert 'durabletask' not in sys.modules
assert 'pydantic_ai' not in sys.modules

import pytest
status = pytest.main({args!r})
if missing == 'durabletask':
    from tests.append_only import discover_capabilities, skip_unless_installed
    tree = ast.parse(Path(r'tests\\harness\\test_history_append_only.py').read_text(encoding='utf-8'))
    entry = next(
        node for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.target.id == 'OPTIONAL_DEPENDENCIES'
    )
    optional = ast.literal_eval(entry.value)
    assert optional['DurableTaskDurability'] == 'durabletask'
    assert 'DurableTaskDurability' in discover_capabilities(('pydantic_ai_harness',))
    try:
        skip_unless_installed('DurableTaskDurability', optional)
    except pytest.skip.Exception:
        pass
    else:
        raise AssertionError('automatic discovery did not skip the missing extra')
raise SystemExit(status)
"""
    result = subprocess.run(
        [sys.executable, '-c', code],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        timeout=120,
        env={key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')},
    )
    output = result.stdout + result.stderr
    assert result.returncode == pytest.ExitCode.OK, output
    assert 'skipped' in output, output
    assert 'ERROR' not in output, output
