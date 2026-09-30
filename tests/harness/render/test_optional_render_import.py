"""Collection distinguishes an absent Render extra from a broken installation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

RENDER_TEST_SUBTREE = 'tests/harness/render'
MISSING_TRANSITIVE_MODULE = 'render_transitive_dependency_that_is_missing'
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_BLOCK_RENDER_LAUNCHER = f"""
import importlib.abc
import importlib.util
import sys


class BlockRender(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'render' or fullname.startswith('render.'):
            raise ModuleNotFoundError("No module named 'render'", name='render')
        return None


for cached in [name for name in sys.modules if name == 'render' or name.startswith('render.')]:
    del sys.modules[cached]

sys.meta_path.insert(0, BlockRender())

try:
    importlib.util.find_spec('render')
except ModuleNotFoundError as exc:
    if exc.name != 'render':
        raise
else:
    print('SIMULATION FAILED: the render package is still discoverable')
    raise SystemExit(99)

import pytest

raise SystemExit(
    pytest.main(['--collect-only', '-q', '--assert=plain', {RENDER_TEST_SUBTREE!r}])
)
""".lstrip()


def _collect(arguments: list[str], environment: dict[str, str] | None = None) -> tuple[int, str]:
    """Run a collection-only pytest subprocess and return its code and combined output."""
    completed = subprocess.run(
        arguments,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.returncode, completed.stdout + completed.stderr


def test_render_test_subtree_collects_when_the_extra_is_installed() -> None:
    """The subtree really holds tests, so an empty collection can only come from absence."""
    code, output = _collect([sys.executable, '-m', 'pytest', '--collect-only', '-q', RENDER_TEST_SUBTREE])

    assert code == 0, output
    assert 'no tests collected' not in output
    assert 'tests collected' in output


def test_present_but_broken_render_package_fails_collection(tmp_path: Path) -> None:
    """A discoverable Render package's missing transitive import is not an extra skip."""
    package = tmp_path / 'broken' / 'render'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(f'import {MISSING_TRANSITIVE_MODULE}\n')
    environment = dict(os.environ)
    existing_path = environment.get('PYTHONPATH')
    paths = [str(package.parent)]
    if existing_path:
        paths.append(existing_path)
    environment['PYTHONPATH'] = os.pathsep.join(paths)

    code, output = _collect(
        [sys.executable, '-m', 'pytest', '--collect-only', '-q', RENDER_TEST_SUBTREE],
        environment,
    )

    assert code != 0, output
    assert code != 5, output
    assert MISSING_TRANSITIVE_MODULE in output
    assert 'no tests collected' not in output
    assert 'Skipped:' not in output


def test_truly_absent_render_package_is_ignored_without_errors(tmp_path: Path) -> None:
    """With no importable top-level render package the whole subtree is ignored safely."""
    launcher = tmp_path / 'collect_without_render.py'
    launcher.write_text(_BLOCK_RENDER_LAUNCHER)

    code, output = _collect([sys.executable, str(launcher)])

    assert 'SIMULATION FAILED' not in output
    assert code == 5, output
    assert 'no tests collected' in output
    assert 'Traceback' not in output
    assert 'ERROR' not in output
    assert MISSING_TRANSITIVE_MODULE not in output
