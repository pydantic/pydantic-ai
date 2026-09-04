#!/usr/bin/env python
"""Type-check the files a change can reach, instead of the whole project.

`make typecheck` runs Pyright over every file in `[tool.pyright] include`, and takes about
as long whether one file changed or a thousand. That is the right trade in CI and the wrong
one on every commit, so the pre-commit hook runs this instead: it narrows the run to the
files whose content changed since Pyright last passed, plus everything that transitively
imports them.

What passed is recorded in a checkpoint under the git directory, so it is per-worktree and
never committed. Anything the checkpoint cannot account for falls back to
`make typecheck-pyright`, the same full run CI performs: a first run, a dependency or
configuration change, an import that would resolve somewhere new, an interpreter older than
the 3.11 this needs to read `pyproject.toml`, or a change large enough that narrowing stops
paying for itself.

Usage:
    python scripts/typecheck_changed.py
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypedDict

Runner = Callable[[Sequence[str]], int]
"""Runs a command, streams its output, and returns its exit code."""

# The checkpoint lives in the git directory, which is per-worktree and never tracked.
CHECKPOINT_NAME = 'pyright-checkpoint.json'

# Files that change what Pyright reports without appearing in its file list: its own
# configuration, the locked dependency versions it resolves imports against, and the
# recipe that invokes it.
_CONFIGURATION_FILES = ('pyproject.toml', 'uv.lock', 'Makefile')

# Pyright applies these to whatever an `include` entry sweeps up, but only while `exclude`
# is unset; setting `exclude` replaces them rather than adding to them.
_SKIPPED_DIRECTORIES = frozenset({'__pycache__', 'node_modules'})

_GLOB_CHARACTERS = frozenset('*?[')


class _FileState(TypedDict):
    hash: str
    imports: list[str]


class _Checkpoint(TypedDict):
    keys: dict[str, str]
    files: dict[str, _FileState]


class _ExecutionEnvironment(TypedDict, total=False):
    extraPaths: list[str]


class _PyrightSettings(TypedDict, total=False):
    include: list[str]
    exclude: list[str]
    extends: str
    executionEnvironments: list[_ExecutionEnvironment]


class _Workspace(TypedDict, total=False):
    members: list[str]


class _Uv(TypedDict, total=False):
    workspace: _Workspace


class _Tools(TypedDict, total=False):
    pyright: _PyrightSettings
    uv: _Uv


class _Pyproject(TypedDict, total=False):
    tool: _Tools


_CHECKPOINT_ADAPTER = TypeAdapter(_Checkpoint)
_PYPROJECT_ADAPTER = TypeAdapter(_Pyproject)


@dataclass(frozen=True)
class _Project:
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    # The project root, the workspace packages and every execution environment's
    # `extraPaths`, longest first so the most specific one names a file's own module.
    #
    # An execution environment's own `root` is a search path for Pyright too, and adding it
    # here looks like the more faithful model, but it is not: naming `tests` a root makes
    # every file under it stop being part of the `tests` package, so each one loses the
    # relative-import edge that ties it to `tests/__init__.py`. Measured on this repository,
    # adding it drops 228 edges and adds 1.
    import_roots: tuple[str, ...]


def run_command(command: Sequence[str]) -> int:
    """Run `command` the way the Makefile would, streaming its output, and return its exit code."""
    sys.stdout.flush()
    # Without `PYRIGHT_PYTHON_IGNORE_WARNINGS` the Pyright wrapper asks GitHub for the
    # latest release on every invocation, which is what the Makefile also avoids.
    return subprocess.run(command, env={**os.environ, 'PYRIGHT_PYTHON_IGNORE_WARNINGS': '1'}).returncode


def main(run: Runner = run_command) -> int:
    """Type-check the files the working tree's changes can reach, and return Pyright's exit code."""
    if os.environ.get('CI'):
        # CI keeps no checkpoint between runs, so there is nothing to narrow against.
        return _check_everything(run, 'CI is set')

    if sys.version_info < (3, 11):
        # Reading Pyright's file list out of pyproject.toml needs `tomllib`, added in 3.11.
        return _check_everything(run, 'this interpreter is older than Python 3.11')

    project = _load_project()
    if project is None:
        return _check_everything(run, 'the Pyright file list is not one this script can reproduce')

    universe = _tracked_files()
    # `exclude` silences a file's own diagnostics and `include` bounds what Pyright looks
    # at, but either file is still read for whoever imports it. So both stay in the graph
    # and neither is ever a check target.
    checkable = [path for path in universe if _is_checked(path, project)]
    hashes = {path: _file_hash(path) for path in universe}
    # The Makefile turns this into `--pythonversion`, so it decides what Pyright answers.
    requested_version = os.environ.get('PYRIGHT_PYTHON', '')
    keys = _invalidation_keys(requested_version)
    checkpoint_path = Path(_git('rev-parse', '--absolute-git-dir').strip()) / CHECKPOINT_NAME
    checkpoint = _load_checkpoint(checkpoint_path)
    stored: dict[str, _FileState] = checkpoint['files'] if checkpoint is not None else {}
    changed = [path for path in universe if path not in stored or stored[path]['hash'] != hashes[path]]

    reason = _reason_to_check_everything(checkpoint, keys, stored, universe, project.import_roots)
    imports: dict[str, list[str]] | None = None
    affected: list[str] = []
    if reason is None:
        imports = _parse_imports(changed, universe, project.import_roots)
        deleted = [path for path in stored if path not in hashes]
        affected = _affected(changed, deleted, stored, imports, checkable)
        if not affected:
            # Either nothing changed, or what changed is only read by files Pyright reports
            # nothing about, which comes to the same answer.
            print('Nothing to type-check: no change since Pyright last passed reaches a file it reports on.')
            return 0
        if len(affected) * 2 > len(checkable):
            reason = f'{len(affected)} of {len(checkable)} files are affected, so a full run costs no more'

    if reason is not None:
        code = _check_everything(run, reason)
    else:
        options = ['--pythonversion', requested_version] if requested_version else []
        print(f'Type-checking {len(affected)} of {len(checkable)} files, reached from {len(changed)} changed.')
        # No `--threads`, unlike the full run: the narrowed set is at most half the project,
        # and at that size the workers do not pay for themselves. Measured on this repo,
        # 31 files take 5 seconds single-process.
        code = run([sys.executable, '-m', 'pyright', *options, *affected])

    if code != 0:
        # The checkpoint records what Pyright accepted, so a failing run leaves it alone.
        return code

    if imports is None:
        imports = _parse_imports(changed, universe, project.import_roots)
    files = {
        path: _FileState(hash=hashes[path], imports=imports[path] if path in imports else stored[path]['imports'])
        for path in universe
    }
    checkpoint_path.write_bytes(_CHECKPOINT_ADAPTER.dump_json(_Checkpoint(keys=keys, files=files)))
    return 0


def _check_everything(run: Runner, reason: str) -> int:
    print(f'Type-checking every file: {reason}.')
    return run(['make', 'typecheck-pyright'])


def _reason_to_check_everything(
    checkpoint: _Checkpoint | None,
    keys: Mapping[str, str],
    stored: Mapping[str, _FileState],
    universe: Sequence[str],
    roots: Sequence[str],
) -> str | None:
    """Say why the checkpoint cannot be narrowed against, or `None` when it can."""
    if checkpoint is None:
        return 'there is no checkpoint from an earlier passing run'

    stale = sorted(name for name in {*keys, *checkpoint['keys']} if keys.get(name) != checkpoint['keys'].get(name))
    if stale:
        return f'{" and ".join(f"`{name}`" for name in stale)} changed since the last passing run'

    # A file's stored imports are paths, resolved when it was last parsed. Adding or moving
    # a file can point an unchanged import at a different one, and nothing in that file's
    # own content would say so.
    was = _module_map(sorted(stored), roots)
    now = _module_map(universe, roots)
    moved = sorted(name for name, path in was.items() if now.get(name, path) != path)
    if moved:
        return f'`{moved[0]}` now resolves to a different file'
    shadowing = sorted(name for name in now.keys() - was.keys() if '.' not in name)
    if shadowing:
        return f'`{shadowing[0]}` is a new top-level module and can shadow an installed one'
    return None


def _affected(
    changed: Sequence[str],
    deleted: Sequence[str],
    stored: Mapping[str, _FileState],
    imports: Mapping[str, list[str]],
    checkable: Sequence[str],
) -> list[str]:
    """Return the checkable files that changed, plus those transitively importing a changed or deleted one."""
    # Both graphs count: a deleted file has importers only in the stored one, and a file
    # that has just stopped importing another still has to be re-checked for having done so.
    importers: defaultdict[str, set[str]] = defaultdict(set)
    edges = chain(((path, state['imports']) for path, state in stored.items()), imports.items())
    for source, targets in edges:
        for target in targets:
            importers[target].add(source)

    reached = {*changed, *deleted}
    queue = deque(reached)
    while queue:
        for importer in importers[queue.popleft()]:
            if importer not in reached:
                reached.add(importer)
                queue.append(importer)
    return sorted(reached.intersection(checkable))


def _parse_imports(paths: Sequence[str], universe: Sequence[str], roots: Sequence[str]) -> dict[str, list[str]]:
    modules = _module_map(universe, roots)
    return {path: _imports_of(path, modules, roots) for path in paths}


def _imports_of(path: str, modules: Mapping[str, str], roots: Sequence[str]) -> list[str]:
    """Return the first-party files `path` imports, wherever in the file the import appears."""
    try:
        tree = ast.parse(Path(path).read_bytes(), filename=path)
    except (SyntaxError, ValueError):
        # Pyright reports the unparsable file itself; an empty import list keeps it a leaf
        # until it parses again.
        return []

    package = _package_of(path, roots)
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                targets.update(_prefixes(alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = _absolute_module(node, package)
            if module is None:
                continue
            targets.update(_prefixes(module))
            # `from a.b import c` reads `a/b/c.py` where that module exists, and an
            # attribute of `a/b` where it does not.
            targets.update(f'{module}.{alias.name}' for alias in node.names if alias.name != '*')

    return sorted({modules[target] for target in targets if target in modules} - {path})


def _absolute_module(node: ast.ImportFrom, package: str) -> str | None:
    if not node.level:
        return node.module or None
    parts = package.split('.') if package else []
    ascend = node.level - 1
    if ascend > len(parts):
        return None
    parts = parts[: len(parts) - ascend]
    if node.module:
        parts = [*parts, node.module]
    return '.'.join(parts) or None


def _prefixes(module: str) -> Iterator[str]:
    """Yield every module along a dotted path: `a`, then `a.b`, then `a.b.c`."""
    parts = module.split('.')
    for index in range(len(parts)):
        yield '.'.join(parts[: index + 1])


def _module_map(paths: Iterable[str], roots: Sequence[str]) -> dict[str, str]:
    """Map each dotted module name onto the file that defines it."""
    modules: dict[str, str] = {}
    for path in paths:
        for name in _module_names(path, roots):
            known = modules.get(name)
            if known is None or (known.endswith('.py') and path.endswith('.pyi')):
                modules[name] = path
    return modules


def _module_names(path: str, roots: Sequence[str]) -> Iterator[str]:
    """Yield the names `path` answers to, most specific root first."""
    for root in roots:
        prefix = f'{root}/' if root else ''
        if not path.startswith(prefix):
            continue
        parts = path[len(prefix) :].rsplit('.', 1)[0].split('/')
        if parts[-1] == '__init__':
            parts = parts[:-1]
        if parts:
            yield '.'.join(parts)


def _package_of(path: str, roots: Sequence[str]) -> str:
    name = next(_module_names(path, roots), '')
    if Path(path).stem == '__init__':
        return name
    return name.rpartition('.')[0]


def _load_project() -> _Project | None:
    """Read Pyright's file list, or `None` when this script cannot reproduce it."""
    # 3.11+, which is why `main` turns an older interpreter away before reaching here.
    import tomllib

    pyproject = _PYPROJECT_ADAPTER.validate_python(tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8')))
    tools = pyproject.get('tool') or _Tools()
    pyright = tools.get('pyright') or _PyrightSettings()
    workspace = (tools.get('uv') or _Uv()).get('workspace') or _Workspace()

    # A `pyrightconfig.json` takes precedence over `[tool.pyright]`, and `extends` names a
    # file this does not read. Either way pyproject.toml has stopped describing what Pyright
    # checks, so there is nothing here to narrow against.
    if pyright.get('extends') or Path('pyrightconfig.json').exists():
        return None

    include = pyright.get('include') or []
    exclude = pyright.get('exclude') or []
    members = workspace.get('members') or []
    environments = pyright.get('executionEnvironments') or []
    extra_paths = [path for environment in environments for path in environment.get('extraPaths') or []]
    # An absent `include` means Pyright reads the whole project, and a glob is a pattern
    # this script does not expand; either way it cannot say which files Pyright would read.
    if not include or any(
        _GLOB_CHARACTERS.intersection(entry) for entry in chain(include, exclude, members, extra_paths)
    ):
        return None

    roots = sorted({'', *members, *extra_paths}, key=len, reverse=True)
    return _Project(tuple(include), tuple(exclude), tuple(roots))


def _tracked_files() -> list[str]:
    """Return every tracked Python file that is on disk.

    The import graph is a property of the source tree, not of Pyright's file list: a file
    Pyright reports nothing about is still read for whoever imports it, so it belongs in the
    graph even though it never belongs on the command line. A narrowed run never sees an
    untracked file; a fallback run does, because Pyright walks the project itself.
    """
    listed = _git('ls-files', '-z').split('\0')
    # A file removed from the working tree but not yet from the index is still listed, and
    # counts as deleted: handing Pyright a path that is not there would fail the whole run.
    return sorted(path for path in listed if path.endswith(('.py', '.pyi')) and Path(path).is_file())


def _is_checked(path: str, project: _Project) -> bool:
    """Say whether Pyright reports diagnostics for `path`, which is what makes it worth checking."""
    if any(_covers(entry, path) for entry in project.exclude):
        return False
    for entry in project.include:
        if not _covers(entry, path):
            continue
        if project.exclude:
            return True
        swept_up = path[len(entry) :].strip('/').split('/')
        if not any(part.startswith('.') or part in _SKIPPED_DIRECTORIES for part in swept_up):
            return True
    return False


def _covers(entry: str, path: str) -> bool:
    return path == entry or path.startswith(f'{entry}/')


def _invalidation_keys(requested_version: str) -> dict[str, str]:
    """Return what has to hold for the checkpoint to still describe a passing run."""
    keys = {
        'pyright': importlib.metadata.version('pyright'),
        'python': platform.python_version(),
        'PYRIGHT_PYTHON': requested_version,
    }
    keys.update({name: _file_hash(name) for name in _CONFIGURATION_FILES})
    return keys


def _file_hash(path: str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return ''


def _load_checkpoint(path: Path) -> _Checkpoint | None:
    try:
        return _CHECKPOINT_ADAPTER.validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return None


def _git(*arguments: str) -> str:
    return subprocess.run(['git', *arguments], capture_output=True, text=True, check=True).stdout


if __name__ == '__main__':
    sys.exit(main())
