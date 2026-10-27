"""A plugin's one-paragraph description, read from its docstring without running its code."""

import ast
import os
import sys
from functools import lru_cache
from pathlib import Path


def describe(factory: str, path: Path | None) -> str:
    """The first paragraph of the factory class's docstring, else the module's; empty when there is none.

    `factory` and `path` are a declaration's: `module:Class`, and the drop-in file if it is one. The source
    is parsed, never imported, so an off or unapproved plugin runs no code to describe itself. A parse is
    cached until the file changes, as `/plugins` is listed on every web UI refresh.
    """
    source = path if path is not None else _source(factory.partition(':')[0])
    if source is None:
        return ''
    try:
        stat = source.stat()
    except OSError:
        return ''
    return _docstring(source, stat.st_mtime_ns, stat.st_size, factory.partition(':')[2])


@lru_cache(maxsize=256)
def _docstring(source: Path, mtime_ns: int, size: int, attr: str) -> str:
    """The description in `source` as it was at `mtime_ns` and `size`, which only key the cache."""
    try:
        tree = ast.parse(source.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return ''
    owner = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == attr), None)
    doc = (ast.get_docstring(owner) if owner else None) or ast.get_docstring(tree) or ''
    text = ' '.join(doc.split('\n\n')[0].split()).replace('`', '')
    return ''.join(char for char in text if char.isprintable())


def _source(module: str) -> Path | None:
    """The source file of `module` on the import path, as the import system would find it."""
    # Only inspect ordinary source files. Import finders may execute parent packages or custom loaders.
    # Zip imports and custom import hooks deliberately get no description.
    # `os.path` rather than `Path`: listing plugins walks this for each one on every web UI refresh.
    relative = os.path.join(*module.split('.'))
    for directory in sys.path:
        candidate = os.path.join(directory, relative)
        for source in (os.path.join(candidate, '__init__.py'), f'{candidate}.py'):
            if os.path.isfile(source):
                return Path(source)
    return None
