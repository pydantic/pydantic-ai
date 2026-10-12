"""A plugin's one-paragraph description, read from its docstring without running its code."""

import ast
import sys
from pathlib import Path

from pydantic_clai2.plugins import DepsT
from pydantic_clai2.plugins.loader import PluginEntry


def describe(entry: PluginEntry[DepsT]) -> str:
    """The first paragraph of the factory class's docstring, else the module's; empty when there is none.

    The source is parsed, never imported, so an off or unapproved plugin runs no code to describe itself.
    """
    source = _source(entry)
    if source is None:
        return ''
    try:
        tree = ast.parse(source.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return ''
    attr = entry.declaration.factory.partition(':')[2]
    owner = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == attr), None)
    doc = (ast.get_docstring(owner) if owner else None) or ast.get_docstring(tree) or ''
    text = ' '.join(doc.split('\n\n')[0].split()).replace('`', '')
    return ''.join(char for char in text if char.isprintable())


def _source(entry: PluginEntry[DepsT]) -> Path | None:
    if entry.path is not None:
        return entry.path
    # Only inspect ordinary source files. Import finders may execute parent packages or custom loaders.
    # Zip imports and custom import hooks deliberately get no description.
    parts = entry.declaration.factory.partition(':')[0].split('.')
    for directory in sys.path:
        module = Path(directory).joinpath(*parts)
        for source in (module / '__init__.py', module.with_suffix('.py')):
            if source.is_file():
                return source
    return None
