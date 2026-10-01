"""A plugin's one-paragraph description, read from its docstring without running its code."""

import ast
import importlib.util
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
    if entry.project:
        return None  # Finding a module imports its parent packages; a project plugin must not run before approval.
    try:
        spec = importlib.util.find_spec(entry.declaration.factory.partition(':')[0])
    except (ImportError, ValueError):
        return None
    return Path(spec.origin) if spec is not None and spec.origin else None
