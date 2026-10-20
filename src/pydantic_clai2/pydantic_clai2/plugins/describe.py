"""A plugin's one-paragraph description, read from its docstring without running its code."""

import ast
import sys
from pathlib import Path


def describe(factory: str, path: Path | None) -> str:
    """The first paragraph of the factory class's docstring, else the module's; empty when there is none.

    `factory` and `path` are a declaration's: `module:Class`, and the drop-in file if it is one. The source
    is parsed, never imported, so an off or unapproved plugin runs no code to describe itself.
    """
    source = _source(factory, path)
    if source is None:
        return ''
    try:
        tree = ast.parse(source.read_bytes())
    except (OSError, SyntaxError, ValueError):
        return ''
    attr = factory.partition(':')[2]
    owner = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == attr), None)
    doc = (ast.get_docstring(owner) if owner else None) or ast.get_docstring(tree) or ''
    text = ' '.join(doc.split('\n\n')[0].split()).replace('`', '')
    return ''.join(char for char in text if char.isprintable())


def _source(factory: str, path: Path | None) -> Path | None:
    if path is not None:
        return path
    # Only inspect ordinary source files. Import finders may execute parent packages or custom loaders.
    # Zip imports and custom import hooks deliberately get no description.
    parts = factory.partition(':')[0].split('.')
    for directory in sys.path:
        module = Path(directory).joinpath(*parts)
        for source in (module / '__init__.py', module.with_suffix('.py')):
            if source.is_file():
                return source
    return None
