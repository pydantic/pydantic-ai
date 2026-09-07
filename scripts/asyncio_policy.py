"""Collect asyncio references and lint exceptions for the migration policy."""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def asyncio_sites(source: str) -> Counter[str]:
    """Count imports, references by scope, and relevant lint suppressions."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    aliases: dict[str, str] = {}
    sites: Counter[str] = Counter()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'asyncio' or alias.name.startswith('asyncio.'):
                    aliases[alias.asname or 'asyncio'] = alias.name if alias.asname else 'asyncio'
                    sites[f'import:{ast.unparse(node)}'] += 1
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == 'asyncio' or node.module.startswith('asyncio.'):
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f'{node.module}.{alias.name}'
                sites[f'import:{ast.unparse(node)}'] += 1

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.Attribute) and parent.value is node:
            continue
        root = node
        attributes: list[str] = []
        while isinstance(root, ast.Attribute):
            attributes.append(root.attr)
            root = root.value
        if not isinstance(root, ast.Name) or root.id not in aliases or not isinstance(root.ctx, ast.Load):
            continue
        scopes: list[str] = []
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                scopes.append(parent.name)
            parent = parents.get(parent)
        scope = '.'.join(reversed(scopes)) or '<module>'
        symbol = '.'.join([aliases[root.id], *reversed(attributes)])
        sites[f'{scope}:{symbol}'] += 1

    sites.update(lint_suppressions(source))
    return sites


def lint_suppressions(source: str) -> Counter[str]:
    """Count source comments that can disable the asyncio import rule."""
    sites: Counter[str] = Counter()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        match = re.search(
            r'#\s*(?:(?:ruff|flake8):\s*)?noqa\b(?:\s*:\s*([A-Z]+\d+(?:\s*,\s*[A-Z]+\d+)*))?',
            token.string,
            re.IGNORECASE,
        )
        if match and (match[1] is None or 'TID251' in re.split(r'\s*,\s*', match[1].upper())):
            sites[f'suppression:{token.string}'] += 1
    return sites


def config_suppressions(path: Path) -> Counter[str]:
    """Find Ruff settings which disable the asyncio import rule."""
    config = tomllib.loads(path.read_text(encoding='utf-8'))
    if path.name == 'pyproject.toml':
        config = config.get('tool', {}).get('ruff', {})
    sites: Counter[str] = Counter()
    for lint in (config, config.get('lint', {})):
        for setting in ('per-file-ignores', 'extend-per-file-ignores'):
            ignores = TypeAdapter(dict[str, list[str]]).validate_python(lint.get(setting, {}))
            for pattern, rules in ignores.items():
                for rule in rules:
                    if 'TID251'.startswith(rule) or rule == 'ALL':
                        sites[f'config:{setting}:{pattern}:{rule}'] += 1
        for setting in ('ignore', 'extend-ignore'):
            rules = TypeAdapter(list[str]).validate_python(lint.get(setting, []))
            for rule in rules:
                if 'TID251'.startswith(rule) or rule == 'ALL':
                    sites[f'config:{setting}:{rule}'] += 1
    return sites
