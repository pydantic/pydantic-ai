"""Reload CLAI's modules before rebuilding the shell, without reloading its dependencies."""

import ast
import importlib
import importlib.util
import operator
import os
import sys
import tokenize
from collections.abc import Callable, Iterable
from graphlib import TopologicalSorter
from pathlib import Path
from types import ModuleType
from typing import TypeAlias, TypeVar

import pydantic_clai2

T = TypeVar('T')
_GuardValue: TypeAlias = str | int | tuple[str | int, ...]


class _Unknown:
    """A guard value that cannot be determined without executing source."""


_UNKNOWN = _Unknown()
_COMPARISONS: dict[type[ast.cmpop], Callable[[_GuardValue, _GuardValue], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_IMPORT_VALUES: dict[str, _GuardValue] = {
    'sys.platform': sys.platform,
    'sys.version_info': tuple(sys.version_info),
    'os.name': os.name,
    'typing.TYPE_CHECKING': False,
}


class _Imports(ast.NodeVisitor):
    """Collect module-scope imports without treating lazy or type-only imports as eager dependencies."""

    def __init__(self, *, package: str, name: str) -> None:
        self.package = package
        self.names: set[str] = set()
        self.values: dict[str, _GuardValue] = {'__name__': name, '__package__': package}

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.name for alias in node.names)
        for alias in node.names:
            imported = alias.name if alias.asname else alias.name.partition('.')[0]
            bound = alias.asname or imported
            self._forget(bound)
            for key, value in _IMPORT_VALUES.items():
                if key.startswith(imported + '.'):
                    self.values[key.replace(imported, bound, 1)] = value

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        name = importlib.util.resolve_name('.' * node.level + (node.module or ''), self.package)
        self.names.add(name)
        self.names.update(f'{name}.{alias.name}' for alias in node.names)
        for alias in node.names:
            self._forget(alias.asname or alias.name)
            key = f'{name}.{alias.name}'
            if key in _IMPORT_VALUES:
                self.values[alias.asname or alias.name] = _IMPORT_VALUES[key]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class bodies execute eagerly, but their bindings do not escape the class scope.
        before = self.values.copy()
        for statement in node.body:
            self.visit(statement)
        self.values = before
        self._forget(node.name)

    def _forget(self, name: str) -> None:
        self.values = {
            key: value for key, value in self.values.items() if key != name and not key.startswith(name + '.')
        }

    def _forget_target(self, target: ast.expr) -> None:
        for child in ast.walk(target):
            if isinstance(child, ast.Name):
                self._forget(child.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._forget_target(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._forget_target(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._forget_target(node.target)

    def _value(self, node: ast.expr) -> _GuardValue | _Unknown:
        key = ast.unparse(node)
        if key in self.values:
            return self.values[key]
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Tuple):
            values: list[str | int] = []
            for item in node.elts:
                value = self._value(item)
                if not isinstance(value, (str, int)):
                    return _UNKNOWN
                values.append(value)
            return tuple(values)
        return _UNKNOWN

    def _condition(self, node: ast.expr) -> bool | None:
        value = self._value(node)
        if not isinstance(value, _Unknown):
            return bool(value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            condition = self._condition(node.operand)
            return None if condition is None else not condition
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            compare = _COMPARISONS.get(type(node.ops[0]))
            left, right = self._value(node.left), self._value(node.comparators[0])
            if compare is not None and not isinstance(left, _Unknown) and not isinstance(right, _Unknown):
                try:
                    return compare(left, right)
                except TypeError:
                    return None
        return None

    def visit_If(self, node: ast.If) -> None:
        condition = self._condition(node.test)
        if condition is None:
            before = self.values.copy()
            for statement in node.body:
                self.visit(statement)
            after_body = self.values
            self.values = before
            for statement in node.orelse:
                self.visit(statement)
            self.values = {key: value for key, value in self.values.items() if after_body.get(key, _UNKNOWN) == value}
        else:
            for statement in node.body if condition else node.orelse:
                self.visit(statement)


def _reload_plan(names: Iterable[str]) -> tuple[tuple[str, Path], ...]:
    sources: dict[str, Path] = {}
    for directory in pydantic_clai2.__path__:
        root = Path(directory)
        for path in sorted(root.rglob('*.py')):
            parts = path.relative_to(root).with_suffix('').parts
            if parts[-1] == '__init__':
                parts = parts[:-1]
            sources.setdefault('.'.join(('pydantic_clai2', *parts)), path)

    dependencies: dict[str, set[str]] = {}
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in dependencies:
            continue
        path = sources[name]
        package = name if path.name == '__init__.py' else name.rpartition('.')[0]
        imports = _Imports(package=package, name=name)
        with tokenize.open(path) as source:
            imports.visit(ast.parse(source.read(), filename=str(path)))
        targets = {target for target in imports.names if target in sources and target != name}
        # Importing a submodule can also execute a previously unloaded package initializer.
        for target in tuple(targets):
            parent = target.rpartition('.')[0]
            while parent in sources:
                if parent != name:
                    targets.add(parent)
                parent = parent.rpartition('.')[0]
        dependencies[name] = targets
        pending.extend(targets - dependencies.keys())

    return tuple((name, sources[name]) for name in TopologicalSorter(dependencies).static_order())


def reload_clai(build: Callable[[], T]) -> T:
    """Order reloads from current source imports; restore bindings if reload or rebuild fails."""
    modules = {
        name: module
        for name, module in sys.modules.copy().items()
        if name == 'pydantic_clai2' or name.startswith('pydantic_clai2.')
    }
    snapshots: dict[ModuleType, dict[str, object]] = {module: vars(module).copy() for module in modules.values()}
    ordered = _reload_plan(modules)
    importlib.invalidate_caches()
    try:
        # Invalidate newly referenced modules too, before an importer can load their bytecode.
        for _, path in ordered:
            Path(importlib.util.cache_from_source(str(path))).unlink(missing_ok=True)
        for name, _ in ordered:
            if name in modules:
                importlib.reload(modules[name])
        return build()
    except BaseException:
        for module, namespace in snapshots.items():
            vars(module).clear()
            vars(module).update(namespace)
        for name in sys.modules.copy():
            if name.startswith('pydantic_clai2.') and name not in modules:
                del sys.modules[name]
        raise
