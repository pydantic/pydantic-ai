#!/usr/bin/env python
"""Generate the durable-exec contract manifest: the core definitions durable engines depend on.

Durable engines import core symbols, override core methods, read core attributes and project core
dataclasses across their durable boundaries. A core change to one of those definitions can break an
engine without touching any file under `durable_exec/`, so a review routed by path never sees it.
The manifest records, per core definition, how the engines depend on it, so a change to a core
definition can be matched against it instead.

The walk is pure `ast` over the engine package roots in `ENGINES`; nothing is imported. "Core" is
`pydantic_ai` outside `pydantic_ai.durable_exec`. Each engine site depends on a core definition in
one of five ways:

- `import`: the engine names a core class, function or variable, by import or as a module attribute,
  and uses the name at runtime: a call, an attribute read, `isinstance`, a base class, a default, a
  decorator.
- `annotation_only`: the engine module names the definition only in annotation positions: parameter,
  return and variable annotations, string annotations, and `if TYPE_CHECKING:` blocks.
- `override`: an engine class redefines a member of a core class in its MRO.
- `attribute_read`: an engine reads or assigns a core member, passes it as a constructor or
  `dataclasses.replace` keyword, or reflects over a core dataclass's fields. The member is resolved
  from annotations: `ctx: RunContext[...]` makes `ctx.usage` a read of `RunContext.usage`, and
  `self.wrapped` on a `WrapperModel` subclass is a `Model`, so `self.wrapped.profile` reads
  `Model.profile`.
- `projected_field`: an engine class declares a field of the same name as a member of the core class
  one of its methods takes as a parameter, as `ModelRequestContextProjection.from_context` does.

Inside a function, a name gets its classes from a parameter or variable annotation, an assignment,
a `for` loop or comprehension over an iterable whose annotation or return annotation is known, or
`isinstance` narrowing, and a module-level type alias stands for the type it names. A name bound any
other way, such as a `with ... as` target, a tuple-unpacked loop variable or a module-level loop
variable, has no class, so a member read through it records nothing. An access on a typed name that
the walk cannot resolve to a member, such as `getattr(obj, name)` or `obj.__dict__[name]`, falls back
to the class, unless the class has a base outside the walked packages (`BaseModel`, `Exception`),
which may define the member: those accesses record nothing. Reflection over a whole instance is read
only through `fields()` and `__dataclass_fields__`; `vars(obj)` and iterating `obj.__dict__` record
nothing. A name a module serves only through a module-level `__getattr__` (PEP 562) resolves to
nothing, so a dependency on it records nothing. The classes `_ClassInfo.member_level_only` names
never get a class-level entry: nearly every change to core touches them, so a class-level entry would
match every change.

Only the packages in `ENGINES` are walked. A Harness capability that reads durability state, such as
`RunContext.in_durable_context`, without subclassing `BaseDurabilityCapability` is not one, so its
dependencies are not recorded. TODO: https://github.com/pydantic/pydantic-ai/issues/11198

The manifest maps `core_file -> qualname -> {kinds, engines, sites}`. A change to a core definition
matches an entry when the changed definition's qualname is the entry's qualname or nested under it:
an entry for `Model` matches a change to `Model.request`; an entry for `RunContext.usage` matches
only that field. A matched entry gates review only when `gates` holds for it: an engine that names a
class only in a signature does not depend on it at runtime, so an entry whose only kind is
`annotation_only` stays in the manifest but does not gate.

`--touched` prints the gating entries a pull request touches, in the manifest's shape, reading the
pull request through `gh api` and only parsing what it reads (`touched_entries` has the rules).
`.github/workflows/bots.yml` labels a pull request that touches any of them
`needs-durable-exec-review`, and `scripts/gather-pydantic-ai-review-context.sh` hands them to CI
Review. Both run this file and the manifest as the base branch has them, never the pull request's
copies.

Usage:
    uv run python scripts/durable_exec_contract.py          # exit 1 if the manifest is stale
    uv run python scripts/durable_exec_contract.py --write  # regenerate it
    uv run python scripts/durable_exec_contract.py --touched REPO PR BASE_SHA HEAD_SHA
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import re
import subprocess
import sys
import tokenize
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = 'scripts/durable_exec_contract.json'

# Source roots, relative to the repository root, that `_Index` searches for the modules the walk reads.
SOURCE_ROOTS = ('pydantic_ai_slim', 'src/pydantic_ai_harness')

# Engine name -> glob of its modules, relative to the repository root. An engine is a package that
# defines a `BaseDurabilityCapability` subclass; `durable_exec` is the shared base every engine builds
# on (`BaseDurabilityCapability`, the operation backends, the projections).
ENGINES: dict[str, str] = {
    'durable_exec': 'pydantic_ai_slim/pydantic_ai/durable_exec/*.py',
    'temporal': 'pydantic_ai_slim/pydantic_ai/durable_exec/temporal/**/*.py',
    'dbos': 'pydantic_ai_slim/pydantic_ai/durable_exec/dbos/**/*.py',
    'prefect': 'pydantic_ai_slim/pydantic_ai/durable_exec/prefect/**/*.py',
    'render': 'src/pydantic_ai_harness/pydantic_ai_harness/render/**/*.py',
    'absurd': 'src/pydantic_ai_harness/pydantic_ai_harness/absurd/**/*.py',
    'pgtask': 'src/pydantic_ai_harness/pydantic_ai_harness/pgtask/**/*.py',
    'aws_lambda': 'src/pydantic_ai_harness/pydantic_ai_harness/aws_lambda/**/*.py',
    'azure_durable_task': 'src/pydantic_ai_harness/pydantic_ai_harness/azure_durable_task/**/*.py',
}

# Annotation constructors whose arguments, not themselves, are the annotated type.
_TYPE_WRAPPERS = frozenset(
    {'Annotated', 'ClassVar', 'Final', 'NotRequired', 'Optional', 'ReadOnly', 'Required', 'Type', 'Union', 'type'}
)

# What `--touched` accepts before it makes any request: a pull request's repository, number and commits, and the
# manifest's core files, which are the only paths it fetches.
_REPOSITORY = re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+')
_PULL_REQUEST = re.compile(r'[1-9][0-9]*')
_COMMIT = re.compile(r'[0-9a-f]{40}')
_CORE_FILE = re.compile(r'(?:[A-Za-z0-9_]+/)+[A-Za-z0-9_]+\.py')

_HUNK_HEADER = re.compile(r'@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@')

Kind = Literal['import', 'annotation_only', 'override', 'attribute_read', 'projected_field']
FileStatus = Literal['added', 'removed', 'modified', 'renamed', 'copied', 'changed', 'unchanged']
Manifest = dict[str, dict[str, dict[str, list[str]]]]
_EntrySets = dict[str, dict[str, dict[str, set[str]]]]
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
ImportTarget = tuple[str, str | None]
"""The module an import binds a name to, and the name in it, or `None` for the module itself."""


@dataclass(eq=False)
class _Member:
    """A name a class defines, with what reading it yields."""

    annotation: ast.expr | None
    """The type of the attribute when read: a field's annotation or a property's return annotation."""
    returns: ast.expr | None = None
    """A method's return annotation, which is what calling the attribute yields."""
    is_field: bool = False
    """Whether this is a class-body annotated field."""
    in_body: bool = True
    """Whether the class body defines it, rather than a `self.name = ...` assignment in a method."""


@dataclass(eq=False)
class _Module:
    """A parsed module and its top-level names."""

    name: str
    path: str
    is_package: bool
    tree: ast.Module
    definitions: dict[str, ast.stmt] = field(default_factory=dict[str, ast.stmt])
    imports: dict[str, ImportTarget] = field(default_factory=dict[str, ImportTarget])


@dataclass(eq=False)
class _ClassInfo:
    """A class and the members its body and methods define."""

    module: _Module
    qualname: str
    node: ast.ClassDef
    members: dict[str, _Member] = field(default_factory=dict[str, _Member])

    @property
    def member_level_only(self) -> bool:
        return (self.module.name, self.qualname) in {
            ('pydantic_ai._run_context', 'RunContext'),
            ('pydantic_ai.models', 'ModelRequestContext'),
            ('pydantic_ai.models', 'ModelRequestParameters'),
            ('pydantic_ai.agent', 'Agent'),
        }


class _Ancestry(NamedTuple):
    """Classes found by following a class's bases, and whether every base on the way resolved."""

    classes: list[_ClassInfo]
    closed: bool


@dataclass(frozen=True)
class _ModuleValue:
    """A module-level function or variable, the other definitions an import can name."""

    module: _Module
    name: str
    returns: ast.expr | None
    value: ast.expr | None = None
    """A variable's assigned value, which an annotation naming the variable as a type alias stands for."""
    annotation: ast.expr | None = None
    """A variable's annotation, which is what reading the variable yields."""


Symbol = _Module | _ClassInfo | _ModuleValue


class _Index:
    """Lazily parsed modules under `SOURCE_ROOTS`, and name resolution across them."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._modules: dict[str, _Module | None] = {}
        self._classes: dict[int, _ClassInfo] = {}

    def module(self, name: str) -> _Module | None:
        if name in self._modules:
            return self._modules[name]
        self._modules[name] = None
        relative = Path(*name.split('.'))
        for source_root in SOURCE_ROOTS:
            for path, is_package in ((relative / '__init__.py', True), (relative.with_suffix('.py'), False)):
                file = self.root / source_root / path
                if not file.is_file():
                    continue
                module = self._modules[name] = _Module(
                    name,
                    (Path(source_root) / path).as_posix(),
                    is_package,
                    ast.parse(file.read_text(encoding='utf-8'), filename=str(file)),
                )
                self.bind_top_level(module)
                self.bind_local_imports(module)
                return module
        return None

    def bind_top_level(self, module: _Module) -> None:
        """The names the module's top-level statements define and import."""
        for statement in _top_level(module.tree.body):
            match statement:
                case ast.ClassDef(name=bound) | ast.FunctionDef(name=bound) | ast.AsyncFunctionDef(name=bound):
                    module.definitions[bound] = statement
                case ast.Assign(targets=targets):
                    for target in targets:
                        if isinstance(target, ast.Name):
                            module.definitions[target.id] = statement
                case ast.AnnAssign(target=ast.Name(id=bound)):
                    module.definitions[bound] = statement
                case ast.Import() | ast.ImportFrom():
                    module.imports.update(self.import_targets(module, statement))
                case _:
                    pass

    def bind_local_imports(self, module: _Module) -> None:
        """A function-local import binds module-wide, unless the module binds the name itself."""
        for node in ast.walk(module.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for bound, target in self.import_targets(module, node):
                    module.imports.setdefault(bound, target)

    def import_targets(
        self, module: _Module, statement: ast.Import | ast.ImportFrom
    ) -> Iterator[tuple[str, ImportTarget]]:
        """Each name an import statement binds, with the module and name it binds to."""
        if isinstance(statement, ast.ImportFrom):
            base = self.import_base(module, statement)
            for alias in statement.names:
                yield alias.asname or alias.name, (base, alias.name)
            return
        for alias in statement.names:
            top = alias.name.split('.')[0]
            yield (alias.asname, (alias.name, None)) if alias.asname else (top, (top, None))

    @staticmethod
    def import_base(module: _Module, statement: ast.ImportFrom) -> str:
        """The absolute module a `from ... import` statement imports from."""
        if not statement.level:
            return statement.module or ''
        return importlib.util.resolve_name(
            '.' * statement.level + (statement.module or ''),
            module.name if module.is_package else module.name.rpartition('.')[0],
        )

    def class_info(self, module: _Module, node: ast.ClassDef, qualname: str) -> _ClassInfo:
        if (info := self._classes.get(id(node))) is None:
            info = self._classes[id(node)] = _ClassInfo(module, qualname, node)
            _collect_body_members(info)
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _collect_instance_attributes(info.members, method)
        return info

    def resolve(self, module: _Module, name: str, seen: frozenset[tuple[str, str]] = frozenset()) -> Symbol | None:
        """What `name` refers to at the top level of `module`, following re-exports."""
        if (module.name, name) in seen:
            return None
        statement = module.definitions.get(name)
        match statement:
            case ast.ClassDef():
                return self.class_info(module, statement, name)
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                return _ModuleValue(module, name, statement.returns)
            case ast.AnnAssign(annotation=annotation, value=value):
                return _ModuleValue(module, name, None, value, annotation)
            case ast.Assign(value=value):
                return _ModuleValue(module, name, None, value)
            case _:
                pass
        if (target := module.imports.get(name)) is None:
            return self.module(f'{module.name}.{name}')
        target_module, target_name = target
        if target_name is None:
            return self.module(target_module)
        if target_module == module.name:
            # `from . import name` in a package's `__init__` imports its submodule.
            return self.module(f'{module.name}.{target_name}')
        if (source := self.module(target_module)) is None:
            return None
        return self.resolve(source, target_name, seen | {(module.name, name)})

    def symbol(self, module: _Module, expr: ast.expr) -> Symbol | None:
        """What a dotted name expression refers to at the top level of `module`."""
        match expr:
            case ast.Name(id=name):
                return self.resolve(module, name)
            case ast.Attribute(value=value, attr=attr):
                base = self.symbol(module, value)
                return self.resolve(base, attr) if isinstance(base, _Module) else None
            case _:
                return None

    def annotation_classes(self, module: _Module, annotation: ast.expr | None) -> frozenset[_ClassInfo]:
        """The classes an annotation, written in `module`, says a value is an instance of."""
        match annotation:
            case ast.Constant(value=str(source)):
                return self.annotation_classes(module, _parse_annotation(source))
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return self.annotation_classes(module, left) | self.annotation_classes(module, right)
            case ast.Subscript(value=value, slice=index):
                if _last_name(value) not in _TYPE_WRAPPERS:
                    return self.annotation_classes(module, value)
                arguments = index.elts if isinstance(index, ast.Tuple) else [index]
                if _last_name(value) == 'Annotated':
                    arguments = arguments[:1]
                return frozenset[_ClassInfo]().union(
                    *(self.annotation_classes(module, argument) for argument in arguments)
                )
            case ast.Name() | ast.Attribute():
                match self.symbol(module, annotation):
                    case _ClassInfo() as info:
                        return frozenset({info})
                    case _ModuleValue(module=alias_module, value=ast.expr() as value):
                        return self.annotation_classes(alias_module, value)
                    case _:
                        return frozenset()
            case _:
                return frozenset()

    def element_classes(self, module: _Module, annotation: ast.expr | None) -> frozenset[_ClassInfo]:
        """The classes the items of an iterable with this annotation, written in `module`, are instances of."""
        match annotation:
            case ast.Constant(value=str(source)):
                return self.element_classes(module, _parse_annotation(source))
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return self.element_classes(module, left) | self.element_classes(module, right)
            case ast.Subscript(value=value, slice=index):
                arguments = index.elts if isinstance(index, ast.Tuple) else [index]
                if _last_name(value) not in _TYPE_WRAPPERS:
                    return self.annotation_classes(module, arguments[0])
                if _last_name(value) == 'Annotated':
                    arguments = arguments[:1]
                return frozenset[_ClassInfo]().union(
                    *(self.element_classes(module, argument) for argument in arguments)
                )
            case _:
                return frozenset()

    def bases(self, info: _ClassInfo) -> _Ancestry:
        """The class's resolved direct bases, and whether every base resolved."""
        resolved: list[_ClassInfo] = []
        closed = True
        for base in info.node.bases:
            target = _strip_subscript(base)
            symbol = self.symbol(info.module, target)
            if isinstance(symbol, _ClassInfo):
                resolved.append(symbol)
            # These bases contribute no members a core class could be looked up through.
            elif _last_name(target) not in {'ABC', 'Generic', 'Protocol', 'object'}:
                closed = False
        return _Ancestry(resolved, closed)

    def mro(self, info: _ClassInfo) -> _Ancestry:
        """The class and its resolved ancestors, nearest first, and whether every ancestor resolved."""
        order: list[_ClassInfo] = []
        closed = True
        pending = [info]
        while pending:
            current = pending.pop(0)
            if current in order:
                continue
            order.append(current)
            bases = self.bases(current)
            closed = closed and bases.closed
            pending.extend(bases.classes)
        return _Ancestry(order, closed)

    def lookup(self, info: _ClassInfo, name: str) -> tuple[_ClassInfo, _Member] | None:
        """The nearest class in the MRO that defines `name`, and its member."""
        for ancestor in self.mro(info).classes:
            if (member := ancestor.members.get(name)) is not None:
                return ancestor, member
        return None


def _top_level(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Module-level statements, including those under `if TYPE_CHECKING:` and `try:` blocks."""
    for statement in body:
        match statement:
            case ast.If(body=if_body, orelse=orelse):
                yield from _top_level([*if_body, *orelse])
            case ast.Try(body=try_body, handlers=handlers, orelse=orelse, finalbody=finalbody):
                yield from _top_level(
                    [*try_body, *(handled for handler in handlers for handled in handler.body), *orelse, *finalbody]
                )
            case _:
                yield statement


def _parse_annotation(source: str) -> ast.expr | None:
    """The expression a string annotation spells, or `None` if it is not valid Python."""
    try:
        return ast.parse(source, mode='eval').body
    except SyntaxError:
        return None


def _last_name(expr: ast.expr) -> str | None:
    match expr:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return None


def _collect_body_members(info: _ClassInfo) -> None:
    """The fields, class variables, methods and properties the class body defines."""
    members = info.members
    for statement in info.node.body:
        match statement:
            case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation) if name != '_':
                members[name] = _Member(annotation, is_field=_last_name(_strip_subscript(annotation)) != 'ClassVar')
            case ast.Assign(targets=targets, value=value):
                for target in targets:
                    if isinstance(target, ast.Name):
                        members[target.id] = _Member(value.func if isinstance(value, ast.Call) else None)
            case ast.FunctionDef() | ast.AsyncFunctionDef() if statement.name not in members:
                if any(
                    _last_name(decorator) in {'property', 'cached_property'} for decorator in statement.decorator_list
                ):
                    members[statement.name] = _Member(statement.returns)
                else:
                    members[statement.name] = _Member(None, returns=statement.returns)
            case _:
                pass


def _instance_attribute_assignments(method: FunctionNode) -> Iterator[tuple[ast.AnnAssign | ast.Assign, str]]:
    """The assignments `method` makes to an attribute of its first parameter, each with the attribute's name."""
    arguments = [*method.args.posonlyargs, *method.args.args]
    if not arguments:
        return
    for node in ast.walk(method):
        match node:
            case (
                ast.AnnAssign(target=ast.Attribute(value=ast.Name(id=owner), attr=name))
                | ast.Assign(targets=[ast.Attribute(value=ast.Name(id=owner), attr=name)])
            ) if owner == arguments[0].arg:
                yield node, name
            case _:
                pass


def _collect_instance_attributes(members: dict[str, _Member], method: FunctionNode) -> None:
    """Attributes a method assigns on `self`, typed by their annotation or the annotated parameter assigned."""
    parameters = {argument.arg: argument.annotation for argument in [*method.args.posonlyargs, *method.args.args]}
    for assignment, name in _instance_attribute_assignments(method):
        match assignment:
            case ast.AnnAssign(annotation=annotation):
                annotation_or_value = annotation
            case ast.Assign(value=ast.Name(id=parameter)):
                annotation_or_value = parameters.get(parameter)
            case ast.Assign(value=ast.Call(func=func)):
                annotation_or_value = func
            case _:
                annotation_or_value = None
        if name not in members:
            members[name] = _Member(annotation_or_value, in_body=False)


def _annotation_nodes(tree: ast.Module) -> set[int]:
    """The ids of the nodes in annotation positions: annotations, return annotations and `if TYPE_CHECKING:` bodies."""
    annotations: list[ast.AST] = []
    for node in ast.walk(tree):
        match node:
            case ast.arg(annotation=ast.expr() as annotation) | ast.AnnAssign(annotation=annotation):
                annotations.append(annotation)
            case ast.FunctionDef(returns=ast.expr() as returns) | ast.AsyncFunctionDef(returns=ast.expr() as returns):
                annotations.append(returns)
            case ast.If(test=test, body=body) if _last_name(test) == 'TYPE_CHECKING':
                annotations.extend(body)
            case _:
                pass
    return {id(inner) for annotation in annotations for inner in ast.walk(annotation)}


def _runtime_names(tree: ast.Module, annotation_nodes: set[int]) -> set[str]:
    """The names the module loads outside annotation positions."""
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) not in annotation_nodes
    }


def _strip_subscript(expr: ast.expr) -> ast.expr:
    return expr.value if isinstance(expr, ast.Subscript) else expr


def _is_static(function: FunctionNode) -> bool:
    return any(_last_name(decorator) == 'staticmethod' for decorator in function.decorator_list)


@dataclass
class _Scope:
    qualname: str
    locals: dict[str, frozenset[_ClassInfo]] = field(default_factory=dict[str, frozenset[_ClassInfo]])
    owner: _ClassInfo | None = None
    """The class whose body this scope is, or whose method this scope is."""


class _EngineWalker(ast.NodeVisitor):
    """Records the core dependencies of one engine module."""

    def __init__(self, index: _Index, module: _Module, engine: str, entries: _EntrySets) -> None:
        self.index = index
        self.module = module
        self.engine = engine
        self.entries = entries
        self.scopes = [_Scope('<module>')]
        self.annotation_nodes = _annotation_nodes(module.tree)
        self.runtime_names = _runtime_names(module.tree, self.annotation_nodes)

    # --- Recording ------------------------------------------------------------------------------------------------

    def record(self, definer: _ClassInfo | _Module, qualname: str, kind: Kind, site: str | None = None) -> None:
        module = definer.module if isinstance(definer, _ClassInfo) else definer
        # The trailing dot makes `pydantic_ai.durable_exec` itself match the exclusion below.
        dotted = f'{module.name}.'
        if dotted.startswith('pydantic_ai.') and not dotted.startswith('pydantic_ai.durable_exec.'):
            entry = self.entries.setdefault(module.path, {}).setdefault(
                qualname, {'kinds': set[str](), 'engines': set[str](), 'sites': set[str]()}
            )
            entry['kinds'].add(kind)
            entry['engines'].add(self.engine)
            entry['sites'].add(f'{self.module.path}::{site or self.scopes[-1].qualname}')

    def record_symbol(self, symbol: Symbol | None, kind: Literal['import', 'annotation_only']) -> None:
        """An `import` or `annotation_only` dependency on a named core definition."""
        match symbol:
            case _ClassInfo() if not symbol.member_level_only:
                self.record(symbol, symbol.qualname, kind)
            case _ModuleValue():
                self.record(symbol.module, symbol.name, kind)
            case _:
                pass

    def record_member(self, expr: ast.expr, name: str) -> None:
        """An `attribute_read` of `name` on whatever `expr` is, falling back to its core classes."""
        classes = self.types(expr)
        found = [hit for info in classes if (hit := self.index.lookup(info, name)) is not None]
        for definer, _ in found:
            self.record(definer, f'{definer.qualname}.{name}', 'attribute_read')
        if not found and not (name.startswith('__') and name.endswith('__')):
            self.record_classes(classes)

    def record_named(self, owner: ast.expr, name: ast.expr) -> None:
        """A reflective access of the member `name` names on `owner`, at class level when the name is not a literal."""
        if isinstance(name, ast.Constant) and isinstance(name.value, str):
            self.record_member(owner, name.value)
        else:
            self.record_classes(self.types(owner))

    def record_keywords(self, owner: ast.expr, keywords: list[ast.keyword]) -> None:
        """Keyword arguments naming members of `owner`, at class level for a `**` mapping."""
        for keyword in keywords:
            if keyword.arg is not None:
                self.record_member(owner, keyword.arg)
            else:
                self.record_classes(self.types(owner))

    def record_classes(self, classes: frozenset[_ClassInfo]) -> None:
        """The class-level fallback for an access the walk cannot resolve to a member."""
        for info in classes:
            if not info.member_level_only and self.index.mro(info).closed:
                self.record(info, info.qualname, 'attribute_read')

    def record_fields(self, expr: ast.expr) -> None:
        """Reflection over a dataclass's fields depends on every field it has."""
        for info in self.types(expr):
            for ancestor in self.index.mro(info).classes:
                for name, member in ancestor.members.items():
                    if member.is_field:
                        self.record(ancestor, f'{ancestor.qualname}.{name}', 'attribute_read')

    # --- Types ----------------------------------------------------------------------------------------------------

    def types(self, expr: ast.expr) -> frozenset[_ClassInfo]:
        """The classes `expr` may be an instance of, or may be."""
        match expr:
            case ast.Name(id=name):
                for scope in reversed(self.scopes):
                    if name in scope.locals:
                        return scope.locals[name]
                symbol = self.index.resolve(self.module, name)
                return frozenset({symbol}) if isinstance(symbol, _ClassInfo) else frozenset()
            case ast.Attribute(value=value, attr=attr):
                return self.attribute_types(value, attr, returns=False)
            case ast.Call(func=ast.Name(id='super')):
                owner = next((scope.owner for scope in reversed(self.scopes) if scope.owner is not None), None)
                return frozenset(self.index.bases(owner).classes) if owner is not None else frozenset()
            case ast.Call(func=func, args=args) if _last_name(func) == 'cast' and args:
                return self.index.annotation_classes(self.module, args[0])
            case ast.Call(func=ast.Attribute(value=value, attr=attr)) if not isinstance(
                self.index.symbol(self.module, value), _Module
            ):
                return self.attribute_types(value, attr, returns=True)
            case ast.Call(func=func):
                symbol = self.index.symbol(self.module, func)
                if isinstance(symbol, _ModuleValue):
                    return self.index.annotation_classes(symbol.module, symbol.returns)
                return frozenset({symbol}) if isinstance(symbol, _ClassInfo) else frozenset()
            case ast.Await(value=value) | ast.NamedExpr(value=value):
                return self.types(value)
            case ast.BoolOp(values=values):
                return frozenset[_ClassInfo]().union(*(self.types(value) for value in values))
            case ast.IfExp(body=body, orelse=orelse):
                return self.types(body) | self.types(orelse)
            case _:
                return frozenset()

    def attribute_types(self, value: ast.expr, attr: str, *, returns: bool) -> frozenset[_ClassInfo]:
        symbol = self.index.symbol(self.module, value)
        if isinstance(symbol, _Module):
            resolved = self.index.resolve(symbol, attr)
            return frozenset({resolved}) if isinstance(resolved, _ClassInfo) else frozenset()
        result: frozenset[_ClassInfo] = frozenset()
        for definer, member in self.member_hits(value, attr):
            annotation = member.returns if returns else member.annotation
            result |= self.index.annotation_classes(definer.module, annotation)
        return result

    def member_hits(self, value: ast.expr, attr: str) -> Iterator[tuple[_ClassInfo, _Member]]:
        """The member `attr` names on each class `value` may be an instance of, with the class that defines it."""
        for info in self.types(value):
            if (found := self.index.lookup(info, attr)) is not None:
                yield found

    def element_types(self, iterable: ast.expr, annotated: Mapping[str, ast.expr | None]) -> frozenset[_ClassInfo]:
        """The classes the items of `iterable` are instances of, from the annotation of what it reads or calls."""
        match iterable:
            case ast.Name(id=name) if name in annotated:
                return self.index.element_classes(self.module, annotated[name])
            case ast.Name() | ast.Attribute() | ast.Call() if isinstance(
                symbol := self.index.symbol(self.module, iterable.func if isinstance(iterable, ast.Call) else iterable),
                _ModuleValue,
            ):
                annotation = symbol.returns if isinstance(iterable, ast.Call) else symbol.annotation
                return self.index.element_classes(symbol.module, annotation)
            case ast.Attribute(value=value, attr=attr) | ast.Call(func=ast.Attribute(value=value, attr=attr)):
                result: frozenset[_ClassInfo] = frozenset()
                for definer, member in self.member_hits(value, attr):
                    annotation = member.returns if isinstance(iterable, ast.Call) else member.annotation
                    result |= self.index.element_classes(definer.module, annotation)
                return result
            case _:
                return frozenset()

    def bind(self, scope: _Scope, name: str, classes: frozenset[_ClassInfo]) -> None:
        if classes:
            scope.locals[name] = scope.locals.get(name, frozenset()) | classes

    def narrow(self, scope: _Scope, call: ast.Call) -> None:
        """Flow-insensitive narrowing: `isinstance(x, C)` and a `TypeGuard[C]` call make `x` a `C` too."""
        match call:
            case ast.Call(func=ast.Name(id='isinstance'), args=[ast.Name(id=name), classes]):
                elements = classes.elts if isinstance(classes, ast.Tuple) else [classes]
                for element in elements:
                    self.bind(scope, name, self.types(element))
            case ast.Call(func=func, args=[ast.Name(id=name), *_]):
                match self.index.symbol(self.module, func):
                    case _ModuleValue(module=module, returns=ast.Subscript(value=guard, slice=guarded)) if _last_name(
                        guard
                    ) in {'TypeGuard', 'TypeIs'}:
                        self.bind(scope, name, self.index.annotation_classes(module, guarded))
                    case _:
                        pass
            case _:
                pass

    def record_overrides(self, info: _ClassInfo) -> None:
        """An `override` of each ancestor member the class body redefines."""
        ancestors = self.index.mro(info).classes[1:]
        for name, member in info.members.items():
            if not member.in_body:
                continue
            for ancestor in ancestors:
                if name in ancestor.members:
                    self.record(ancestor, f'{ancestor.qualname}.{name}', 'override', f'{info.qualname}.{name}')

    def record_projections(self, info: _ClassInfo) -> None:
        """A `projected_field` for each field named like a member of a class one of the methods takes."""
        fields = [name for name, member in info.members.items() if member.is_field]
        for method in info.node.body:
            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = [*method.args.posonlyargs, *method.args.args][0 if _is_static(method) else 1 :]
                for parameter in [*parameters, *method.args.kwonlyargs]:
                    for source in self.index.annotation_classes(self.module, parameter.annotation):
                        for name in fields:
                            if (found := self.index.lookup(source, name)) is not None:
                                definer = found[0]
                                site = f'{info.qualname}.{name}'
                                self.record(definer, f'{definer.qualname}.{name}', 'projected_field', site)

    # --- Scopes ---------------------------------------------------------------------------------------------------

    def qualname(self, name: str) -> str:
        parent = self.scopes[-1].qualname
        return name if parent == '<module>' else f'{parent}.{name}'

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        info = self.index.class_info(self.module, node, self.qualname(node.name))
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        self.record_overrides(info)
        self.record_projections(info)
        self.scopes.append(_Scope(info.qualname, owner=info))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: FunctionNode) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        enclosing = self.scopes[-1]
        scope = _Scope(self.qualname(node.name), owner=enclosing.owner)
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        self.bind_parameters(scope, enclosing, node, arguments)
        self.scopes.append(scope)
        annotated = {argument.arg: argument.annotation for argument in arguments}
        if (vararg := node.args.vararg) is not None and vararg.annotation is not None:
            # `*items: T` makes `items` a tuple of `T`.
            annotated[vararg.arg] = ast.Subscript(value=ast.Name(id='tuple'), slice=vararg.annotation)
        self.bind_locals(scope, node, annotated)
        for argument in [*arguments, node.args.vararg, node.args.kwarg]:
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def bind_parameters(self, scope: _Scope, enclosing: _Scope, node: FunctionNode, arguments: list[ast.arg]) -> None:
        """The classes of the function's parameters: a method's first one is its class, the rest are annotated."""
        owner = enclosing.owner
        for position, argument in enumerate(arguments):
            if position == 0 and owner is not None and enclosing.qualname == owner.qualname and not _is_static(node):
                self.bind(scope, argument.arg, frozenset({owner}))
            else:
                self.bind(scope, argument.arg, self.index.annotation_classes(self.module, argument.annotation))

    def bind_locals(self, scope: _Scope, node: FunctionNode, annotated: dict[str, ast.expr | None]) -> None:
        """Flow-insensitively, the classes each local name of the function is bound to anywhere in it."""
        annotated.update(
            {
                statement.target.id: statement.annotation
                for statement in ast.walk(node)
                if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
            }
        )
        for statement in ast.walk(node):
            match statement:
                case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation):
                    self.bind(scope, name, self.index.annotation_classes(self.module, annotation))
                case (
                    ast.Assign(targets=[ast.Name(id=name)], value=value)
                    | ast.NamedExpr(target=ast.Name(id=name), value=value)
                ):
                    self.bind(scope, name, self.types(value))
                case ast.Call():
                    self.narrow(scope, statement)
                case (
                    ast.For(target=ast.Name(id=name), iter=iterable)
                    | ast.AsyncFor(target=ast.Name(id=name), iter=iterable)
                    | ast.comprehension(target=ast.Name(id=name), iter=iterable)
                ):
                    self.bind(scope, name, self.element_types(iterable, annotated))
                case _:
                    pass

    # --- Dependencies ---------------------------------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.record_symbol(self.index.module(alias.name), 'import')

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = self.index.import_base(self.module, node)
        if (source := self.index.module(base)) is not None:
            for alias in node.names:
                self.record_symbol(
                    self.index.resolve(source, alias.name),
                    'import' if (alias.asname or alias.name) in self.runtime_names else 'annotation_only',
                )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        symbol = self.index.symbol(self.module, node.value)
        if isinstance(symbol, _Module):
            kind = 'annotation_only' if id(node) in self.annotation_nodes else 'import'
            self.record_symbol(self.index.resolve(symbol, node.attr), kind)
        elif node.attr == '__dataclass_fields__':
            self.record_fields(node.value)
        else:
            self.record_member(node.value, node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        match node:
            case ast.Subscript(value=ast.Attribute(value=owner, attr='__dict__'), slice=name):
                self.record_named(owner, name)
            case _:
                pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        match node:
            case (
                ast.Call(func=ast.Name(id='getattr' | 'setattr' | 'hasattr' | 'delattr'), args=[owner, name, *_])
                | ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id='object'), attr='__getattribute__' | '__setattr__' | '__delattr__'
                    ),
                    args=[owner, name, *_],
                )
                | ast.Call(
                    func=ast.Attribute(
                        value=ast.Attribute(value=owner, attr='__dict__'), attr='get' | 'setdefault' | 'pop'
                    ),
                    args=[name, *_],
                )
            ):
                self.record_named(owner, name)
            case ast.Call(func=func, args=[owner]) if _last_name(func) == 'fields':
                self.record_fields(owner)
            case ast.Call(func=func, args=[owner, *_], keywords=keywords) if _last_name(func) == 'replace':
                self.record_keywords(owner, keywords)
            case ast.Call(func=func, keywords=keywords) if isinstance(self.index.symbol(self.module, func), _ClassInfo):
                self.record_keywords(func, keywords)
            case _:
                pass
        self.generic_visit(node)


def build_manifest(root: Path = REPO_ROOT) -> Manifest:
    """Walk the engines under `root` and return the manifest of core definitions they depend on."""
    index = _Index(root)
    entries: _EntrySets = {}
    for engine, pattern in ENGINES.items():
        for path in sorted(root.glob(pattern)):
            source_root = next(root / source for source in SOURCE_ROOTS if (root / source) in path.parents)
            parts = path.relative_to(source_root).with_suffix('').parts
            module = index.module('.'.join(parts[:-1] if parts[-1] == '__init__' else parts))
            assert module is not None, path
            _EngineWalker(index, module, engine, entries).visit(module.tree)
    return {
        core_file: {
            qualname: {key: sorted(values) for key, values in entry.items()}
            for qualname, entry in sorted(definitions.items())
        }
        for core_file, definitions in sorted(entries.items())
    }


def gates(entry: Mapping[str, Sequence[str]]) -> bool:
    """Whether a change that matches this manifest entry gates review: any of its kinds but `annotation_only`."""
    return any(kind != 'annotation_only' for kind in entry['kinds'])


@dataclass(frozen=True)
class ChangedFile:
    """A file a pull request changes, as GitHub's `pulls/{number}/files` endpoint lists it."""

    filename: str
    status: FileStatus
    previous_filename: str | None = None
    """The path a `renamed` or `copied` file had before."""
    patch: str | None = None
    """The diff's hunks, which GitHub leaves out when the diff is too large to show."""


def touched_entries(
    manifest: Manifest, files: Iterable[ChangedFile], read_base: Callable[[str], str], read_head: Callable[[str], str]
) -> Manifest:
    """The entries of `manifest` that gate review and that a pull request changing `files` touches.

    `read_base` and `read_head` return a core file's source at the pull request's merge base and head. They are
    only called with a path that is both a manifest key and in `files`.

    A removed line is placed in the merge-base source and an added line in the head source, each in the innermost
    definition around it. A line of a `self.name = ...` assignment in a method is also placed in `Class.name`,
    where the generator records an instance attribute. A comment, docstring or blank line changes no behavior and
    is placed nowhere. A change placed in qualname `T` touches entry `Q` when `T == Q` or `T` is nested under `Q`.

    A core file the pull request removes or renames touches all of its entries: the engines import its
    definitions from the path it leaves. So does one whose lines cannot be placed, because GitHub left its patch
    out, the patch does not match the fetched source, or the source does not parse.
    """
    changed: dict[str, set[str]] = {}
    for file in files:
        left = {'removed': file.filename, 'renamed': file.previous_filename}.get(file.status)
        if left in manifest:
            changed.setdefault(left, set()).update(manifest[left])
        if file.status == 'removed' or file.filename not in manifest:
            continue
        definitions = changed.setdefault(file.filename, set())
        if file.patch is None:
            definitions.update(manifest[file.filename])
            continue
        removed, added = _changed_lines(file.patch)
        # Only a `modified` or `changed` file's patch compares it with its own merge-base version.
        sides = [(read_head, added), *([(read_base, removed)] if file.status in {'modified', 'changed'} else [])]
        for read, lines in sides:
            if not lines:
                continue
            placed = _definitions_at(read(file.filename), lines)
            if placed is None:
                definitions.update(manifest[file.filename])
                break
            definitions.update(placed)
    return {
        core_file: touched
        for core_file, qualnames in changed.items()
        if (
            touched := {
                qualname: entry
                for qualname, entry in manifest[core_file].items()
                if gates(entry) and any(name == qualname or name.startswith(f'{qualname}.') for name in qualnames)
            }
        )
    }


def _changed_lines(patch: str) -> tuple[dict[int, str], dict[int, str]]:
    """A patch's removed lines by their merge-base line number, and its added lines by their head line number."""
    removed: dict[int, str] = {}
    added: dict[int, str] = {}
    old = new = 0
    for line in patch.split('\n'):
        if header := _HUNK_HEADER.match(line):
            old, new = int(header[1]), int(header[2])
        elif line.startswith('-'):
            removed[old] = line[1:]
            old += 1
        elif line.startswith('+'):
            added[new] = line[1:]
            new += 1
        elif not line.startswith('\\'):  # `\ No newline at end of file` belongs to neither side.
            old += 1
            new += 1
    return removed, added


def _definitions_at(source: str, lines: Mapping[int, str]) -> set[str] | None:
    """The qualnames `lines` of `source` are placed in, or `None` if they cannot be placed in it."""
    source_lines = source.split('\n')
    if any(not 0 < number <= len(source_lines) or source_lines[number - 1] != text for number, text in lines.items()):
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    code = {
        line
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type not in {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
        for line in range(token.start[0], token.end[0] + 1)
    }
    spans = _Spans()
    spans.collect(tree.body, '', None)
    return {qualname for number in lines if number in code for qualname in spans.at(number)}


@dataclass
class _Spans:
    """The line spans of a module's definitions, as `(first line, last line, qualname)`."""

    definitions: list[tuple[int, int, str]] = field(default_factory=list[tuple[int, int, str]])
    attributes: list[tuple[int, int, str]] = field(default_factory=list[tuple[int, int, str]])
    """`self.name = ...` assignments in methods, each spanning the instance attribute `Class.name`."""
    docstrings: set[int] = field(default_factory=set[int])
    """The lines of string statements, which do nothing when run."""

    def collect(self, body: list[ast.stmt], prefix: str, owner: str | None) -> None:
        """Record the definitions in `body`, nested ones included; `owner` is the class whose body it is."""
        for statement in _top_level(body):
            start, end = statement.lineno, statement.end_lineno or statement.lineno
            match statement:
                case ast.Expr(value=ast.Constant(value=str())):
                    self.docstrings.update(range(start, end + 1))
                case ast.ClassDef(name=name, decorator_list=decorators):
                    self.definitions.append((min([start, *(d.lineno for d in decorators)]), end, prefix + name))
                    self.collect(statement.body, f'{prefix}{name}.', prefix + name)
                case (
                    ast.FunctionDef(name=name, decorator_list=decorators)
                    | ast.AsyncFunctionDef(name=name, decorator_list=decorators)
                ):
                    self.definitions.append((min([start, *(d.lineno for d in decorators)]), end, prefix + name))
                    if owner is not None:
                        self.collect_instance_attributes(statement, owner)
                    self.collect(statement.body, f'{prefix}{name}.', None)
                case ast.AnnAssign(target=ast.Name(id=name)):
                    self.definitions.append((start, end, prefix + name))
                case ast.Assign(targets=targets):
                    self.definitions.extend((start, end, prefix + t.id) for t in targets if isinstance(t, ast.Name))
                case _:
                    pass

    def collect_instance_attributes(self, method: FunctionNode, owner: str) -> None:
        """Record the attributes of `owner` that `method` assigns, as `_instance_attribute_assignments` finds them."""
        for assignment, attribute in _instance_attribute_assignments(method):
            self.attributes.append(
                (assignment.lineno, assignment.end_lineno or assignment.lineno, f'{owner}.{attribute}')
            )

    def at(self, line: int) -> set[str]:
        """The qualnames a change to `line` is placed in."""
        if line in self.docstrings:
            return set()
        around = [(start, qualname) for start, end, qualname in self.definitions if start <= line <= end]
        innermost = max((start for start, _ in around), default=None)
        return {qualname for start, qualname in around if start == innermost} | {
            qualname for start, end, qualname in self.attributes if start <= line <= end
        }


# Runs `gh`; the tests pass a fake.
def _gh(*arguments: str) -> str:  # pragma: no cover
    """What `gh api` prints for `arguments`."""
    return subprocess.run(['gh', 'api', *arguments], stdout=subprocess.PIPE, check=True, encoding='utf-8').stdout


def render_manifest(manifest: Manifest) -> str:
    """The manifest as the committed JSON text."""
    return json.dumps(manifest, indent=2, sort_keys=True) + '\n'


def main(argv: Sequence[str] | None = None, *, root: Path = REPO_ROOT, gh: Callable[..., str] = _gh) -> int:
    """Check the committed manifest is current, rewrite it with `--write`, or print what `--touched` names."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--write', action='store_true', help='regenerate the manifest instead of checking it')
    mode.add_argument(
        '--touched',
        nargs=4,
        metavar=('REPO', 'PR', 'BASE_SHA', 'HEAD_SHA'),
        help='print the entries pull request PR touches that gate review; BASE_SHA is the base branch commit',
    )
    args = parser.parse_args(argv)
    if args.touched:
        repository, pull_request, base, head = args.touched
        manifest: Manifest = json.loads((root / MANIFEST_PATH).read_text(encoding='utf-8'))
        shapes = [(repository, _REPOSITORY), (pull_request, _PULL_REQUEST), (base, _COMMIT), (head, _COMMIT)]
        shapes += [(core_file, _CORE_FILE) for core_file in manifest]
        if invalid := [value for value, pattern in shapes if not pattern.fullmatch(value)]:
            parser.error(f'refusing to request anything for {", ".join(map(repr, invalid))}')
        merge_base = gh(f'repos/{repository}/compare/{base}...{head}', '--jq', '.merge_base_commit.sha').strip()
        pages = json.loads(gh('--paginate', '--slurp', f'repos/{repository}/pulls/{pull_request}/files?per_page=100'))
        files = [
            ChangedFile(item['filename'], item['status'], item.get('previous_filename'), item.get('patch'))
            for page in pages
            for item in page
        ]
        if not _COMMIT.fullmatch(merge_base) or len(files) >= 3000:
            # GitHub lists at most 3000 files, so a pull request at the cap may change core files it leaves out.
            print(f'cannot classify: merge base {merge_base!r}, {len(files)} files listed', file=sys.stderr)
            return 1
        raw = ('-H', 'Accept: application/vnd.github.raw+json')
        touched = touched_entries(
            manifest,
            files,
            lambda path: gh(*raw, f'repos/{repository}/contents/{path}?ref={merge_base}'),
            lambda path: gh(*raw, f'repos/{repository}/contents/{path}?ref={head}'),
        )
        print(render_manifest(touched), end='')
        return 0
    expected = render_manifest(build_manifest(root))
    manifest_path = root / MANIFEST_PATH
    if args.write:
        manifest_path.write_text(expected, encoding='utf-8')
        return 0
    if manifest_path.is_file() and manifest_path.read_text(encoding='utf-8') == expected:
        return 0
    print(f'`{MANIFEST_PATH}` is stale; run `uv run python scripts/durable_exec_contract.py --write`', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
