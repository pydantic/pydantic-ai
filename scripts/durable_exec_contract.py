#!/usr/bin/env python
"""Generate the durable-exec contract manifest: the core definitions durable engines depend on.

Durable engines import core symbols, override core methods, read core attributes and project core
dataclasses across their durable boundaries. A core change to one of those definitions can break an
engine without touching any file under `durable_exec/`, so a review routed by path never sees it.
The manifest records, per core definition, how the engines depend on it, so a change to a core
definition can be matched against it instead.

The walk is pure `ast` over the engine package roots in `ENGINES`; nothing is imported. "Core" is
`pydantic_ai` outside `pydantic_ai.durable_exec`. Each engine site depends on a core definition in
one of four ways:

- `import`: the engine names a core class, function or variable, by import or as a module attribute.
- `override`: an engine class redefines a member of a core class in its MRO.
- `attribute_read`: an engine reads or assigns a core member, passes it as a constructor or
  `dataclasses.replace` keyword, or reflects over a core dataclass's fields. The member is resolved
  from annotations: `ctx: RunContext[...]` makes `ctx.usage` a read of `RunContext.usage`, and
  `self.wrapped` on a `WrapperModel` subclass is a `Model`, so `self.wrapped.profile` reads
  `Model.profile`.
- `projected_field`: an engine class declares a field of the same name as a member of the core class
  one of its methods takes as a parameter, as `ModelRequestContextProjection.from_context` does.

What the walk cannot resolve to a member, such as `getattr(obj, name)`, falls back to the class,
unless the class has a base outside the walked packages (`BaseModel`, `Exception`), which may define
the member: those accesses record nothing. The classes in `MEMBER_LEVEL_ONLY` never get a
class-level entry: nearly every change to core touches them, so a class-level entry would match
every change.

The manifest maps `core_file -> qualname -> {kinds, engines, sites}`. A change to a core definition
matches an entry when the changed definition's qualname is the entry's qualname or nested under it:
an entry for `Model` matches a change to `Model.request`; an entry for `RunContext.usage` matches
only that field.

Usage:
    uv run python scripts/durable_exec_contract.py          # exit 1 if the manifest is stale
    uv run python scripts/durable_exec_contract.py --write  # regenerate it
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = 'scripts/durable_exec_contract.json'
WRITE_COMMAND = 'uv run python scripts/durable_exec_contract.py --write'

# Directories on `sys.path` for the packages the walk reads, relative to the repository root.
SOURCE_ROOTS = ('pydantic_ai_slim', 'src/pydantic_ai_harness')
CORE_PACKAGE = 'pydantic_ai'
DURABLE_PACKAGE = 'pydantic_ai.durable_exec'

# Engine name -> glob of its modules, relative to the repository root. `durable_exec` is the shared
# base every engine builds on (`BaseDurabilityCapability`, the operation backends, the projections).
ENGINES: dict[str, str] = {
    'durable_exec': 'pydantic_ai_slim/pydantic_ai/durable_exec/*.py',
    'temporal': 'pydantic_ai_slim/pydantic_ai/durable_exec/temporal/**/*.py',
    'dbos': 'pydantic_ai_slim/pydantic_ai/durable_exec/dbos/**/*.py',
    'prefect': 'pydantic_ai_slim/pydantic_ai/durable_exec/prefect/**/*.py',
    'render': 'src/pydantic_ai_harness/pydantic_ai_harness/render/**/*.py',
    'absurd': 'src/pydantic_ai_harness/pydantic_ai_harness/absurd/**/*.py',
    'pgtask': 'src/pydantic_ai_harness/pydantic_ai_harness/pgtask/**/*.py',
    'aws_lambda': 'src/pydantic_ai_harness/pydantic_ai_harness/aws_lambda/**/*.py',
}

MEMBER_LEVEL_ONLY: frozenset[tuple[str, str]] = frozenset(
    {
        ('pydantic_ai._run_context', 'RunContext'),
        ('pydantic_ai.models', 'ModelRequestContext'),
        ('pydantic_ai.models', 'ModelRequestParameters'),
        ('pydantic_ai.agent', 'Agent'),
    }
)

# Bases that contribute no members a core class could be looked up through.
_EMPTY_BASES = frozenset({'ABC', 'Generic', 'Protocol', 'object'})
# Annotation constructors whose arguments, not themselves, are the annotated type.
_TYPE_WRAPPERS = frozenset(
    {'Annotated', 'ClassVar', 'Final', 'NotRequired', 'Optional', 'ReadOnly', 'Required', 'Type', 'Union', 'type'}
)
_TYPE_GUARDS = frozenset({'TypeGuard', 'TypeIs'})

Kind = Literal['import', 'override', 'attribute_read', 'projected_field']
Manifest = dict[str, dict[str, dict[str, list[str]]]]
FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(eq=False)
class Member:
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
class Module:
    """A parsed module and its top-level names."""

    name: str
    path: str
    is_package: bool
    tree: ast.Module
    definitions: dict[str, ast.stmt] = field(default_factory=dict[str, ast.stmt])
    imports: dict[str, tuple[str, str | None]] = field(default_factory=dict[str, tuple[str, str | None]])

    @property
    def is_core(self) -> bool:
        in_core = self.name == CORE_PACKAGE or self.name.startswith(f'{CORE_PACKAGE}.')
        return in_core and not (self.name == DURABLE_PACKAGE or self.name.startswith(f'{DURABLE_PACKAGE}.'))


@dataclass(eq=False)
class ClassInfo:
    """A class and the members its body and methods define."""

    module: Module
    qualname: str
    node: ast.ClassDef
    members: dict[str, Member] = field(default_factory=dict[str, Member])

    @property
    def member_level_only(self) -> bool:
        return (self.module.name, self.qualname) in MEMBER_LEVEL_ONLY


@dataclass(frozen=True)
class Function:
    """A module-level function or variable, the other definitions an import can name."""

    module: Module
    name: str
    returns: ast.expr | None


Symbol = Module | ClassInfo | Function


def module_name(path: Path, source_root: Path) -> str:
    """The dotted module name of `path` under `source_root`."""
    parts = path.relative_to(source_root).with_suffix('').parts
    return '.'.join(parts[:-1] if parts[-1] == '__init__' else parts)


class Index:
    """Lazily parsed modules under `SOURCE_ROOTS`, and name resolution across them."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._modules: dict[str, Module | None] = {}
        self._classes: dict[int, ClassInfo] = {}

    def module(self, name: str) -> Module | None:
        if name not in self._modules:
            self._modules[name] = self._load(name)
        return self._modules[name]

    def _load(self, name: str) -> Module | None:
        relative = Path(*name.split('.'))
        for source_root in SOURCE_ROOTS:
            for path, is_package in ((relative / '__init__.py', True), (relative.with_suffix('.py'), False)):
                file = self.root / source_root / path
                if file.is_file():
                    tree = ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
                    module = Module(name, (Path(source_root) / path).as_posix(), is_package, tree)
                    for statement in _top_level(tree.body):
                        self._bind(module, statement)
                    return module
        return None

    def _bind(self, module: Module, statement: ast.stmt) -> None:
        match statement:
            case ast.ClassDef(name=name) | ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name):
                module.definitions[name] = statement
            case ast.Assign(targets=targets):
                for target in targets:
                    if isinstance(target, ast.Name):
                        module.definitions[target.id] = statement
            case ast.AnnAssign(target=ast.Name(id=name)):
                module.definitions[name] = statement
            case ast.ImportFrom():
                base = self.import_base(module, statement)
                for alias in statement.names:
                    module.imports[alias.asname or alias.name] = (base, alias.name)
            case ast.Import():
                for alias in statement.names:
                    if alias.asname:
                        module.imports[alias.asname] = (alias.name, None)
                    else:
                        top = alias.name.split('.')[0]
                        module.imports[top] = (top, None)
            case _:
                pass

    @staticmethod
    def import_base(module: Module, statement: ast.ImportFrom) -> str:
        """The absolute module a `from ... import` statement imports from."""
        if not statement.level:
            return statement.module or ''
        package = module.name.split('.') if module.is_package else module.name.split('.')[:-1]
        package = package[: len(package) - statement.level + 1]
        return '.'.join([*package, *([statement.module] if statement.module else [])])

    def class_info(self, module: Module, node: ast.ClassDef, qualname: str) -> ClassInfo:
        if (info := self._classes.get(id(node))) is None:
            info = self._classes[id(node)] = ClassInfo(module, qualname, node)
            _collect_members(info)
        return info

    def resolve(self, module: Module, name: str, seen: frozenset[tuple[str, str]] = frozenset()) -> Symbol | None:
        """What `name` refers to at the top level of `module`, following re-exports."""
        if (module.name, name) in seen:
            return None
        statement = module.definitions.get(name)
        match statement:
            case ast.ClassDef():
                return self.class_info(module, statement, name)
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                return Function(module, name, statement.returns)
            case None:
                pass
            case _:
                return Function(module, name, None)
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

    def symbol(self, module: Module, expr: ast.expr) -> Symbol | None:
        """What a dotted name expression refers to at the top level of `module`."""
        match expr:
            case ast.Name(id=name):
                return self.resolve(module, name)
            case ast.Attribute(value=value, attr=attr):
                base = self.symbol(module, value)
                return self.resolve(base, attr) if isinstance(base, Module) else None
            case _:
                return None

    def annotation_classes(self, module: Module, annotation: ast.expr | None) -> frozenset[ClassInfo]:
        """The classes an annotation, written in `module`, says a value is an instance of."""
        match annotation:
            case ast.Constant(value=str(source)):
                try:
                    parsed = ast.parse(source, mode='eval').body
                except SyntaxError:
                    return frozenset()
                return self.annotation_classes(module, parsed)
            case ast.BinOp(left=left, op=ast.BitOr(), right=right):
                return self.annotation_classes(module, left) | self.annotation_classes(module, right)
            case ast.Subscript(value=value, slice=index):
                if _last_name(value) not in _TYPE_WRAPPERS:
                    return self.annotation_classes(module, value)
                arguments = index.elts if isinstance(index, ast.Tuple) else [index]
                if _last_name(value) == 'Annotated':
                    arguments = arguments[:1]
                return frozenset[ClassInfo]().union(
                    *(self.annotation_classes(module, argument) for argument in arguments)
                )
            case ast.Name() | ast.Attribute():
                symbol = self.symbol(module, annotation)
                return frozenset({symbol}) if isinstance(symbol, ClassInfo) else frozenset()
            case _:
                return frozenset()

    def bases(self, info: ClassInfo) -> tuple[list[ClassInfo], bool]:
        """The class's resolved direct bases, and whether every base resolved."""
        resolved: list[ClassInfo] = []
        closed = True
        for base in info.node.bases:
            target = _strip_subscript(base)
            symbol = self.symbol(info.module, target)
            if isinstance(symbol, ClassInfo):
                resolved.append(symbol)
            elif _last_name(target) not in _EMPTY_BASES:
                closed = False
        return resolved, closed

    def mro(self, info: ClassInfo) -> tuple[list[ClassInfo], bool]:
        """The class and its resolved ancestors, nearest first, and whether every ancestor resolved."""
        order: list[ClassInfo] = []
        closed = True
        pending = [info]
        while pending:
            current = pending.pop(0)
            if current in order:
                continue
            order.append(current)
            bases, bases_closed = self.bases(current)
            closed = closed and bases_closed
            pending.extend(bases)
        return order, closed

    def lookup(self, info: ClassInfo, name: str) -> tuple[ClassInfo, Member] | None:
        """The nearest class in the MRO that defines `name`, and its member."""
        for ancestor in self.mro(info)[0]:
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
                handler_bodies = [handled for handler in handlers for handled in handler.body]
                yield from _top_level([*try_body, *handler_bodies, *orelse, *finalbody])
            case _:
                yield statement


def _last_name(expr: ast.expr) -> str | None:
    match expr:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
        case _:
            return None


def _collect_members(info: ClassInfo) -> None:
    members = info.members
    for statement in info.node.body:
        match statement:
            case ast.AnnAssign(target=ast.Name(id=name), annotation=annotation) if name != '_':
                members[name] = Member(annotation, is_field=_last_name(_strip_subscript(annotation)) != 'ClassVar')
            case ast.Assign(targets=targets, value=value):
                for target in targets:
                    if isinstance(target, ast.Name):
                        members[target.id] = Member(value.func if isinstance(value, ast.Call) else None)
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                is_property = any(
                    _last_name(decorator) in {'property', 'cached_property'} for decorator in statement.decorator_list
                )
                if statement.name not in members:
                    members[statement.name] = (
                        Member(statement.returns) if is_property else Member(None, returns=statement.returns)
                    )
            case _:
                pass
    for method in info.node.body:
        if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
            _collect_instance_attributes(members, method)


def _collect_instance_attributes(members: dict[str, Member], method: FunctionNode) -> None:
    """Attributes a method assigns on `self`, typed by their annotation or the annotated parameter assigned."""
    arguments = [*method.args.posonlyargs, *method.args.args]
    if not arguments:
        return
    parameters = {argument.arg: argument.annotation for argument in arguments}
    for node in ast.walk(method):
        match node:
            case ast.AnnAssign(target=ast.Attribute(value=ast.Name(id=owner), attr=name), annotation=annotation):
                annotation_or_value = annotation
            case ast.Assign(targets=[ast.Attribute(value=ast.Name(id=owner), attr=name)], value=value):
                match value:
                    case ast.Name(id=parameter):
                        annotation_or_value = parameters.get(parameter)
                    case ast.Call(func=func):
                        annotation_or_value = func
                    case _:
                        annotation_or_value = None
            case _:
                continue
        if owner == arguments[0].arg and name not in members:
            members[name] = Member(annotation_or_value, in_body=False)


def _strip_subscript(expr: ast.expr) -> ast.expr:
    return expr.value if isinstance(expr, ast.Subscript) else expr


def _is_static(function: FunctionNode) -> bool:
    return any(_last_name(decorator) == 'staticmethod' for decorator in function.decorator_list)


@dataclass
class _Scope:
    qualname: str
    locals: dict[str, frozenset[ClassInfo]] = field(default_factory=dict[str, frozenset[ClassInfo]])
    owner: ClassInfo | None = None
    """The class whose body this scope is, or whose method this scope is."""


class _EngineWalker(ast.NodeVisitor):
    """Records the core dependencies of one engine module."""

    def __init__(self, index: Index, module: Module, engine: str, entries: _Entries) -> None:
        self.index = index
        self.module = module
        self.engine = engine
        self.entries = entries
        self.scopes = [_Scope('<module>')]

    # --- Recording ------------------------------------------------------------------------------------------------

    def record(self, definer: ClassInfo | Module, qualname: str, kind: Kind, site: str | None = None) -> None:
        module = definer.module if isinstance(definer, ClassInfo) else definer
        if module.is_core:
            location = site or self.scopes[-1].qualname
            self.entries.add(module.path, qualname, kind, self.engine, f'{self.module.path}::{location}')

    def record_symbol(self, symbol: Symbol | None) -> None:
        """An `import` dependency on a named core definition."""
        match symbol:
            case ClassInfo() if not symbol.member_level_only:
                self.record(symbol, symbol.qualname, 'import')
            case Function():
                self.record(symbol.module, symbol.name, 'import')
            case _:
                pass

    def record_member(self, expr: ast.expr, name: str) -> None:
        """An `attribute_read` of `name` on whatever `expr` is, falling back to its core classes."""
        classes = self.types(expr)
        found = [found for info in classes if (found := self.index.lookup(info, name)) is not None]
        for definer, _ in found:
            self.record(definer, f'{definer.qualname}.{name}', 'attribute_read')
        if not found and not (name.startswith('__') and name.endswith('__')):
            self.record_classes(classes)

    def record_classes(self, classes: frozenset[ClassInfo]) -> None:
        """The class-level fallback for an access the walk cannot resolve to a member."""
        for info in classes:
            if not info.member_level_only and self.index.mro(info)[1]:
                self.record(info, info.qualname, 'attribute_read')

    def record_fields(self, expr: ast.expr) -> None:
        """Reflection over a dataclass's fields depends on every field it has."""
        for info in self.types(expr):
            for ancestor in self.index.mro(info)[0]:
                for name, member in ancestor.members.items():
                    if member.is_field:
                        self.record(ancestor, f'{ancestor.qualname}.{name}', 'attribute_read')

    # --- Types ----------------------------------------------------------------------------------------------------

    def types(self, expr: ast.expr) -> frozenset[ClassInfo]:
        """The classes `expr` may be an instance of, or may be."""
        match expr:
            case ast.Name(id=name):
                for scope in reversed(self.scopes):
                    if name in scope.locals:
                        return scope.locals[name]
                symbol = self.index.resolve(self.module, name)
                return frozenset({symbol}) if isinstance(symbol, ClassInfo) else frozenset()
            case ast.Attribute(value=value, attr=attr):
                return self.attribute_types(value, attr, returns=False)
            case ast.Call(func=ast.Name(id='super')):
                owner = next((scope.owner for scope in reversed(self.scopes) if scope.owner is not None), None)
                return frozenset(self.index.bases(owner)[0]) if owner is not None else frozenset()
            case ast.Call(func=func, args=args) if _last_name(func) == 'cast' and args:
                return self.index.annotation_classes(self.module, args[0])
            case ast.Call(func=ast.Attribute(value=value, attr=attr)) if not isinstance(
                self.index.symbol(self.module, value), Module
            ):
                return self.attribute_types(value, attr, returns=True)
            case ast.Call(func=func):
                symbol = self.index.symbol(self.module, func)
                if isinstance(symbol, Function):
                    return self.index.annotation_classes(symbol.module, symbol.returns)
                return frozenset({symbol}) if isinstance(symbol, ClassInfo) else frozenset()
            case ast.Await(value=value) | ast.NamedExpr(value=value):
                return self.types(value)
            case ast.BoolOp(values=values):
                return frozenset[ClassInfo]().union(*(self.types(value) for value in values))
            case ast.IfExp(body=body, orelse=orelse):
                return self.types(body) | self.types(orelse)
            case _:
                return frozenset()

    def attribute_types(self, value: ast.expr, attr: str, *, returns: bool) -> frozenset[ClassInfo]:
        symbol = self.index.symbol(self.module, value)
        if isinstance(symbol, Module):
            resolved = self.index.resolve(symbol, attr)
            return frozenset({resolved}) if isinstance(resolved, ClassInfo) else frozenset()
        result: frozenset[ClassInfo] = frozenset()
        for info in self.types(value):
            if (found := self.index.lookup(info, attr)) is not None:
                definer, member = found
                annotation = member.returns if returns else member.annotation
                result |= self.index.annotation_classes(definer.module, annotation)
        return result

    def bind(self, scope: _Scope, name: str, classes: frozenset[ClassInfo]) -> None:
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
                    case Function(module=module, returns=ast.Subscript(value=guard, slice=guarded)) if (
                        _last_name(guard) in _TYPE_GUARDS
                    ):
                        self.bind(scope, name, self.index.annotation_classes(module, guarded))
                    case _:
                        pass
            case _:
                pass

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
        mro = self.index.mro(info)[0][1:]
        for name, member in info.members.items():
            if not member.in_body:
                continue
            for ancestor in mro:
                if name in ancestor.members:
                    self.record(ancestor, f'{ancestor.qualname}.{name}', 'override', f'{info.qualname}.{name}')
        fields = [name for name, member in info.members.items() if member.is_field]
        for method in node.body:
            if isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                parameters = [*method.args.posonlyargs, *method.args.args][0 if _is_static(method) else 1 :]
                for parameter in [*parameters, *method.args.kwonlyargs]:
                    for source in self.index.annotation_classes(self.module, parameter.annotation):
                        for name in fields:
                            if (found := self.index.lookup(source, name)) is not None:
                                definer = found[0]
                                site = f'{info.qualname}.{name}'
                                self.record(definer, f'{definer.qualname}.{name}', 'projected_field', site)
        self.scopes.append(_Scope(info.qualname, owner=info))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: FunctionNode) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        enclosing = self.scopes[-1]
        scope = _Scope(self.qualname(node.name), owner=enclosing.owner)
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for position, argument in enumerate(arguments):
            if position == 0 and enclosing.owner is not None and enclosing.qualname == enclosing.owner.qualname:
                if not _is_static(node):
                    self.bind(scope, argument.arg, frozenset({enclosing.owner}))
                    continue
            self.bind(scope, argument.arg, self.index.annotation_classes(self.module, argument.annotation))
        self.scopes.append(scope)
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
                case _:
                    pass
        for argument in [*arguments, node.args.vararg, node.args.kwarg]:
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    # --- Dependencies ---------------------------------------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.asname:
                self.record_symbol(self.index.module(alias.name))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = self.index.import_base(self.module, node)
        if (source := self.index.module(base)) is not None:
            for alias in node.names:
                self.record_symbol(self.index.resolve(source, alias.name))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        symbol = self.index.symbol(self.module, node.value)
        if isinstance(symbol, Module):
            self.record_symbol(self.index.resolve(symbol, node.attr))
        elif node.attr == '__dataclass_fields__':
            self.record_fields(node.value)
        else:
            self.record_member(node.value, node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        match node:
            case ast.Subscript(value=ast.Attribute(value=owner, attr='__dict__'), slice=ast.Constant(value=str(name))):
                self.record_member(owner, name)
            case _:
                pass
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        match node:
            case ast.Call(func=ast.Name(id='getattr' | 'setattr' | 'hasattr' | 'delattr'), args=[owner, name, *_]):
                if isinstance(name, ast.Constant) and isinstance(name.value, str):
                    self.record_member(owner, name.value)
                else:
                    self.record_classes(self.types(owner))
            case ast.Call(
                func=ast.Attribute(
                    value=ast.Attribute(value=owner, attr='__dict__'), attr='get' | 'setdefault' | 'pop'
                ),
                args=[ast.Constant(value=str(name)), *_],
            ):
                self.record_member(owner, name)
            case ast.Call(func=func, args=[owner]) if _last_name(func) == 'fields':
                self.record_fields(owner)
            case ast.Call(func=func, args=[owner, *_], keywords=keywords) if _last_name(func) == 'replace':
                for keyword in keywords:
                    if keyword.arg is not None:
                        self.record_member(owner, keyword.arg)
            case ast.Call(func=func, keywords=keywords) if isinstance(self.index.symbol(self.module, func), ClassInfo):
                for keyword in keywords:
                    if keyword.arg is not None:
                        self.record_member(func, keyword.arg)
            case _:
                pass
        self.generic_visit(node)


class _Entries:
    def __init__(self) -> None:
        self.manifest: dict[str, dict[str, dict[str, set[str]]]] = {}

    def add(self, core_file: str, qualname: str, kind: Kind, engine: str, site: str) -> None:
        entry = self.manifest.setdefault(core_file, {}).setdefault(
            qualname, {'kinds': set[str](), 'engines': set[str](), 'sites': set[str]()}
        )
        entry['kinds'].add(kind)
        entry['engines'].add(engine)
        entry['sites'].add(site)


def build_manifest(root: Path = REPO_ROOT) -> Manifest:
    """Walk the engines under `root` and return the manifest of core definitions they depend on."""
    index = Index(root)
    entries = _Entries()
    for engine, pattern in ENGINES.items():
        for path in sorted(root.glob(pattern)):
            source_root = next(root / source for source in SOURCE_ROOTS if (root / source) in path.parents)
            module = index.module(module_name(path, source_root))
            assert module is not None, path
            _EngineWalker(index, module, engine, entries).visit(module.tree)
    return {
        core_file: {
            qualname: {key: sorted(values) for key, values in entry.items()}
            for qualname, entry in sorted(definitions.items())
        }
        for core_file, definitions in sorted(entries.manifest.items())
    }


def render_manifest(manifest: Manifest) -> str:
    """The manifest as the committed JSON text."""
    return json.dumps(manifest, indent=2, sort_keys=True) + '\n'


def main(argv: Sequence[str] | None = None, *, root: Path = REPO_ROOT) -> int:
    """Check the committed manifest is current, or rewrite it with `--write`."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='regenerate the manifest instead of checking it')
    args = parser.parse_args(argv)
    expected = render_manifest(build_manifest(root))
    manifest_path = root / MANIFEST_PATH
    if args.write:
        manifest_path.write_text(expected, encoding='utf-8')
        return 0
    if manifest_path.is_file() and manifest_path.read_text(encoding='utf-8') == expected:
        return 0
    print(f'{MANIFEST_PATH} is stale; run `{WRITE_COMMAND}`', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
