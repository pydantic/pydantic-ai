"""Checks rendered tool catalogs against the JSON schemas they were rendered from.

The checker reads the rendered text the way a model or a type checker does, and resolves the schemas on its own,
so it shares no logic with `pydantic_ai.function_signature`. `test_function_signature_properties.py` holds the
renderer to it, and `test_function_signature_invariants.py` pins a hand-built input for each code.

Codes:

- `catalog.syntax_error`: Python can't compile the text;
- `catalog.duplicate_name`: two definitions share a name;
- `catalog.shadowed_helper`: a definition is named like something code using the text relies on, such as `Any`;
- `catalog.unresolved_name`: an annotation names something nothing defines;
- `catalog.field_shadows_type`: a class body field is named like a type that class uses, so a type checker reads
  later annotations as the field;
- `catalog.unsafe_field_name`: a class body field whose written name is not its key at runtime;
- `tool.missing`: a tool has no function;
- `tool.wrong_params`: a tool's parameter names or required flags differ from its schema;
- `tool.wrong_param_shape`: a parameter's annotation differs from its schema;
- `tool.wrong_return_shape`: the return annotation differs from its schema;
- `catalog.not_stable`: rendering the same tools twice gave different text.
"""

# Schemas are arbitrary JSON, so most values here are unknown to the type checker.
# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations as _annotations
import __future__ as _future

import ast
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote

HELPERS = frozenset(
    {'TypedDict', 'Any', 'NotRequired', 'Literal', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple'}
)

_NOT_DEFINABLE = HELPERS | {'asyncio'}

Schema = dict[str, Any]

_PRIMITIVES = {'string': 'str', 'integer': 'int', 'number': 'float', 'boolean': 'bool'}


@dataclass
class Tool:
    """The schemas one rendered function comes from."""

    name: str
    params: Schema
    returns: Schema | None = None


@dataclass
class _Class:
    fields: dict[str, tuple[ast.expr, bool]] = field(default_factory=dict[str, tuple[ast.expr, bool]])


def _resolve(root: Schema, ref: Any) -> Any:
    """Follow a `$ref` within its document, or return `None` when it does not land on a schema."""
    if not isinstance(ref, str) or not ref.startswith('#'):
        return None
    pointer = unquote(ref[1:])
    if pointer and not pointer.startswith('/'):
        return None
    node: Any = root
    for raw in pointer.split('/')[1:]:
        token = raw.replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and re.fullmatch('0|[1-9][0-9]{0,8}', token) and int(token) < len(node):
            node = node[int(token)]
        else:
            return None
    return node if isinstance(node, dict) else None


def _unquote_annotation(annotation: ast.expr) -> ast.expr:
    """Parse a forward reference written as a string."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return ast.parse(annotation.value, mode='eval').body
    return annotation


def _split_required(annotation: ast.expr) -> tuple[ast.expr, bool]:
    annotation = _unquote_annotation(annotation)
    if isinstance(annotation, ast.Subscript) and getattr(annotation.value, 'id', '') == 'NotRequired':
        return _unquote_annotation(annotation.slice), False
    return annotation, True


def _flatten_union(annotation: ast.expr) -> list[ast.expr]:
    if isinstance(annotation, ast.BinOp):
        return _flatten_union(annotation.left) + _flatten_union(annotation.right)
    return [annotation]


def _is_name(annotation: ast.expr, name: str) -> bool:
    return isinstance(annotation, ast.Name) and annotation.id == name


@dataclass
class _Catalog:
    """The definitions in a rendered catalog, and the violations found while reading them."""

    violations: list[str] = field(default_factory=list[str])
    classes: dict[str, _Class] = field(default_factory=dict[str, _Class])
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = field(
        default_factory=dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
    )

    def read(self, text: str) -> None:
        try:
            tree = ast.parse(text)
            compile(text, '<catalog>', 'exec', flags=_future.annotations.compiler_flag)
        except SyntaxError:
            self.violations.append('catalog.syntax_error')
            return
        names: list[str] = []
        annotations: list[ast.expr] = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                names.append(node.name)
                annotations.extend(self._read_class_body(node))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                names.append(node.name)
                self.functions[node.name] = node
                annotations.extend(_function_annotations(node))
            elif isinstance(node, ast.Assign):
                name, values = self._read_functional(node)
                names.append(name)
                annotations.extend(values)
        self.violations.extend('catalog.duplicate_name' for n in set(names) if names.count(n) > 1)
        self.violations.extend('catalog.shadowed_helper' for _ in set(names) & _NOT_DEFINABLE)
        known = set(names) | HELPERS
        for annotation in annotations:
            for node in ast.walk(_unquote_annotation(annotation)):
                if isinstance(node, ast.Name) and node.id not in known:
                    self.violations.append('catalog.unresolved_name')

    def _read_class_body(self, node: ast.ClassDef) -> list[ast.expr]:
        cls = self.classes[node.name] = _Class()
        annotations: list[ast.expr] = []
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
                annotations.append(statement.annotation)
                cls.fields[statement.target.id] = _split_required(statement.annotation)
        used = {n.id for a in annotations for n in ast.walk(_unquote_annotation(a)) if isinstance(n, ast.Name)}
        self.violations.extend('catalog.field_shadows_type' for name in cls.fields if name in used)
        self.violations.extend(
            'catalog.unsafe_field_name' for name in cls.fields if name.startswith('__') and not name.endswith('__')
        )
        return annotations

    def _read_functional(self, node: ast.Assign) -> tuple[str, list[ast.expr]]:
        call = node.value
        target = node.targets[0]
        assert isinstance(target, ast.Name) and isinstance(call, ast.Call) and _is_name(call.func, 'TypedDict')
        spec = call.args[1]
        assert isinstance(spec, ast.Dict)
        cls = self.classes[target.id] = _Class()
        for key, value in zip(spec.keys, spec.values):
            assert isinstance(key, ast.Constant)
            cls.fields[str(key.value)] = _split_required(value)
        return target.id, list(spec.values)

    def check_tool(self, tool: Tool) -> None:
        function = self.functions.get(tool.name)
        if function is None:
            self.violations.append('tool.missing')
            return
        self._check_params(function, tool.params)
        if tool.returns is not None:
            returns = function.returns
            if returns is None or not self._matches(returns, tool.returns, tool.returns):
                self.violations.append('tool.wrong_return_shape')

    def _check_params(self, function: ast.FunctionDef | ast.AsyncFunctionDef, schema: Schema) -> None:
        params = self._rendered_params(function)
        document = schema
        followed = {id(schema)}
        while (target := _resolve(document, schema.get('$ref'))) is not None and id(target) not in followed:
            schema = target
            followed.add(id(schema))
        properties: dict[str, Any] = schema.get('properties', {})
        if set(params) != set(properties):
            self.violations.append('tool.wrong_params')
            return
        required = set(schema.get('required', []))
        for name, (annotation, is_required) in params.items():
            node = properties[name]
            has_default = isinstance(node, dict) and 'default' in node
            if is_required != (name in required and not has_default):
                self.violations.append('tool.wrong_params')
            if name not in required and not has_default:
                # An optional parameter with no default has to accept the `None` it is given.
                node = {'anyOf': [node, {'type': 'null'}]}
            if not self._matches(annotation, node, document):
                self.violations.append('tool.wrong_param_shape')

    def _rendered_params(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, tuple[ast.expr, bool]]:
        if function.args.kwarg is not None:
            # Keywords Python can't write go in `**kwargs`, and the docstring names the type describing them.
            described = re.search(r'`(\w+)` describes them', ast.get_docstring(function) or '')
            cls = self.classes.get(described.group(1)) if described else None
            return dict(cls.fields) if cls else {}
        out: dict[str, tuple[ast.expr, bool]] = {}
        defaults = dict(zip(function.args.kwonlyargs, function.args.kw_defaults))
        for arg in function.args.kwonlyargs:
            assert arg.annotation is not None
            out[arg.arg] = (arg.annotation, defaults[arg] is None)
        return out

    def _matches(  # noqa: C901
        self,
        annotation: ast.expr,
        node: Any,
        root: Schema,
        seen: frozenset[tuple[str, int]] = frozenset(),
        inlining: frozenset[int] = frozenset(),
    ) -> bool:
        """Whether `annotation` means the same as schema `node`, assuming any pair in `seen` already matches."""
        annotation = _unquote_annotation(annotation)
        if not isinstance(node, dict):
            return _is_name(annotation, 'Any')
        if '$ref' in node:
            target = _resolve(root, node['$ref'])
            if target is None or id(target) in inlining:
                return _is_name(annotation, 'Any')
            if not _is_class_schema(target):
                inlining = inlining | {id(target)}
            return self._matches(annotation, target, root, seen, inlining)
        if 'const' in node:
            return _literal_values(annotation) == [node['const']]
        union_key = 'anyOf' if 'anyOf' in node else 'oneOf'
        if union_key in node:
            members = node[union_key]
            if not members:
                return _is_name(annotation, 'Any')
            rendered = _flatten_union(annotation)
            expected = [n for m in members for n in self._flatten_expected(m, root, inlining)]
            return all(any(self._matches(r, e, root, seen, inlining) for e in expected) for r in rendered) and all(
                any(self._matches(r, e, root, seen, inlining) for r in rendered) for e in expected
            )
        if 'enum' in node:
            if not node['enum']:
                return _is_name(annotation, 'Any')
            return _literal_values(annotation) == node['enum']
        kind = node.get('type')
        if isinstance(kind, list) and 'properties' not in node:
            return self._matches(annotation, {'anyOf': [{'type': k} for k in kind]}, root, seen, inlining)
        if kind == 'null':
            return isinstance(annotation, ast.Constant) and annotation.value is None
        if kind in _PRIMITIVES:
            return _is_name(annotation, _PRIMITIVES[kind])
        if kind == 'array':
            if not (isinstance(annotation, ast.Subscript) and _is_name(annotation.value, 'list')):
                return False
            return self._matches(annotation.slice, node.get('items', {}), root, seen, inlining)
        if _is_class_schema(node):
            return self._matches_class(annotation, node, root, seen)
        if kind == 'object':
            if not (isinstance(annotation, ast.Subscript) and _is_name(annotation.value, 'dict')):
                return False
            assert isinstance(annotation.slice, ast.Tuple)
            value = node.get('additionalProperties', True)
            return self._matches(annotation.slice.elts[1], {} if value is True else value, root, seen, inlining)
        return _is_name(annotation, 'Any')

    def _flatten_expected(self, node: Any, root: Schema, inlining: frozenset[int]) -> list[Any]:
        """Nested unions render flat, so compare against their members."""
        if isinstance(node, dict) and (node.get('anyOf') or node.get('oneOf')):
            return [n for m in node.get('anyOf') or node['oneOf'] for n in self._flatten_expected(m, root, inlining)]
        if isinstance(node, dict) and isinstance(node.get('type'), list) and 'properties' not in node:
            return [{'type': kind} for kind in node['type']]
        if isinstance(node, dict) and '$ref' in node:
            target = _resolve(root, node['$ref'])
            if (
                target is not None
                and id(target) not in inlining
                and (target.get('anyOf') or target.get('oneOf') or isinstance(target.get('type'), list))
            ):
                return self._flatten_expected(target, root, inlining | {id(target)})
        return [node]

    def _matches_class(
        self, annotation: ast.expr, node: Schema, root: Schema, seen: frozenset[tuple[str, int]]
    ) -> bool:
        if not isinstance(annotation, ast.Name) or annotation.id not in self.classes:
            return False
        key = (annotation.id, id(node))
        if key in seen:
            return True
        seen = seen | {key}
        rendered = self.classes[annotation.id].fields
        properties: dict[str, Any] = node['properties']
        required = set(node.get('required', []))
        if set(rendered) != set(properties):
            return False
        return all(
            rendered[name][1] == (name in required) and self._matches(rendered[name][0], properties[name], root, seen)
            for name in properties
        )


def _is_class_schema(node: Schema) -> bool:
    return node.get('type') == 'object' and 'properties' in node


def _function_annotations(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.expr]:
    arguments = [*function.args.kwonlyargs, *([function.args.kwarg] if function.args.kwarg else [])]
    out = [a.annotation for a in arguments if a.annotation is not None]
    if function.returns is not None:
        out.append(function.returns)
    return out


def _literal_values(annotation: ast.expr) -> list[Any] | None:
    if not (isinstance(annotation, ast.Subscript) and _is_name(annotation.value, 'Literal')):
        return None
    spec = annotation.slice
    return [ast.literal_eval(e) for e in spec.elts] if isinstance(spec, ast.Tuple) else [ast.literal_eval(spec)]


def catalog_violations(text: str, tools: list[Tool]) -> list[str]:
    """Return the code of every way `text` misdescribes `tools`, or an empty list."""
    catalog = _Catalog()
    catalog.read(text)
    if 'catalog.syntax_error' not in catalog.violations:
        for tool in tools:
            catalog.check_tool(tool)
    return catalog.violations


def stability_violations(first: str, second: str) -> list[str]:
    """Return `catalog.not_stable` when two renders of the same tools differ."""
    return [] if first == second else ['catalog.not_stable']
