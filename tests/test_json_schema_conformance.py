"""Every JSON Schema walker reaches each keyword in `SUBSCHEMA_KEYWORDS`, and the lossless walkers keep a schema's meaning.

`SUBSCHEMA_KEYWORDS` in `pydantic_ai._json_schema_keywords` lists where a schema keeps its subschemas. `CORPUS` puts a
`$ref`, a recursive `$ref`, a JSON pointer `$ref`, boolean schemas and plain schemas at each of those positions, and:

- every `JsonSchemaTransformer` subclass the package defines walks each corpus schema, and every `$ref` left in its
  output resolves;
- `InlineDefsJsonSchemaTransformer`'s output of each corpus schema accepts what the schema accepts, among a few values;
- `merge_json_schema_defs` points every `$ref` at the definition it pointed at before the merge;
- the lossless walkers, `InlineDefsJsonSchemaTransformer` and `merge_json_schema_defs`, keep the verdict of every test
  in a pinned copy of the official JSON-Schema-Test-Suite (see `json_schema_test_suite/README.md`).

Provider transformers strip keywords their APIs reject, so only the lossless walkers are held to the suite.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import re
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import assert_never
from urllib.parse import unquote

import pytest
from jsonschema.protocols import Validator
from jsonschema.validators import Draft201909Validator, Draft202012Validator, validator_for
from pydantic import JsonValue
from referencing.exceptions import Unresolvable

from pydantic_ai._json_schema import (
    InlineDefsJsonSchemaTransformer,
    JsonSchema,
    JsonSchemaTransformer,
    UnwalkedRefWriter,
)
from pydantic_ai._json_schema_keywords import (
    SUBSCHEMA_KEYWORDS,
    Subschema,
    SubschemaShape,
    iter_subschemas,
    map_subschemas,
)
from pydantic_ai._utils import is_str_dict, merge_json_schema_defs
from pydantic_ai.exceptions import UserError

_TRANSFORMER_MODULES = (
    'pydantic_ai.profiles.google',
    'pydantic_ai.profiles.openai',
    'pydantic_ai.providers.anthropic',
    'pydantic_ai.providers.azure',
    'pydantic_ai.providers.bedrock',
    'pydantic_ai.providers.groq',
    'pydantic_ai.providers.openrouter',
    'pydantic_ai.realtime.elevenlabs_agents',
)
"""The modules that define a `JsonSchemaTransformer` subclass, imported so `__subclasses__()` finds them.

`test_transformer_modules_lists_every_defining_module` keeps this in step with the source.
"""


def _transformer_classes() -> dict[str, type[JsonSchemaTransformer]]:
    for module in _TRANSFORMER_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:  # the provider's SDK isn't installed, as in the slim test lane
            pass

    found: dict[str, type[JsonSchemaTransformer]] = {}
    pending = [JsonSchemaTransformer]
    while pending:
        for subclass in pending.pop().__subclasses__():
            pending.append(subclass)
            # Only the package's own: a subclass a test module defines would make collection depend on test order.
            if not inspect.isabstract(subclass) and subclass.__module__.startswith('pydantic_ai.'):
                found[f'{subclass.__module__}.{subclass.__qualname__}'] = subclass
    return found


TRANSFORMERS = _transformer_classes()

_LEAF: JsonSchema = {'type': 'string', 'minLength': 1}
_DEFS: JsonSchema = {
    'Leaf': {'title': 'Leaf', **_LEAF},
    'Node': {'title': 'Node', 'type': 'object', 'properties': {'next': {'$ref': '#/$defs/Node'}}},
}
FILLERS: dict[str, Subschema] = {
    'ref': {'$ref': '#/$defs/Leaf'},
    'recursive_ref': {'$ref': '#/$defs/Node'},
    'pointer_ref': {'$ref': '#/properties/target'},
    'true': True,
    'false': False,
    'plain': _LEAF,
    'single_member_union': {'anyOf': [_LEAF]},
}
"""What the corpus puts at each subschema position."""

_HOST_TYPES: dict[str, str] = {
    **dict.fromkeys(
        (
            'properties',
            'patternProperties',
            'additionalProperties',
            'propertyNames',
            'dependentSchemas',
            'dependencies',
            'unevaluatedProperties',
        ),
        'object',
    ),
    **dict.fromkeys(('prefixItems', 'items', 'contains', 'additionalItems', 'unevaluatedItems'), 'array'),
    'contentSchema': 'string',
}
"""The instance type each keyword applies to, given to the schema holding it; the others apply to every type."""
_UNTYPED_HOST_KEYWORDS = frozenset(
    (
        'dependentSchemas',
        'dependencies',
        'unevaluatedProperties',
        'additionalItems',
        'unevaluatedItems',
        'contentSchema',
    )
)
"""The `_HOST_TYPES` keywords that also go under a host without a `type`, as a `$ref` under them resolves either way."""


def _value(shape: SubschemaShape, filler: Subschema) -> object:
    match shape:
        case 'schema' | 'schema_or_list':
            return deepcopy(filler)
        case 'schema_map':
            return {'name': deepcopy(filler)}
        case 'schema_list':
            return [deepcopy(filler)]
        case _:
            assert_never(shape)


def _root(
    field: JsonSchema,
    *,
    defs: JsonSchema | None = None,
    definitions: JsonSchema | None = None,
    target_flag: bool = False,
    target_name: bool = False,
) -> JsonSchema:
    """A tool-like object schema with `field` as a required property, a `target` for JSON pointers, and `$defs`.

    `target_flag` gives `target` a boolean subschema, and `target_name` puts a property named `target name` beside it.
    """
    target: JsonSchema = {'type': 'integer'}
    if target_flag:
        target = {'type': 'object', 'properties': {'flag': True}}
    properties: JsonSchema = {'field': field, 'target': target}
    if target_name:
        properties['target name'] = {'type': 'integer'}
    root: JsonSchema = {
        'type': 'object',
        'properties': properties,
        'required': ['field'],
        '$defs': {**deepcopy(_DEFS), **(defs or {})},
    }
    if definitions is not None:
        root['definitions'] = definitions
    return root


def _corpus() -> Iterator[tuple[str, JsonSchema]]:
    for keyword, shape in SUBSCHEMA_KEYWORDS.items():
        for filler_name, filler in FILLERS.items():
            case = f'{keyword}-{filler_name}'
            if keyword == '$defs':
                yield case, _root({'$ref': '#/$defs/Wrapper'}, defs={'Wrapper': deepcopy(filler)})
                continue
            if keyword == 'definitions':
                yield case, _root({'$ref': '#/definitions/Wrapper'}, definitions={'Wrapper': deepcopy(filler)})
                continue
            host: JsonSchema = {keyword: _value(shape, filler)}
            if (host_type := _HOST_TYPES.get(keyword)) is not None:
                host = {'type': host_type, **host}
            yield case, _root(host)
            if keyword in _UNTYPED_HOST_KEYWORDS:
                yield f'{case}-untyped', _root({keyword: _value(shape, filler)})
            if shape == 'schema_or_list':
                yield f'{case}-list', _root({**host, keyword: [deepcopy(filler)]})


CORPUS = {
    **dict(_corpus()),
    # https://github.com/pydantic/pydantic-ai/pull/12924: a JSON pointer `$ref` at a boolean schema.
    'pointer_ref-to-boolean': _root({'$ref': '#/properties/target/properties/flag'}, target_flag=True),
    # https://github.com/pydantic/pydantic-ai/pull/13009: a percent-encoded JSON pointer `$ref`.
    'pointer_ref-percent-encoded': _root({'$ref': '#/properties/target%20name'}, target_name=True),
}


def _refs(node: JsonValue, path: str = '') -> Iterator[tuple[str, str]]:
    """Each `$ref` under `node` and the JSON path to it."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == '$ref' and isinstance(value, str):
                yield path, value
            else:
                yield from _refs(value, f'{path}/{key}')
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _refs(value, f'{path}/{index}')


_MISSING = object()


def _pointer_target(schema: JsonSchema, ref: str) -> object:
    """What a local JSON pointer `$ref` (RFC 6901) points at, booleans included, or `_MISSING`.

    The gate resolves pointers itself, so it doesn't depend on `resolve_json_pointer`, which two open PRs change.
    """
    node: JsonValue = schema
    for token in ref.removeprefix('#').split('/')[1:]:
        # A `$ref` is a URI, so its fragment is percent-decoded before the pointer's own escapes (RFC 6901, section 6).
        token = unquote(token).replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return _MISSING
    return node


def _dangling_refs(cls: type[JsonSchemaTransformer]) -> dict[str, list[str]]:
    """The `$ref`s `cls` leaves pointing at nothing in its output, by corpus case.

    A schema a transformer rejects with `UserError` is left out: rejecting it is loud, unlike a dangling `$ref`.
    """
    dangling: dict[str, list[str]] = {}
    for case, schema in CORPUS.items():
        try:
            output = cls(deepcopy(schema)).walk()
        except UserError:
            continue
        if refs := [ref for _, ref in _refs(output) if _pointer_target(output, ref) is _MISSING]:
            dangling[case] = refs
    return dangling


def test_transformer_modules_lists_every_defining_module():
    """`_TRANSFORMER_MODULES` names each module whose source defines a `*SchemaTransformer` subclass.

    The source is read rather than imported, so the check needs no provider SDK.
    """
    package = Path(inspect.getfile(JsonSchemaTransformer)).parent
    defining: set[str] = set()
    for path in package.rglob('*.py'):
        if 'SchemaTransformer' not in (source := path.read_text(encoding='utf-8')):
            continue
        if any(
            isinstance(node, ast.ClassDef)
            and any(ast.unparse(base).endswith('SchemaTransformer') for base in node.bases)
            for node in ast.walk(ast.parse(source))
        ):
            defining.add('.'.join(('pydantic_ai', *path.relative_to(package).with_suffix('').parts)))
    # `pydantic_ai._json_schema` defines the base class, which this module imports.
    assert defining - {'pydantic_ai._json_schema'} == set(_TRANSFORMER_MODULES)


@pytest.mark.parametrize('name', sorted(TRANSFORMERS))
def test_walk_resolves_every_ref(name: str):
    """Every `$ref` a transformer leaves in its output points at something in that output."""
    dangling = _dangling_refs(TRANSFORMERS[name])
    assert not dangling, f'`{name}` leaves `$ref`s pointing at nothing: {dangling}'


def test_gate_finds_dangling_refs():
    """The gate's own checks: a transformer that drops `$defs` without inlining fails, and pointers resolve."""

    class DropsDefs(InlineDefsJsonSchemaTransformer):
        def walk(self) -> JsonSchema:
            return {key: value for key, value in self.schema.items() if key != '$defs'}

    assert _dangling_refs(DropsDefs)['properties-ref'] == ['#/$defs/Leaf']
    assert _pointer_target({'a': [True, {'b': 1}]}, '#/a/1/b') == 1
    assert _pointer_target({'a': [True]}, '#/a/1') is _MISSING


_UNRESOLVED_POINTERS = {
    'pointer_ref-to-boolean': 'https://github.com/pydantic/pydantic-ai/pull/12924',
    'definitions-true': 'https://github.com/pydantic/pydantic-ai/pull/12924',
    'definitions-false': 'https://github.com/pydantic/pydantic-ai/pull/12924',
    'pointer_ref-percent-encoded': 'https://github.com/pydantic/pydantic-ai/pull/13009',
}
"""Corpus cases whose JSON pointer `$ref` inlining can't resolve yet, by the open PR that resolves it."""


@pytest.mark.parametrize('case', sorted(_UNRESOLVED_POINTERS))
def test_unresolved_pointers_raise(case: str):
    """Each case is expected to raise until its PR lands, which then removes the entry."""
    with pytest.raises(UserError, match='Could not find \\$ref definition'):
        InlineDefsJsonSchemaTransformer(CORPUS[case]).walk()


def _titled(schema: JsonSchema, origin: str) -> JsonSchema:
    """`schema` with each of its object definitions titled by `origin`, so a merge that repoints a `$ref` shows."""
    titled = dict(schema)
    for key in ('$defs', 'definitions'):
        if key in schema:
            titled[key] = {
                name: {**body, 'title': f'{name} from {origin}'} if is_str_dict(body) else body
                for name, body in schema[key].items()
            }
    return titled


def _ref_targets(schema: JsonSchema) -> dict[str, object]:
    """What each `$ref` in `schema` points at, keyed by its path, with a definition's title standing for its name."""
    outside = {key: value for key, value in schema.items() if key != '$defs'}
    targets = {path: _target(schema, ref) for path, ref in _refs(outside)}
    for body in schema['$defs'].values():
        if is_str_dict(body):
            targets |= {f'{body["title"]}{path}': _target(schema, ref) for path, ref in _refs(body)}
    return targets


@pytest.mark.parametrize('case', sorted(CORPUS))
def test_merge_keeps_each_ref_target(case: str):
    """After a merge renames colliding definitions, each `$ref` points at the definition it pointed at before."""
    schema = _titled(CORPUS[case], 'schema')
    other = _titled({'$defs': {name: {'type': 'boolean'} for name in schema['$defs']}}, 'other')
    (_, rewritten), definitions = merge_json_schema_defs([other, schema])
    merged: JsonSchema = {**rewritten, '$defs': definitions}

    before = _ref_targets(schema)
    after = {path: target for path, target in _ref_targets(merged).items() if path in before}
    assert after == before


def _target(schema: JsonSchema, ref: str) -> object:
    """What `ref` points at, told apart by its `title` when it has one, since a renamed definition's own refs differ."""
    target = _pointer_target(schema, ref)
    return target.get('title', target) if is_str_dict(target) else target


_SUITE = Path(__file__).parent / 'json_schema_test_suite' / 'draft2020-12'

_WALKER_MERGES = 'https://github.com/pydantic/pydantic-ai/issues/13985'
_DYNAMIC_REF = "`$dynamicRef` isn't supported."
KNOWN_LOSSY: dict[str, dict[str, str]] = {
    'additionalProperties': {'additionalProperties does not look in applicators': _WALKER_MERGES},
    'allOf': {'allOf combined with anyOf, oneOf': _WALKER_MERGES},
    'items': {'items does not look in applicators, valid case': _WALKER_MERGES},
    'unevaluatedItems': {
        'unevaluatedItems with nested tuple': _WALKER_MERGES,
        'unevaluatedItems with $ref': _WALKER_MERGES,
        'unevaluatedItems before $ref': _WALKER_MERGES,
        'unevaluatedItems with $dynamicRef': _DYNAMIC_REF,
    },
    'unevaluatedProperties': {
        'unevaluatedProperties with nested properties': _WALKER_MERGES,
        'unevaluatedProperties with nested unevaluatedProperties': _WALKER_MERGES,
        'unevaluatedProperties with $ref': _WALKER_MERGES,
        'unevaluatedProperties before $ref': _WALKER_MERGES,
        'unevaluatedProperties with $dynamicRef': _DYNAMIC_REF,
        'nested unevaluatedProperties, outer false, inner true, properties outside': _WALKER_MERGES,
        'nested unevaluatedProperties, outer false, inner true, properties inside': _WALKER_MERGES,
        'nested unevaluatedProperties, outer true, inner false, properties outside': _WALKER_MERGES,
        'nested unevaluatedProperties, outer true, inner false, properties inside': _WALKER_MERGES,
        'in-place applicator siblings, anyOf has unevaluated': _WALKER_MERGES,
    },
}
"""The suite's groups, by file, whose verdicts `InlineDefsJsonSchemaTransformer` changes, with why.

Each entry expects the change, so it has to go once the walker is fixed. `merge_json_schema_defs` changes none.
"""


def _lossless_walks(schema: JsonSchema) -> dict[str, JsonSchema | UserError]:
    other: JsonSchema = {'$defs': {name: {'title': 'collides'} for name in schema.get('$defs', {})}}
    (_, merged), definitions = merge_json_schema_defs([other, schema])
    walks: dict[str, JsonSchema | UserError] = {
        'merge_json_schema_defs': {**merged, '$defs': definitions} if definitions else merged
    }
    try:
        walks['InlineDefsJsonSchemaTransformer'] = InlineDefsJsonSchemaTransformer(schema).walk()
    except UserError as e:
        walks['InlineDefsJsonSchemaTransformer'] = e
    return walks


def _verdict(schema: JsonSchema, data: JsonValue, draft: type[Validator] = Draft202012Validator) -> bool | None:
    """Whether `schema` accepts `data`, or `None` when `jsonschema` can't tell: a `$ref` resolves nowhere, or a
    `pattern` is ECMA-262 syntax Python's `re` doesn't support."""
    try:
        return validator_for(schema, default=draft)(schema).is_valid(data)
    except (Unresolvable, re.error):
        return None


def test_walked_keywords_are_the_keywords_the_walk_enters():
    """`UnwalkedRefWriter.WALKED_KEYWORDS` names exactly the keywords under which the walk transforms an inlined `$ref`.

    Under any other keyword the `$ref` is written out as written, without `transform()`. If the walk starts entering
    another keyword, this fails until `WALKED_KEYWORDS` lists it. `$defs` and `definitions` hold definitions, which
    the walk reaches through `$ref`s instead of by entering them.
    """

    class TitleMarking(InlineDefsJsonSchemaTransformer):
        def transform(self, schema: JsonSchema) -> JsonSchema:
            if 'title' in schema:
                schema['title'] += ' (transformed)'
            return schema

    entered = {
        keyword
        for keyword in SUBSCHEMA_KEYWORDS.keys() - {'$defs', 'definitions'}
        if 'Leaf (transformed)' in json.dumps(TitleMarking(deepcopy(CORPUS[f'{keyword}-ref'])).walk())
    }
    assert entered == UnwalkedRefWriter.WALKED_KEYWORDS


_INSTANCES: list[JsonValue] = [
    None,
    1,
    '',
    'x',
    [],
    [''],
    ['x'],
    {},
    {'name': ''},
    {'name': 'x'},
    {'next': 1},
    {'next': {'next': {}}},
]
"""Values for a corpus schema's `field`, which tell the fillers apart: `Leaf` needs a non-empty string, `Node` an object
whose `next` is a `Node`, and the map shape puts a filler under the name `name`."""


@pytest.mark.parametrize('case', sorted(CORPUS.keys() - _UNRESOLVED_POINTERS.keys()))
def test_inline_defs_keeps_corpus_verdicts(case: str):
    """`InlineDefsJsonSchemaTransformer`'s output accepts each `_INSTANCES` value of `field` its schema accepts.

    This holds the `$ref`s the walk inlines, the ones it writes out where it doesn't enter, and the recursive ones it
    keeps, to the definitions they point at.
    """
    schema = CORPUS[case]
    output = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()
    # Drafts before 2020-12 spell a tuple as an `items` list, which a 2020-12 validator can't evaluate.
    draft = Draft201909Validator if case.endswith('-list') else Draft202012Validator
    changed = [
        value
        for value in _INSTANCES
        if _verdict(output, {'field': value, 'target': 1}, draft)
        != _verdict(schema, {'field': value, 'target': 1}, draft)
    ]
    assert not changed


@pytest.mark.parametrize('suite_file', sorted(path.stem for path in _SUITE.glob('*.json')))
def test_lossless_walkers_keep_suite_verdicts(suite_file: str):
    """Each test in the pinned suite gets the same verdict from a lossless walker's output as from its schema.

    A test whose own schema `jsonschema` gets wrong or can't evaluate is skipped.
    """
    changed: dict[str, set[str]] = {}
    for group in json.loads((_SUITE / f'{suite_file}.json').read_text()):
        if not is_str_dict(schema := group['schema']):
            continue
        tests = [test for test in group['tests'] if _verdict(schema, test['data']) == test['valid']]
        for walker, output in _lossless_walks(schema).items():
            if isinstance(output, UserError) or any(_verdict(output, test['data']) != test['valid'] for test in tests):
                changed.setdefault(walker, set()).add(group['description'])
    assert changed == (
        {'InlineDefsJsonSchemaTransformer': set(known)} if (known := KNOWN_LOSSY.get(suite_file)) else {}
    )


def test_iter_subschemas_reads_each_shape():
    """`iter_subschemas` yields every subschema of every shape, and skips a member or a value that isn't one."""
    node: dict[str, object] = {
        'not': {'type': 'string'},
        'properties': {'a': {}, 'b': True},
        'allOf': [{}, 'not a schema'],
        'items': [{}],
        'dependencies': {'a': ['b']},
        'anyOf': 'not a list',
        'dependentSchemas': [{}],
    }
    assert list(iter_subschemas(node, list(node))) == [{'type': 'string'}, {}, True, {}, {}]


def test_map_subschemas_maps_each_shape():
    """`map_subschemas` replaces every subschema of every shape, and keeps a member or a value that isn't one."""
    node: dict[str, object] = {
        'title': 'kept',
        'not': {'type': 'string'},
        'properties': {'a': {}, 'b': True},
        'allOf': [{}, 'not a schema'],
        'items': [{}],
        'contains': 'not a schema',
        'dependencies': {'a': ['b']},
        'anyOf': 'not a list',
        'dependentSchemas': [{}],
    }
    assert map_subschemas(node, lambda keyword, _: keyword) == {
        'title': 'kept',
        'not': 'not',
        'properties': {'a': 'properties', 'b': 'properties'},
        'allOf': ['allOf', 'not a schema'],
        'items': ['items'],
        'contains': 'not a schema',
        'dependencies': {'a': ['b']},
        'anyOf': 'not a list',
        'dependentSchemas': [{}],
    }
