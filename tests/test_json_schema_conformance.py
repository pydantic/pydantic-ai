"""Every JSON Schema walker reaches each keyword in `SUBSCHEMA_KEYWORDS`, and the lossless walkers keep a schema's meaning.

`SUBSCHEMA_KEYWORDS` in `pydantic_ai._json_schema_keywords` lists where a schema keeps its subschemas. `CORPUS` puts a
`$ref`, a recursive `$ref`, a JSON pointer `$ref`, boolean schemas and plain schemas at each of those positions, and:

- every `JsonSchemaTransformer` subclass the package defines walks each corpus schema, and every `$ref` left in its
  output resolves;
- `merge_json_schema_defs` points every `$ref` at the definition it pointed at before the merge;
- the lossless walkers, `InlineDefsJsonSchemaTransformer` and `merge_json_schema_defs`, keep the verdict of every test
  in a pinned copy of the official JSON-Schema-Test-Suite (see `json_schema_test_suite/README.md`).

Provider transformers strip keywords their APIs reject, so only the lossless walkers are held to the suite.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from collections.abc import Iterator
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from typing import Annotated, Literal, assert_never
from urllib.parse import unquote
from uuid import UUID

import pytest
from jsonschema import FormatChecker
from jsonschema.validators import validator_for
from pydantic import AnyUrl, BaseModel, EmailStr, Field, JsonValue, SecretStr, TypeAdapter, ValidationError
from referencing.exceptions import Unresolvable
from typing_extensions import TypeForm

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer, JsonSchema, JsonSchemaTransformer
from pydantic_ai._json_schema_keywords import (
    SUBSCHEMA_KEYWORDS,
    Subschema,
    SubschemaShape,
    iter_subschemas,
    map_subschemas,
)
from pydantic_ai._utils import is_str_dict, merge_json_schema_defs
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import _JsonSchemaTestData  # pyright: ignore[reportPrivateUsage]

_TRANSFORMER_MODULES = (
    'pydantic_ai.profiles.google',
    'pydantic_ai.profiles.openai',
    'pydantic_ai.providers.anthropic',
    'pydantic_ai.providers.bedrock',
    'pydantic_ai.providers.groq',
    'pydantic_ai.providers.openrouter',
    'pydantic_ai.realtime.elevenlabs_agents',
)
"""The modules that define a `JsonSchemaTransformer` subclass, imported so `__subclasses__()` finds them."""


def _transformer_classes() -> dict[str, type[JsonSchemaTransformer]]:
    for module in _TRANSFORMER_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:  # pragma: no cover - the provider's SDK isn't installed
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


def _verdict(schema: JsonSchema, data: JsonValue) -> bool | None:
    """Whether `schema` accepts `data`, or `None` when `jsonschema` can't tell: a `$ref` resolves nowhere, or a
    `pattern` is ECMA-262 syntax Python's `re` doesn't support."""
    try:
        return validator_for(schema)(schema).is_valid(data)
    except (Unresolvable, re.error):
        return None


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


class _Inner(BaseModel):
    x: int


class _Tree(BaseModel):
    children: list[_Tree] = []


class _Color(Enum):
    RED = 'r'
    BLUE = 'b'


TEST_MODEL_TYPES: dict[str, TypeForm[object]] = {
    'int': int,
    'int_multiple_of': Annotated[int, Field(multiple_of=7)],
    'int_multiple_of_bounded': Annotated[int, Field(multiple_of=7, ge=10, le=30)],
    'int_exclusive_bounds': Annotated[int, Field(gt=3, lt=6)],
    'float_multiple_of': Annotated[float, Field(multiple_of=0.5, gt=1)],
    'float_exclusive_bounds': Annotated[float, Field(gt=0.1, lt=0.2)],
    'decimal': Decimal,
    'str_length': Annotated[str, Field(min_length=2, max_length=3)],
    'str_pattern': Annotated[str, Field(pattern=r'^\d{3}$')],
    'email': EmailStr,
    'url': AnyUrl,
    'uuid': UUID,
    'datetime': datetime,
    'date': date,
    'time': time,
    'timedelta': timedelta,
    'ipv4': IPv4Address,
    'ipv6': IPv6Address,
    'bytes': bytes,
    'secret': SecretStr,
    'dict_min_length': Annotated[dict[str, int], Field(min_length=2)],
    'dict_max_length_0': Annotated[dict[str, int], Field(max_length=0)],
    'list_length': Annotated[list[int], Field(min_length=2, max_length=3)],
    'set': set[int],
    'frozenset_min_length': Annotated[frozenset[int], Field(min_length=2)],
    'tuple': tuple[int, str],
    'tuple_variadic': tuple[int, ...],
    'literal': Literal['a', 'b'],
    'enum': _Color,
    'optional': int | None,
    'union': int | str,
    'model': _Inner,
    'recursive_model': _Tree,
    'list_of_models': list[_Inner],
    'bool': bool,
    'none': None,
}
"""Field types whose Pydantic schemas `TestModel` has to generate valid arguments for."""

TEST_MODEL_SCHEMAS: dict[str, JsonSchema] = {
    'allOf': {'allOf': [{'type': 'integer', 'minimum': 3}]},
    'allOf-ref': {'allOf': [{'$ref': '#/$defs/Leaf'}], 'description': 'A leaf'},
    'allOf-properties': {'allOf': [{'type': 'object', 'properties': {'a': _LEAF}, 'required': ['a']}]},
    'allOf-boolean': {'allOf': [True, {'type': 'integer'}]},
    'allOf-ref-to-boolean': {'allOf': [{'$ref': '#/$defs/Flag'}, {'type': 'integer'}]},
    'allOf-ref-to-ref': {'type': 'string', 'allOf': [{'$ref': '#/$defs/Alias'}]},
    'allOf-anyOf': {'type': 'string', 'allOf': [{'anyOf': [{'type': 'string'}, {'type': 'integer'}]}]},
    'minProperties': {'type': 'object', 'minProperties': 1},
    'minProperties-optional': {'type': 'object', 'properties': {'a': _LEAF}, 'minProperties': 1},
    'maxProperties': {'type': 'object', 'additionalProperties': True, 'maxProperties': 0},
    'multipleOf-within-exclusive': {'type': 'integer', 'multipleOf': 5, 'exclusiveMinimum': 5, 'exclusiveMaximum': 11},
    'multipleOf-below-maximum': {'type': 'integer', 'multipleOf': 5, 'maximum': 8},
    'multipleOf-fractional-integer': {'type': 'integer', 'multipleOf': 2.5, 'minimum': 0, 'maximum': 10},
    'format-longer-than-maxLength': {'type': 'string', 'format': 'uri', 'maxLength': 3},
    'not': {'type': 'string', 'not': {'const': 'a'}},
}
"""Hand-written schemas Pydantic doesn't generate, for the keywords `TestModel` handles."""

_TEST_MODEL_DEFS: JsonSchema = {
    'Flag': True,
    'Alias': {'$ref': '#/$defs/TwoCharacters'},
    'TwoCharacters': {'minLength': 2},
}
"""Definitions beside `TEST_MODEL_SCHEMAS`."""

UNSUPPORTED_KEYWORDS = {
    'pattern': 'Generating a string that matches a regular expression needs a regex-inverting dependency.',
    'not': 'Only the schema itself drives generation, so a value can match what `not` rejects.',
}
"""Keywords `TestModel` doesn't generate for, so a schema using one can get an argument it rejects."""

PYDANTIC_STRICTER = {
    'decimal': "`Decimal`'s schema accepts any string, so `TestModel` can generate one that isn't a number.",
}
"""Types whose schema accepts an argument their Pydantic validation rejects, which an entry expects."""

_FORMATS = FormatChecker(formats=('date', 'email', 'ipv4', 'ipv6', 'uuid'))
"""The formats `jsonschema` checks with the standard library alone, so the result doesn't depend on what's installed."""


def _test_model_schemas() -> Iterator[tuple[str, JsonSchema, TypeAdapter[object] | None]]:
    for name, field_type in TEST_MODEL_TYPES.items():
        adapter = TypeAdapter[object](field_type)
        field_schema = adapter.json_schema()
        defs = field_schema.pop('$defs', {})
        yield (
            name,
            {'type': 'object', 'properties': {'field': field_schema}, 'required': ['field'], '$defs': defs},
            adapter,
        )
    for name, field_schema in TEST_MODEL_SCHEMAS.items():
        yield name, _root(field_schema, defs=_TEST_MODEL_DEFS), None


def _keywords(schema: Subschema) -> Iterator[str]:
    """Every keyword in `schema` and its subschemas."""
    if isinstance(schema, dict):
        yield from schema
        for subschema in iter_subschemas(schema):
            yield from _keywords(subschema)


@pytest.mark.parametrize(
    ('name', 'schema', 'adapter'), [pytest.param(*case, id=case[0]) for case in _test_model_schemas()]
)
def test_test_model_arguments_validate(name: str, schema: JsonSchema, adapter: TypeAdapter[object] | None):
    """`TestModel`'s arguments for each seed validate against their schema, and against the Pydantic type behind it.

    A schema using an `UNSUPPORTED_KEYWORDS` keyword, or a `PYDANTIC_STRICTER` type, is expected to get an invalid
    argument for some seed, so the entry has to go once `TestModel` generates valid ones.
    """
    invalid: dict[int, list[str]] = {}
    for seed in range(3):
        arguments = _JsonSchemaTestData(schema, seed).generate()
        errors = [
            error.message for error in validator_for(schema)(schema, format_checker=_FORMATS).iter_errors(arguments)
        ]
        if adapter is not None:
            try:
                # Tool arguments arrive as JSON, so they're validated in JSON mode.
                adapter.validate_json(json.dumps(arguments['field']))
            except ValidationError as e:
                errors.extend(error['msg'] for error in e.errors())
        if errors:
            invalid[seed] = errors
    if expected := sorted(UNSUPPORTED_KEYWORDS.keys() & set(_keywords(schema))) or PYDANTIC_STRICTER.get(name):
        assert invalid, f'`TestModel` now generates valid arguments for `{name}`: remove the entry for {expected}'
    else:
        assert not invalid, f'`TestModel` generates invalid arguments for `{name}`: {invalid}'


def test_map_subschemas_maps_each_shape():
    """`map_subschemas` replaces every subschema of every shape, and keeps a member that isn't one."""
    node: dict[str, object] = {
        'not': {'type': 'string'},
        'properties': {'a': {}, 'b': True},
        'allOf': [{}, 'not a schema'],
        'items': [{}],
        'dependencies': {'a': ['b']},
        'anyOf': 'not a list',
        'dependentSchemas': [{}],
    }
    map_subschemas(node, lambda subschema: {'mapped': subschema}, list(node))
    assert node == {
        'not': {'mapped': {'type': 'string'}},
        'properties': {'a': {'mapped': {}}, 'b': {'mapped': True}},
        'allOf': [{'mapped': {}}, 'not a schema'],
        'items': [{'mapped': {}}],
        'dependencies': {'a': ['b']},
        'anyOf': 'not a list',
        'dependentSchemas': [{}],
    }


@pytest.mark.parametrize(
    'schema',
    [
        pytest.param(_root({'allOf': [{'$ref': '#/$defs/Missing'}]}), id='allOf-missing-ref'),
        pytest.param(
            _root({'$ref': '#/$defs/Loop'}, defs={'Loop': {'allOf': [{'$ref': '#/$defs/Loop'}]}}), id='allOf-cycle'
        ),
        pytest.param(_root({'type': 'number', 'multipleOf': 3e-308, 'minimum': 1e10}), id='multipleOf-overflows'),
        pytest.param(_root({'type': 'string', 'allOf': True}), id='allOf-not-a-list'),
        pytest.param(_root({'type': 'string', 'allOf': [{'allOf': None}]}), id='nested-allOf-not-a-list'),
        pytest.param(_root({'type': 'string', 'allOf': [{'$ref': 5}]}), id='allOf-ref-not-a-string'),
        pytest.param(
            {**_root({'type': 'string', 'allOf': [{'$ref': '#/$defs/Leaf'}]}), '$defs': []}, id='defs-not-an-object'
        ),
        pytest.param(
            _root({'type': 'string', 'allOf': [{'properties': 5, 'required': 5}]}),
            id='allOf-member-properties-not-an-object',
        ),
        pytest.param(_root({'type': 'string', 'format': ['email']}), id='format-not-a-string'),
    ],
)
def test_test_model_generates_for_schemas_it_cant_satisfy(schema: JsonSchema):
    """`TestModel` still generates arguments where `allOf`, `multipleOf` or `format` can't be honored, as it did when it
    ignored them, rather than raising."""
    assert 'field' in _JsonSchemaTestData(schema).generate()


def test_ref_in_data_is_not_walked():
    """A `$ref` in a value like `const` is data, so the subschema holding it is sent as written."""
    field: JsonSchema = {'not': {'const': {'$ref': '#/$defs/Leaf'}}}
    output = InlineDefsJsonSchemaTransformer(_root(deepcopy(field))).walk()
    assert output['properties']['field'] == field


def test_malformed_subschema_under_a_ref_keyword_names_its_keyword():
    """A node under a keyword the up-front shape check skips is checked when it's walked, and the error says where."""
    schema = _root({'not': {'$ref': '#/$defs/Leaf', 'properties': 'not an object'}})
    with pytest.raises(UserError, match=re.escape('Invalid JSON Schema at `not.properties`')):
        InlineDefsJsonSchemaTransformer(schema).walk()
