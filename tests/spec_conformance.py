"""Check a spec registry's JSON schema against the classes it names and against the loader that reads it.

`AgentSpec.model_json_schema_with_capabilities()` and `Dataset.model_json_schema_with_evaluators()` publish, for each
registered class, the spec forms that can load it: the bare name, a short form holding one argument, and a keyword
form holding its parameters. Nothing at runtime checks that these forms agree with the class, and a parameter can
drop out of them without an error in two ways:

- when `__init__`'s type hints don't resolve, as with a `TYPE_CHECKING` import, the schema builder reads the variadic
  base `from_spec` instead and publishes only the name;
- when one parameter's type has no JSON schema, Pydantic leaves the whole keyword form out of the union.

A form can also be published that the loader reads differently, or that the writer can't write back.

- `spec_parameters` and `published_parameters` say which parameters a class takes that a spec can set, and which of
  them the schema publishes.
- `form_instances` builds instances of every form the schema publishes for a name, each valid for the schema: one
  per branch of each union in a form, and for the keyword form, one per value of each parameter.
- `argument_errors` binds a loaded spec's arguments to the function the loader calls, and validates each one
  against its parameter's type.
- `check_capability_forms_load` also builds each instance through `from_spec`, as the loader does. Only a `TypeError`
  fails it: a class raises that when it rejects the shape of its arguments, and a value error, a missing credential
  or another platform when it rejects their values or its environment.

`tests/test_spec_conformance.py` (core capabilities, and native tools through `NativeTool`),
`tests/harness/test_spec_conformance.py` (Harness capabilities) and `tests/evals/test_spec_conformance.py`
(evaluators) run them over every registered class.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from jsonschema.validators import validator_for
from pydantic import ConfigDict, JsonValue, TypeAdapter
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, PydanticUserError
from pydantic_core import SchemaError, ValidationError

from pydantic_ai._spec import CapabilitySpec, filter_serializable_type
from pydantic_ai._utils import get_function_type_hints, is_str_dict
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError

JsonSchema = dict[str, JsonValue]

_VARIADIC = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
_NO_SCHEMA = (PydanticSchemaGenerationError, PydanticInvalidForJsonSchema, PydanticUserError, SchemaError)
_ARBITRARY_TYPES = ConfigDict(arbitrary_types_allowed=True)
_VALUES_OR_ENVIRONMENT = (UserError, ValueError, LookupError, NotImplementedError)


def _hinted(target: Callable[..., object]) -> Callable[..., object]:
    """The function whose type hints describe `target`'s parameters: a class's `__init__`, or `target` itself."""
    return target.__init__ if isinstance(target, type) else target


def capability_loader_target(cls: type[AbstractCapability[None]]) -> Callable[..., object]:
    """What the loader calls with a capability spec's arguments: an overriding `from_spec`, or the class itself."""
    owner = next(klass for klass in cls.__mro__ if 'from_spec' in vars(klass))
    return cls if owner is AbstractCapability else cls.from_spec


def spec_parameters(function: Callable[..., object]) -> dict[str, str]:
    """The parameters of `function` a spec can set, each with why it counts.

    A parameter counts when its type, once callables and type variables are filtered out as the schema builder does,
    has a JSON schema and takes more than `None`. One whose type hint doesn't resolve counts too: nothing can show
    it has no JSON form.
    """
    parameters: dict[str, str] = {}
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in ('self', 'cls') or parameter.kind in _VARIADIC:
            continue
        try:
            hint: object = get_function_type_hints(function, include_keys={parameter.name}).get(parameter.name, Any)
        except (NameError, TypeError, AttributeError) as error:
            parameters[parameter.name] = f'its type hint does not resolve ({type(error).__name__}: {error})'
            continue
        serializable = filter_serializable_type(hint)
        if serializable is None or serializable is type(None):
            continue
        try:
            TypeAdapter(serializable).json_schema()
        except _NO_SCHEMA:
            continue
        parameters[parameter.name] = 'its type has a JSON schema'
    return parameters


def name_forms(items: JsonSchema, name: str) -> list[JsonSchema]:
    """The members of the spec-entry union `items` that load the class registered as `name`."""
    members = items['anyOf']
    assert isinstance(members, list)
    refs = {f'#/$defs/spec_{name}', f'#/$defs/short_spec_{name}'}
    forms: list[JsonSchema] = []
    for member in members:
        assert is_str_dict(member)
        if member.get('const') == name or member.get('$ref') in refs:
            forms.append(member)
    return forms


def published_parameters(
    target: Callable[..., object], name: str, forms: list[JsonSchema], defs: JsonSchema
) -> set[str]:
    """The parameters `forms` publish: a keyword form's properties, or else the one parameter of a short form."""
    keyword_form = defs.get(f'spec_params_{name}')
    if is_str_dict(keyword_form):
        properties = keyword_form['properties']
        assert is_str_dict(properties)
        return set(properties)
    if any(form.get('$ref') == f'#/$defs/short_spec_{name}' for form in forms):
        [parameter] = [p.name for p in inspect.signature(target).parameters.values() if p.kind not in _VARIADIC]
        return {parameter}
    return set()


class _Instances:
    """Instances of the subschemas of one schema, which holds the definitions in `defs`."""

    FORMAT_EXAMPLES: dict[str, str] = {
        'date': '2026-01-01',
        'date-time': '2026-01-01T00:00:00Z',
        'duration': 'PT1S',
        'email': 'a@example.com',
        'path': 'a',
        'time': '00:00:00',
        'uri': 'https://example.com/',
        'uuid': '00000000-0000-4000-8000-000000000000',
    }
    """A value for each string `format` Pydantic emits in these schemas, since `jsonschema` doesn't check formats."""

    def __init__(self, defs: JsonSchema):
        self.defs = defs

    def of(self, node: JsonValue, *, vary: int = 0, refs: frozenset[str] = frozenset()) -> list[JsonValue]:
        """Instances of `node` that it accepts, one per branch of each union in it.

        An object's properties are varied one at a time down to `vary` levels: below that, an object gets its
        required properties only. A definition is not entered again inside itself, so a recursive one gets the
        instances of its branches that don't recurse.
        """
        valid: list[JsonValue] = []
        schema: JsonSchema = {'$defs': self.defs, 'allOf': [node]}
        validator = validator_for(schema)(schema)
        for candidate in self._candidates(node, vary, refs):
            if candidate not in valid and validator.is_valid(candidate):
                valid.append(candidate)
        return valid

    def _candidates(self, node: JsonValue, vary: int, refs: frozenset[str]) -> list[JsonValue]:
        if node is True:
            return ['a']
        assert is_str_dict(node), f'No instances for the subschema {node!r}'
        if isinstance(ref := node.get('$ref'), str):
            if ref in refs:
                return []
            target = self.defs[ref.removeprefix('#/$defs/')]
            siblings = {key: value for key, value in node.items() if key != '$ref'}
            return self._candidates({**target, **siblings} if is_str_dict(target) else target, vary, refs | {ref})
        if 'const' in node:
            return [node['const']]
        enum: JsonValue = node.get('enum')
        if isinstance(enum, list):
            return enum[:1]
        candidates: list[JsonValue] = [node['default']] if 'default' in node else []
        for union in ('anyOf', 'oneOf', 'allOf'):
            members: JsonValue = node.get(union)
            if isinstance(members, list):
                siblings = {key: value for key, value in node.items() if key != union}
                for member in members[:1] if union == 'allOf' else members:
                    merged = {**member, **siblings} if is_str_dict(member) else member
                    candidates += self._candidates(merged, vary, refs)
                return candidates
        types: JsonValue = node.get('type')
        for type_ in types if isinstance(types, list) else [types]:
            candidates += self._of_type(node, type_, vary, refs)
        return candidates

    def _of_type(self, node: JsonSchema, type_: JsonValue, vary: int, refs: frozenset[str]) -> list[JsonValue]:
        match type_:
            case 'string':
                if isinstance(format_ := node.get('format'), str) and format_ in self.FORMAT_EXAMPLES:
                    return [self.FORMAT_EXAMPLES[format_]]
                length = node.get('minLength')
                return ['a' * length if isinstance(length, int) and length > 0 else 'a']
            case 'integer' | 'number':
                minimum = node.get('minimum', node.get('exclusiveMinimum', 0))
                assert isinstance(minimum, int | float)
                return [minimum + 1 if 'exclusiveMinimum' in node else minimum]
            case 'boolean':
                return [True, False]
            case 'null':
                return [None]
            case 'array':
                items = self.of(node.get('items', True), refs=refs)
                count = node.get('minItems', 1)
                assert isinstance(count, int)
                return [[items[0]] * max(count, 1)] if items else [[]]
            case 'object':
                return self._objects(node, vary, refs)
            case None:
                return ['a']
            case _:
                assert False, f'No instances for the JSON type {type_!r}'

    def _objects(self, node: JsonSchema, vary: int, refs: frozenset[str]) -> list[JsonValue]:
        properties = node.get('properties', {})
        assert is_str_dict(properties)
        required = node.get('required', [])
        assert isinstance(required, list)
        minimal: dict[str, JsonValue] = {}
        for key in required:
            assert isinstance(key, str)
            if not (values := self.of(properties[key], vary=vary - 1, refs=refs)):
                return []
            minimal[key] = values[0]
        objects: list[JsonValue] = [minimal]
        if vary > 0:
            for key, subschema in properties.items():
                objects += [{**minimal, key: value} for value in self.of(subschema, vary=vary - 1, refs=refs)]
        return objects


def form_instances(forms: list[JsonSchema], defs: JsonSchema) -> list[JsonValue]:
    """Instances of each of `forms`, each valid for it: see `_Instances.of`."""
    instances = _Instances(defs)
    return [instance for form in forms for instance in instances.of(form, vary=2)]


def argument_errors(target: Callable[..., object], args: Sequence[object], kwargs: dict[str, object]) -> dict[str, str]:
    """Why the loader can't pass `args` and `kwargs` to `target`, by parameter, or an empty dict.

    The arguments must bind to `target`'s signature, under the key `''` when they don't, and each must validate
    against its parameter's type, which must resolve. That type is filtered as the schema builder filters it, and a
    class in it that has no core schema matches only its instances, as when `arbitrary_types_allowed` is set.
    """
    signature = inspect.signature(target)
    hinted = _hinted(target)
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError as error:
        return {'': str(error)}
    errors: dict[str, str] = {}
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind in _VARIADIC:
            continue
        try:
            hint = filter_serializable_type(get_function_type_hints(hinted, include_keys={name}).get(name, Any))
            if hint is None:
                errors[name] = 'its type has no JSON form'
                continue
            try:
                adapter: TypeAdapter[object] = TypeAdapter(hint, config=_ARBITRARY_TYPES)
            except PydanticUserError:  # A model, dataclass or `TypedDict` type takes no config.
                adapter = TypeAdapter(hint)
            adapter.rebuild(raise_errors=True)
        except (NameError, TypeError, AttributeError, *_NO_SCHEMA) as error:
            errors[name] = f'its type can not be checked ({type(error).__name__}: {error})'
            continue
        try:
            adapter.validate_python(value)
        except ValidationError as error:
            errors[name] = str(error)
    return errors


def capability_forms(schema: JsonSchema, name: str) -> list[JsonSchema]:
    """The forms an `AgentSpec` schema publishes for the capability registered as `name`."""
    properties = schema['properties']
    assert is_str_dict(properties)
    capabilities = properties['capabilities']
    assert is_str_dict(capabilities)
    items = capabilities['items']
    assert is_str_dict(items)
    return name_forms(items, name)


def check_capability_parameters(cls: type[AbstractCapability[None]], schema: JsonSchema, left_out: Mapping[str, str]):
    """Fail unless `schema` publishes every parameter of `cls` a spec can set, except those in `left_out`.

    Every parameter in `left_out` must be one a spec can set, so an entry has to go when its parameter does.
    """
    name = cls.get_serialization_name()
    assert name is not None
    settable = spec_parameters(cls.__init__)
    assert left_out.keys() <= settable.keys(), f'Remove the left-out entries `{name}` no longer takes'
    target = capability_loader_target(cls)
    forms = capability_forms(schema, name)
    if not forms:
        target_settable = spec_parameters(_hinted(target))
        unspecifiable = [
            p.name
            for p in inspect.signature(target).parameters.values()
            if p.default is p.empty and p.kind not in _VARIADIC and p.name not in target_settable
        ]
        assert unspecifiable, f'The agent spec schema publishes no form of `{name}`'
        return
    published = published_parameters(target, name, forms, definitions(schema))
    missing = {p: why for p, why in settable.items() if p not in published and p not in left_out}
    assert not missing, f'The agent spec schema leaves out parameters of `{name}`: {missing}'
    assert not published & left_out.keys(), f'Remove the left-out entries the schema now publishes for `{name}`'


def check_capability_forms_load(
    cls: type[AbstractCapability[None]],
    schema: JsonSchema,
    directory: Path,
    custom_capability_types: Sequence[type[AbstractCapability[None]]] = (),
    known_unloadable: Collection[str] = (),
    known_rejected: bool = False,
):
    """Fail unless every instance of every form `schema` publishes for `cls` loads, and survives `to_file()`.

    An instance loads when its arguments pass `argument_errors` and `cls.from_spec` takes them without a `TypeError`.
    A parameter in `known_unloadable` is expected to fail its type for some instance, and `known_rejected` expects
    `from_spec` to reject some instance, so each has to go once its class is fixed. All the instances are written
    with `AgentSpec.to_file()`, and the file must validate against the schema written beside it and read back to the
    same specs.
    """
    name = cls.get_serialization_name()
    assert name is not None
    target = capability_loader_target(cls)
    specs: list[CapabilitySpec] = []
    unloadable: set[str] = set()
    rejected: dict[str, str] = {}
    forms = capability_forms(schema, name)
    instances = form_instances(forms, definitions(schema))
    assert instances or not forms, f'No instance of the published `{name}` forms {forms!r}'
    for instance in instances:
        assert not schema_errors(schema, {'capabilities': [instance]})
        spec = CapabilitySpec.model_validate(instance)
        errors = argument_errors(target, spec.args, spec.kwargs)
        unloadable.update(errors.keys() & set(known_unloadable))
        errors = {parameter: error for parameter, error in errors.items() if parameter not in known_unloadable}
        assert not errors, f'The published `{name}` form instance {instance!r} does not load: {errors}'
        try:
            cls.from_spec(*spec.args, **spec.kwargs)
        except TypeError as error:
            rejected[repr(instance)] = str(error)
        except _VALUES_OR_ENVIRONMENT:
            pass
        specs.append(spec)
    assert unloadable == set(known_unloadable), f'Remove the `{name}` entries that now load'
    if known_rejected:
        assert rejected, f'Remove the `{name}` entry: `from_spec` now takes every published form'
    else:
        assert not rejected, f'`{name}.from_spec` rejects published form instances: {rejected}'

    path, schema_path = directory / 'agent.yaml', directory / 'agent_schema.json'
    AgentSpec(capabilities=specs).to_file(
        path, schema_path=schema_path, custom_capability_types=custom_capability_types
    )
    written = yaml.safe_load(path.read_text())
    assert not schema_errors(json.loads(schema_path.read_text()), written), f'`to_file()` wrote an invalid `{name}`'
    reloaded = AgentSpec.from_file(path).capabilities
    assert [(c.name, c.args, c.kwargs) for c in reloaded] == [(c.name, c.args, c.kwargs) for c in specs]


def schema_errors(schema: JsonSchema, instance: JsonValue) -> list[str]:
    """Every error `schema` finds in `instance`."""
    return [
        f'{"/".join(map(str, error.absolute_path))}: {error.message}'
        for error in validator_for(schema)(schema).iter_errors(instance)
    ]


def definitions(schema: JsonSchema) -> JsonSchema:
    """The `$defs` of `schema`."""
    defs = schema.get('$defs', {})
    assert is_str_dict(defs)
    return defs
