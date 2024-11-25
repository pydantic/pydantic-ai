"""Used to build pydantic validators and JSON schemas from functions.

This module has to use numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from __future__ import annotations as _annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, TypedDict, cast, get_origin

from pydantic import ConfigDict, TypeAdapter
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema

from ._griffe import doc_descriptions
from ._utils import ObjectJsonSchema, check_object_json_schema, is_model_like

if TYPE_CHECKING:
    from . import _retriever
    from .dependencies import AgentDeps, RetrieverParams


__all__ = 'function_schema', 'LazyTypeAdapter'


class FunctionSchema(TypedDict):
    """Internal information about a function schema."""

    description: str
    validator: SchemaValidator
    json_schema: ObjectJsonSchema
    # if not None, the function takes a single by that name (besides potentially `info`)
    single_arg_name: str | None
    positional_fields: list[str]
    var_positional_field: str | None


def function_schema(either_function: _retriever.RetrieverEitherFunc[AgentDeps, RetrieverParams]) -> FunctionSchema:  # noqa: C901
    """Build a Pydantic validator and JSON schema from a retriever function.

    Args:
        either_function: The function to build a validator and JSON schema for.

    Returns:
        A `FunctionSchema` instance.
    """
    function = either_function.whichever()
    takes_ctx = either_function.is_left()
    config = ConfigDict(title=function.__name__)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)

    sig = signature(function)

    type_hints = _typing_extra.get_function_type_hints(function)

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    positional_fields: list[str] = []
    var_positional_field: str | None = None
    errors: list[str] = []
    decorators = _decorators.DecoratorInfos()
    description, field_descriptions = doc_descriptions(function, sig)

    for index, (name, p) in enumerate(sig.parameters.items()):
        if p.annotation is sig.empty:
            if takes_ctx and index == 0:
                # should be the `context` argument, skip
                continue
            # TODO warn?
            annotation = Any
        else:
            annotation = type_hints[name]

            if index == 0 and takes_ctx:
                if not _is_call_ctx(annotation):
                    errors.append('First argument must be a CallContext instance when using `.retriever`')
                continue
            elif not takes_ctx and _is_call_ctx(annotation):
                errors.append('CallContext instance can only be used with `.retriever`')
                continue
            elif index != 0 and _is_call_ctx(annotation):
                errors.append('CallContext instance can only be used as the first argument')
                continue

        field_name = p.name
        if p.kind == Parameter.VAR_KEYWORD:
            var_kwargs_schema = gen_schema.generate_schema(annotation)
        else:
            if p.kind == Parameter.VAR_POSITIONAL:
                annotation = list[annotation]

            # FieldInfo.from_annotation expects a type, `annotation` is Any
            annotation = cast(type[Any], annotation)
            field_info = FieldInfo.from_annotation(annotation)
            if field_info.description is None:
                field_info.description = field_descriptions.get(field_name)

            fields[field_name] = td_schema = gen_schema._generate_td_field_schema(  # pyright: ignore[reportPrivateUsage]
                field_name,
                field_info,
                decorators,
            )
            # noinspection PyTypeChecker
            td_schema.setdefault('metadata', {})['is_model_like'] = is_model_like(annotation)

            if p.kind == Parameter.POSITIONAL_ONLY:
                positional_fields.append(field_name)
            elif p.kind == Parameter.VAR_POSITIONAL:
                var_positional_field = field_name

    if errors:
        from .exceptions import UserError

        error_details = '\n  '.join(errors)
        raise UserError(f'Error generating schema for {function.__qualname__}:\n  {error_details}')

    core_config = config_wrapper.core_config(None)
    # noinspection PyTypedDict
    core_config['extra_fields_behavior'] = 'allow' if var_kwargs_schema else 'forbid'

    schema, single_arg_name = _build_schema(fields, var_kwargs_schema, gen_schema, core_config)
    schema = gen_schema.clean_schema(schema)
    # noinspection PyUnresolvedReferences
    schema_validator = create_schema_validator(
        schema,
        function,
        function.__module__,
        function.__qualname__,
        'validate_call',
        core_config,
        config_wrapper.plugin_settings,
    )
    # PluggableSchemaValidator is api compatible with SchemaValidator
    schema_validator = cast(SchemaValidator, schema_validator)
    json_schema = GenerateJsonSchema().generate(schema)

    # workaround for https://github.com/pydantic/pydantic/issues/10785
    # if we build a custom TypeDict schema (matches when `single_arg_name` is None), we manually set
    # `additionalProperties` in the JSON Schema
    if single_arg_name is None:
        json_schema['additionalProperties'] = bool(var_kwargs_schema)

    # instead of passing `description` through in core_schema, we just add it here
    if description:
        json_schema = {'description': description} | json_schema

    return FunctionSchema(
        description=description,
        validator=schema_validator,
        json_schema=check_object_json_schema(json_schema),
        single_arg_name=single_arg_name,
        positional_fields=positional_fields,
        var_positional_field=var_positional_field,
    )


def _build_schema(
    fields: dict[str, core_schema.TypedDictField],
    var_kwargs_schema: core_schema.CoreSchema | None,
    gen_schema: _generate_schema.GenerateSchema,
    core_config: core_schema.CoreConfig,
) -> tuple[core_schema.CoreSchema, str | None]:
    """Generate a typed dict schema for function parameters.

    Args:
        fields: The fields to generate a typed dict schema for.
        var_kwargs_schema: The variable keyword arguments schema.
        gen_schema: The `GenerateSchema` instance.
        core_config: The core configuration.

    Returns:
        tuple of (generated core schema, single arg name).
    """
    if len(fields) == 1 and var_kwargs_schema is None:
        name = next(iter(fields))
        td_field = fields[name]
        if td_field['metadata']['is_model_like']:  # type: ignore
            return td_field['schema'], name

    td_schema = core_schema.typed_dict_schema(
        fields,
        config=core_config,
        extras_schema=gen_schema.generate_schema(var_kwargs_schema) if var_kwargs_schema else None,
    )
    return td_schema, None


def _is_call_ctx(annotation: Any) -> bool:
    from .dependencies import CallContext

    return annotation is CallContext or (
        _typing_extra.is_generic_alias(annotation) and get_origin(annotation) is CallContext
    )


if TYPE_CHECKING:
    LazyTypeAdapter = TypeAdapter
else:

    class LazyTypeAdapter:
        __slots__ = '_args', '_kwargs', '_type_adapter'

        def __init__(self, *args, **kwargs):
            self._args = args
            self._kwargs = kwargs
            self._type_adapter = None

        def __getattr__(self, item):
            if self._type_adapter is None:
                self._type_adapter = TypeAdapter(*self._args, **self._kwargs)
            return getattr(self._type_adapter, item)
