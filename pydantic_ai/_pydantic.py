"""Used to build pydantic validators and JSON schemas from functions.

This module has to use numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from __future__ import annotations as _annotations

from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict, cast, get_origin

from _griffe.enumerations import DocstringSectionKind
from _griffe.models import Docstring, Object as GriffeObject
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.config import ConfigDict
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema

from ._utils import ObjectJsonSchema, check_object_json_schema, is_model_like

if TYPE_CHECKING:
    from . import _retriever
    from .call import AgentDeps


__all__ = ('function_schema',)


class FunctionSchema(TypedDict):
    """Internal information about a function schema."""

    description: str
    validator: SchemaValidator
    json_schema: ObjectJsonSchema
    # if not None, the function takes a single by that name (besides potentially `info`)
    single_arg_name: str | None
    positional_fields: list[str]
    var_positional_field: str | None


def function_schema(either_function: _retriever.RetrieverEitherFunc[AgentDeps, _retriever.P]) -> FunctionSchema:
    """Build a Pydantic validator and JSON schema from a retriever function.

    Args:
        either_function: The function to build a validator and JSON schema for.

    Returns:
        A `FunctionSchema` instance.
    """
    function = either_function.whichever()
    takes_ctx = either_function.is_left()
    namespace = _typing_extra.get_module_ns_of(function)
    config = ConfigDict(title=function.__name__)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper, namespace)
    core_config = config_wrapper.core_config(None)

    sig = signature(function)

    type_hints = _typing_extra.get_function_type_hints(function)

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    positional_fields: list[str] = []
    var_positional_field: str | None = None
    errors: list[str] = []
    decorators = _decorators.DecoratorInfos()
    description, field_descriptions = _doc_descriptions(function, sig)

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
                    errors.append('First argument must be a CallContext instance when using `.retriever_context`')
                continue
            elif not takes_ctx and _is_call_ctx(annotation):
                errors.append('CallContext instance can only be used with `.retriever_context`')
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
            extra_metadata = {'is_model_like': is_model_like(annotation)}
            if metadata := td_schema.get('metadata'):
                metadata.update(extra_metadata)
            else:
                td_schema['metadata'] = extra_metadata
            if p.kind == Parameter.POSITIONAL_ONLY:
                positional_fields.append(field_name)
            elif p.kind == Parameter.VAR_POSITIONAL:
                var_positional_field = field_name

    if errors:
        error_details = '\n  '.join(errors)
        raise ValueError(f'Error generating schema for {function.__qualname__}:\n{error_details}')

    schema, single_arg_name = _build_schema(fields, var_kwargs_schema, gen_schema, core_config)
    schema_validator = create_schema_validator(
        schema,
        function,
        function.__module__,
        function.__qualname__,
        'validate_call',
        core_config,
        config_wrapper.plugin_settings,
    )
    # PluggableSchemaValidator is api compat with SchemaValidator
    schema_validator = cast(SchemaValidator, schema_validator)
    json_schema = GenerateJsonSchema().generate(schema)

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
        extra_behavior='allow' if var_kwargs_schema else 'forbid',
    )
    return td_schema, None


DocstringStyle = Literal['google', 'numpy', 'sphinx']


def _doc_descriptions(
    func: Callable[..., Any], sig: Signature, *, style: DocstringStyle | None = None
) -> tuple[str, dict[str, str]]:
    """Extract the function description and parameter descriptions from a function's docstring.

    Returns:
        A tuple of (main function description, parameter descriptions).
    """
    doc = func.__doc__
    if doc is None:
        return '', {}

    # see https://github.com/mkdocstrings/griffe/issues/293
    parent = cast(GriffeObject, sig)

    docstring = Docstring(doc, lineno=1, parser=style or _infer_docstring_style(doc), parent=parent)
    sections = docstring.parse()

    params = {}
    if parameters := next((p for p in sections if p.kind == DocstringSectionKind.parameters), None):
        params = {p.name: p.description for p in parameters.value}

    main_desc = ''
    if main := next((p for p in sections if p.kind == DocstringSectionKind.text), None):
        main_desc = main.value

    return main_desc, params


def _infer_docstring_style(doc: str) -> DocstringStyle:
    """Simplistic docstring style inference."""
    if '  Args:' in doc:
        return 'google'
    elif '  :param ' in doc:
        return 'sphinx'
    elif '  Parameters' in doc:
        return 'numpy'
    else:
        # fallback to google style
        return 'google'


def _is_call_ctx(annotation: Any) -> bool:
    from .call import CallContext

    return annotation is CallContext or (
        _typing_extra.is_generic_alias(annotation) and get_origin(annotation) is CallContext
    )
