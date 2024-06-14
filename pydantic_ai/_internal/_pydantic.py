"""Used to build pydantic validators and JSON schemas from functions.

This module has to use numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from inspect import Parameter, Signature, signature
from typing import Any, Callable, Literal, TypedDict, cast

from griffe.dataclasses import Docstring, Object as GriffeObject
from griffe.enumerations import DocstringSectionKind
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.config import ConfigDict
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema

__all__ = ('function_schema',)


class FunctionSchema(TypedDict):
    """Internal information about a function schema."""

    description: str
    validator: SchemaValidator
    json_schema: JsonSchemaValue
    takes_info: bool
    # if not None, the function takes a single by that name
    single_arg_name: str | None


def function_schema(function: Callable[..., Any]) -> FunctionSchema:
    """Build a Pydantic validator and JSON schema from a function.

    Args:
        function: The function to build a validator and JSON schema for.

    Returns:
        A `FunctionSchema` instance.
    """
    namespace = _typing_extra.add_module_globals(function, None)
    config = ConfigDict(title=function.__name__)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper, namespace)
    core_config = config_wrapper.core_config(None)
    schema, description, takes_info, single_arg_name = _parameters_dict_schema(function, gen_schema, core_config)

    schema_validator = create_schema_validator(
        schema,
        function,
        function.__module__,
        function.__qualname__,
        'validate_call',
        core_config,
        config_wrapper.plugin_settings,
    )
    json_schema = GenerateJsonSchema().generate(schema)
    return FunctionSchema(
        description=description,
        validator=schema_validator,
        json_schema=json_schema,
        takes_info=takes_info,
        single_arg_name=single_arg_name,
    )


def _parameters_dict_schema(
    function: Callable[..., Any],
    gen_schema: _generate_schema.GenerateSchema,
    core_config: core_schema.CoreConfig,
) -> tuple[core_schema.CoreSchema, str, bool, str | None]:
    """Generate a typed dict schema for function parameters.

    Args:
        function: The function to generate a typed dict schema for.
        gen_schema: The `GenerateSchema` instance.
        core_config: The core configuration.

    Returns:
        tuple of (generated core schema, description, takes info argument, single arg name).
    """
    from ..retrievers import CallInfo

    sig = signature(function)

    type_hints = _typing_extra.get_function_type_hints(function)

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    errors: list[str] = []
    decorators = _decorators.DecoratorInfos()
    description, field_descriptions = _doc_descriptions(function, sig)
    takes_info = False

    for index, (name, p) in enumerate(sig.parameters.items()):
        if p.annotation is sig.empty:
            # TODO warn?
            annotation = Any
        else:
            annotation = type_hints[name]

            if index == 0 and annotation is CallInfo:
                takes_info = True
                continue

        field_name = p.name
        if p.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY):
            field_info = FieldInfo.from_annotation(annotation)  # type: ignore
            if field_info.description is None:
                field_info.description = field_descriptions.get(field_name)

            fields[field_name] = gen_schema._generate_td_field_schema(field_name, field_info, decorators)  # type: ignore
        elif p.kind == Parameter.VAR_KEYWORD:
            # OK
            var_kwargs_schema = gen_schema.generate_schema(annotation)
        elif p.kind == Parameter.POSITIONAL_ONLY:
            errors.append(f'{p.name}: positional only function parameters are not supported')
        else:
            assert p.kind == Parameter.VAR_POSITIONAL, p.kind
            errors.append(f'{p.name}: *args function parameters are not supported')

    if errors:
        error_details = '\n  '.join(errors)
        raise ValueError(f'Error generating schema for {function.__qualname__}:\n{error_details}')

    if len(fields) == 1 and var_kwargs_schema is None:
        name = next(iter(fields))
        # we can only use a single argument if it allows positional use
        if sig.parameters[name].kind == Parameter.POSITIONAL_OR_KEYWORD:
            # and it's a model, dataclass or typed dict (so it's JSON Schema is an "object")
            field_schema = fields[name]['schema']
            if field_schema['type'] in {'typed-dict', 'model', 'dataclass'}:
                return field_schema, description, takes_info, name

    td_schema = core_schema.typed_dict_schema(
        fields,
        config=core_config,
        extras_schema=gen_schema.generate_schema(var_kwargs_schema) if var_kwargs_schema else None,
        extra_behavior='allow' if var_kwargs_schema else 'forbid',
    )
    return td_schema, description, takes_info, None


# @dataclass
# class _ModifyJsonSchema:
#     """Add title and description JSON schema."""
#
#     title: str
#     description: str
#
#     def __call__(self, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
#         json_schema = handler(schema)
#         json_schema = handler.resolve_ref_schema(json_schema)
#         json_schema.update(
#             title=self.title,
#             description=self.description,
#         )
#         return json_schema


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
