"""Used to build pydantic validators and JSON schemas from functions.

This module has to use numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Concatenate, cast, get_origin

from pydantic import ConfigDict
from pydantic._internal import _decorators, _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema
from typing_extensions import ParamSpec, TypeIs, TypeVar

from ._griffe import doc_descriptions
from ._run_context import RunContext
from ._utils import check_object_json_schema, is_async_callable, is_model_like, run_in_executor

if TYPE_CHECKING:
    from .tools import DocstringFormat, ObjectJsonSchema


__all__ = ('function_schema',)


@dataclass(kw_only=True)
class FunctionSchema:
    """Internal information about a function schema."""

    function: Callable[..., Any]
    description: str | None
    validator: SchemaValidator
    json_schema: ObjectJsonSchema
    # if not None, the function takes a single by that name (besides potentially `info`)
    takes_ctx: bool
    is_async: bool
    single_arg_name: str | None = None
    positional_fields: list[str] = field(default_factory=list)
    var_positional_field: str | None = None

    async def call(self, args_dict: dict[str, Any], ctx: RunContext[Any]) -> Any:
        args, kwargs = self._call_args(args_dict, ctx)
        if self.is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args, **kwargs)
        else:
            function = cast(Callable[[Any], str], self.function)
            return await run_in_executor(function, *args, **kwargs)

    def _call_args(
        self,
        args_dict: dict[str, Any],
        ctx: RunContext[Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        if self.single_arg_name:
            args_dict = {self.single_arg_name: args_dict}

        args = [ctx] if self.takes_ctx else []
        for positional_field in self.positional_fields:
            args.append(args_dict.pop(positional_field))  # pragma: no cover
        if self.var_positional_field:
            args.extend(args_dict.pop(self.var_positional_field))

        return args, args_dict


def function_schema(  # noqa: C901
    function: Callable[..., Any],
    schema_generator: type[GenerateJsonSchema],
    takes_ctx: bool | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
) -> FunctionSchema:
    """Build a Pydantic validator and JSON schema from a tool function.

    Args:
        function: The function to build a validator and JSON schema for.
        takes_ctx: Whether the function takes a `RunContext` first argument.
        docstring_format: The docstring format to use.
        require_parameter_descriptions: Whether to require descriptions for all tool function parameters.
        schema_generator: The JSON schema generator class to use.

    Returns:
        A `FunctionSchema` instance.
    """
    if takes_ctx is None:
        takes_ctx = _takes_ctx(function)

    config = ConfigDict(title=function.__name__, use_attribute_docstrings=True)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)
    errors: list[str] = []

    try:
        sig = signature(function)
    except ValueError as e:
        errors.append(str(e))
        sig = signature(lambda: None)

    type_hints = _typing_extra.get_function_type_hints(function)

    var_kwargs_schema: core_schema.CoreSchema | None = None
    fields: dict[str, core_schema.TypedDictField] = {}
    positional_fields: list[str] = []
    var_positional_field: str | None = None
    decorators = _decorators.DecoratorInfos()

    description, field_descriptions = doc_descriptions(function, sig, docstring_format=docstring_format)

    if require_parameter_descriptions:
        if takes_ctx:
            parameters_without_ctx = set(
                name for name in sig.parameters if not _is_call_ctx(sig.parameters[name].annotation)
            )
            missing_params = parameters_without_ctx - set(field_descriptions)
        else:
            missing_params = set(sig.parameters) - set(field_descriptions)

        if missing_params:
            errors.append(f'Missing parameter descriptions for {", ".join(missing_params)}')

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
                    errors.append('First parameter of tools that take context must be annotated with RunContext[...]')
                continue
            elif not takes_ctx and _is_call_ctx(annotation):
                errors.append('RunContext annotations can only be used with tools that take context')
                continue
            elif index != 0 and _is_call_ctx(annotation):
                errors.append('RunContext annotations can only be used as the first argument')
                continue

        field_name = p.name
        if p.kind == Parameter.VAR_KEYWORD:
            var_kwargs_schema = gen_schema.generate_schema(annotation)
        else:
            if p.kind == Parameter.VAR_POSITIONAL:
                annotation = list[annotation]

            required = p.default is Parameter.empty
            # FieldInfo.from_annotated_attribute expects a type, `annotation` is Any
            annotation = cast(type[Any], annotation)
            if required:
                field_info = FieldInfo.from_annotation(annotation)
            else:
                field_info = FieldInfo.from_annotated_attribute(annotation, p.default)
            if field_info.description is None:
                field_info.description = field_descriptions.get(field_name)

            fields[field_name] = td_schema = gen_schema._generate_td_field_schema(  # pyright: ignore[reportPrivateUsage]
                field_name,
                field_info,
                decorators,
                required=required,
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
    json_schema = schema_generator().generate(schema)

    # workaround for https://github.com/pydantic/pydantic/issues/10785
    # if we build a custom TypedDict schema (matches when `single_arg_name is None`), we manually set
    # `additionalProperties` in the JSON Schema
    if single_arg_name is not None and not description:
        # if the tool description is not set, and we have a single parameter, take the description from that
        # and set it on the tool
        description = json_schema.pop('description', None)

    return FunctionSchema(
        description=description,
        validator=schema_validator,
        json_schema=check_object_json_schema(json_schema),
        single_arg_name=single_arg_name,
        positional_fields=positional_fields,
        var_positional_field=var_positional_field,
        takes_ctx=takes_ctx,
        is_async=is_async_callable(function),
        function=function,
    )


P = ParamSpec('P')
R = TypeVar('R')


WithCtx = Callable[Concatenate[RunContext[Any], P], R]
WithoutCtx = Callable[P, R]
TargetCallable = WithCtx[P, R] | WithoutCtx[P, R]


def _takes_ctx(callable_obj: TargetCallable[P, R]) -> TypeIs[WithCtx[P, R]]:
    """Check if a callable takes a `RunContext` first argument.

    Args:
        callable_obj: The callable to check.

    Returns:
        `True` if the callable takes a `RunContext` as first argument, `False` otherwise.
    """
    try:
        sig = signature(callable_obj)
    except ValueError:
        return False
    try:
        first_param_name = next(iter(sig.parameters.keys()))
    except StopIteration:
        return False
    else:
        # See https://github.com/pydantic/pydantic/pull/11451 for a similar implementation in Pydantic
        if not isinstance(callable_obj, _decorators._function_like):  # pyright: ignore[reportPrivateUsage]
            call_func = getattr(type(callable_obj), '__call__', None)
            if call_func is not None:
                callable_obj = call_func
            else:
                return False  # pragma: no cover

        type_hints = _typing_extra.get_function_type_hints(_decorators.unwrap_wrapped_function(callable_obj))
        annotation = type_hints.get(first_param_name)
        if annotation is None:
            return False
        return True is not sig.empty and _is_call_ctx(annotation)


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
    """Return whether the annotation is the `RunContext` class, parameterized or not."""
    return annotation is RunContext or get_origin(annotation) is RunContext
