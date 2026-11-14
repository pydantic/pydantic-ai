"""Used to build pydantic validators and JSON schemas from functions.

This module has to use numerous internal Pydantic APIs and is therefore brittle to changes in Pydantic.
"""

from __future__ import annotations as _annotations

import warnings
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import cached_property, partial
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Concatenate, Literal, Self, cast, get_args, get_origin

from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic._internal import _decorators, _generate_schema
from pydantic._internal._config import ConfigWrapper
from pydantic.errors import PydanticSchemaGenerationError, PydanticUserError
from pydantic.fields import FieldInfo
from pydantic.json_schema import GenerateJsonSchema
from pydantic.plugin._schema_validator import create_schema_validator
from pydantic_core import SchemaValidator, core_schema
from typing_extensions import ParamSpec, TypeIs, TypeVar, get_type_hints

from ._griffe import doc_descriptions
from ._run_context import RunContext
from ._utils import (
    await_maybe,
    check_object_json_schema,
    is_async_callable,
    is_model_like,
    run_in_executor,
    takes_run_context,
)
from .messages import ToolReturn

if TYPE_CHECKING:
    from .tools import DocstringFormat, ObjectJsonSchema


__all__ = ('function_schema',)


@dataclass(kw_only=True)
class FunctionSchema:
    """Internal information about a function schema."""

    function: Callable[..., Any]
    name: str
    description: str | None
    validator: SchemaValidator
    json_schema: ObjectJsonSchema
    takes_ctx: bool
    is_async: bool
    parameter_fields: dict[str, core_schema.TypedDictField] = field(
        default_factory=dict[str, core_schema.TypedDictField]
    )
    """The parameters' fields in the cleaned core schema of `validator`. Empty for a model-like single arg."""
    definitions: list[core_schema.CoreSchema] = field(default_factory=list[core_schema.CoreSchema])
    """The definitions the cleaned core schema keeps for `parameter_fields` to refer to."""
    # if not None, the function takes a single by that name (besides potentially `info`)
    single_arg_name: str | None = None
    single_field_name: str | None = None
    """Name of the single argument if the function takes exactly one value-carrying arg, else `None`.

    Covers both model-like single args (where it equals `single_arg_name`, which uses a wrap validator
    to normalize to `{name: value}`) and primitive single args (where the schema is a
    one-field TypedDict). `None` for multi-arg functions, any function taking `**kwargs`, and a
    single primitive arg the JSON schema leaves out (e.g. one marked `SkipJsonSchema`).

    This is the Python parameter name, which keys the validated args. The JSON schema property is
    named after the parameter's alias instead.

    The "field name" is the wrapper key only — e.g. for `def f(data: dict[str, str])`,
    this is `'data'`. The dict the user sends as `data` keeps all its keys; only the
    outer `{data: ...}` envelope is the wrapper.
    """
    positional_fields: list[str] = field(default_factory=list[str])
    var_positional_field: str | None = None
    return_schema: ObjectJsonSchema = field(default_factory=dict[str, Any])
    """JSON schema for the function's return type. At minimum `{}` (equivalent to `Any`)."""

    @cached_property
    def parameter_names(self) -> dict[str, str]:
        """Map each JSON schema property named after a parameter's validation alias to that parameter's Python name.

        Each aliased field's JSON schema is generated on its own, on first access, so Pydantic names its property and
        leaves out a field the JSON schema hides, like one marked `SkipJsonSchema`. A property named after a
        parameter's Python name is left out.
        """
        parameter_names: dict[str, str] = {}
        for name, td_field in self.parameter_fields.items():
            if 'validation_alias' not in td_field:
                continue
            field_schema = core_schema.definitions_schema(
                core_schema.typed_dict_schema({name: td_field}), self.definitions
            )
            for alias in GenerateJsonSchema().generate(field_schema).get('properties', {}):
                # A property named after a parameter's Python name keeps naming that parameter.
                if alias not in self.parameter_fields:
                    parameter_names[alias] = name
        return parameter_names

    def revalidate(self, args_dict: dict[str, Any], *, context: Any = None) -> dict[str, Any]:
        """Validate arguments that `validator` already produced, after they crossed a boundary as JSON.

        For a single model-like argument they're the `{name: value}` envelope, which `validator` reads as
        unwrapped input when the model also has a field called `name` that accepts a dict. Here the
        envelope's content is validated as the model's input, which is what the first validation produced.
        """
        if self.single_arg_name is None:
            return self.validator.validate_python(args_dict, context=context)
        return self.validator.validate_python(_Revalidated(args_dict[self.single_arg_name]), context=context)

    async def call(self, args_dict: dict[str, Any], ctx: RunContext[Any]) -> Any:
        args, kwargs = self._call_args(args_dict, ctx)
        if self.is_async:
            function = cast(Callable[[Any], Awaitable[str]], self.function)
            return await function(*args, **kwargs)
        else:
            # A plain `def` may still return an awaitable, which `run_in_executor` would leave un-awaited.
            function = cast(Callable[[Any], str | Awaitable[str]], self.function)
            return await await_maybe(await run_in_executor(function, *args, **kwargs))

    def _call_args(
        self,
        args_dict: dict[str, Any],
        ctx: RunContext[Any],
    ) -> tuple[list[Any], dict[str, Any]]:
        args = [ctx] if self.takes_ctx else []
        if self.positional_fields or self.var_positional_field:
            # Copy before popping so we never mutate the caller's dict. The same validated-args
            # dict is later handed to tool-execute hooks (e.g. `after_tool_execute`), which must
            # still observe the full set of arguments.
            args_dict = dict(args_dict)
        for positional_field in self.positional_fields:
            args.append(args_dict.pop(positional_field))
        if self.var_positional_field:
            args.extend(args_dict.pop(self.var_positional_field))

        return args, args_dict


def function_schema(
    function: Callable[..., Any],
    schema_generator: type[GenerateJsonSchema],
    *,
    tool_name: str | None = None,
    takes_ctx: bool | None = None,
    docstring_format: DocstringFormat = 'auto',
    require_parameter_descriptions: bool = False,
) -> FunctionSchema:
    """Build a Pydantic validator and JSON schema from a tool function.

    Args:
        function: The function to build a validator and JSON schema for.
        tool_name: The tool name. Defaults to `function.__name__`.
        takes_ctx: Whether the function takes a `RunContext` first argument.
        docstring_format: The docstring format to use.
        require_parameter_descriptions: Whether to require descriptions for all tool function parameters.
        schema_generator: The JSON schema generator class to use.

    Returns:
        A `FunctionSchema` instance.
    """
    config = ConfigDict(title=function.__name__, use_attribute_docstrings=True)
    config_wrapper = ConfigWrapper(config)
    gen_schema = _generate_schema.GenerateSchema(config_wrapper)
    errors: list[str] = []

    try:
        sig = signature(function)
    except ValueError as e:
        errors.append(str(e))
        sig = signature(lambda: None)
    original_func = function.func if isinstance(function, partial) else function
    function = cast(Callable[..., Any], function)  # cope with pyright changing the type from the isinstance() check.

    type_hints = get_type_hints(original_func, include_extras=True)

    description, field_descriptions = doc_descriptions(original_func, docstring_format=docstring_format)
    schema_context = _FunctionSchemaContext(
        gen_schema=gen_schema,
        type_hints=type_hints,
        field_descriptions=field_descriptions,
        require_parameter_descriptions=require_parameter_descriptions,
        # A `POSITIONAL_OR_KEYWORD` parameter that precedes `*args` must be passed positionally at call
        # time; passing it as a keyword would double-bind with the values unpacked into `*args`. When
        # there's no `*args`, such parameters keep being passed as keywords (the historical behavior).
        has_var_positional=any(p.kind is Parameter.VAR_POSITIONAL for p in sig.parameters.values()),
        errors=errors,
    )

    for index, (name, p) in enumerate(sig.parameters.items()):
        if index == 0 and takes_ctx is None:
            takes_ctx = p.annotation is not sig.empty and is_call_ctx(type_hints[name])
        _add_parameter(schema_context, index, name, p, takes_ctx=takes_ctx)

    if schema_context.missing_param_descriptions:
        errors.append(f'Missing parameter descriptions for {", ".join(schema_context.missing_param_descriptions)}')

    if errors:
        from .exceptions import UserError

        error_details = '\n  '.join(errors)
        raise UserError(f'Error generating schema for {function.__qualname__}:\n  {error_details}')

    core_config = config_wrapper.core_config(None)

    schema, single_field_name, single_arg_name, single_arg_keys = _build_schema(
        schema_context.fields, schema_context.var_kwargs_schema, core_config
    )
    schema = gen_schema.clean_schema(schema)
    parameter_fields, definitions = _cleaned_fields(schema)
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

    if single_arg_keys is not None:
        # For a single model-like arg the tool's JSON schema *is* the model's, so its property names
        # are exactly the top-level keys the model accepts (aliases already resolved by Pydantic).
        # `_validate_single_arg` reads this to tell unwrapped input from a wrapper envelope.
        single_arg_keys.update(json_schema.get('properties', {}))
    elif not json_schema.get('properties'):
        # The model wasn't asked for a field the JSON schema leaves out, so the validated args stay whole.
        single_field_name = None

    # workaround for https://github.com/pydantic/pydantic/issues/10785
    # if we build a custom TypedDict schema (matches when `single_arg_name is None`), we manually set
    # `additionalProperties` in the JSON Schema
    if single_arg_name is not None and not description:
        # if the tool description is not set, and we have a single parameter, take the description from that
        # and set it on the tool
        description = json_schema.pop('description', None)

    name = tool_name or function.__name__
    checked_json_schema = check_object_json_schema(json_schema)

    # Compute return schema eagerly (before Temporal sandbox where TypeAdapter is too slow)
    return_annotation = type_hints.get('return')
    return_schema_type = extract_return_schema_type(return_annotation, function)
    try:
        return_schema: ObjectJsonSchema = TypeAdapter(return_schema_type).json_schema(
            schema_generator=schema_generator, mode='serialization'
        )
    except (PydanticSchemaGenerationError, PydanticUserError):
        warnings.warn(
            f'Could not generate return schema for {original_func.__qualname__!r}: '
            f'unsupported return type {return_annotation!r}. Falling back to unconstrained schema.',
            UserWarning,
            stacklevel=2,
        )
        return_schema = {}

    return FunctionSchema(
        name=name,
        description=description,
        validator=schema_validator,
        json_schema=checked_json_schema,
        parameter_fields=parameter_fields,
        definitions=definitions,
        single_arg_name=single_arg_name,
        single_field_name=single_field_name,
        positional_fields=schema_context.positional_fields,
        var_positional_field=schema_context.var_positional_field,
        takes_ctx=bool(takes_ctx),
        is_async=is_async_callable(function),
        function=function,
        return_schema=return_schema,
    )


def _cleaned_fields(
    schema: core_schema.CoreSchema,
) -> tuple[dict[str, core_schema.TypedDictField], list[core_schema.CoreSchema]]:
    """Return the fields of a cleaned parameters schema and the definitions it keeps for them."""
    match schema:
        case {'type': 'definitions', 'schema': {'type': 'typed-dict', 'fields': fields}, 'definitions': definitions}:
            return fields, definitions
        case {'type': 'typed-dict', 'fields': fields}:
            return fields, []
        case _:
            return {}, []


@dataclass(kw_only=True)
class _FunctionSchemaContext:
    """The validator fields `function_schema` builds from a tool function's parameters, one parameter at a time."""

    gen_schema: _generate_schema.GenerateSchema
    type_hints: dict[str, Any]
    field_descriptions: dict[str, str]
    require_parameter_descriptions: bool
    has_var_positional: bool
    errors: list[str]
    """Where each parameter that can't be a tool argument is reported, shared with `function_schema`."""
    fields: dict[str, core_schema.TypedDictField] = field(default_factory=dict[str, core_schema.TypedDictField])
    positional_fields: list[str] = field(default_factory=list[str])
    var_positional_field: str | None = None
    var_kwargs_schema: core_schema.CoreSchema | None = None
    missing_param_descriptions: set[str] = field(default_factory=set[str])
    decorators: _decorators.DecoratorInfos = field(default_factory=_decorators.DecoratorInfos)


def _add_parameter(
    schema_context: _FunctionSchemaContext, index: int, name: str, p: Parameter, *, takes_ctx: bool | None
) -> None:
    """Add the parameter `p` at `index`, unless it is the `RunContext` argument or misuses `RunContext`."""
    if p.annotation is Parameter.empty:
        if takes_ctx and index == 0:
            # should be the `context` argument, skip
            return
        # TODO warn?
        annotation = Any
    else:
        annotation = schema_context.type_hints[name]

        if index == 0 and takes_ctx:
            if not is_call_ctx(annotation):
                schema_context.errors.append(
                    'First parameter of tools that take context must be annotated with RunContext[...]'
                )
            return
        elif not takes_ctx and is_call_ctx(annotation):
            schema_context.errors.append('RunContext annotations can only be used with tools that take context')
            return
        elif index != 0 and is_call_ctx(annotation):
            schema_context.errors.append('RunContext annotations can only be used as the first argument')
            return

    _add_field(schema_context, p, annotation)


def _add_field(schema_context: _FunctionSchemaContext, p: Parameter, annotation: Any) -> None:
    field_name = p.name

    if schema_context.require_parameter_descriptions and field_name not in schema_context.field_descriptions:
        schema_context.missing_param_descriptions.add(field_name)

    if p.kind == Parameter.VAR_KEYWORD:
        schema_context.var_kwargs_schema = schema_context.gen_schema.generate_schema(annotation)
        return

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
        field_info.description = schema_context.field_descriptions.get(field_name)

    td_schema = schema_context.gen_schema._generate_td_field_schema(  # pyright: ignore[reportPrivateUsage]
        field_name,
        field_info,
        schema_context.decorators,
        required=required,
    )
    schema_context.fields[field_name] = td_schema
    # noinspection PyTypeChecker
    metadata = td_schema.setdefault('metadata', {})
    metadata['is_model_like'] = is_model_like(annotation)

    if p.kind == Parameter.POSITIONAL_ONLY or (
        schema_context.has_var_positional and p.kind == Parameter.POSITIONAL_OR_KEYWORD
    ):
        schema_context.positional_fields.append(field_name)
    elif p.kind == Parameter.VAR_POSITIONAL:
        schema_context.var_positional_field = field_name


P = ParamSpec('P')
R = TypeVar('R')


WithCtx = Callable[Concatenate[RunContext[Any], P], R]
WithoutCtx = Callable[P, R]
TargetCallable = WithCtx[P, R] | WithoutCtx[P, R]


def takes_ctx(callable_obj: TargetCallable[P, R]) -> TypeIs[WithCtx[P, R]]:
    """Check if a callable takes a `RunContext` first argument.

    Args:
        callable_obj: The callable to check.

    Returns:
        `True` if the callable takes a `RunContext` as first argument, `False` otherwise.
    """
    return takes_run_context(callable_obj)


def _build_schema(
    fields: dict[str, core_schema.TypedDictField],
    var_kwargs_schema: core_schema.CoreSchema | None,
    core_config: core_schema.CoreConfig,
) -> tuple[core_schema.CoreSchema, str | None, str | None, set[str] | None]:
    """Generate a typed dict schema for function parameters.

    Args:
        fields: The fields to generate a typed dict schema for.
        var_kwargs_schema: The variable keyword arguments schema.
        core_config: The core configuration.

    Returns:
        tuple of (generated core schema, single field name, single arg name, single arg model keys).
        The single field name is set when there is exactly one field and no `**kwargs`, and the single
        arg name only when that field is also model-like. `function_schema` drops the single field name
        when the generated JSON schema leaves a primitive field out. The keys set is empty here and filled
        in by `function_schema` from the generated JSON schema.
    """
    single_field_name: str | None = None
    if len(fields) == 1 and var_kwargs_schema is None:
        name = single_field_name = next(iter(fields))
        td_field = fields[name]
        metadata = td_field.get('metadata') or {}
        if metadata.get('is_model_like'):
            # The JSON schema sent to the model is the model-like parameter's schema directly (unwrapped),
            # so the model generates its fields at the top level rather than inside a redundant wrapper.
            # The validator output is wrapped to `{name: value}` so validated args are always a dict
            # keyed by parameter name — matching the contract that hooks and `call_tool` rely on.
            # Use a wrap validator so we also accept the already-wrapped `{name: value}` shape, which
            # a caller re-validating previously validated args after a serialization round-trip passes
            # unless it goes through `FunctionSchema.revalidate`.
            # `accepted_keys` lets the validator tell that wrapper shape apart from genuine unwrapped
            # input for a model with a field (or alias) named `name`; `function_schema` fills it from
            # the generated JSON schema so we don't rebuild the model's schema just to read its keys.
            accepted_keys: set[str] = set()
            return (
                core_schema.no_info_wrap_validator_function(
                    partial(_validate_single_arg, name=name, accepted_keys=accepted_keys),
                    td_field['schema'],
                ),
                name,
                name,
                accepted_keys,
            )

    extra_behavior: Literal['allow', 'forbid'] = 'allow' if var_kwargs_schema else 'forbid'
    td_schema = core_schema.typed_dict_schema(
        fields,
        config=core_config,
        extra_behavior=extra_behavior,
        extras_schema=var_kwargs_schema,
    )
    return td_schema, single_field_name, None, None


@dataclass
class _Revalidated:
    """A single model-like argument's already-validated value, passed in by `FunctionSchema.revalidate`."""

    value: Any


def _is_wrapped_single_arg(value: Any, name: str) -> TypeIs[dict[Any, Any]]:
    return isinstance(value, dict) and list(cast(dict[Any, Any], value)) == [name]


def _validate_single_arg(
    value: Any,
    handler: core_schema.ValidatorFunctionWrapHandler,
    *,
    name: str,
    accepted_keys: set[str],
) -> dict[str, Any]:
    if isinstance(value, _Revalidated):
        return {name: handler(value.value)}
    if not _is_wrapped_single_arg(value, name):
        # Plain unwrapped model input, as emitted against the flattened JSON schema.
        return {name: handler(value)}
    if name not in accepted_keys:
        # `name` isn't a key the model accepts, so `{name: ...}` can only be a wrapper envelope (e.g.
        # re-validated args after a serialization round-trip). Unwrap it; a bad payload still raises here.
        return {name: handler(value[name])}
    # `name` is a real field or alias, so `{name: ...}` is normally genuine unwrapped input. Validate it
    # as-is, falling back to unwrapping the envelope only when that fails (the round-trip of such a model).
    # If the field accepts both shapes (e.g. it's typed `Any`) the two are indistinguishable; we prefer
    # the unwrapped reading, so re-validating such a model's envelope here isn't idempotent. Callers that
    # know they hold an envelope use `FunctionSchema.revalidate`, which doesn't guess.
    try:
        return {name: handler(value)}
    except ValidationError:
        return {name: handler(value[name])}


def extract_return_schema_type(return_annotation: Any, function: Callable[..., Any]) -> Any:
    """Extract the type to generate a return schema for.

    Always returns a type — every function has a return schema:
    - No annotation (`None` from `get()`) → `Any` (produces `{}`)
    - `-> None` (`type(None)`) → `type(None)` (produces `{"type": "null"}`)
    - `-> Any` → `Any` (produces `{}`)
    - `-> Self` → resolved to owning class for bound methods
    - Bare `ToolReturn` → `Any` (pre-generic legacy form)
    - `ToolReturn[Any]` → `Any` (produces `{}`)
    - `ToolReturn[T]` → `T`
    - Other types → the type itself
    """
    if return_annotation is None:
        # No annotation — untyped, same as Any
        return Any
    if return_annotation is type(None):
        return type(None)
    # Bare ToolReturn without type parameter — pre-generic legacy form
    if return_annotation is ToolReturn:
        return Any
    # Resolve Self to the owning class for bound methods.
    # Only works when the function is already bound (e.g. instance.method);
    # unbound methods and classmethods fall back to Any since there's no
    # instance to infer the class from.
    if return_annotation is Self:
        self_obj = getattr(function, '__self__', None)
        if self_obj is not None:
            return cast(type[Any], type(self_obj))
        return Any
    if get_origin(return_annotation) is ToolReturn:
        type_args = get_args(return_annotation)
        inner_type = type_args[0] if type_args else Any
        return inner_type
    return return_annotation


def is_call_ctx(annotation: Any) -> bool:
    """Return whether the annotation is the `RunContext` class, parameterized or not."""
    return annotation is RunContext or get_origin(annotation) is RunContext


def find_typed_parameter(
    function: Callable[..., Any],
    type_hints: dict[str, Any],
    predicate: Callable[[Any], bool],
    type_name: str,
    callable_kind: str = 'Callable',
) -> str | None:
    """Find the sole parameter matching an annotation predicate, rejecting ambiguous signatures."""
    parameters = [name for name, annotation in type_hints.items() if name != 'return' and predicate(annotation)]
    if len(parameters) > 1:
        from .exceptions import UserError

        raise UserError(f'{callable_kind} {function.__qualname__!r} cannot take more than one `{type_name}` parameter.')
    return parameters[0] if parameters else None


def validate_schema_signature(
    function: Callable[..., Any],
    sig: Signature,
    type_hints: dict[str, Any],
    ctx_parameter: str | None,
) -> None:
    """Validate annotations needed to build a schema around an optional `RunContext` parameter."""
    if ctx_parameter is not None and sig.parameters[ctx_parameter].kind is Parameter.VAR_POSITIONAL:
        from .exceptions import UserError

        raise UserError('RunContext cannot be used as a variadic positional parameter (`*args`)')
    for parameter in sig.parameters.values():
        if parameter.name != ctx_parameter and parameter.name not in type_hints:
            from .exceptions import UserError

            raise UserError(
                f'Error generating schema for {function.__qualname__}:\n'
                f'  Parameter {parameter.name!r} must have a type annotation'
            )
