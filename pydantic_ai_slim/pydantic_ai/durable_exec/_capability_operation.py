from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, ParamSpec, TypeVar, cast, get_type_hints, overload

from pydantic_ai._function_schema import (
    FunctionSchema,
    _extract_return_schema_type,  # pyright: ignore[reportPrivateUsage]
    _is_call_ctx,  # pyright: ignore[reportPrivateUsage]
    function_schema,
)
from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.capabilities.abstract import AbstractCapability, leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import GenerateToolJsonSchema, RunContext
from pydantic_ai.usage import RunUsage

from ._operation_backend import BoundDurableOperation

R = TypeVar('R')
P = ParamSpec('P')
A = TypeVar('A', bound=Awaitable[Any])


@dataclass(frozen=True)
class CapabilityOperationParams:
    run_context: RunContext[Any]
    arguments: dict[str, Any]
    model_id: str | None = None


@dataclass(frozen=True)
class _CapabilityOperationResult(Generic[R]):
    value: R
    usage_delta: RunUsage


def _operation_result_type(result_type: object) -> object:  # pyright: ignore[reportUnusedFunction]
    return cast(Any, _CapabilityOperationResult)[result_type]


@dataclass
class ModelRequestContextProjection:
    messages: list[ModelMessage]
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    model_id: str | None
    streaming: bool

    @classmethod
    def from_context(cls, context: ModelRequestContext) -> ModelRequestContextProjection:
        return cls(
            context.messages,
            cast(dict[str, Any] | None, context.model_settings),
            context.model_request_parameters,
            context.model_id,
            context.streaming,
        )

    def apply(self, context: ModelRequestContext) -> None:
        context.messages = self.messages
        context.model_settings = cast(ModelSettings | None, self.model_settings)
        context.model_request_parameters = self.model_request_parameters
        context.model_id = self.model_id
        context.streaming = self.streaming


@dataclass(frozen=True)
class CapabilityMethodDeclaration:
    name: str
    function: Callable[..., Awaitable[Any]]
    signature: inspect.Signature
    schema: FunctionSchema
    result_type: object
    ctx_parameter: str | None
    model_request_hook: bool = False


class CapabilityCacheIdentity:
    """Project the model, validated arguments, and run context into Prefect's cache identity.

    The model separates registered model targets, the arguments identify the operation input,
    and the run context contributes durable run and step identity through Prefect's policy.
    """

    def project(self, params: CapabilityOperationParams) -> tuple[object, ...]:
        return (params.model_id, params.arguments, params.run_context)


@dataclass(frozen=True)
class _DurableOperationMarker:
    name: str
    function: Callable[..., Awaitable[Any]]
    tier_one: bool = False


def _operation_name(function: Callable[..., Any], name: str | None) -> str:
    return name or function.__name__.removeprefix('_')


@overload
def durable_operation(function: Callable[P, A], /) -> Callable[P, A]: ...


@overload
def durable_operation(*, name: str | None = None) -> Callable[[Callable[P, A]], Callable[P, A]]: ...


def durable_operation(function: Any = None, /, *, name: str | None = None) -> Any:
    """Declare an async capability method as a durable operation.

    The method keeps its original signature. During a run with a durability capability, calls
    dispatch through that engine's activity, step, or task. Without durability, calls await the
    original method directly. The optional `name` is the stable operation name within the
    capability's required stable `id`.

    ```python
    from pydantic_ai.capabilities import AbstractCapability, durable_operation
    from pydantic_ai.tools import RunContext


    class Audit(AbstractCapability[None]):
        id = 'audit'

        @durable_operation
        async def record(self, ctx: RunContext[None], message: str) -> bool:
            return bool(message)
    ```

    Args:
        function: The async capability method when used as `@durable_operation`.
        name: An explicit stable name when used as `@durable_operation(name='...')`.

    Returns:
        The marked method with its parameter and return types preserved.

    Raises:
        TypeError: If the decorated method is synchronous.
    """

    def decorate(target: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
        if not inspect.iscoroutinefunction(target):
            if target.__name__ in ('get_toolset', 'get_wrapper_toolset'):
                setattr(
                    target,
                    '__pydantic_ai_durable_operation__',
                    _DurableOperationMarker(_operation_name(target, name), cast(Callable[..., Awaitable[Any]], target)),
                )
                return target
            raise TypeError('`durable_operation` can only decorate async methods')
        marker = _DurableOperationMarker(_operation_name(target, name), target)

        @wraps(target)
        async def decorated(self: AbstractCapability[Any], *args: Any, **kwargs: Any) -> R:
            bound = inspect.signature(target).bind(self, *args, **kwargs)
            ctx: RunContext[Any] | None = get_current_run_context()
            for value in bound.arguments.values():
                if isinstance(value, RunContext):
                    ctx = cast(RunContext[Any], value)
                    break
            if ctx is None:
                return await target(self, *args, **kwargs)
            request_context = next(
                (value for value in bound.arguments.values() if isinstance(value, ModelRequestContext)), None
            )
            if isinstance(request_context, ModelRequestContext):
                projection = ModelRequestContextProjection.from_context(request_context)
                dispatch_args = tuple(projection if value is request_context else value for value in args)
                dispatch_kwargs = {
                    key: projection if value is request_context else value for key, value in kwargs.items()
                }
            else:
                dispatch_args = args
                dispatch_kwargs = kwargs
            handler = target.__get__(self, type(self))
            result = await ctx.durable_operation(self, marker.name, handler)(*dispatch_args, **dispatch_kwargs)
            if request_context is not None and isinstance(result, ModelRequestContextProjection):
                result.apply(request_context)
                return cast(R, request_context)
            return cast(R, result)

        setattr(decorated, '__pydantic_ai_durable_operation__', marker)
        return decorated

    return decorate(function) if function is not None else decorate


def tier_one_durable_operation(function: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
    """Mark a base hook whose overrides are inherently durable."""
    setattr(
        function,
        '__pydantic_ai_durable_operation__',
        _DurableOperationMarker(_operation_name(function, None), cast(Callable[..., Awaitable[Any]], function), True),
    )
    return function


_NEVER_DURABLE_HOOKS = {
    'get_toolset': '`get_toolset` returns a live toolset and cannot be a durable operation.',
    'get_wrapper_toolset': '`get_wrapper_toolset` returns a live toolset and cannot be a durable operation.',
    'wrap_run': '`wrap_run` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_node_run': '`wrap_node_run` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_model_request': '`wrap_model_request` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_tool_validate': '`wrap_tool_validate` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_tool_execute': '`wrap_tool_execute` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_output_validate': '`wrap_output_validate` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_output_process': '`wrap_output_process` receives a handler callable, which cannot cross a durable boundary.',
    'wrap_run_event_stream': '`wrap_run_event_stream` receives a live stream and cannot be a durable operation.',
}


def collect_capability_operations(  # noqa: C901
    capability: AbstractCapability[Any],
) -> dict[str, CapabilityMethodDeclaration]:
    handlers = dict(capability.get_durable_operations() or {})
    for base in type(capability).__mro__[1:]:
        for method_name, base_member in vars(base).items():
            marker = cast(
                _DurableOperationMarker | None, getattr(base_member, '__pydantic_ai_durable_operation__', None)
            )
            if marker is None or not marker.tier_one:
                continue
            member = getattr(type(capability), method_name)
            if member is not base_member:
                if marker.name in handlers:
                    raise UserError(
                        f'Duplicate durable operation name {marker.name!r} on capability {capability.id!r}. '
                        'Use `@durable_operation(name=...)` or change the hook key.'
                    )
                handlers[marker.name] = cast(Callable[..., Awaitable[Any]], member)

    for method_name, member in inspect.getmembers(type(capability)):
        marker = cast(_DurableOperationMarker | None, getattr(member, '__pydantic_ai_durable_operation__', None))
        if marker is None:
            continue
        if marker.tier_one and member is marker.function:
            continue
        if reason := _NEVER_DURABLE_HOOKS.get(method_name):
            raise UserError(reason)
        if marker.name in handlers:
            raise UserError(
                f'Duplicate durable operation name {marker.name!r} on capability {capability.id!r}. '
                'Use `@durable_operation(name=...)` or change the hook key.'
            )
        handlers[marker.name] = cast(Callable[..., Awaitable[Any]], member)

    declarations: dict[str, CapabilityMethodDeclaration] = {}
    for operation_name, handler in handlers.items():
        if not callable(handler):
            raise UserError(f'Durable operation {operation_name!r} must be an async callable.')
        handler = cast(Callable[..., Awaitable[Any]], handler)
        if not inspect.iscoroutinefunction(handler):
            raise UserError(
                f'Durable operation {operation_name!r} on capability {capability.id!r} must be an async callable.'
            )
        original = cast(_DurableOperationMarker | None, getattr(handler, '__pydantic_ai_durable_operation__', None))
        function = original.function if original is not None else handler
        bound = function.__get__(capability, type(capability))
        model_request_hook = function.__name__ == 'before_model_request'
        schema_target = _model_request_schema if model_request_hook else bound
        signature = inspect.signature(schema_target)
        ctx_parameters = [
            name
            for name, annotation in get_type_hints(bound, include_extras=True).items()
            if name != 'return' and _is_call_ctx(annotation)
        ]
        if len(ctx_parameters) > 1:
            raise UserError(
                f'Durable operation {function.__qualname__!r} cannot take more than one `RunContext` parameter.'
            )
        if model_request_hook and ctx_parameters and ctx_parameters[0] != 'ctx':
            signature = signature.replace(
                parameters=[
                    parameter.replace(name=ctx_parameters[0]) if parameter.name == 'ctx' else parameter
                    for parameter in signature.parameters.values()
                ]
            )
        for parameter in signature.parameters.values():
            if parameter.name not in ctx_parameters and parameter.annotation is inspect.Parameter.empty:
                raise UserError(
                    f'Error generating schema for {function.__qualname__}:\n'
                    f'  Parameter {parameter.name!r} must have a type annotation'
                )
        schema = _capability_operation_schema(schema_target, signature, ctx_parameters[0] if ctx_parameters else None)
        declarations[operation_name] = CapabilityMethodDeclaration(
            operation_name,
            function,
            signature,
            schema,
            ModelRequestContextProjection
            if model_request_hook
            else _extract_return_schema_type(get_type_hints(bound, include_extras=True).get('return'), bound),
            ctx_parameters[0] if ctx_parameters else None,
            model_request_hook,
        )
    return declarations


def _capability_operation_schema(
    function: Callable[..., Awaitable[Any]], signature: inspect.Signature, ctx_parameter: str | None
) -> FunctionSchema:
    if ctx_parameter is None:
        return function_schema(function, GenerateToolJsonSchema)

    context = signature.parameters[ctx_parameter]
    if context.kind is inspect.Parameter.VAR_POSITIONAL:
        raise UserError('RunContext cannot be used as a variadic positional parameter (`*args`)')

    type_hints = get_type_hints(function, include_extras=True)
    if ctx_parameter not in type_hints:
        source_ctx_parameter = next(name for name, annotation in type_hints.items() if _is_call_ctx(annotation))
        type_hints[ctx_parameter] = type_hints.pop(source_ctx_parameter)

    async def schema_target(**kwargs: Any) -> Any:  # pragma: no cover
        return kwargs

    schema_target.__name__ = function.__name__
    schema_target.__qualname__ = function.__qualname__
    schema_target.__doc__ = function.__doc__
    schema_target.__annotations__ = {
        name: annotation for name, annotation in type_hints.items() if name != ctx_parameter
    }
    schema_signature = signature.replace(
        parameters=[p for p in signature.parameters.values() if p.name != ctx_parameter]
    )
    cast(Any, schema_target).__signature__ = schema_signature
    return function_schema(schema_target, GenerateToolJsonSchema, takes_ctx=False)


def bind_arguments(
    declaration: CapabilityMethodDeclaration,
    ctx: RunContext[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    bound = declaration.signature.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    if declaration.ctx_parameter is not None:
        arguments.pop(declaration.ctx_parameter)
    for name, parameter in declaration.signature.parameters.items():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(arguments.pop(name))
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            arguments[name] = list(arguments[name])
    return cast(dict[str, Any], declaration.schema.validator.validate_python(arguments))


async def call_declaration(
    declaration: CapabilityMethodDeclaration,
    capability: AbstractCapability[Any],
    params: CapabilityOperationParams,
) -> Any:
    if declaration.model_request_hook:
        raise RuntimeError('Model-request hook declarations require the durability model scope')
    bound = declaration.function.__get__(capability, type(capability))
    arguments = dict(params.arguments)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, parameter in declaration.signature.parameters.items():
        if name == declaration.ctx_parameter:
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = params.run_context
            else:
                args.append(params.run_context)
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(arguments)
            continue
        value = arguments.pop(name)

        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            args.append(value)
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        else:
            kwargs[name] = value
    return await bound(*args, **kwargs)


def recover_capability(ctx: RunContext[Any], capability_id: str) -> AbstractCapability[Any]:
    run_capabilities = cast(dict[str, AbstractCapability[Any]], ctx.__dict__.get('_run_capabilities_by_id', {}))
    if capability := run_capabilities.get(capability_id):
        return capability
    agent = ctx.agent
    if agent is None:
        raise RuntimeError('A durable capability operation requires the worker agent on `RunContext`.')
    matches = [cap for cap in leaf_capabilities(agent.root_capability) if cap.id == capability_id]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one bound capability with id {capability_id!r}, found {len(matches)}.')
    return matches[0]


CapabilityBoundOperation = BoundDurableOperation[CapabilityOperationParams, Any, Any]


async def _model_request_schema(
    ctx: RunContext[Any], request_context: ModelRequestContextProjection
) -> ModelRequestContextProjection:
    return request_context
