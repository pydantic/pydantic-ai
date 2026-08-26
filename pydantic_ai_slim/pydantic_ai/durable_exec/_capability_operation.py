from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast, get_type_hints, overload

from pydantic_ai._function_schema import (
    FunctionSchema,
    _extract_return_schema_type,  # pyright: ignore[reportPrivateUsage]
    function_schema,
)
from pydantic_ai.capabilities.abstract import AbstractCapability, leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import GenerateToolJsonSchema, RunContext

from ._operation_backend import BoundDurableOperation

R = TypeVar('R')
P = ParamSpec('P')
A = TypeVar('A', bound=Awaitable[Any])


@dataclass(frozen=True)
class CapabilityOperationParams:
    run_context: RunContext[Any]
    arguments: dict[str, Any]


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
    model_request_hook: bool = False


class CapabilityCacheIdentity:
    """Use every validated parameter as the Prefect cache identity."""

    def project(self, params: CapabilityOperationParams) -> tuple[object, ...]:
        return (params.arguments,)


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
    """Declare an async capability method as a durable operation."""

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
        async def decorated(self: AbstractCapability[Any], ctx: RunContext[Any], *args: Any, **kwargs: Any) -> R:
            agent = ctx.agent
            if agent is not None:
                from ._base import BaseDurabilityCapability

                durability = next(
                    (
                        cap
                        for cap in leaf_capabilities(agent.root_capability)
                        if isinstance(cap, BaseDurabilityCapability)
                    ),
                    None,
                )
                if durability is not None and durability.in_durable_context:
                    request_context = args[0] if target.__name__ == 'before_model_request' and args else None
                    dispatch_args = (
                        (ModelRequestContextProjection.from_context(request_context),)
                        if isinstance(request_context, ModelRequestContext)
                        else args
                    )
                    return cast(
                        R,
                        await _dispatch_and_apply_model_request_projection(
                            durability, self, marker.name, ctx, dispatch_args, kwargs, request_context
                        ),
                    )
            return await target(self, ctx, *args, **kwargs)

        setattr(decorated, '__pydantic_ai_durable_operation__', marker)
        return decorated

    return decorate(function) if function is not None else decorate


async def _dispatch_and_apply_model_request_projection(
    durability: Any,
    capability: AbstractCapability[Any],
    operation: str,
    ctx: RunContext[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    request_context: ModelRequestContext | None,
) -> Any:
    result = await durability._invoke_capability_operation(capability, operation, ctx, args, kwargs)
    if request_context is not None and isinstance(result, ModelRequestContextProjection):
        result.apply(request_context)
        return request_context
    return result


async def call_tier_one_operation(
    capability: AbstractCapability[Any], operation: str, ctx: RunContext[Any], *args: Any
) -> Any:
    """Call an inherently durable hook through the bound engine when one is active."""
    if getattr(type(capability), operation) is getattr(AbstractCapability, operation):
        return await getattr(capability, operation)(ctx, *args)
    agent = ctx.agent
    if agent is not None:
        from ._base import BaseDurabilityCapability

        durability = next(
            (cap for cap in leaf_capabilities(agent.root_capability) if isinstance(cap, BaseDurabilityCapability)),
            None,
        )
        if durability is not None and durability.in_durable_context:
            return await durability._invoke_capability_operation(  # pyright: ignore[reportPrivateUsage]
                capability, operation, ctx, args, {}
            )
    return await getattr(capability, operation)(ctx, *args)


def tier_one_durable_operation(function: Callable[..., Awaitable[R]]) -> Callable[..., Awaitable[R]]:
    """Mark a base hook whose overrides are inherently durable."""
    setattr(
        function,
        '__pydantic_ai_durable_operation__',
        _DurableOperationMarker(_operation_name(function, None), cast(Callable[..., Awaitable[Any]], function), True),
    )
    return function


_NEVER_DURABLE_HOOKS = {
    'get_sandbox': '`get_sandbox` returns a live sandbox connection, which cannot cross a durable boundary.',
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


tier_one_durable_operation(AbstractCapability.create_sandbox)
tier_one_durable_operation(AbstractCapability.destroy_sandbox)


def collect_capability_operations(capability: AbstractCapability[Any]) -> dict[str, CapabilityMethodDeclaration]:
    from pydantic_ai.capabilities import WrapperCapability

    handlers = dict(capability.get_durable_operations() or {})
    for base in type(capability).__mro__[1:]:
        for method_name, base_member in vars(base).items():
            marker = cast(
                _DurableOperationMarker | None, getattr(base_member, '__pydantic_ai_durable_operation__', None)
            )
            if marker is None or not marker.tier_one:
                continue
            member = getattr(type(capability), method_name)
            if member is base_member or (
                isinstance(capability, WrapperCapability) and member is getattr(WrapperCapability, method_name)
            ):
                continue
            if marker.name in handlers:
                raise UserError(f'Duplicate durable operation name {marker.name!r} on capability {capability.id!r}.')
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
            raise UserError(f'Duplicate durable operation name {marker.name!r} on capability {capability.id!r}.')
        handlers[marker.name] = cast(Callable[..., Awaitable[Any]], member)

    declarations: dict[str, CapabilityMethodDeclaration] = {}
    for operation_name, handler in handlers.items():
        original = cast(_DurableOperationMarker | None, getattr(handler, '__pydantic_ai_durable_operation__', None))
        function = original.function if original is not None else handler
        bound = function.__get__(capability, type(capability))
        model_request_hook = function.__name__ == 'before_model_request'
        schema_target = _model_request_schema if model_request_hook else bound
        signature = inspect.signature(schema_target)
        for parameter in signature.parameters.values():
            if (
                parameter.name != next(iter(signature.parameters), None)
                and parameter.annotation is inspect.Parameter.empty
            ):
                raise UserError(
                    f'Error generating schema for {function.__qualname__}:\n'
                    f'  Parameter {parameter.name!r} must have a type annotation'
                )
        schema = function_schema(schema_target, GenerateToolJsonSchema)
        if not schema.takes_ctx:
            raise UserError(
                f'Durable operation {function.__qualname__!r} must take `RunContext` as its first argument.'
            )
        declarations[operation_name] = CapabilityMethodDeclaration(
            operation_name,
            function,
            signature,
            schema,
            ModelRequestContextProjection
            if model_request_hook
            else _extract_return_schema_type(get_type_hints(bound, include_extras=True).get('return'), bound),
            model_request_hook,
        )
    return declarations


def bind_arguments(
    declaration: CapabilityMethodDeclaration,
    ctx: RunContext[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    bound = declaration.signature.bind(ctx, *args, **kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop(next(iter(declaration.signature.parameters)))
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
    return await replace(declaration.schema, function=bound).call(params.arguments, params.run_context)


def recover_capability(ctx: RunContext[Any], capability_id: str) -> AbstractCapability[Any]:
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
