from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass, replace
from functools import update_wrapper, wraps
from typing import Any, Generic, ParamSpec, TypeVar, cast, get_type_hints

from pydantic_ai._function_schema import (
    FunctionSchema,
    extract_return_schema_type,
    find_typed_parameter,
    function_schema,
    is_call_ctx,
    validate_schema_signature,
)
from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.capabilities.abstract import AbstractCapability, leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import GenerateToolJsonSchema, RunContext
from pydantic_ai.usage import RunUsage

from ._operation import CacheIdentity
from ._operation_backend import BoundDurableOperation

ResultT = TypeVar('ResultT')
P = ParamSpec('P')
A = TypeVar('A', bound=Awaitable[Any])

_SYNC_NEVER_DURABLE_HOOKS = frozenset({'get_toolset', 'get_wrapper_toolset'})
"""Sync hooks marked so collection can report why they cannot be durable.

The decorator still attaches a marker to these sync hooks so binding raises the specific
never-durable error instead of the generic sync-method error.
"""
_WRAP_NEVER_DURABLE_HOOKS = (
    'wrap_run',
    'wrap_node_run',
    'wrap_model_request',
    'wrap_tool_validate',
    'wrap_tool_execute',
    'wrap_output_validate',
    'wrap_output_process',
)
_NEVER_DURABLE_HOOKS = {
    **{
        name: f'`{name}` receives a handler callable, which cannot cross a durable boundary.'
        for name in _WRAP_NEVER_DURABLE_HOOKS
    },
    'wrap_run_event_stream': '`wrap_run_event_stream` receives a live stream and cannot be a durable operation.',
    'get_toolset': '`get_toolset` returns a live toolset and cannot be a durable operation.',
    'get_wrapper_toolset': '`get_wrapper_toolset` returns a live toolset and cannot be a durable operation.',
}


@dataclass(frozen=True)
class CapabilityOperationParams:
    run_context: RunContext[Any]
    _: KW_ONLY
    arguments: dict[str, Any]
    model_id: str | None = None


@dataclass(frozen=True)
class CapabilityOperationResult(Generic[ResultT]):
    value: ResultT
    _: KW_ONLY
    usage_delta: RunUsage


def capability_operation_result_type(result_type: object) -> type[CapabilityOperationResult[Any]]:
    """Build `CapabilityOperationResult[ResultT]` for the operation's declared result type.

    Codecs use the parametrized wrapper to validate both the operation value and its usage delta
    when serializing and reconstructing `CapabilityOperationResult[ResultT]` across a durable boundary.
    """
    return cast(type[CapabilityOperationResult[Any]], cast(Any, CapabilityOperationResult)[result_type])


@dataclass
class ModelRequestContextProjection:
    """Serializable projection of a model request context for a durable boundary.

    `ModelRequestContext` carries a live `Model`, which cannot cross a durable boundary. This
    projection carries the serializable request state plus the registered `model_id`, then rebuilds
    or updates the live context on the other side.
    """

    messages: list[ModelMessage]
    _: KW_ONLY
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    model_id: str | None
    streaming: bool

    @classmethod
    def from_context(cls, context: ModelRequestContext) -> ModelRequestContextProjection:
        return cls(
            messages=context.messages,
            model_settings=cast(dict[str, Any] | None, context.model_settings),
            model_request_parameters=context.model_request_parameters,
            model_id=context.model_id,
            streaming=context.streaming,
        )

    def build_context(self, model: Model) -> ModelRequestContext:
        """Build the worker-side live model request context from this projection."""
        context = ModelRequestContext(
            model=model,
            messages=self.messages,
            model_settings=cast(ModelSettings | None, self.model_settings),
            model_request_parameters=self.model_request_parameters,
        )
        context.model_id = self.model_id
        context.streaming = self.streaming
        return context

    def apply(self, context: ModelRequestContext, model: Model | None = None) -> None:
        if model is not None:
            context.model = model
        context.messages = self.messages
        context.model_settings = cast(ModelSettings | None, self.model_settings)
        context.model_request_parameters = self.model_request_parameters
        context.model_id = self.model_id
        context.streaming = self.streaming


@dataclass(frozen=True)
class _ResolvedModelRequestContext:
    projection: ModelRequestContextProjection
    _: KW_ONLY
    model: Model | None = None


@dataclass(frozen=True)
class CapabilityMethodDeclaration:
    name: str
    _: KW_ONLY
    function: Callable[..., Awaitable[Any]]
    signature: inspect.Signature
    schema: FunctionSchema
    result_type: object
    ctx_parameter: str | None
    model_request_parameter: str | None = None

    @property
    def model_request_hook(self) -> bool:
        return self.model_request_parameter is not None


class CapabilityCacheIdentity(CacheIdentity[CapabilityOperationParams]):
    """Project the model, validated arguments, and run context into the cache identity.

    Hash-keyed engines consult this projection, while sequence-keyed engines ignore it. The model
    separates registered targets, the arguments identify the input, and the run context contributes
    durable run and step identity where the engine's cache policy supports it.
    """

    def project(self, params: CapabilityOperationParams) -> tuple[object, ...]:
        return (params.model_id, params.arguments, params.run_context)


@dataclass(frozen=True)
class _DurableOperationMarker:
    name: str
    _: KW_ONLY
    function: Callable[..., Awaitable[Any]]
    base_hook: bool = False


Marker = _DurableOperationMarker
_MARKER_ATTRIBUTE = '__pydantic_ai_durable_operation__'


def get_durable_operation_marker(obj: object) -> Marker | None:
    """Return the durable-operation marker attached to `obj`, if present."""
    return cast(Marker | None, getattr(obj, _MARKER_ATTRIBUTE, None))


def set_durable_operation_marker(obj: object, marker: Marker) -> None:
    """Attach a durable-operation `marker` to `obj`."""
    setattr(obj, _MARKER_ATTRIBUTE, marker)


def _validate_operation_name(name: object) -> str:
    if not isinstance(name, str):
        raise TypeError(f'`durable_operation` name must be a string, got {type(name).__name__}')
    if not name:
        raise ValueError('`durable_operation` name must not be empty')
    return name


def durable_operation(name: str) -> Callable[[Callable[P, A]], Callable[P, A]]:
    """Declare an async capability method as a durable operation.

    The method keeps its original signature. During a run with a durability capability, calls
    dispatch through that engine's activity, step, or task. Without durability, calls await the
    original method directly. The required `name` becomes part of persisted durable-unit names
    within the capability's required stable `id`, so it must essentially never change. The Python
    method itself can be renamed freely as long as `name` stays the same.

    ```python
    from pydantic_ai.capabilities import AbstractCapability, durable_operation
    from pydantic_ai.tools import RunContext


    class Audit(AbstractCapability[None]):
        id = 'audit'

        @durable_operation(name='record')
        async def record(self, ctx: RunContext[None], message: str) -> bool:
            return bool(message)
    ```

    Args:
        name: The stable operation name. This can be passed positionally or by keyword.

    Returns:
        The marked method with its parameter and return types preserved.

    Raises:
        TypeError: If used without an explicit name or if the decorated method is synchronous.
        ValueError: If `name` is empty.
    """
    if callable(name):
        raise TypeError(
            '`durable_operation` requires an explicit operation name because it becomes persisted compatibility data '
            "and must not change when the function is renamed. Use `@durable_operation(name='operation_name')`."
        )
    name = _validate_operation_name(name)

    def decorate(target: Callable[P, A]) -> Callable[P, A]:
        if not inspect.iscoroutinefunction(target):
            if target.__name__ in _SYNC_NEVER_DURABLE_HOOKS:
                set_durable_operation_marker(
                    target,
                    Marker(
                        name=name,
                        function=cast(Callable[..., Awaitable[Any]], target),
                    ),
                )
                return target
            raise TypeError('`durable_operation` can only decorate async methods')
        marker = Marker(name=name, function=cast(Callable[..., Awaitable[Any]], target))

        @wraps(target)
        async def decorated(self: AbstractCapability[Any], *args: Any, **kwargs: Any) -> Any:
            # Bind the call so context parameters are visible regardless of calling style.
            bound = inspect.signature(target).bind(self, *args, **kwargs)

            # Find the explicit or ambient run context that selects durable dispatch.
            ctx: RunContext[Any] | None = get_current_run_context()
            for value in bound.arguments.values():
                if isinstance(value, RunContext):
                    ctx = cast(RunContext[Any], value)
                    break
            if ctx is None:
                # Agent runs and durable scopes set the ambient context. Calls outside either scope
                # deliberately pass through to the undecorated method.
                return await cast(Callable[..., Awaitable[Any]], target)(self, *args, **kwargs)

            request_context = next(
                (value for value in bound.arguments.values() if isinstance(value, ModelRequestContext)), None
            )

            # Resolve the per-run operation first, then the agent-bound fallback.
            handler = target.__get__(self, type(self))
            operations = ctx._durable_operations  # pyright: ignore[reportPrivateUsage]
            operation = (
                operations.get((self.id, marker.name)) if operations is not None and self.id is not None else None
            )
            if operation is not None:
                result = await operation(*args, **kwargs)
            else:
                dispatcher = (
                    self._get_durable_operation_bindings().get(ctx.agent, {}).get(marker.name)  # pyright: ignore[reportPrivateUsage]
                    if ctx.agent is not None
                    else None
                )
                if dispatcher is None:
                    result = await handler(*args, **kwargs)
                else:
                    result = await dispatcher(
                        ctx,
                        cast(tuple[object, ...], args),
                        cast(dict[str, object], kwargs),
                    )

            # Apply worker-side model-request mutations back to the live context.
            if request_context is not None and isinstance(result, _ResolvedModelRequestContext):
                result.projection.apply(request_context, result.model)
                return request_context
            return result

        set_durable_operation_marker(decorated, marker)
        return cast(Callable[P, A], decorated)

    return decorate


def base_hook_durable_operation(
    name: str,
) -> Callable[[Callable[..., Awaitable[ResultT]]], Callable[..., Awaitable[ResultT]]]:
    """Mark a base hook so every override inherits durable execution automatically."""
    name = _validate_operation_name(name)

    def decorate(function: Callable[..., Awaitable[ResultT]]) -> Callable[..., Awaitable[ResultT]]:
        set_durable_operation_marker(
            function,
            Marker(
                name=name,
                function=cast(Callable[..., Awaitable[Any]], function),
                base_hook=True,
            ),
        )
        return function

    return decorate


def collect_capability_operations(
    capability: AbstractCapability[Any],
) -> dict[str, CapabilityMethodDeclaration]:
    """Collect durable declarations with a two-phase MRO scan.

    The first phase finds overridden base hooks marked durable. The second finds directly marked
    methods, validates never-durable hooks and duplicate names, then builds typed declarations.
    """
    handlers: dict[str, Callable[..., Awaitable[Any]]] = {}
    for base in type(capability).__mro__[1:]:
        for method_name, base_member in vars(base).items():
            marker = get_durable_operation_marker(base_member)
            if marker is None or not marker.base_hook:
                continue
            member = getattr(type(capability), method_name)
            if member is not base_member:
                handlers[marker.name] = cast(Callable[..., Awaitable[Any]], member)

    for method_name, member in inspect.getmembers(type(capability)):
        marker = get_durable_operation_marker(member)
        if marker is None:
            continue
        if marker.base_hook and member is marker.function:
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
        original = get_durable_operation_marker(handler)
        function = original.function if original is not None else handler
        bound = function.__get__(capability, type(capability))
        signature = inspect.signature(bound)
        type_hints = get_type_hints(bound, include_extras=True)
        ctx_parameter = find_typed_parameter(
            function, type_hints, is_call_ctx, 'RunContext', callable_kind='Durable operation'
        )
        model_request_parameter = find_typed_parameter(
            function,
            type_hints,
            lambda annotation: annotation is ModelRequestContext,
            'ModelRequestContext',
            callable_kind='Durable operation',
        )
        validate_schema_signature(function, signature, type_hints, ctx_parameter)
        if (
            model_request_parameter is not None
            and signature.parameters[model_request_parameter].kind is inspect.Parameter.VAR_POSITIONAL
        ):
            raise UserError('ModelRequestContext cannot be used as a variadic positional parameter (`*args`)')
        replacements = (
            {model_request_parameter: ModelRequestContextProjection, 'return': ModelRequestContextProjection}
            if model_request_parameter is not None
            else {}
        )
        schema = _capability_operation_schema(
            bound, signature, ctx_parameter, type_hints, annotation_replacements=replacements
        )
        result_type = (
            ModelRequestContextProjection
            if model_request_parameter is not None
            else extract_return_schema_type(type_hints.get('return'), bound)
        )
        declarations[operation_name] = CapabilityMethodDeclaration(
            name=operation_name,
            function=function,
            signature=signature,
            schema=schema,
            result_type=result_type,
            ctx_parameter=ctx_parameter,
            model_request_parameter=model_request_parameter,
        )
    return declarations


def _capability_operation_schema(
    function: Callable[..., Awaitable[Any]],
    signature: inspect.Signature,
    ctx_parameter: str | None,
    type_hints: dict[str, Any],
    *,
    annotation_replacements: dict[str, Any],
) -> FunctionSchema:
    if ctx_parameter is None and not annotation_replacements:
        return function_schema(function, GenerateToolJsonSchema)

    schema_target = _schema_target(function)
    schema_target.__annotations__ = {
        name: annotation_replacements.get(name, annotation)
        for name, annotation in type_hints.items()
        if name != ctx_parameter
    }
    schema_signature = signature.replace(
        parameters=[p for p in signature.parameters.values() if p.name != ctx_parameter]
    )
    cast(Any, schema_target).__signature__ = schema_signature
    return function_schema(schema_target, GenerateToolJsonSchema, takes_ctx=False)


def _schema_target(function: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Create a metadata-preserving callable whose signature can be adapted for schema generation."""

    async def target(**kwargs: Any) -> Any:  # pragma: no cover
        return kwargs

    return update_wrapper(target, function)


def bind_arguments(
    declaration: CapabilityMethodDeclaration,
    *,
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
    *,
    params: CapabilityOperationParams,
    model_request_context: ModelRequestContext | None = None,
) -> Any:
    bound = declaration.function.__get__(capability, type(capability))
    # The operation body runs as the capability, so name it on the context the way the hook chain
    # does: a context that crossed a durable boundary was rebuilt without the emitting capability,
    # and `RunContext.emit` resolves a `CapabilityEvent`'s owner through it.
    run_context = replace(params.run_context, _capability=capability)
    arguments = dict(params.arguments)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, parameter in declaration.signature.parameters.items():
        if name == declaration.ctx_parameter:
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                kwargs[name] = run_context
            else:
                args.append(run_context)
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(arguments)
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(arguments.pop(name))
            continue
        if name == declaration.model_request_parameter:
            if model_request_context is None:
                raise AssertionError(
                    'A model-request durable declaration was called without its model scope. '
                    'This is an internal Pydantic AI bug; please report it.'
                )
            value = model_request_context
        else:
            value = arguments.pop(name)

        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            args.append(value)
        else:
            kwargs[name] = value
    return await bound(*args, **kwargs)


async def recover_capability(ctx: RunContext[Any], *, capability_id: str) -> AbstractCapability[Any]:
    run_capabilities = ctx._run_capabilities_by_id or {}  # pyright: ignore[reportPrivateUsage]
    if capability := run_capabilities.get(capability_id):
        return capability
    agent = ctx.agent
    if agent is None:
        raise RuntimeError('A durable capability operation requires the worker agent on `RunContext`.')
    matches = [cap for cap in leaf_capabilities(agent.root_capability) if cap.id == capability_id]
    if len(matches) != 1:
        raise RuntimeError(f'Expected one bound capability with id {capability_id!r}, found {len(matches)}.')
    capability = matches[0]
    if type(capability).for_run is AbstractCapability.for_run:
        return capability
    return await capability.for_run(ctx)


CapabilityBoundOperation = BoundDurableOperation[CapabilityOperationParams, Any, Any]
