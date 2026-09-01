from __future__ import annotations

import functools
from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import ConfigDict, PydanticSchemaGenerationError, TypeAdapter, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import ModelMessage, ModelResponse, models
from pydantic_ai._agent_graph import _clean_message_history  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import (
    CompletedStreamedResponse,
    Model,
    ModelRequestParameters,
    StreamedResponse,
    infer_model_profile,
    parse_model_id,
)
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext

from .._utils import managed_model_scope
from ._activity_execution import execute_activity
from ._durability import (
    IMAGE_OUTPUT_UNSUPPORTED_MESSAGE,
    _RequestParams,  # pyright: ignore[reportPrivateUsage]
)
from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import model_response_payload_errors

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

__all__ = [
    'TemporalModel',
    'TemporalProviderFactory',
]


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CancelParams:
    response: ModelResponse
    model_id: str | None = None
    serialized_run_context: Any = None
    deps: Any = None


TemporalProviderFactory = Callable[[RunContext[AgentDepsT], str], Provider[Any]]


class TemporalModel(WrapperModel):
    def __init__(
        self,
        model: Model | None,
        *,
        activity_name_prefix: str,
        activity_config: ActivityConfig,
        deps_type: type[AgentDepsT],
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
        event_stream_handler: EventStreamHandler[Any] | None = None,
        models: Mapping[str, Model] | None = None,
        provider_factory: TemporalProviderFactory[AgentDepsT] | None = None,
        agent: AbstractAgent[Any, Any] | None = None,
    ):
        # Build models_by_id registry from wrapped model and models parameter
        self._models_by_id: dict[str, Model] = {}
        if model is not None:
            self._models_by_id['default'] = model
        if models:
            for model_id, model_instance in models.items():
                if model_id == 'default':
                    raise UserError("Model ID 'default' is reserved for the agent's primary model.")
                self._models_by_id[model_id] = model_instance

        if not self._models_by_id:
            raise UserError(
                "The wrapped agent's `model` or the TemporalAgent's `models` parameter must provide at least one Model instance to be used with Temporal. Models cannot be set at agent run time."
            )

        # Use provided model if available, otherwise first registered model
        primary_model = model or next(iter(self._models_by_id.values()))
        super().__init__(primary_model)
        self.activity_config = activity_config
        self.run_context_type = run_context_type
        self.event_stream_handler = event_stream_handler
        self._model_id_var: ContextVar[str | None] = ContextVar('_temporal_model_id', default=None)
        self._provider_factory = provider_factory
        self._agent = agent

        async def request_activity(params: _RequestParams, deps: Any | None = None) -> ModelResponse:
            run_context = deserialize_run_context(
                self.run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            async with managed_model_scope(
                model_for_request, owned=not self._is_registered_model(model_for_request)
            ) as active_model:
                messages = self._reprepare_messages(params, active_model)
                return await active_model.request(
                    messages,
                    cast(ModelSettings | None, params.model_settings),
                    params.model_request_parameters,
                )

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        # Union with None for backward compatibility with activity payloads created before deps was added
        request_activity.__annotations__['deps'] = deps_type | None

        self.request_activity = activity.defn(name=f'{activity_name_prefix}__model_request')(request_activity)

        async def request_stream_activity(params: _RequestParams, deps: AgentDepsT) -> ModelResponse:
            # An error is raised in `request_stream` if no `event_stream_handler` is set.
            assert self.event_stream_handler is not None
            run_context = deserialize_run_context(
                self.run_context_type, params.serialized_run_context, deps=deps, agent=self._agent
            )
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            async with managed_model_scope(
                model_for_request, owned=not self._is_registered_model(model_for_request)
            ) as active_model:
                messages = self._reprepare_messages(params, active_model)
                async with active_model.request_stream(
                    messages,
                    cast(ModelSettings | None, params.model_settings),
                    params.model_request_parameters,
                    run_context,
                ) as streamed_response:
                    await self.event_stream_handler(run_context, streamed_response)

                    async for _ in streamed_response:
                        pass
            return streamed_response.get()

        # Set type hint explicitly so that Temporal can take care of serialization and deserialization
        # Union with None for backward compatibility with activity payloads created before deps was added
        request_stream_activity.__annotations__['deps'] = deps_type | None

        self.request_stream_activity = activity.defn(name=f'{activity_name_prefix}__model_request_stream')(
            request_stream_activity
        )

        try:
            deps_type_adapter = TypeAdapter(deps_type)
        except PydanticSchemaGenerationError:
            deps_type_adapter = TypeAdapter(deps_type, config=ConfigDict(defer_build=True))

        async def cancel_suspended_response_activity(params: _CancelParams) -> None:
            # Resolve the model that produced the response (mirrors `request_activity`'s use of
            # `model_id`) so a multi-model registry cancels on the right client. The teardown is a
            # raw HTTP call to the provider, so it must run in an activity rather than the workflow
            # sandbox. The run context and deps travel inside the single params payload to preserve
            # the activity command shape for replay; old payloads omit them and keep the previous
            # environment-inference behavior.
            run_context = None
            if params.serialized_run_context is not None:
                deps = deps_type_adapter.validate_python(params.deps) if params.deps is not None else None
                run_context = deserialize_run_context(
                    self.run_context_type,
                    params.serialized_run_context,
                    deps=deps,
                    agent=self._agent,
                )
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            async with managed_model_scope(
                model_for_request, owned=not self._is_registered_model(model_for_request)
            ) as active_model:
                await active_model.cancel_suspended_response(params.response)

        self.cancel_suspended_response_activity = activity.defn(
            name=f'{activity_name_prefix}__model_cancel_suspended_response'
        )(cancel_suspended_response_activity)

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.request_activity, self.request_stream_activity, self.cancel_suspended_response_activity]

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if not workflow.in_workflow():
            return await super().request(messages, model_settings, model_request_parameters)

        self._validate_model_request_parameters(model_request_parameters)

        model_id = self._current_model_id()
        run_context = get_current_run_context()
        if run_context is None:  # pragma: no cover
            raise UserError(
                'A Temporal model cannot be used with `pydantic_ai.direct.model_request()` as it requires a `run_context`. Use `agent.run()` instead.'
            )
        serialized_run_context = self.run_context_type.serialize_run_context(run_context)
        deps = run_context.deps

        model_name = model_id or self.model_id
        activity_config: ActivityConfig = {'summary': f'request model: {model_name}', **self.activity_config}
        with model_response_payload_errors(model_name):
            return await execute_activity(
                activity=self.request_activity,
                args=[
                    _RequestParams(
                        messages=messages,
                        model_settings=cast(dict[str, Any] | None, model_settings),
                        model_request_parameters=model_request_parameters,
                        serialized_run_context=serialized_run_context,
                        model_id=model_id,
                    ),
                    deps,
                ],
                **activity_config,
            )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        if not workflow.in_workflow():
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed_response:
                yield streamed_response
                return

        if run_context is None:
            raise UserError(
                'A Temporal model cannot be used with `pydantic_ai.direct.model_request_stream()` as it requires a `run_context`. Set an `event_stream_handler` on the agent and use `agent.run()` instead.'
            )

        # We can never get here without an `event_stream_handler`, as `TemporalAgent.run_stream` and `TemporalAgent.iter` raise an error saying to use `TemporalAgent.run` instead,
        # and that only calls `request_stream` if `event_stream_handler` is set.
        assert self.event_stream_handler is not None

        self._validate_model_request_parameters(model_request_parameters)

        model_id = self._current_model_id()
        serialized_run_context = self.run_context_type.serialize_run_context(run_context)
        model_name = model_id or self.model_id
        activity_config: ActivityConfig = {'summary': f'request model: {model_name} (stream)', **self.activity_config}
        with model_response_payload_errors(model_name):
            response = await execute_activity(
                activity=self.request_stream_activity,
                args=[
                    _RequestParams(
                        messages=messages,
                        model_settings=cast(dict[str, Any] | None, model_settings),
                        model_request_parameters=model_request_parameters,
                        serialized_run_context=serialized_run_context,
                        model_id=model_id,
                    ),
                    run_context.deps,
                ],
                **activity_config,
            )
        yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        if not workflow.in_workflow():
            return await super().cancel_suspended_response(response)

        model_id = self._current_model_id()
        run_context = get_current_run_context()
        model_name = model_id or self.model_id
        activity_config: ActivityConfig = {
            'summary': f'cancel suspended response: {model_name}',
            **self.activity_config,
        }
        await execute_activity(
            activity=self.cancel_suspended_response_activity,
            args=[
                _CancelParams(
                    response=response,
                    model_id=model_id,
                    serialized_run_context=(
                        self.run_context_type.serialize_run_context(run_context) if run_context is not None else None
                    ),
                    deps=run_context.deps if run_context is not None else None,
                )
            ],
            **activity_config,
        )

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError(IMAGE_OUTPUT_UNSUPPORTED_MESSAGE)

    def _get_model_id(self, model: models.Model | models.KnownModelName | str | None = None) -> str | None:
        """Get the model ID for the given model parameter.

        Returns a string that will be checked against registered model IDs,
        or passed to infer_model if not found. Returns None to use the default model.
        """
        if model in (None, 'default'):
            return None

        if isinstance(model, Model):
            # Check if this model instance is already registered
            model_id = next((model_id for model_id, m in self._models_by_id.items() if m is model), ...)
            if model_id is ...:
                raise UserError(
                    'Arbitrary model instances cannot be used at runtime inside a Temporal workflow. '
                    'Register the model via `models` or reference a registered model by id.'
                )
            return None if model_id == 'default' else model_id

        return model

    def resolve_model(self, model: models.Model | models.KnownModelName | str | None = None) -> Model:
        """Resolve a model parameter to a Model instance.

        This is typically used outside of a workflow to resolve model parameters
        before passing them to the underlying agent methods.

        Args:
            model: The model to resolve. Can be a Model instance, model name string,
                   or None for the default model.

        Returns:
            The resolved Model instance.
        """
        # Handle Model instances directly - outside a workflow, unregistered
        # Model instances are allowed since there's no serialization constraint.
        if isinstance(model, Model):
            return model

        # For strings and None, use _get_model_id + _resolve_model_id
        model_id = self._get_model_id(model)
        return self._resolve_model_id(model_id)

    @contextmanager
    def using_model(self, model: models.Model | models.KnownModelName | str | None) -> Generator[None]:
        """Context manager to set the model for the duration of a block.

        Accepts a Model instance, model name string, or None for the default model.
        """
        model_id = self._get_model_id(model)
        token = self._model_id_var.set(model_id)
        try:
            yield
        finally:
            self._model_id_var.reset(token)

    def _current_model_id(self) -> str | None:
        return self._model_id_var.get()

    def _current_model(self) -> models.Model | str:
        """Get the current model, or the unregistered model ID string."""
        model_id = self._current_model_id()
        if model_id is None:
            return self.wrapped
        if model_id in self._models_by_id:
            return self._models_by_id[model_id]
        return model_id

    @property
    def model_name(self) -> str:
        """Get the model name, inferring from raw strings without provider construction."""
        current = self._current_model()
        if isinstance(current, str):
            _, model_name = parse_model_id(current)
            return model_name
        return current.model_name

    @property
    def system(self) -> str:
        """Get the system (provider) name, inferring from raw strings without provider construction."""
        current = self._current_model()
        if isinstance(current, str):
            provider_name, _ = parse_model_id(current)
            return provider_name or self.wrapped.system
        return current.system

    @property
    def model_id(self) -> str:
        """Get the ID of the currently active model, rather than the default one's."""
        current = self._current_model()
        if isinstance(current, str):
            # `system` and `model_name` already parse the raw ID, falling back to the default model's
            # provider when the string names none, so recombining them is the answer here.
            return f'{self.system}:{self.model_name}'
        return current.model_id

    @property
    def profile(self) -> ModelProfile:
        """Get the model profile, inferring from raw strings without provider construction.

        Note: This overrides a cached_property with a regular property because the profile
        depends on _current_model_id() which can change dynamically via using_model().
        """
        current = self._current_model()
        if isinstance(current, str):
            # Unlike Model.profile, this returns the raw provider profile without intersecting
            # supported_native_tools with the model class's supported_native_tools(). This is
            # acceptable because TemporalModel delegates to the wrapped model for actual requests,
            # and this profile is only used for capability checks, not request preparation.
            return infer_model_profile(current)
        return current.profile

    @property
    def base_url(self) -> str | None:
        """Get the base URL of the currently active model, rather than the default one's."""
        current = self._current_model()
        if isinstance(current, str):
            # A raw model ID carries no URL, and the default model's would name the wrong server.
            return None
        return current.base_url

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        current = self._current_model()
        if isinstance(current, str):
            return Model.customize_request_parameters(self, model_request_parameters)
        return current.customize_request_parameters(model_request_parameters)

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        """Prepare request using the currently active model's profile.

        This override ensures that when a different model is specified at runtime
        via `using_model()`, we use that model's profile for validation and
        parameter preparation, not the default wrapped model's profile.
        """
        current = self._current_model()

        # For unregistered model strings, use Model.prepare_request (grandparent's method)
        # with our overridden profile property. This allows validation to use the correct
        # profile inferred from the model string, without constructing a full model instance.
        if isinstance(current, str):
            return Model.prepare_request(self, model_settings, model_request_parameters)

        return current.prepare_request(model_settings, model_request_parameters)

    def prepare_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        """Pre-process messages using the currently active model's profile.

        When `using_model()` selects a registered model, delegate to that concrete model's
        profile. For an unregistered model string, defer preparation until the activity has
        resolved the concrete model: preparation can be lossy, so an inferred workflow-side
        profile must not transform the history first.
        """
        current = self._current_model()
        if isinstance(current, str):
            return messages
        return current.prepare_messages(messages, model_request_parameters)

    def _reprepare_messages(self, params: _RequestParams, model_for_request: Model) -> list[ModelMessage]:
        """Re-run `prepare_messages` against the concrete model, where the workflow couldn't.

        `prepare_messages` decides from `self.profile`, and a registered model is the same instance on
        both sides of the boundary, so its workflow-side pass already saw the right profile. A runtime
        model *string* isn't: the workflow infers it without constructing a provider, and a client can
        narrow the inferred profile — `AsyncAnthropicFoundry` turns off `supports_inline_system_prompts`,
        which the string-inferred profile advertises. Whatever the workflow prepared was decided against
        a profile the request is not being sent under.

        The second cleanup pass mirrors `_agent_graph`'s, for the reason given there: tool-search
        synthesis can split a response into a response plus a request, leaving two same-role messages
        adjacent. Preparing here and skipping the merge would make the activity path render history
        differently from every non-durable path, which is the whole thing this is here to avoid.
        """
        if params.model_id is None or params.model_id in self._models_by_id:
            return params.messages

        prepared = model_for_request.prepare_messages(params.messages, params.model_request_parameters)
        if prepared is params.messages:
            return prepared
        return _clean_message_history(prepared, repair_last_response=True)

    def _resolve_model_id(self, model_id: str | None, run_context: RunContext[Any] | None = None) -> Model:
        """Resolve a model ID to a Model instance.

        Args:
            model_id: The model ID string, or None for the default model.
            run_context: Optional run context for provider factory usage.

        Returns:
            The resolved Model instance.
        """
        if model_id is None:
            return self.wrapped

        if model_id in self._models_by_id:
            return self._models_by_id[model_id]

        return self._infer_model(model_id, run_context)  # pragma: lax no cover

    def _is_registered_model(self, model: Model) -> bool:
        return any(registered is model for registered in self._models_by_id.values())

    def _infer_model(self, model_id: str, run_context: RunContext[Any] | None) -> Model:  # pragma: lax no cover
        provider_factory = self._provider_factory
        if provider_factory is None or run_context is None:
            return models.infer_model(model_id)

        return models.infer_model(model_id, provider_factory=functools.partial(provider_factory, run_context))
