from __future__ import annotations

import functools
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from pydantic import ConfigDict, with_config
from temporalio import activity, workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import ModelMessage, ModelResponse, ModelResponseStreamEvent, models
from pydantic_ai._run_context import get_current_run_context
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RequestUsage

from ._run_context import TemporalRunContext


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    messages: list[ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    model_id: str | None = None


TemporalProviderFactory = Callable[[RunContext[AgentDepsT], str], Provider[Any]]


class TemporalStreamedResponse(StreamedResponse):
    def __init__(self, model_request_parameters: ModelRequestParameters, response: ModelResponse):
        super().__init__(model_request_parameters)
        self.response = response

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        return
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        return self.response

    def usage(self) -> RequestUsage:
        return self.response.usage  # pragma: no cover

    @property
    def model_name(self) -> str:
        return self.response.model_name or ''  # pragma: no cover

    @property
    def provider_name(self) -> str:
        return self.response.provider_name or ''  # pragma: no cover

    @property
    def provider_url(self) -> str | None:
        return self.response.provider_url  # pragma: no cover

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp  # pragma: no cover


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
        provider_factory: TemporalProviderFactory | None = None,
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

        @activity.defn(name=f'{activity_name_prefix}__model_request')
        async def request_activity(params: _RequestParams, deps: Any | None = None) -> ModelResponse:
            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            return await model_for_request.request(
                params.messages,
                cast(ModelSettings | None, params.model_settings),
                params.model_request_parameters,
            )

        self.request_activity = request_activity
        # Union with None for backward compatibility with activity payloads created before deps was added
        self.request_activity.__annotations__['deps'] = deps_type | None

        async def request_stream_activity(params: _RequestParams, deps: AgentDepsT) -> ModelResponse:
            # An error is raised in `request_stream` if no `event_stream_handler` is set.
            assert self.event_stream_handler is not None
            run_context = self.run_context_type.deserialize_run_context(params.serialized_run_context, deps=deps)
            model_for_request = self._resolve_model_id(params.model_id, run_context)
            async with model_for_request.request_stream(
                params.messages,
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

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        return [self.request_activity, self.request_stream_activity]

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

        model_name = model_id or f'{self.system}:{self.model_name}'
        activity_config: ActivityConfig = {'summary': f'request model: {model_name}', **self.activity_config}
        return await workflow.execute_activity(
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
    ) -> AsyncIterator[StreamedResponse]:
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
        model_name = model_id or f'{self.system}:{self.model_name}'
        activity_config: ActivityConfig = {'summary': f'request model: {model_name} (stream)', **self.activity_config}
        response = await workflow.execute_activity(
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
        yield TemporalStreamedResponse(model_request_parameters, response)

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError('Image output is not supported with Temporal because of the 2MB payload size limit.')

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

        # For strings and None, use _get_model_id + _resolve_model
        model_id = self._get_model_id(model)
        return self._resolve_model_id(model_id)

    @contextmanager
    def using_model(self, model: models.Model | models.KnownModelName | str | None) -> Iterator[None]:
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

    def _infer_model(self, model_id: str, run_context: RunContext[Any] | None) -> Model:  # pragma: lax no cover
        provider_factory = self._provider_factory
        if provider_factory is None or run_context is None:
            return models.infer_model(model_id)

        return models.infer_model(model_id, provider_factory=functools.partial(provider_factory, run_context))
