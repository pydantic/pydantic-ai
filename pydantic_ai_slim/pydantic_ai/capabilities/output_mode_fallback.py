from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_core import to_json

from pydantic_ai._unsupported_settings import suppress_warnings as suppress_unsupported_setting_warnings
from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import ModelHTTPError, RetryModelRequest
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models._anthropic_schema_limits import reports_schema_too_complex
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability, CapabilityOrdering
from .model_fallback import ModelFallback

if TYPE_CHECKING:
    from pydantic_ai.models import Model, ModelRequestContext
    from pydantic_ai.output import OutputObjectDefinition

__all__ = ('OutputModeFallback',)

_MAX_REMEMBERED_SCHEMAS = 256

_too_complex_schemas: OrderedDict[str, None] = OrderedDict()
"""The output schemas, per model, that this process saw rejected in Native Output, most recently seen last."""
_too_complex_schemas_lock = threading.Lock()
"""Guards `_too_complex_schemas`, which agents running in different threads share."""


@dataclass
class OutputModeFallback(AbstractCapability[AgentDepsT]):
    """Retry a request in Tool Output when the provider can't compile the Native Output schema it inferred.

    Anthropic compiles a Native Output schema into a grammar, and can reject one that is within its
    [documented limits](../models/anthropic.md#schema-complexity-limits) as too complex, after spending
    up to minutes trying. A request's output mode is inferred when the `output_type` doesn't choose one,
    and Anthropic infers Native Output for a request that thinks. On that rejection, this capability
    makes another attempt at the request on the same model in Tool Output, which isn't compiled, and the
    model never learns the first attempt happened:

    ```python
    from pydantic import BaseModel

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import OutputModeFallback


    class Survey(BaseModel):
        name: str
        age: int | None = None


    agent = Agent('anthropic:claude-sonnet-5', output_type=Survey, capabilities=[OutputModeFallback()])
    ```

    The process then remembers the schema for that model, so later requests use Tool Output right away
    instead of waiting for the compile to fail again. It remembers only within a run under durable
    execution, where workflow code has to make the same choices when it's replayed.

    An output mode you chose, with [`NativeOutput`][pydantic_ai.output.NativeOutput] for example, is
    left alone, so the rejection is raised as it is without this capability. The capability is placed
    inside [`ModelFallback`][pydantic_ai.capabilities.ModelFallback], so the same model is attempted
    in Tool Output before another model is.
    """

    _attempt_schema: str | None = field(default=None, init=False, repr=False)
    """The key of this attempt's output schema, while the attempt infers its output mode."""
    _run_schemas: set[str] = field(default_factory=set[str], init=False, repr=False)
    """The schemas rejected during this run, which it uses Tool Output for even under durable execution."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost', wrapped_by=[ModelFallback])

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> OutputModeFallback[AgentDepsT]:
        return replace_no_init(self, _attempt_schema=None, _run_schemas=set[str]())

    async def prepare_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self._attempt_schema = None
        params = request_context.model_request_parameters
        if params.output_mode != 'auto' or params.output_object is None or not params.output_tools:
            return request_context
        key = _schema_key(request_context.model, params.output_object)
        if key in self._run_schemas or (not ctx.in_durable_context and _recall(key)):
            return replace(request_context, model_request_parameters=params.with_default_output_mode('tool'))
        self._attempt_schema = key
        return request_context

    async def on_model_request_error(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        key, self._attempt_schema = self._attempt_schema, None
        if key is None or not _is_schema_too_complex(error) or not _sent_native_output(request_context):
            raise error
        self._run_schemas.add(key)
        if not ctx.in_durable_context:
            _remember(key)
        raise RetryModelRequest()


def _schema_key(model: Model, output_object: OutputObjectDefinition) -> str:
    schema = hashlib.sha256(to_json(output_object.json_schema)).hexdigest()
    return f'{model.model_id}:{schema}'


def _recall(key: str) -> bool:
    """Whether this process remembers the schema as rejected, which keeps it from being forgotten soon."""
    with _too_complex_schemas_lock:
        if key not in _too_complex_schemas:
            return False
        _too_complex_schemas.move_to_end(key)
        return True


def _remember(key: str) -> None:
    with _too_complex_schemas_lock:
        _too_complex_schemas[key] = None
        _too_complex_schemas.move_to_end(key)
        if len(_too_complex_schemas) > _MAX_REMEMBERED_SCHEMAS:
            _too_complex_schemas.popitem(last=False)


def _is_schema_too_complex(error: Exception) -> bool:
    """Whether the provider rejected the request because its strict schemas are too complex to compile."""
    return isinstance(error, ModelHTTPError) and error.status_code == 400 and reports_schema_too_complex(error.body)


def _sent_native_output(request_context: ModelRequestContext) -> bool:
    """Whether the failed attempt used Native Output, as the model resolves the mode it inferred."""
    params = request_context.model_request_parameters
    if params.output_mode == 'auto':
        with suppress_unsupported_setting_warnings():
            _, params = request_context.model.prepare_request(request_context.model_settings, params)
    return params.output_mode == 'native'
