"""Low-level methods to make requests directly to models with minimal abstraction.

These methods allow you to make requests to LLMs where the only abstraction is input and output schema
translation so you can request all models with the same API.

These methods are thin wrappers around [`Model`][pydantic_ai.models.Model] implementations.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pydantic_graph._utils import get_event_loop as _get_event_loop

from . import messages, models, settings, usage


async def model_request(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
) -> tuple[messages.ModelResponse, usage.Usage]:
    """Make a non-streamed request to a model.

    This method is roughly equivalent to [`Agent.run`][pydantic_ai.Agent.run].

    ```py title="model_request_example.py"
    from pydantic_ai.low_level import model_request
    from pydantic_ai.messages import ModelRequest


    async def main():
        model_response, request_usage = await model_request(
            'anthropic:claude-3-5-haiku-latest',
            [ModelRequest.user_text_prompt('What is the capital of France?')]  # (1)!
        )
        print(model_response)
        '''
        ModelResponse(
            parts=[TextPart(content='Paris', part_kind='text')],
            model_name='claude-3-5-haiku-latest',
            timestamp=datetime.datetime(...),
            kind='response',
        )
        '''
        print(request_usage)
        '''
        Usage(
            requests=0, request_tokens=56, response_tokens=1, total_tokens=57, details=None
        )
        '''
    ```

    1. See [`ModelRequest.user_text_prompt`][pydantic_ai.messages.ModelRequest.user_text_prompt] for details.

    Then

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters

    Returns:
        The model response and token usage associated with the request.
    """
    model_instance = models.infer_model(model)
    return await model_instance.request(
        messages,
        model_settings,
        model_request_parameters or models.ModelRequestParameters(),
    )


def model_request_sync(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
) -> tuple[messages.ModelResponse, usage.Usage]:
    """Make a Synchronous, non-streamed request to a model.

    This is a convenience method that wraps [`model_request`][pydantic_ai.low_level.model_request] with
    `loop.run_until_complete(...)`. You therefore can't use this method inside async code or if there's an active event loop.

    This method is roughly equivalent to [`Agent.run_sync`][pydantic_ai.Agent.run_sync].


    ```py title="model_request_sync_example.py"
    from pydantic_ai.low_level import model_request_sync
    from pydantic_ai.messages import ModelRequest

    model_response, _ = model_request_sync(
        'anthropic:claude-3-5-haiku-latest',
        [ModelRequest.user_text_prompt('What is the capital of France?')]
    )
    print(model_response)
    '''
    ModelResponse(
        parts=[TextPart(content='Paris', part_kind='text')],
        model_name='claude-3-5-haiku-latest',
        timestamp=datetime.datetime(...),
        kind='response',
    )
    '''
    ```

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters

    Returns:
        The model response and token usage associated with the request.
    """
    return _get_event_loop().run_until_complete(
        model_request(model, messages, model_settings=model_settings, model_request_parameters=model_request_parameters)
    )


@asynccontextmanager
async def model_request_stream(
    model: models.Model | models.KnownModelName | str,
    messages: list[messages.ModelMessage],
    *,
    model_settings: settings.ModelSettings | None = None,
    model_request_parameters: models.ModelRequestParameters | None = None,
) -> AsyncIterator[models.StreamedResponse]:
    """Make a streamed request to a model.

    This method is roughly equivalent to [`Agent.run_stream`][pydantic_ai.Agent.run_stream].

    ```py title="model_request_stream_example.py"

    from pydantic_ai.low_level import model_request_stream
    from pydantic_ai.messages import ModelRequest


    async def main():
        messages = [ModelRequest.user_text_prompt('Who was Albert Einstein?')]
        async with model_request_stream( 'openai:gpt-4.1-mini', messages) as stream:
            async for chunk in stream:
                print(chunk)
                '''
                PartStartEvent(
                    index=0,
                    part=TextPart(content='Albert Einstein was ', part_kind='text'),
                    event_kind='part_start',
                )
                '''
                '''
                PartDeltaEvent(
                    index=0,
                    delta=TextPartDelta(
                        content_delta='a German-born theoretical ', part_delta_kind='text'
                    ),
                    event_kind='part_delta',
                )
                '''
                '''
                PartDeltaEvent(
                    index=0,
                    delta=TextPartDelta(content_delta='physicist.', part_delta_kind='text'),
                    event_kind='part_delta',
                )
                '''
    ```

    Args:
        model: The model to make a request to. We allow `str` here since the actual list of allowed models changes frequently.
        messages: Messages to send to the model
        model_settings: optional model settings
        model_request_parameters: optional model request parameters

    Returns:
        A [stream response][pydantic_ai.models.StreamedResponse] async context manager.
    """
    model_instance = models.infer_model(model)
    stream_cxt_mgr = model_instance.request_stream(
        messages,
        model_settings,
        model_request_parameters or models.ModelRequestParameters(),
    )
    async with stream_cxt_mgr as streamed_response:
        yield streamed_response
