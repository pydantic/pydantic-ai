from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from httpx import AsyncClient as AsyncHTTPClient

from . import (
    AbstractToolDefinition,
    AgentModel,
    Model,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from openai import AsyncOpenAI
    from openai.types import chat
except ImportError as e:
    raise ImportError(
        'Please install `openai` to use the OpenAI model, '
        "you can use the `openai` optional group — `pip install 'pydantic-ai[openai]'`"
    ) from e

# Only required in the below typing is enforced
# try:
#     from ollama import list, ListResponse
# except ImportError as e:
#     raise ImportError(
#         'Please install `ollama` to use the Ollama model, '
#         "you can use the `ollama` optional group — `pip install 'pydantic-ai[ollama]'`"
#     ) from e


from .openai import OpenAIAgentModel

# NB: Currently commented out but this is
# a possible approach to type the ollama model names
# # This is really nasty, but ollama doesn't
# # expose a type of all available models, and as
# # there are 100's and being udpated all the time,
# # it seems difficult to maintain a list of all possible
# # versions statically.
# # This code gets the ones that are _currently_ available
# # to the user as they are ones that they have already downloaded
# ollama_ls: ListResponse = list()
# ollama_available_models: list[str] = [x.model for x in ollama_ls.models]
# OllamaModelName = Literal[tuple(ollama_available_models)]


# Inherits from this model
@dataclass(init=False)
class OllamaModel(Model):
    """A model that uses the OpenAI API.

    Internally, this uses the [OpenAI Python client](https://github.com/openai/openai-python) to interact with the API.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    model_name: str
    client: AsyncOpenAI = field(repr=False)

    def __init__(
        self,
        model_name: str,
        *,
        base_url: str = 'http://localhost:11434/v1/',
        openai_client: AsyncOpenAI | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        """Initialize an Ollama model.

        Ollama has built-in compatability with the OpenAI chat completions API ([source](https://ollama.com/blog/openai-compatibility)), and so these models are built ontop of that

        Args:
            model_name: The name of the Ollama model to use. List of models available [here](https://ollama.com/library)
                NB: You must first download the model (ollama pull <MODEL-NAME>) in order to use the model
            base_url: The base url for the ollama requests. The default value is the ollama default
            openai_client: An existing
                [`AsyncOpenAI`](https://github.com/openai/openai-python?tab=readme-ov-file#async-usage)
                client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name: str = model_name
        if openai_client is not None:
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self.client = openai_client
            print('Using existing client')
        elif http_client is not None:
            # API key is not required for ollama but a value is required to create the client
            self.client = AsyncOpenAI(base_url=base_url, api_key='ollama', http_client=http_client)
        else:
            # API key is not required for ollama but a value is required to create the client
            self.client = AsyncOpenAI(base_url=base_url, api_key='ollama', http_client=cached_async_http_client())

    async def agent_model(
        self,
        function_tools: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        check_allow_model_requests()
        tools = [self._map_tool_definition(r) for r in function_tools.values()]
        if result_tools is not None:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return OpenAIAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'ollama:{self.model_name}'

    @staticmethod
    def _map_tool_definition(f: AbstractToolDefinition) -> chat.ChatCompletionToolParam:
        return {
            'type': 'function',
            'function': {
                'name': f.name,
                'description': f.description,
                'parameters': f.json_schema,
            },
        }
