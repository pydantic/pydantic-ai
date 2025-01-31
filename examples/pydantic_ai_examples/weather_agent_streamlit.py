# pyright: reportUnknownMemberType=false
import asyncio
import os
from dataclasses import dataclass
from typing import Literal

import streamlit as st
from httpx import AsyncClient

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.gemini import GeminiModel, GeminiModelName
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelName
from pydantic_ai_examples.weather_agent import Deps, weather_agent

# Available models and their respective model class
models: dict[GeminiModelName | OpenAIModelName, type[GeminiModel | OpenAIModel]] = {
    'gemini-1.5-flash': GeminiModel,
    'gpt-4o': OpenAIModel,
}


@dataclass
class ChatMessage:
    role: Literal['user', 'assistant']
    content: str


def to_chat_message(message: ModelMessage) -> ChatMessage | None:
    """Convert a ModelMessage to a ChatMessage.

    Args:
        message (ModelMessage): The model message to convert.

    Returns:
        ChatMessage | None: The converted chat message, or None if the message type is not recognized or does not contain a UserPromptPart or TextPart.

    Notes:
        Messages will be ignored if they do not contain a UserPromptPart (for ModelRequests) or a TextPart (for ModelResponses).
        This means that messages with other part types (e.g. ToolCallPart) will not be converted to ChatMessages.
    """
    if isinstance(message, ModelRequest):
        user_prompt_part = next(
            (part for part in message.parts if isinstance(part, UserPromptPart)), None
        )
        if user_prompt_part:
            return ChatMessage(role='user', content=user_prompt_part.content)
    elif isinstance(message, ModelResponse):
        text_part = next(
            (
                part
                for part in message.parts
                if isinstance(part, TextPart) and part.has_content()
            ),
            None,
        )
        if text_part:
            return ChatMessage(role='assistant', content=text_part.content)
    return None


# Initialize the 'messages' list in session_state if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages: list[ModelMessage] = []  # type: ignore[reportInvalidTypeForm]

with st.sidebar:
    model_name = st.selectbox('Model', models.keys())
    model_api_key = st.text_input(
        'Model API Key', os.getenv('MODEL_API_KEY'), type='password'
    )
    weather_api_key = st.text_input(
        'Weather API Key', os.getenv('WEATHER_API_KEY'), type='password'
    )
    geocoding_api_key = st.text_input(
        'Geocoding API Key', os.getenv('GEO_API_KEY'), type='password'
    )


st.logo('https://ai.pydantic.dev/img/logo-white.svg', size='medium')
st.subheader('Weather AI Assistant made with Pydantic AI')

# display chat messages from history on app rerun
for model_message in st.session_state.messages:
    chat_message = to_chat_message(model_message)
    if chat_message:
        st.chat_message(chat_message.role).write(chat_message.content)

if prompt := st.chat_input('What is the weather like in London and in Wiltshire?'):
    if not model_api_key:
        st.info('LLM API Key must be provided to continue')
        st.stop()

    if not weather_api_key:
        st.info(
            'Please add a Weather API Key to continue. You can obtain the key [here](https://www.tomorrow.io/weather-api/)'
        )
        st.stop()

    if not geocoding_api_key:
        st.info(
            'Please add a Geocoding API key to continue. You can obtain the key [here](https://geocode.maps.co/)'
        )
        st.stop()

    with st.chat_message('user'):
        st.write(prompt)

    client = AsyncClient()
    deps = Deps(
        client=client, weather_api_key=weather_api_key, geo_api_key=geocoding_api_key
    )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if model_class := models.get(model_name):
        result = weather_agent.run_sync(
            prompt,
            model=model_class(model_name, api_key=model_api_key),  # type: ignore[reportInvalidTypeForm]
            deps=deps,
            message_history=st.session_state.messages,
        )
        st.session_state.messages.extend(result.new_messages())
        st.chat_message('assistant').write(result.data)
    else:
        st.error('Invalid Model')
