from __future__ import annotations as _annotations

import json

from httpx import AsyncClient
from pydantic import BaseModel

from pydantic_ai import ToolCallPart, ToolReturnPart
from pydantic_ai_examples.weather_agent import Deps, weather_agent

try:
    import gradio as gr
except ImportError as e:
    raise ImportError(
        'Please install gradio with `pip install gradio`. You must use python>=3.10.'
    ) from e

TOOL_TO_DISPLAY_NAME = {'get_lat_lng': 'Geocoding API', 'get_weather': 'Weather API'}

client = AsyncClient()
deps = Deps(client=client)


async def stream_from_agent(prompt: str, chatbot: list[dict], past_messages: list):
    chatbot.append({'role': 'user', 'content': prompt})
    yield gr.Textbox(interactive=False, value=''), chatbot, gr.skip()
    async with weather_agent.run_stream(
        prompt, deps=deps, message_history=past_messages
    ) as result:
        for message in result.new_messages():
            for call in message.parts:
                if isinstance(call, ToolCallPart):
                    call_args = call.args_as_json_str()
                    metadata = {
                        'title': f'🛠️ Using {TOOL_TO_DISPLAY_NAME[call.tool_name]}',
                    }
                    if call.tool_call_id is not None:
                        metadata['id'] = call.tool_call_id

                    gr_message = {
                        'role': 'assistant',
                        'content': 'Parameters: ' + call_args,
                        'metadata': metadata,
                    }
                    chatbot.append(gr_message)
                if isinstance(call, ToolReturnPart):
                    for gr_message in chatbot:
                        if (
                            gr_message.get('metadata', {}).get('id', '')
                            == call.tool_call_id
                        ):
                            if isinstance(call.content, BaseModel):
                                json_content = call.content.model_dump_json()
                            else:
                                json_content = json.dumps(call.content)
                            gr_message['content'] += f'\nOutput: {json_content}'
                yield gr.skip(), chatbot, gr.skip()
        chatbot.append({'role': 'assistant', 'content': ''})
        async for message in result.stream_text():
            chatbot[-1]['content'] = message
            yield gr.skip(), chatbot, gr.skip()
        past_messages = result.all_messages()

        yield gr.Textbox(interactive=True), gr.skip(), past_messages


async def handle_retry(chatbot, past_messages: list, retry_data: gr.RetryData):
    new_history = chatbot[: retry_data.index]
    previous_prompt = chatbot[retry_data.index]['content']
    past_messages = past_messages[: retry_data.index]
    async for update in stream_from_agent(previous_prompt, new_history, past_messages):
        yield update


def undo(chatbot, past_messages: list, undo_data: gr.UndoData):
    new_history = chatbot[: undo_data.index]
    past_messages = past_messages[: undo_data.index]
    return chatbot[undo_data.index]['content'], new_history, past_messages


def select_data(message: gr.SelectData) -> str:
    return message.value['text']


with gr.Blocks() as demo:
    gr.HTML(
        """
<div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; width: 100%">
    <img src="https://ai.pydantic.dev/img/logo-white.svg" style="max-width: 200px; height: auto">
    <div>
        <h1 style="margin: 0 0 1rem 0">Weather Assistant</h1>
        <h3 style="margin: 0 0 0.5rem 0">
            This assistant answer your weather questions.
        </h3>
    </div>
</div>
"""
    )
    past_messages = gr.State([])
    chatbot = gr.Chatbot(
        label='Packing Assistant',
        type='messages',
        avatar_images=(None, 'https://ai.pydantic.dev/img/logo-white.svg'),
        examples=[
            {'text': 'What is the weather like in Miami?'},
            {'text': 'What is the weather like in London?'},
        ],
    )
    with gr.Row():
        prompt = gr.Textbox(
            lines=1,
            show_label=False,
            placeholder='What is the weather like in New York City?',
        )
    generation = prompt.submit(
        stream_from_agent,
        inputs=[prompt, chatbot, past_messages],
        outputs=[prompt, chatbot, past_messages],
    )
    chatbot.example_select(select_data, None, [prompt])
    chatbot.retry(
        handle_retry, [chatbot, past_messages], [prompt, chatbot, past_messages]
    )
    chatbot.undo(undo, [chatbot, past_messages], [prompt, chatbot, past_messages])


if __name__ == '__main__':
    demo.launch()
