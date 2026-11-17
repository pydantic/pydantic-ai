"""`/refine` runs one review round over the last answer."""

import io

from rich.console import Console

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart
from pydantic_ai_harness import DEFAULT_REFINE_OUTPUT_INSTRUCTIONS
from pydantic_clai2.builtin_plugins.refinement import RefinementPlugin
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, Transcript, load_plugin

ANSWERED: list[ModelMessage] = [
    ModelRequest.user_text_prompt('Explain the change'),
    ModelResponse(parts=[TextPart('It adds a flag.')]),
]


def refinement(messages: list[ModelMessage], submitted: list[str]) -> LoadedPlugin[None]:
    host = PluginHost[None](
        name='refinement',
        console=Console(file=io.StringIO()),
        settings={},
        conversation=Transcript(messages=messages),
        submit_prompt=submitted.append,
    )
    return load_plugin(RefinementPlugin, host)


async def test_refine_submits_the_review_instructions() -> None:
    submitted: list[str] = []

    assert await refinement(ANSWERED, submitted).commands.execute_async('/refine') == ''

    assert submitted == [DEFAULT_REFINE_OUTPUT_INSTRUCTIONS]


async def test_refine_adds_the_focus_as_typed() -> None:
    submitted: list[str] = []

    assert await refinement(ANSWERED, submitted).commands.execute_async('/refine the "auth" notes') == ''

    assert submitted == [f'{DEFAULT_REFINE_OUTPUT_INSTRUCTIONS}\n\nFocus on: the "auth" notes']


async def test_refine_needs_an_answer() -> None:
    submitted: list[str] = []
    plugin = refinement([ModelRequest.user_text_prompt('Explain the change')], submitted)

    assert await plugin.commands.execute_async('/refine') == 'Nothing to refine: the model has not answered yet.'

    assert submitted == []
