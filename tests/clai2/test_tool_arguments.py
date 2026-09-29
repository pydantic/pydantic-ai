"""Generic tool headers list their arguments as styled, clipped `name=value` pairs."""

import io

import pytest
from rich.console import Console

from pydantic_ai import FunctionToolCallEvent
from pydantic_ai.messages import ToolCallPart
from pydantic_clai2 import StreamRenderer, theme
from pydantic_clai2.config import Settings, resolve_settings
from pydantic_clai2.tool_output import tool_arguments_text


async def render(args: dict[str, object] | str, *, width: int = 200, tool_arg_chars: int = 40) -> str:
    output = io.StringIO()
    renderer = StreamRenderer(
        Console(file=output, width=width), stop_loading=lambda: None, tool_arg_chars=tool_arg_chars
    )
    await renderer.on_stream_event(FunctionToolCallEvent(part=ToolCallPart('search_repositories', args)))
    return output.getvalue()


async def test_generic_tool_shows_arguments() -> None:
    args: dict[str, object] = {
        'query': 'pydantic',
        'page': 2,
        'archived': False,
        'topics': ['ai', 'llm'],
        'owner': None,
    }
    assert await render(args) == (
        '● search_repositories query="pydantic" page=2 archived=false topics=["ai","llm"] owner=null\n\n'
    )


@pytest.mark.parametrize('args', [{}, '', '{}'])
async def test_empty_arguments_show_only_the_name(args: dict[str, object] | str) -> None:
    assert await render(args) == '● search_repositories\n\n'


async def test_each_value_is_clipped_to_the_configured_limit() -> None:
    text = await render({'code': 'x' * 100, 'n': 12345}, tool_arg_chars=8)
    assert text == '● search_repositories code="xxxxxx… n=12345\n\n'


async def test_zero_limit_hides_arguments() -> None:
    assert await render({'query': 'pydantic'}, tool_arg_chars=0) == '● search_repositories\n\n'


async def test_multiline_values_stay_on_one_line() -> None:
    text = await render({'code': 'import os\nprint(os.getcwd())\n' * 20, 'tag': '\x1b[2J'}, width=60)
    lines = text.splitlines()
    assert lines[1:] == ['']
    assert len(lines[0]) <= 60
    assert '\x1b' not in text


async def test_invalid_json_is_shown_rather_than_dropped() -> None:
    assert await render('{oops', tool_arg_chars=10) == '● search_repositories INVALID_JSON="{oops"\n\n'


def test_names_use_the_accent_and_values_are_muted() -> None:
    text = tool_arguments_text({'path': 'a.py', 'limit': 3}, max_chars=40)
    assert text.plain == 'path="a.py" limit=3'
    styles = {text.plain[span.start : span.end]: span.style for span in text.spans}
    accent, muted = theme.color(theme.ACCENT), theme.color(theme.MUTED)
    assert styles == {'path': accent, '="a.py"': muted, 'limit': accent, '=3': muted}
    console = Console(file=io.StringIO(), force_terminal=True, color_system='truecolor')
    with console.capture() as capture:
        console.print(text)
    assert '\x1b[1;38;2;229;32;233mpath' in capture.get()  # bold Lithium, like the tool name


def test_setting_is_validated_and_defaults_to_forty() -> None:
    assert Settings().tool_arg_chars == 40
    assert resolve_settings({'display.tool_arg_chars': 12}).tool_arg_chars == 12
    with pytest.raises(ValueError):
        resolve_settings({'display.tool_arg_chars': -1})
