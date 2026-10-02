"""Choose an earlier prompt without replaying tools or undoing external effects."""

from collections.abc import Sequence
from dataclasses import dataclass

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_clai2.plugins import Conversation
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.prompt.image_input import ImageInput
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from pydantic_clai2.ui.rendering._rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text


@dataclass(kw_only=True)
class RewindPoint:
    """An editable user prompt at a run boundary in retained history."""

    message_index: int
    text: str
    images: tuple[BinaryContent, ...]


def build_rewind_menu(messages: Sequence[ModelMessage]) -> Menu:
    """Offer run-start prompts, never a steering message amid unfinished tool calls."""
    items: list[MenuItem] = []
    seen_runs: set[str] = set()
    for index, message in enumerate(messages):
        if message.run_id is not None:
            if message.run_id in seen_runs:
                continue
            seen_runs.add(message.run_id)
        if not isinstance(message, ModelRequest) or any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for part in message.parts
        ):
            continue
        prompts = [part for part in message.parts if isinstance(part, UserPromptPart)]
        if not prompts:
            continue
        content = [
            item for part in prompts for item in ([part.content] if isinstance(part.content, str) else part.content)
        ]
        text = '\n'.join(item for item in content if isinstance(item, str))
        images = tuple(item for item in content if isinstance(item, BinaryContent))
        supported = all(isinstance(item, (str, BinaryContent)) for item in content)
        label = ' '.join(terminal_text(text).split()) or '[attachment]'
        items.append(
            MenuItem(
                f'{len(items) + 1}. {label}' + ('' if supported else ' (unsupported attachment)'),
                value=RewindPoint(message_index=index, text=text, images=images),
                disabled=not supported,
                description='' if supported else 'This prompt contains attachments the editor cannot restore.',
            )
        )
    return (
        MenuBuilder('Rewind conversation')
        .style(markdown_style())
        .items(list(reversed(items)) or [MenuItem('No earlier prompts to rewind to.', disabled=True)])
        .preview(
            lambda item: (
                'Remove this prompt and all\nlater messages, then edit it again.\n\n'
                'Files and tool side effects\nare NOT undone.\n\n'
                f'{terminal_text(item.value.text)}\n\n'
                f'Attachments: {len(item.value.images)}'
                if isinstance(item.value, RewindPoint)
                else 'No editable prompts in the retained history.'
            )
        )
        .footer_hint('Up/Down select - Enter rewind - Esc cancel | does NOT undo files')
        .key_source(menu_key)
        .build()
    )


async def rewind(conversation: Conversation, editor: LivePrompt, *, runners: Runners = TERMINAL) -> str:
    """Persist the selected boundary before changing the draft or its attachments."""
    messages = conversation.messages
    result = await run_worker(lambda: runners.run_list(build_rewind_menu(messages)))
    if result.cancelled or result.item is None or not isinstance(result.item.value, RewindPoint):
        return 'No changes.'
    point = result.item.value
    staged = ImageInput()
    staged.pending = dict(editor.images.pending)
    markers = staged.attach(point.images) if point.images else ''
    await conversation.commit_messages(messages[: point.message_index])
    editor.images.pending = staged.pending
    editor.restore_draft(markers + point.text)
    return 'Conversation rewound. Edit the restored prompt; files and tool side effects were not undone.'
