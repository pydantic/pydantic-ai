"""Choose an earlier prompt without replaying tools or undoing external effects."""

from collections.abc import Sequence

from termflow.tui import MenuBuilder, MenuItem
from termflow.tui.menu import Menu

from pydantic_ai.messages import ModelMessage
from pydantic_clai2.plugins import Conversation
from pydantic_clai2.runtime.rewind import RewindPoint, rewind_points
from pydantic_clai2.ui.menus.field_menu import TERMINAL, Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.prompt.image_input import ImageInput
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from pydantic_clai2.ui.rendering.rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text


def build_rewind_menu(messages: Sequence[ModelMessage]) -> Menu:
    """Offer the `rewind_points` of `messages`, newest first."""
    items = [
        MenuItem(
            f'{number}. '
            + (' '.join(terminal_text(point.text).split()) or '[attachment]')
            + (f' ({point.reason})' if point.reason else ''),
            value=point,
            disabled=bool(point.reason),
            description=point.reason,
        )
        for number, point in enumerate(rewind_points(messages), start=1)
    ]
    return (
        MenuBuilder('Rewind conversation')
        .style(markdown_style())
        .items(list(reversed(items)) or [MenuItem('No earlier prompts to rewind to.', disabled=True)])
        .preview(_preview)
        .footer_hint('Up/Down select - Enter rewind - Esc cancel - does NOT undo files')
        .key_source(menu_key)
        .build()
    )


def _preview(item: MenuItem) -> str:
    if not isinstance(item.value, RewindPoint):
        return 'No editable prompts in the retained history.'
    if item.disabled:
        header = f'Cannot rewind this prompt:\n{item.description}.\n\n'
    else:
        header = (
            'Remove this prompt and all\nlater messages, then edit it again.\n\n'
            'Replaces your current draft\nand its attachments.\n\n'
        )
    return (
        header
        + 'Files and tool side effects\nare NOT undone.\n\n'
        + f'{terminal_text(item.value.text)}\n\nAttachments: {len(item.value.images)}'
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
    staged.retain(editor.queued_messages)
    markers = staged.attach(point.images) if point.images else ''
    await conversation.commit_messages(messages[: point.message_index])
    editor.images.pending = staged.pending
    editor.restore_draft(markers + point.text)
    return 'Conversation rewound. Edit the restored prompt; files and tool side effects were not undone.'
