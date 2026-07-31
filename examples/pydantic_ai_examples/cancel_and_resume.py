"""Interactive chat demonstrating how to cancel and resume a streaming agent run.

Press Esc while the agent is responding to cancel the current turn. The next message resumes the
conversation with the history preserved before cancellation.

Run with:

    uv run -m pydantic_ai_examples.cancel_and_resume
"""

from __future__ import annotations as _annotations

import asyncio
from collections.abc import Callable

import logfire
from prompt_toolkit import Application, PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl

from pydantic_ai import Agent, CancellationToken, ModelMessage, RunCancelled

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


async def stream_turn(
    agent: Agent[None],
    prompt: str,
    history: list[ModelMessage],
    token: CancellationToken,
    on_text: Callable[[str], None],
) -> tuple[list[ModelMessage], bool]:
    """Stream one turn, returning resumable history and whether it was cancelled."""
    try:
        async with agent.run_stream(
            prompt, message_history=history, cancellation_token=token
        ) as result:
            async for delta in result.stream_text(delta=True):
                on_text(delta)
        return result.all_messages(), False
    except RunCancelled as exc:
        return exc.all_messages(), True


async def _run_interactive_turn(
    agent: Agent[None],
    prompt: str,
    history: list[ModelMessage],
) -> tuple[list[ModelMessage], bool]:
    token = CancellationToken()
    chunks: list[str] = []
    bindings = KeyBindings()

    def cancel_turn(_event: KeyPressEvent) -> None:
        token.cancel()

    bindings.add('escape')(cancel_turn)

    control = FormattedTextControl(
        lambda: FormattedText([('class:answer', f'Agent: {"".join(chunks)}')])
    )
    application: Application[None] = Application(
        layout=Layout(Window(control)),
        key_bindings=bindings,
        full_screen=False,
    )

    def on_text(delta: str) -> None:
        chunks.append(delta)
        application.invalidate()

    async def run_turn() -> tuple[list[ModelMessage], bool]:
        try:
            return await stream_turn(agent, prompt, history, token, on_text)
        finally:
            application.exit()

    turn_task = asyncio.create_task(run_turn())
    await application.run_async()
    messages, was_cancelled = await turn_task
    if was_cancelled:
        print('⏹ cancelled')
    return messages, was_cancelled


async def main() -> None:
    agent = Agent('openai:gpt-5-mini')
    session: PromptSession[str] = PromptSession()
    history: list[ModelMessage] = []

    print(
        'Chat with Pydantic AI. Press Esc to cancel a response; Ctrl-C or Ctrl-D to exit.'
    )
    while True:
        try:
            prompt = await session.prompt_async('\nYou: ')
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            continue

        # A cancelled token stays cancelled, so each turn gets a fresh one.
        history, _ = await _run_interactive_turn(agent, prompt, history)


if __name__ == '__main__':
    asyncio.run(main())
