"""One line of typed input under the transcript, for a command that printed something to read first."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from termflow.tui.layout import truncate
from termflow.tui.terminal import raw_mode

from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.prompt.prompt_buffer import PromptBuffer
from pydantic_clai2.ui.prompt.prompt_surface import PromptSurface
from pydantic_clai2.ui.prompt.question_input import Paste, QuestionKey, TranscriptKey, question_input
from pydantic_clai2.ui.rendering import theme

_ANSWER_ROWS = 3
"""Rows a long answer, such as a pasted URL, wraps onto before it scrolls with the cursor."""


async def read_line(prompt: str, *, console: Console, hidden: bool = False) -> str | None:
    """Read one line of input; `None` when the user presses Esc or Ctrl-C.

    In the live editor, `prompt` and the answer are pinned under the transcript, the way `ask_user`
    shows its questions, so what the command printed just before (a sign-in URL, say) stays on screen
    above them. `hidden` shows how many characters were typed instead of the text, for a value such
    as a URL carrying an authorization code. Without the live editor this is a plain terminal prompt.

    Slash commands can call it directly. A tool or hook that asks mid-run takes the screen first,
    with `async with host.full_screen()`.
    """
    surface = console.file
    if not isinstance(surface, PromptSurface):
        try:
            with patch_stdout():
                return await PromptSession[str]().prompt_async(prompt, is_password=hidden)
        except (KeyboardInterrupt, EOFError):
            return None
    line = LineInput(prompt=prompt, hidden=hidden)
    with question_input() as keys:
        return await run_worker(partial(line.run, surface=surface, keys=keys), inline=True)


@dataclass(kw_only=True)
class LineInput:
    """The prompt and answer being typed, independent of terminal input."""

    prompt: str
    hidden: bool = False
    answer: PromptBuffer = field(default_factory=PromptBuffer)

    def frame(self, *, width: int) -> tuple[str, ...]:
        """The prompt, then the answer with its painted cursor."""
        prompt = theme.sgr(theme.ACCENT) + truncate(self.prompt, width) + '\x1b[0m'
        if self.hidden:
            count = len(self.answer.text)
            shown = f'[{count} character{"" if count == 1 else "s"}]' if count else ''
            return prompt, shown + '\x1b[7m \x1b[27m'
        return prompt, *self.answer.rows(width=width, limit=_ANSWER_ROWS)

    def feed(self, key: str | Paste) -> str | None:
        """Apply a key; return the answer when Enter submits it."""
        if isinstance(key, Paste):
            self.answer.insert(''.join(key.text.splitlines()))
        elif key == 'enter':
            return self.answer.text
        elif key != 'ctrl-r':
            # The answer has no history to search.
            self.answer.edit(key)
        if self.hidden:
            # The count's cursor is always at the end, so a hidden answer is only ever edited there.
            self.answer.cursor = len(self.answer.text)
        return None

    def run(self, *, surface: PromptSurface, keys: Callable[[], QuestionKey]) -> str | None:
        """Paint through the released editor's panel until Enter, Esc, or Ctrl-C."""
        try:
            with raw_mode():
                while True:
                    width, _ = surface.size()
                    surface.paint(self.frame(width=width))
                    key = keys()
                    if isinstance(key, TranscriptKey):
                        # Reading back through or copying the transcript must not type into the answer.
                        surface.transcript_key(key.key, key.data)
                    elif key in ('escape', 'ctrl-c'):
                        return None
                    elif (answer := self.feed(key)) is not None:
                        return answer
        finally:
            surface.release()
