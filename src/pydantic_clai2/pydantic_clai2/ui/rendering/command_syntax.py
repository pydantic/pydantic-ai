"""Highlight a shell command as Bash, and the code embedded in it (heredocs, `python -c`) in its own language."""

import re
from dataclasses import dataclass

from rich.style import Style
from rich.syntax import Syntax, SyntaxTheme
from rich.text import Text

from pydantic_clai2.ui.rendering import theme

_INTERPRETERS = {
    'python': 'python',
    'pypy': 'python',
    'ipython': 'python',
    'node': 'javascript',
    'nodejs': 'javascript',
    'ruby': 'ruby',
    'perl': 'perl',
    'psql': 'sql',
    'sqlite': 'sql',
    'mysql': 'sql',
    'duckdb': 'sql',
    'bash': 'bash',
    'sh': 'bash',
    'zsh': 'bash',
    'dash': 'bash',
}
"""The Pygments lexer for the code a command reads, by its name without a path or version (`python3.12`)."""

_NAMES = {
    'py': 'python',
    'python': 'python',
    'js': 'javascript',
    'mjs': 'javascript',
    'cjs': 'javascript',
    'javascript': 'javascript',
    'ts': 'typescript',
    'rb': 'ruby',
    'ruby': 'ruby',
    'pl': 'perl',
    'perl': 'perl',
    'sql': 'sql',
    'sh': 'bash',
    'bash': 'bash',
    'zsh': 'bash',
    'json': 'json',
    'yaml': 'yaml',
    'yml': 'yaml',
    'toml': 'toml',
    'html': 'html',
    'css': 'css',
}
"""The Pygments lexer for a file extension or a heredoc delimiter's name (`PYEOF`, `EOF_SQL`), lowercase."""

_INLINE_FLAGS = {'python': 'c', 'javascript': 'ep', 'ruby': 'e', 'perl': 'eE', 'sql': 'ce', 'bash': 'c'}
"""The flags whose argument is code, by language: `python -c`, `node -e`, `psql -c`, `bash -c`."""

_HEREDOC = re.compile(r'(?<!<)<<(-?)[ \t]*\\?([\'"]?)([A-Za-z_]\w*)\2')
_LITERAL = re.compile(r'\$\'(?:[^\'\\]|\\.)*\'|\'[^\']*\'|"(?:[^"\\]|\\.)*"|\\.|(?<![^\s;&|(])#[^\n]*\n')
"""Closed quotes, escaped characters, and whole comment lines, in which `<<` is text."""
_OPEN_QUOTE_OR_COMMENT = re.compile(r'[\'"]|(?<![^\s;&|(])#')
_VERSION = re.compile(r'[\d.]+$')
"""A version suffix on a command name: `python3.12`, `sqlite3`."""
_SEPARATOR = re.compile(r'[;&|(){}`]|\$\(')
_TARGET = re.compile(r'(?:>{1,2}|\btee(?:[ \t]+-\w+)*)[ \t]*([^\s<>;&|]+)')
_INLINE = re.compile(
    r'(?<![\w./-])(?:[\w./-]*/)?(?P<command>[A-Za-z]+)[\d.]*'
    r'(?:[ \t]+-[-\w=]+)*?[ \t]+-(?P<flag>[A-Za-z])[ \t]+'
    r'(?:\'(?P<single>[^\']*)\'|"(?P<double>[^"$`\\]*)")'
)
"""A command, a flag, and its quoted argument, when the argument reaches the command as shown: no expansions."""
_SECONDARY = Style(dim=True, underline=False)
"""Lexed code reads as secondary to the answer: dimmed, without the underline `syntax_theme()` gives module names."""


@dataclass(frozen=True, kw_only=True)
class Segment:
    """A run of a shell command and the Pygments lexer for it; `None` when its language is unknown."""

    text: str
    language: str | None
    embedded: tuple[tuple[int, int, str], ...] = ()
    """Code inside a Bash run, by start, end, and language, such as the quoted argument of `python -c`."""


def command_segments(command: str, *, end: int | None = None) -> list[Segment]:
    """Split `command` into Bash runs and the heredoc bodies between them, keeping every character in order.

    A heredoc body takes the language of the command reading it, else of the file it is written to,
    else of a conventional delimiter (`PYEOF`, `SQL`). Delimiter lines stay Bash. Heredocs are not
    looked for on lines ending past `end`, which bounds the work for a long command shown in part.
    """
    limit = len(command) if end is None else end
    segments: list[Segment] = []
    shell = body = ''
    position = 0
    # Heredocs opened and not yet closed: delimiter, whether leading tabs are stripped, language.
    pending: list[tuple[str, bool, str | None]] = []
    *lines, last = command.split('\n')
    for line in [*(f'{line}\n' for line in lines), last]:
        if position > limit and not pending:
            shell += command[position:]  # Nothing from here on is shown.
            break
        position += len(line)
        if pending:
            delimiter, strip_tabs, language = pending[0]
            if (line.lstrip('\t') if strip_tabs else line).removesuffix('\n') != delimiter:
                body += line
                continue
            segments.append(Segment(text=body, language=language))
            body = ''
            pending.pop(0)
        start = len(shell)
        shell += line
        # A body starts on the next line, so a heredoc opened on a line ending past `limit` is never shown.
        if position <= limit:
            pending.extend(
                (match[3], bool(match[1]), _heredoc_language(line, match=match))
                for match in _HEREDOC.finditer(line)
                if _opens_heredoc(shell[: start + match.start()])
            )
        if pending:
            segments.append(_shell_segment(shell))
            shell = ''
    if pending:
        segments.append(Segment(text=body, language=pending[0][2]))
    segments.append(_shell_segment(shell))
    return [segment for segment in segments if segment.text]


def _opens_heredoc(before: str) -> bool:
    """Whether a `<<` after `before`, the Bash run so far, is an operator: not in quotes, a comment, or `$((1<<n))`."""
    code = _LITERAL.sub('', before)
    return not _OPEN_QUOTE_OR_COMMENT.search(code) and code.count('((') <= code.count('))')


def _heredoc_language(line: str, *, match: re.Match[str]) -> str | None:
    """The language of the heredoc `match` opens, from the simple command around it on `line`."""
    before = _SEPARATOR.split(line[: match.start()])[-1]
    after = _SEPARATOR.split(line[match.end() :])[0]
    words = [word.strip('\'"') for word in before.split()]
    for index, word in enumerate(words):
        if word.startswith(('>', '<')):
            break
        language = _INTERPRETERS.get(_command_name(word))
        if language is not None:
            # `python3 script.py <<EOF` and `python3 -c CODE <<EOF` run other code, which reads the heredoc as data.
            rest = words[index + 1 :]
            script = next((word for word in rest if not word.startswith('-')), '')
            code_flags = {f'-{flag}' for flag in _INLINE_FLAGS[language]}
            if _extension_language(script) != language and code_flags.isdisjoint(rest):
                return language
            break
    for target in _TARGET.findall(f'{before} {after}'):
        language = _extension_language(target.strip('\'"'))
        if language is not None:
            return language
    name = ''.join(word for word in match[3].lower().split('_') if word not in ('eof', 'end'))
    return _NAMES.get(name.removesuffix('eof'))


def _command_name(word: str) -> str:
    return _VERSION.sub('', word.rpartition('/')[2])


def _extension_language(path: str) -> str | None:
    name = path.rpartition('/')[2]
    return _NAMES.get(name.rpartition('.')[2].lower()) if '.' in name else None


def _shell_segment(text: str) -> Segment:
    """A Bash run, with each interpreter's inline code (`python -c '...'`) embedded in its own language."""
    embedded: list[tuple[int, int, str]] = []
    for match in _INLINE.finditer(text):
        language = _INTERPRETERS.get(match['command'])
        if language is not None and match['flag'] in _INLINE_FLAGS[language]:
            group = 'single' if match['single'] is not None else 'double'
            embedded.append((match.start(group), match.end(group), language))
    return Segment(text=text, language='bash', embedded=tuple(embedded))


def highlight_command(command: str, *, end: int) -> Text:
    """`command[:end]` in `theme.syntax_theme()`, dimmed and not underlined, with code in an unknown language muted.

    Languages are found in the whole command, but nothing past `end` is lexed. `command` must
    already be terminal-safe: highlighting styles text, it does not escape it.
    """
    syntax_theme = theme.syntax_theme()
    text = Text()
    for segment in command_segments(command, end=end):
        if len(text) >= end:
            break
        code = segment.text[: end - len(text)]
        if segment.language is None:
            text.append(code, style=theme.color(theme.MUTED))
            continue
        # Bash lexes the whole run, quotes included, so code inside quotes only replaces what they hold.
        highlighted = _lex(code, language=segment.language, syntax_theme=syntax_theme)
        for start, stop, language in segment.embedded:
            stop = min(stop, len(code))
            if start < stop:
                before, _, after = highlighted.divide((start, stop))
                highlighted = Text.assemble(
                    before, _lex(code[start:stop], language=language, syntax_theme=syntax_theme), after
                )
        text.append_text(highlighted)
    return text


def _lex(code: str, *, language: str, syntax_theme: SyntaxTheme) -> Text:
    # No tab expansion, so the highlighted text keeps the command's characters and wraps like it.
    highlighted = Syntax(code, language, theme=syntax_theme, tab_size=0).highlight(code)
    if not code.endswith('\n'):
        highlighted.right_crop(1)  # The newline the lexer ends all code with.
    # Last, so it overrides the token styles; it also dims names left in the terminal foreground.
    highlighted.stylize(_SECONDARY)
    return highlighted
