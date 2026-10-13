"""Recognize `!` commands that need the terminal itself, such as editors, pagers, and REPLs.

Captured output is what makes a `!` command useful to the model, so it stays the default. A
program on these lists draws on, or reads from, the terminal, and piping it only garbles it.
"""

import re
import shlex

_FULL_SCREEN = frozenset(
    {
        # Editors.
        'emacs',
        'helix',
        'hx',
        'joe',
        'kak',
        'mg',
        'micro',
        'nano',
        'ne',
        'nvim',
        'pico',
        'vi',
        'view',
        'vim',
        'vimdiff',
        # Pagers.
        'less',
        'man',
        'more',
        'most',
        # Monitors.
        'atop',
        'btop',
        'glances',
        'htop',
        'iotop',
        'nvtop',
        'top',
        'watch',
        # Remote sessions and multiplexers.
        'mosh',
        'screen',
        'ssh',
        'telnet',
        'tmux',
        # Terminal applications.
        'fzf',
        'k9s',
        'lazygit',
        'mc',
        'nnn',
        'ranger',
        'tig',
        'vifm',
    }
)
"""Programs that need the terminal whatever their arguments."""

_REPLS = frozenset(
    {
        'bash',
        'bpython',
        'cmd',
        'dash',
        'deno',
        'fish',
        'ghci',
        'ipython',
        'irb',
        'julia',
        'ksh',
        'lua',
        'mongosh',
        'mysql',
        'node',
        'nu',
        'powershell',
        'pry',
        'psql',
        'ptpython',
        'pwsh',
        'python',
        'redis-cli',
        'sh',
        'sqlite',
        'zsh',
    }
)
"""Programs that start an interactive session when given no arguments."""

_WRAPPERS: dict[str, frozenset[str]] = {
    'command': frozenset(),
    'doas': frozenset({'-C', '-u'}),
    'env': frozenset({'-C', '-S', '-u', '--chdir', '--split-string', '--unset'}),
    'exec': frozenset({'-a'}),
    'nice': frozenset({'-n', '--adjustment'}),
    'sudo': frozenset(
        {
            '-C',
            '-D',
            '-g',
            '-p',
            '-R',
            '-r',
            '-T',
            '-t',
            '-U',
            '-u',
            '--chdir',
            '--chroot',
            '--close-from',
            '--command-timeout',
            '--group',
            '--other-user',
            '--prompt',
            '--role',
            '--type',
            '--user',
        }
    ),
    'time': frozenset({'-f', '-o', '--format', '--output'}),
}
"""Programs that run the command after their options, so the command decides; each with its options that take a value."""

_GIT_PROMPTS: dict[str, tuple[str, tuple[str, ...]]] = {
    'add': ('eip', ('--edit', '--interactive', '--patch')),
    'checkout': ('p', ('--patch',)),
    'rebase': ('i', ('--interactive',)),
    'reset': ('p', ('--patch',)),
    'restore': ('p', ('--patch',)),
    'stash': ('p', ('--patch',)),
}
"""Git subcommands that are interactive with one of these short or long options."""

_GIT_COMMIT_MESSAGE = ('CFm', ('--file', '--fixup', '--message', '--no-edit', '--reuse-message'))
"""`git commit` options that supply the message, so no editor opens."""

_ASSIGNMENT = re.compile(r'[A-Za-z_]\w*=')
_VERSION = re.compile(r'[\d.]+$')


def needs_terminal(command: str) -> bool:
    """Whether any command in `command`'s pipelines and lists is a known interactive program.

    `git log | less` needs the terminal for `less`. A command line that cannot be split, such as one
    with an unclosed quote, does not.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    return any(_interactive(words) for words in _simple_commands(tokens))


def _simple_commands(tokens: list[str]) -> list[list[str]]:
    """The words of each simple command, without operators or redirection targets."""
    commands: list[list[str]] = [[]]
    redirect = False
    for token in tokens:
        if redirect:
            redirect = False
        elif token.strip('();<>|&'):
            commands[-1].append(token)
        elif '<' in token or '>' in token:
            redirect = True
        else:
            commands.append([])
    return commands


def _program(word: str) -> str:
    """The program's name without its directory, `.exe`, or version, so `python3.12` is `python`."""
    name = re.split(r'[\\/]', word)[-1].removesuffix('.exe')
    return _VERSION.sub('', name) or name


def _interactive(words: list[str]) -> bool:
    index = 0
    while index < len(words):
        if _ASSIGNMENT.match(words[index]):
            index += 1
            continue
        valued = _WRAPPERS.get(_program(words[index]))
        if valued is None:
            break
        index += 1
        while index < len(words) and words[index].startswith('-'):
            index += 2 if words[index] in valued else 1
    if index >= len(words):
        return False
    name, arguments = _program(words[index]), words[index + 1 :]
    if name == 'git':
        return _git_interactive(arguments)
    return name in _FULL_SCREEN or (name in _REPLS and not arguments)


def _git_interactive(arguments: list[str]) -> bool:
    index = 0
    while index < len(arguments) and arguments[index].startswith('-'):
        index += 2 if arguments[index] in ('-C', '-c') else 1
    if index >= len(arguments):
        return False
    subcommand, options = arguments[index], arguments[index + 1 :]
    if subcommand == 'commit':
        return not _has_option(options, *_GIT_COMMIT_MESSAGE)
    return subcommand in _GIT_PROMPTS and _has_option(options, *_GIT_PROMPTS[subcommand])


def _has_option(options: list[str], short: str, long: tuple[str, ...]) -> bool:
    """Whether `options` has one of `long`, or a short option bundle (`-am`) with a letter in `short`."""
    for option in options:
        if option.startswith('--'):
            if option.split('=', 1)[0] in long:
                return True
        elif option.startswith('-') and any(letter in short for letter in option[1:]):
            return True
    return False
