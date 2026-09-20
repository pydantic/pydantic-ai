"""CLI settings resolution and interactive application startup."""

import argparse
import asyncio
import os
from pathlib import Path

from pydantic_ai.usage import UsageLimits

from ._app import DEFAULT_PLUGINS, chat, create_agent
from .commands import config_command, plugins_command
from .config import resolve_settings
from .project_settings import load_project_settings
from .settings_store import SettingsStore
from .worktrees import create_worktree


def run() -> None:
    """Parse explicit overrides without replacing persisted preferences."""
    parser = argparse.ArgumentParser(description='CLAI 2.0: streaming Pydantic AI terminal')
    parser.add_argument(
        '--resume', nargs='?', const='', metavar='SESSION-ID', help='Restore a saved session; no ID opens the browser'
    )
    parser.add_argument(
        '--worktree',
        '-w',
        nargs='?',
        const='',
        metavar='NAME',
        help='Start in a new Git worktree; omit NAME to generate one',
    )
    parser.add_argument('--model', help='Provider-qualified model name')
    parser.add_argument('--request-limit', type=int)
    parser.add_argument('--database', type=Path, help='Settings database location')
    parser.add_argument('command', nargs='?', choices=('config', 'plugins'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        if args.command and (args.resume is not None or args.worktree is not None):
            parser.error('--resume and --worktree cannot be combined with config or plugins')
        if args.worktree is not None and args.resume is not None:
            parser.error('--worktree cannot be combined with --resume; resume from an existing worktree directory')
        store = SettingsStore(args.database)
        store.path = store.path.resolve()
        if args.command:
            handler = config_command if args.command == 'config' else plugins_command
            print(handler(store, args.arguments))
            return
        if args.worktree is not None:
            workspace = create_worktree(name=args.worktree)
            print(f'Worktree: {workspace} (branch: clai/{workspace.name}). Kept on exit.')
            os.chdir(workspace)
        project = load_project_settings(Path.cwd())
        overrides = store.overrides() | project.overrides
        if model := args.model or os.getenv('CLAI_MODEL'):
            overrides['model'] = model
        if args.request_limit is not None:
            overrides['run.request_limit'] = args.request_limit
        settings = resolve_settings(overrides)
        asyncio.run(
            chat(
                create_agent(),
                deps=None,
                usage_limits=UsageLimits(request_limit=settings.request_limit),
                settings=settings,
                store=store,
                builtin_plugins=DEFAULT_PLUGINS,
                project=project,
                resume=args.resume,
            )
        )
    except (ValueError, TypeError, ImportError, AttributeError, LookupError, OSError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        pass
