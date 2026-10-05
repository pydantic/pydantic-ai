"""CLI settings resolution and interactive application startup."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from pydantic_clai2.runtime.launch import launch_session_id
from pydantic_clai2.ui.rendering.splash import Splash


def run(*, splash: Splash | None = None) -> None:
    """Parse explicit overrides without replacing persisted preferences."""
    parser = argparse.ArgumentParser(description='CLAI 2.0: streaming Pydantic AI terminal')
    parser.add_argument(
        '--resume', nargs='?', const='', metavar='SESSION-ID', help='Restore a saved session; no ID opens the browser'
    )
    parser.add_argument(
        '--session-id',
        metavar='UUID',
        help='Start the new session under this ID, or with --fork-session, save the copy under it',
    )
    parser.add_argument(
        '--fork-session',
        action='store_true',
        help='With --resume, continue in a copy of the session under a new ID, leaving the original as it was',
    )
    parser.add_argument(
        '--worktree',
        '-w',
        nargs='?',
        const='',
        metavar='NAME',
        help='Start in a Git worktree, reopening NAME if it exists; omit NAME to generate one',
    )
    parser.add_argument(
        '-a',
        '--agent',
        metavar='MODULE:ATTR',
        help=(
            'Chat with an existing Pydantic AI Agent instance, e.g. pydantic_ai.main:my_cool_agent; '
            "loads no plugins for this session and keeps the agent's model unless -m is given"
        ),
    )
    parser.add_argument('-m', '--model', help='Provider-qualified model name')
    parser.add_argument(
        '-p', '--prompt', metavar='TEXT', help='Run one prompt without interaction and print only the answer'
    )
    parser.add_argument('--request-limit', type=int)
    parser.add_argument('--database', type=Path, help='Settings database location')
    parser.add_argument('command', nargs='?', choices=('config', 'plugins'))
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    _validate_args(args, parser)
    if args.database is not None:
        # Before `--worktree` changes directory, so a restart after `/update` reopens the same database.
        args.database = args.database.resolve()
    try:
        from pydantic_ai.usage import UsageLimits
        from pydantic_clai2._app import DEFAULT_PLUGINS, STOCK_PLUGINS, chat, create_stock_agent as create_agent
        from pydantic_clai2.cli.agent_import import import_agent
        from pydantic_clai2.cli.self_update import Relaunch
        from pydantic_clai2.commands import config_command, plugins_command
        from pydantic_clai2.config import resolve_settings
        from pydantic_clai2.config.project_settings import load_project_settings
        from pydantic_clai2.config.settings_store import SettingsStore
        from pydantic_clai2.runtime.worktrees import offer_worktree_cleanup, open_worktree
    finally:
        if splash is not None:
            splash.stop()
    try:
        store = SettingsStore(args.database)
        store.path = store.path.resolve()
        if args.command:
            handler = config_command if args.command == 'config' else plugins_command
            print(handler(store, args.arguments))
            return
        agent = import_agent(args.agent) if args.agent is not None else None
        if args.worktree is not None:
            worktree = open_worktree(name=args.worktree)
            print(
                f'{"Worktree" if worktree.created else "Reopened worktree"}: {worktree.path} '
                f'(branch: {worktree.branch}). Kept unless removal is confirmed on exit.',
                file=sys.stderr if args.prompt is not None else sys.stdout,
            )
            os.chdir(worktree.path)
        project = load_project_settings(Path.cwd())
        overrides = store.overrides() | project.overrides
        if model := args.model or os.getenv('CLAI_MODEL'):
            overrides['model'] = model
        if args.request_limit is not None:
            overrides['run.request_limit'] = args.request_limit
        settings = resolve_settings(overrides)
        if agent is not None and agent.model is not None and not model:
            # Keep the agent's own model over saved, project, and default ones; only -m or CLAI_MODEL replaces it.
            settings = settings.model_copy(update={'model': None})
        if args.prompt is not None:
            from pydantic_clai2.cli.headless import run_headless

            raise SystemExit(
                asyncio.run(
                    run_headless(
                        text=args.prompt,
                        settings=settings,
                        store=store,
                        project=project,
                        resume=args.resume,
                        agent=agent,
                        session_id=args.session_id,
                        fork_session=args.fork_session,
                    )
                )
            )
        asyncio.run(
            chat(
                create_agent() if agent is None else agent,
                deps=None,
                usage_limits=UsageLimits(request_limit=settings.request_limit),
                settings=settings,
                store=store,
                builtin_plugins=DEFAULT_PLUGINS if args.agent else STOCK_PLUGINS,
                project=project,
                resume=args.resume,
                load_plugins=agent is None,
                session_id=args.session_id,
                fork_session=args.fork_session,
            )
        )
        offer_worktree_cleanup()
    except Relaunch as relaunch:
        # Replace this process with the new build; the working directory, a worktree included, carries over.
        argv = relaunch_argv(
            args,
            executable=relaunch.executable,
            session_id=relaunch.session_id,
            new_session_id=relaunch.new_session_id,
        )
        sys.stdout.flush()
        os.execv(relaunch.executable, argv)
    except (ValueError, TypeError, ImportError, AttributeError, LookupError, OSError) as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        if args.prompt is not None:
            raise SystemExit(130) from None


def relaunch_argv(
    args: argparse.Namespace, *, executable: str, session_id: str | None, new_session_id: str | None = None
) -> list[str]:
    """The launch options to restart with after `/update`, resuming `session_id` instead of any `--resume`.

    An unsaved conversation started with `--session-id` restarts under `new_session_id`.
    """
    argv = [executable]
    if args.agent is not None:
        argv += ['--agent', args.agent]
    if args.model is not None:
        argv += ['--model', args.model]
    if args.request_limit is not None:
        argv += ['--request-limit', str(args.request_limit)]
    if args.database is not None:
        argv += ['--database', str(args.database)]
    if session_id is not None:
        argv += ['--resume', session_id]
    elif new_session_id is not None:
        argv += ['--session-id', new_session_id]
    return argv


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command and (
        args.resume is not None
        or args.worktree is not None
        or args.agent is not None
        or args.session_id is not None
        or args.fork_session
    ):
        parser.error(
            '--resume, --session-id, --fork-session, --worktree, and --agent cannot be combined with config or plugins'
        )
    try:
        args.session_id = launch_session_id(resume=args.resume, session_id=args.session_id, fork=args.fork_session)
    except ValueError as exc:
        parser.error(str(exc))
    if args.worktree is not None and args.resume is not None:
        parser.error('--worktree cannot be combined with --resume; resume from an existing worktree directory')
    if args.prompt is not None:
        if args.command:
            parser.error('--prompt cannot be combined with config or plugins')
        if not args.prompt.strip():
            parser.error('--prompt requires non-empty text')
        if args.resume == '':
            parser.error('--prompt requires an explicit --resume SESSION-ID')
