"""CLI settings resolution and interactive application startup."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from pydantic_clai2.ui.rendering.splash import Splash


def run(*, splash: Splash | None = None) -> None:
    """Parse explicit overrides without replacing persisted preferences."""
    parser = argparse.ArgumentParser(description='CLAI 2.0: streaming Pydantic AI terminal')
    resume_flags = parser.add_mutually_exclusive_group()
    resume_flags.add_argument(
        '--resume', nargs='?', const='', metavar='SESSION-ID', help='Restore a saved session; no ID opens the browser'
    )
    for flag, name in (('--resume-claude', 'Claude Code'), ('--resume-codex', 'Codex')):
        resume_flags.add_argument(
            flag,
            nargs='?',
            const='',
            metavar='SESSION-ID',
            help=f'Import and restore a {name} session; no ID browses {name} sessions',
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
    _resume_source(args)
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
    worktree = None
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
            if args.prompt is not None:
                # Headless stdout carries only the answer; the shell shows the notice under its banner instead.
                print(worktree.notice, file=sys.stderr)
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
                        resume_from=args.resume_from,
                        agent=agent,
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
                resume_from=args.resume_from,
                load_plugins=agent is None,
                worktree=worktree,
            )
        )
        offer_worktree_cleanup()
    except Relaunch as relaunch:
        # Replace this process with the new build; the working directory, a worktree included, carries over.
        argv = relaunch_argv(args, executable=relaunch.executable, session_id=relaunch.session_id)
        sys.stdout.flush()
        os.execv(relaunch.executable, argv)
    except (ValueError, TypeError, ImportError, AttributeError, LookupError, OSError) as exc:
        if worktree is not None:
            # A startup error can come before the banner, so name the checkout this launch leaves behind.
            print(f'Worktree kept at {worktree.path} (branch: {worktree.branch}).', file=sys.stderr)
        parser.error(str(exc))
    except KeyboardInterrupt:
        if args.prompt is not None:
            raise SystemExit(130) from None


def relaunch_argv(args: argparse.Namespace, *, executable: str, session_id: str | None) -> list[str]:
    """The launch options to restart with after `/update`, resuming `session_id` instead of any `--resume`."""
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
    return argv


def _resume_source(args: argparse.Namespace) -> None:
    """Fold `--resume-claude` and `--resume-codex` into `resume`, so the same rules check all three."""
    args.resume_from = None
    for source, session_id in (('claude', args.resume_claude), ('codex', args.resume_codex)):
        if session_id is not None:
            args.resume_from, args.resume = source, session_id


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.command and (args.resume is not None or args.worktree is not None or args.agent is not None):
        parser.error('--resume, --worktree, and --agent cannot be combined with config or plugins')
    if args.worktree is not None and args.resume is not None:
        parser.error('--worktree cannot be combined with --resume; resume from an existing worktree directory')
    if args.prompt is not None:
        if args.command:
            parser.error('--prompt cannot be combined with config or plugins')
        if not args.prompt.strip():
            parser.error('--prompt requires non-empty text')
        if args.resume == '':
            parser.error('--prompt requires an explicit --resume SESSION-ID')
