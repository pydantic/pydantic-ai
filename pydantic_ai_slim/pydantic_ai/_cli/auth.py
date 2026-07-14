from __future__ import annotations as _annotations

import argparse
import json
import sys
import webbrowser
from collections.abc import Sequence

import anyio
from rich.console import Console

from ..auth.codex import CodexAuth, CodexAuthError, CodexAuthStatus, CodexDeviceCode


def cli_auth(args_list: Sequence[str], prog_name: str) -> int:
    """Run the `auth` command group."""
    parser = argparse.ArgumentParser(
        prog=f'{prog_name} auth',
        description='Manage model-provider authentication',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    login = subparsers.add_parser('login', help='Sign in to a model provider')
    login.add_argument('provider', choices=['codex'])
    login.add_argument('--method', choices=['browser', 'device'], default='browser')

    status = subparsers.add_parser('status', help='Show authentication status')
    status.add_argument('provider', nargs='?', choices=['codex'], default='codex')
    status.add_argument('--json', action='store_true', dest='json_output')

    refresh = subparsers.add_parser('refresh', help='Refresh managed credentials')
    refresh.add_argument('provider', choices=['codex'])

    logout = subparsers.add_parser('logout', help='Sign out from a model provider')
    logout.add_argument('provider', choices=['codex'])
    logout.add_argument('--local-only', action='store_true')

    args = parser.parse_args(list(args_list))
    console = Console()
    try:
        return anyio.run(_run_auth_command, args, console)
    except KeyboardInterrupt:  # pragma: no cover
        return 130


async def _run_auth_command(args: argparse.Namespace, console: Console) -> int:
    auth = CodexAuth()
    try:
        if args.command == 'login':
            if args.method == 'browser':

                def open_url(url: str) -> None:
                    if not webbrowser.open(url):
                        console.print('Open this URL to continue Codex login:', markup=False)
                        console.print(url, markup=False)

                console.print('Opening a browser to sign in to Codex...', markup=False)
                await auth.login_browser(open_url)
            else:

                def show_code(device_code: CodexDeviceCode) -> None:
                    console.print('Open this URL and enter the one-time code:', markup=False)
                    console.print(device_code.verification_url, markup=False)
                    console.print(device_code.user_code.get_secret_value(), markup=False)

                await auth.login_device(show_code)
            console.print('Codex login complete.', markup=False)
            return 0

        if args.command == 'status':
            status = await auth.status()
            if args.json_output:
                sys.stdout.write(json.dumps(_status_json(status), separators=(',', ':')) + '\n')
            elif status.authenticated:
                state = 'refresh required' if status.needs_refresh else 'ready'
                expiry = status.expires_at.isoformat() if status.expires_at is not None else 'unknown'
                console.print(f'Codex authentication: {state} (expires {expiry})', markup=False)
            else:
                console.print('Codex authentication: not signed in', markup=False)
            return 0

        if args.command == 'refresh':
            credentials = await auth.refresh()
            console.print(f'Codex credentials refreshed (expires {credentials.expires_at.isoformat()}).', markup=False)
            return 0

        if args.command == 'logout':
            result = await auth.logout(local_only=args.local_only)
            if result.revocation_error is not None:
                console.print(f'Warning: {result.revocation_error}', style='yellow', markup=False)
            if result.local_credentials_removed:
                console.print('Codex logout complete.', markup=False)
            else:
                console.print('Codex authentication: not signed in', markup=False)
            return 0

        raise AssertionError(f'Unknown auth command: {args.command}')  # pragma: no cover
    except CodexAuthError as error:
        console.print(f'Error: {error}', style='red', markup=False)
        return 1


def _status_json(status: CodexAuthStatus) -> dict[str, str | bool | None]:
    return {
        'provider': 'codex',
        'authenticated': status.authenticated,
        'expires_at': status.expires_at.isoformat() if status.expires_at is not None else None,
        'needs_refresh': status.needs_refresh,
        'account_is_fedramp': status.account_is_fedramp,
    }
