"""A stand-in for the GitHub CLI, so tests never read or change a real `gh` login."""

import os
import sys
import time
from pathlib import Path

state = Path(os.environ['FAKE_GH_STATE'])
args = sys.argv[1:]
host = args[args.index('--hostname') + 1]
with (state / 'calls').open('a') as calls:
    calls.write(' '.join(args) + '\n')
if args[:2] == ['auth', 'token']:
    if (state / 'token-hang').exists():
        time.sleep(60)
    token = state / host
    if os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN'):
        print('token-from-the-environment')
    elif token.exists():
        print(token.read_text())
    else:
        sys.exit(f'no oauth token found for {host}')
    sys.exit(0)
mode = (state / 'login').read_text() if (state / 'login').exists() else 'ok'
if mode == 'silent':
    time.sleep(60)
if mode == 'early':
    sys.exit('error: authentication failed before a code was issued')
print('\n! One-time code (ABCD-1234) copied to clipboard', file=sys.stderr)
print('Open this URL to continue in your web browser: https://github.com/login/device', file=sys.stderr, flush=True)
if mode == 'hang':
    time.sleep(60)
elif mode == 'fail':
    sys.exit('error: the code expired')
elif mode == 'no-token':
    sys.exit(0)
(state / host).write_text('gho_browser')
if mode == 'stall-token':
    (state / 'token-hang').write_text('')
print('Authentication complete.\nLogged in as octocat', file=sys.stderr)
