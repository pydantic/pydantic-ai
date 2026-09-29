"""Slack browser sign-in: the user's own CLAI Slack app, created from a manifest, signing in with PKCE.

Slack's MCP server has no Dynamic Client Registration and only serves internal or Marketplace apps, so each user
creates an app in their workspace from `create_app_url()`. The manifest enables PKCE, which makes the app a public
client: signing in and refreshing need its Client ID but no client secret. It also turns on MCP access
(`is_mcp_enabled`, without which Slack answers 400) and token rotation. The Client ID is not secret and lives in
plugin settings; the tokens live in the credential store under `ACCOUNT`.
"""

import json
from urllib.parse import quote

from .pkce import PKCESignIn, PublicClient

ACCOUNT = 'slack-oauth'
REDIRECT_URI = 'http://localhost:53118/slack/callback'
"""Registered in the manifest; Slack matches it exactly, so the port is fixed."""
CLIENT_ID_PATTERN = r'^\d+\.\d+$'

READ_SCOPES = (
    'search:read.public',
    'search:read.private',
    'search:read.mpim',
    'search:read.im',
    'search:read.files',
    'search:read.users',
    'files:read',
    'emoji:read',
    'reactions:read',
    'channels:history',
    'groups:history',
    'mpim:history',
    'im:history',
    'channels:read',
    'groups:read',
    'im:read',
    'mpim:read',
    'canvases:read',
    'lists:read',
    'users:read',
    'users:read.email',
)
"""What Slack's read-only MCP tools need (https://docs.slack.dev/ai/slack-mcp-server/)."""
WRITE_SCOPES = (
    'chat:write',
    'reactions:write',
    'channels:write',
    'groups:write',
    'im:write',
    'mpim:write',
    'canvases:write',
    'lists:write',
    'files:write',
)


def create_app_url() -> str:
    """Slack's create-app page, filled in with CLAI's manifest; the user picks a workspace and clicks Create."""
    manifest = {
        'display_information': {'name': 'CLAI', 'description': "Slack's MCP tools in the CLAI coding agent"},
        'oauth_config': {
            'redirect_urls': [REDIRECT_URI],
            'scopes': {'user': [*READ_SCOPES, *WRITE_SCOPES]},
            'pkce_enabled': True,
        },
        'settings': {
            'is_mcp_enabled': True,
            'token_rotation_enabled': True,
            'org_deploy_enabled': False,
            'socket_mode_enabled': False,
        },
    }
    return 'https://api.slack.com/apps?new_app=1&manifest_json=' + quote(json.dumps(manifest, separators=(',', ':')))


def session(client_id: str, *, read_only: bool) -> PKCESignIn:
    """The sign-in for one app. Read-only asks Slack for read scopes only, so the token itself cannot post."""
    client = PublicClient(
        authorize_url='https://slack.com/oauth/v2_user/authorize',
        token_url='https://slack.com/api/oauth.v2.user.access',
        client_id=client_id,
        redirect_uri=REDIRECT_URI,
        scopes=READ_SCOPES if read_only else (*READ_SCOPES, *WRITE_SCOPES),
        scope_separator=',',
    )
    return PKCESignIn(client=client, account=ACCOUNT, service='Slack', setup='/plugins configure slack')
