import os
from typing import Any

import httpx
import logfire

### [send_slack_message]
API_KEY = os.getenv('SLACK_API_KEY')
assert API_KEY, 'SLACK_API_KEY is not set'


@logfire.instrument('Send Slack message')
async def send_slack_message(channel: str, blocks: list[dict[str, Any]]):
    client = httpx.AsyncClient()
    response = await client.post(
        'https://slack.com/api/chat.postMessage',
        json={
            'channel': channel,
            'blocks': blocks,
        },
        headers={
            'Authorization': f'Bearer {API_KEY}',
        },
        timeout=5,
    )
    response.raise_for_status()
    result = response.json()
    if not result.get('ok', False):
        error = result.get('error', 'Unknown error')
        raise Exception(f'Failed to send to Slack: {error}')  ### [/send_slack_message]
