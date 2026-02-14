import argparse
import asyncio
import json
import sys
import threading
import time

from pydantic_ai import Agent

try:
    import pjrpc  # noqa: F401
    import starlette  # noqa: F401
    import uvicorn
    import websockets
except ImportError as e:
    print(f'Missing dependencies: {e}')
    print('Please install pjrpc, starlette, uvicorn, and websockets to run this demo.')
    sys.exit(1)

# --- Agent Setup ---

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a helpful coding assistant.',
)

# --- Demo Logic ---


def run_server():
    print('Starting ACP server on http://localhost:8000/acp', file=sys.stderr)
    app = agent.to_acp(debug=True)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')


async def run_stdio():
    # Run the agent in stdio mode
    from pydantic_ai._acp import run_stdio

    await run_stdio(agent, name='stdio-agent')


async def run_client():
    uri = 'ws://localhost:8000/acp'
    print(f'Connecting to {uri}...', file=sys.stderr)

    async with websockets.connect(uri) as websocket:
        # 1. Initialize
        print('Sending initialize...', file=sys.stderr)
        await websocket.send(
            json.dumps(
                {
                    'jsonrpc': '2.0',
                    'method': 'initialize',
                    'id': 1,
                    'params': {'client_name': 'demo-client', 'client_version': '1.0'},
                }
            )
        )
        resp = await websocket.recv()
        print(f'Initialize Response: {resp}', file=sys.stderr)

        # 2. Create Session
        print('Creating session...', file=sys.stderr)
        await websocket.send(
            json.dumps({'jsonrpc': '2.0', 'method': 'session/new', 'id': 2})
        )
        resp = json.loads(await websocket.recv())
        session_id = resp['result']['sessionId']
        print(f'Session ID: {session_id}', file=sys.stderr)

        # 3. Prompt (V3 Structured)
        print('Sending V3 prompt...', file=sys.stderr)
        await websocket.send(
            json.dumps(
                {
                    'jsonrpc': '2.0',
                    'method': 'session/prompt',
                    'id': 3,
                    'params': {
                        'sessionId': session_id,
                        'prompt': [
                            {'type': 'text', 'text': 'Analyzing file: '},
                            {
                                'type': 'resourceLink',
                                'uri': 'file:///path/to/script.py',
                            },
                            {'type': 'text', 'text': '\nWhat does this file do?'},
                        ],
                    },
                }
            )
        )

        # 4. Listen for updates (V3 Rich Notifications)
        while True:
            msg = json.loads(await websocket.recv())
            if 'method' in msg and msg['method'] == 'session/update':
                # Print chunk
                update = msg['params']['update']
                if update['type'] == 'agent_message_chunk':
                    print(f'{update["delta"]}', end='', flush=True)
            elif 'id' in msg and msg['id'] == 3:
                # Final response
                print(f'\n\nFinal Result: {msg["result"]}', file=sys.stderr)
                break
            else:
                print(f'\nMessage: {msg}', file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stdio', action='store_true', help='Run in stdio mode (server only)'
    )
    parser.add_argument(
        '--client',
        action='store_true',
        help='Run client demo (connects to localhost:8000)',
    )
    args = parser.parse_args()

    if args.stdio:
        # Run standard input/output server
        asyncio.run(run_stdio())
    elif args.client:
        # Run WebSocket client demo
        asyncio.run(run_client())
    else:
        # Default: Run WebSocket Server + Client
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        try:
            asyncio.run(run_client())
        except KeyboardInterrupt:
            pass
