# Testing

This page covers how to test an A2A server implementation, focusing specifically on testing methodologies.

To test an A2A server, you can use the [`A2AClient`][fasta2a.client.A2AClient] with an ASGI test client:

```python
import anyio
import httpx
from asgi_lifespan import LifespanManager
from fasta2a.client import A2AClient
from fasta2a.schema import Message, TextPart


async def test_a2a_server(app):
    """Test an A2A server using the A2AClient with an ASGI transport."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            # Create a test message
            message = Message(role='user', parts=[TextPart(text='Hello, world!', type='text')])

            # Send the task and check response format
            response = await a2a_client.send_task(message=message)
            assert 'result' in response
            assert 'id' in response['result']

            # Get the task ID and verify task completion
            task_id = response['result']['id']
            await anyio.sleep(0.1)  # Wait for processing
            task = await a2a_client.get_task(task_id)
            assert task['result']['status']['state'] in ['submitted', 'completed', 'failed']
```

## Testing Different Message Types

### Text Messages

```python
from fasta2a.schema import Message, TextPart

def create_text_message():
    return Message(role='user', parts=[TextPart(text='Hello, world!', type='text')])
```

### File Messages

```python
from fasta2a.schema import Message, FilePart

def create_file_url_message():
    return Message(
        role='user',
        parts=[
            FilePart(
                type='file',
                file={'url': 'https://example.com/file.txt', 'mime_type': 'text/plain'},
            )
        ],
    )

def create_file_content_message():
    return Message(
        role='user',
        parts=[
            FilePart(type='file', file={'data': 'file content', 'mime_type': 'text/plain'}),
        ],
    )
```

### Data Messages

```python
from fasta2a.schema import Message, DataPart

def create_data_message():
    return Message(
        role='user',
        parts=[DataPart(type='data', data={'key': 'value'})],
    )
```

## Testing Task Management

```python
import anyio
import pytest
from fasta2a.client import A2AClient
from fasta2a.schema import Message


@pytest.fixture()
async def a2a_client():
    """Create an A2A client for testing."""
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

async def test_task_lifecycle(a2a_client: A2AClient, message: Message):
    """Test the full lifecycle of a task."""
    # Send task
    response = await a2a_client.send_task(message=message)
    task_id = response['result']['id']

    # Check initial status (should be 'submitted')
    task = await a2a_client.get_task(task_id)
    initial_state = task['result']['status']['state']

    # Wait for completion
    await anyio.sleep(0.5)

    # Check final status (should be 'completed' or 'failed')
    task = await a2a_client.get_task(task_id)
    final_state = task['result']['status']['state']
    assert final_state == 'completed'

    # Check history
    history = task['result']['history']
    assert len(history) > 0
```

## Using a Custom Storage for Testing

For testing with controlled data, you can use the `InMemoryStorage`:

```python
from fasta2a.storage import InMemoryStorage


def create_test_app(agent):
    """Create an A2A app with in-memory storage for testing."""
    storage = InMemoryStorage()
    return agent.to_a2a(storage=storage), storage
```

You can then use the storage object to inspect or manipulate the task data directly during tests.
