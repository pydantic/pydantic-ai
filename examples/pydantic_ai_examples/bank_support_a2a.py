"""Bank support agent exposed as an A2A server.

Shows how to use deps_factory to provide customer context from task metadata.

Run the server:
    python -m pydantic_ai_examples.bank_support_a2a
    # or
    uvicorn pydantic_ai_examples.bank_support_a2a:app --reload

Test with curl:
    curl -X POST http://localhost:8000/ \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": {
          "id": "test-task-1",
          "message": {
            "role": "user",
            "parts": [{"type": "text", "text": "What is my balance?"}]
          },
          "metadata": {"customer_id": 123}
        },
        "id": "1"
      }'

Then get the result:
    curl -X POST http://localhost:8000/ \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0",
        "method": "tasks/get",
        "params": {"id": "test-task-1"},
        "id": "2"
      }'
"""

from fasta2a.schema import Task

from pydantic_ai_examples.bank_support import (
    DatabaseConn,
    SupportDependencies,
    support_agent,
)


def create_deps(task: Task) -> SupportDependencies:
    """Create dependencies from A2A task metadata.

    In a real application, you might:
    - Validate the customer_id
    - Look up authentication from a session token
    - Connect to a real database with connection pooling
    """
    metadata = task.get('metadata', {})
    customer_id = metadata.get('customer_id', 0)

    # In production, you'd validate the customer exists
    # and the request is authorized
    return SupportDependencies(customer_id=customer_id, db=DatabaseConn())


# Create the A2A application
app = support_agent.to_a2a(
    deps_factory=create_deps,
    name='Bank Support Agent',
    description='AI support agent for banking customers',
)


if __name__ == '__main__':
    # For development convenience
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
