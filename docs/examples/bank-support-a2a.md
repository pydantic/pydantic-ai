Example showing how to expose the [bank support agent](bank-support.md) as an A2A server with dependency injection.

Demonstrates:

* Converting an existing agent to A2A
* Using `deps_factory` to provide customer context
* Passing metadata through A2A protocol

## Running the Example

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
# Start the A2A server
uvicorn pydantic_ai_examples.bank_support_a2a:app --reload

# In another terminal, send a request
curl -X POST http://localhost:8000/tasks.send \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tasks.send",
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
```

## Example Code

```python {title="bank_support_a2a.py"}
#! examples/pydantic_ai_examples/bank_support_a2a.py
```