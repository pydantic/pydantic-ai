# OpenAI Batch API

The OpenAI Batch API allows you to process multiple requests efficiently with 50% cost savings compared to synchronous API calls. Batch jobs are processed within a 24-hour window and require at least 2 requests.

## Installation

The batch API is included with the OpenAI integration:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

### Jupyter Notebook Setup

If you're working in Jupyter notebooks, you'll need `nest_asyncio` to run async code in cells:

```bash
pip/uv-add  nest-asyncio
```

Then add this at the start of your notebook:

```python {test="skip"}
import nest_asyncio
nest_asyncio.apply()
```
(**Recommendation:** Use dated model versions (e.g., `gpt-4.1-2025-04-14`) instead of aliases (`gpt-4.1`) for faster batch processing)

## Basic Usage

### Creating and Submitting a Batch Job

```python
import asyncio
from pydantic_ai.batches.openai import OpenAIBatchModel, create_chat_request

async def basic_batch_example():
    # Initialize batch model
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    # Create batch requests (minimum 2 required)
    requests = [
        create_chat_request(
            custom_id='math-question',
            prompt='What is 2+2?',
            model='gpt-4o-mini',
            max_tokens=50
        ),
        create_chat_request(
            custom_id='creative-writing',
            prompt='Write a short poem about coding',
            model='gpt-4o-mini',
            max_tokens=100,
            temperature=0.8
        ),
        create_chat_request(
            custom_id='explanation',
            prompt='Explain quantum computing in one sentence',
            model='gpt-4o-mini',
            max_tokens=75
        ),
    ]

    # Submit batch job
    batch_id = await batch_model.batch_create_job(
        requests=requests,
        # provide metadata (optional, if supported)
        metadata={'project': 'my-batch-job', 'version': '1.0'}
    )
    print(f'Batch job created: {batch_id}')
    #> Batch job created: batch_test_123

    return batch_id

asyncio.run(basic_batch_example())
```
_(This example is complete, it can be run "as is" )_

### Checking Job Status

```python
from pydantic_ai.batches.openai import OpenAIBatchModel

async def check_batch_status(batch_id: str):
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    # Get job information
    job_info = await batch_model.batch_get_status(batch_id)

    print(f'Status: {job_info.status}')
    print(f'Created at: {job_info.created_at}')
    print(f'Completion window: {job_info.completion_window}')

    if job_info.request_counts:
        print(f'Progress: {job_info.request_counts}')

    return job_info.status
```
_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

### Getting Results

```python
from pydantic_ai.batches.openai import OpenAIBatchModel

async def get_batch_results(batch_id: str):
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    # Check if completed
    job_info = await batch_model.batch_get_status(batch_id)
    if job_info.status != 'completed':
        print(f'Job not ready. Status: {job_info.status}')
        return

    # Get results
    results = await batch_model.batch_retrieve_job(batch_id)

    for result in results:
        print(f'\n{result.custom_id}:')
        if result.error:
            print(f'  Error: {result.error}')
        elif result.response:
            # Extract content from nested response structure
            content = result.output
            print(f'  Response: {content}')
```
_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(get_batch_results(batch_id))` to run it with your `batch_id`)_

## Using Tools

You can include tools in batch requests by extracting `ToolDefinition` objects from pydantic-ai `Tool` instances:

```python
import asyncio
from pydantic_ai import RunContext
from pydantic_ai.batches.openai import OpenAIBatchModel, create_chat_request
from pydantic_ai.tools import Tool

# Define tool functions
def get_weather(ctx: RunContext[None], location: str, units: str = "celsius") -> str:
    """Get current weather information for a location."""
    # In real implementation, this would call a weather API
    return f"Weather in {location}: 22°{units[0].upper()}, sunny, 60% humidity"

def calculate(ctx: RunContext[None], expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        # In real implementation, use safe evaluation
        result = eval(expression)  # Don't use eval in production!
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

async def batch_with_tools():
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    # Create tools and extract definitions
    weather_tool = Tool(get_weather)
    calc_tool = Tool(calculate)
    tools = [weather_tool.tool_def, calc_tool.tool_def]

    # Create requests with tools
    requests = [
        create_chat_request(
            custom_id='weather-tokyo',
            prompt="What's the weather like in Tokyo?",
            model='gpt-4o-mini',
            tools=tools,
            max_tokens=150
        ),
        create_chat_request(
            custom_id='calculation',
            prompt="Calculate 15 * 23 + 7",
            model='gpt-4o-mini',
            tools=tools,
            max_tokens=100
        ),
        create_chat_request(
            custom_id='weather-london',
            prompt="Get weather for London, UK in fahrenheit",
            model='gpt-4o-mini',
            tools=tools,
            max_tokens=150
        ),
    ]

    # Submit batch job
    batch_id = await batch_model.batch_create_job(requests)
    print(f'Batch with tools submitted: {batch_id}')
    #> Batch with tools submitted: batch_test_123

    return batch_id

asyncio.run(batch_with_tools())
```
_(This example is complete, it can be run "as is" )_

## Structured Output

The batch API supports all structured output modes available in pydantic-ai:

### Native Mode (Recommended)

```python
from pydantic import BaseModel
from pydantic_ai.batches.openai import OpenAIBatchModel, create_chat_request

class WeatherResult(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: int

async def batch_with_structured_output():
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    requests = [
        create_chat_request(
            custom_id='structured-paris',
            prompt='Get weather information for Paris and format it properly',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='native',  # Uses OpenAI's structured output
            max_tokens=200
        ),
        create_chat_request(
            custom_id='structured-tokyo',
            prompt='Check weather in Tokyo and return structured data',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='native',
            max_tokens=200
        ),
    ]

    batch_id = await batch_model.batch_create_job(requests)
    return batch_id
```

### Tool Mode

```python
from pydantic import BaseModel
from pydantic_ai.batches.openai import OpenAIBatchModel, create_chat_request

class WeatherResult(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: int

async def batch_with_tool_mode():
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    requests = [
        create_chat_request(
            custom_id='tool-mode-1',
            prompt='Analyze the weather in Berlin',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='tool',  # Forces output through a tool call
            max_tokens=200
        ),
        create_chat_request(
            custom_id='tool-mode-2',
            prompt='Get weather data for Madrid',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='tool',
            max_tokens=200
        ),
    ]

    batch_id = await batch_model.batch_create_job(requests)
    return batch_id
```

### Prompted Mode

```python
from pydantic import BaseModel
from pydantic_ai.batches.openai import OpenAIBatchModel, create_chat_request

class WeatherResult(BaseModel):
    location: str
    temperature: float
    condition: str
    humidity: int

async def batch_with_prompted_mode():
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    requests = [
        create_chat_request(
            custom_id='prompted-1',
            prompt='Get weather for Miami and return as JSON',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='prompted',  # Adds schema to system prompt
            max_tokens=250
        ),
        create_chat_request(
            custom_id='prompted-2',
            prompt='Analyze Rome weather data',
            model='gpt-4o-mini',
            output_type=WeatherResult,
            output_mode='prompted',
            max_tokens=250
        ),
    ]

    batch_id = await batch_model.batch_create_job(requests)
    return batch_id
```

### Getting result for tools (tool requests by LLM)

```python
from pydantic_ai.batches.openai import OpenAIBatchModel

async def get_batch_results_with_tools(batch_id: str):
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    # Check if completed
    job_info = await batch_model.batch_get_status(batch_id)
    if job_info.status != 'completed':
        print(f'Job not ready. Status: {job_info.status}')
        return

    # Get results
    results = await batch_model.batch_retrieve_job(batch_id)

    for result in results:
        print(f'\n{result.custom_id}:')
        if result.error:
            print(f'  Error: {result.error}')
        else:
            # Use convenience properties instead of deep nested access
            if result.output:
                print(f'  Response: {result.output}')
            elif result.tool_calls:
                print(f'  Made {len(result.tool_calls)} tool calls:')
                for i, tool_call in enumerate(result.tool_calls):
                    print(f'{tool_call["function"]["name"]}')
```
_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(get_batch_results_with_tools(batch_id))` to run it with your `batch_id`)_

## Batch Management Methods

The `OpenAIBatchModel` class provides several methods for managing batch jobs:

### Core Methods

- **`batch_create_job(requests, endpoint='...', completion_window='24h', metadata=None)`**
    - Submit a new batch job (requires ≥2 requests)
    - Returns the batch ID for tracking

- **`batch_get_status(batch_id)`**
  - Get current job status and details
  - Returns `BatchJob` object with status, timestamps, and counts

- **`batch_retrieve_job(batch_id)`**
  - Download results from completed jobs
  - Returns list of `BatchResult` objects

- **`batch_cancel_job(batch_id)`**
  - Cancel a pending or in-progress job
  - Returns updated `BatchJob` information

- **`batch_list_jobs(limit=20)`**
  - List recent batch jobs
  - Returns list of `BatchJob` objects

### Job Status Values

- `validating`: Job is being validated
- `in_progress`: Job is being processed
- `finalizing`: Job is being finalized
- `completed`: Job completed successfully
- `failed`: Job failed due to errors
- `expired`: Job expired before completion
- `cancelled`: Job was cancelled

## Best Practices

### Request Design

- **Minimum Requirements**: Always include at least 2 requests
- **Custom IDs**: Use descriptive custom IDs for easy result identification

### Error Handling

```python
import asyncio
import time
from pydantic_ai.batches.openai import OpenAIBatchModel, create_chat_request

async def robust_batch_processing():
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')

    # Example requests
    requests = [
        create_chat_request(
            custom_id='example-1',
            prompt='Hello world',
            model='gpt-4o-mini',
            max_tokens=50
        ),
        create_chat_request(
            custom_id='example-2',
            prompt='Write a haiku',
            model='gpt-4o-mini',
            max_tokens=100
        ),
    ]

    try:
        # Submit batch
        batch_id = await batch_model.batch_create_job(requests)

        # Monitor with timeout
        max_wait_time = 25 * 60 * 60  # 25 hours
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            job_info = await batch_model.batch_retrieve_job(batch_id)

            if job_info.status == 'completed':
                results = await batch_model.batch_get_results(batch_id)
                return results
            elif job_info.status == 'failed':
                print(f'Batch failed: {job_info.errors}')
                break

            await asyncio.sleep(300)  # Check every 5 minutes

    except Exception as e:
        print(f'Batch processing error: {e}')
        # Handle cleanup if needed
```
## Troubleshooting

### Common Issues

1. **Minimum Request Error**: Ensure at least 2 requests in each batch
2. **Tool Definition Errors**: Extract `tool_def` from `Tool` instances correctly
3. **Response Structure**: Use correct nested path for response content
4. **Timeout Handling**: Jobs can take up to 24 hours to complete

### Response Structure

Batch results have a nested structure:

```python
from pydantic_ai.batches.openai import OpenAIBatchModel

async def extract_content_example(batch_id: str):
    batch_model = OpenAIBatchModel('openai:gpt-4o-mini')
    results = await batch_model.batch_retrieve_job(batch_id)

    for result in results:
        # Correct way to extract content
        content = result.output  # Use the convenience property

        # Alternative way to access it
        # content = result.response['body']['choices'][0]['message']['content']  # Manual access

        print(f"Content: {content}")
```
