# Temporal Streaming Example

This example demonstrates how to implement streaming with Pydantic AI agents in Temporal workflows. It showcases the streaming pattern described in the [Temporal documentation](../../../docs/durable_execution/temporal.md#streaming).

## Overview

The example implements a Yahoo Finance search agent that:
- Uses MCP (Model Context Protocol) toolsets for accessing financial data
- Executes Python code in a sandbox for data analysis
- Streams events during execution via Temporal signals and queries
- Provides durable execution with automatic retries

## Architecture

The streaming architecture works as follows:

1. **Agent Configuration** (`agents.py`): Defines the agent with MCP toolsets and custom Python execution tools
2. **Workflow** (`workflow.py`): Temporal workflow that orchestrates agent execution and manages event streams
3. **Streaming Handler** (`streaming_handler.py`): Processes agent events and sends them to the workflow via signals
4. **Main** (`main.py`): Sets up the Temporal client/worker and polls for events via queries

## Key Components

### Event Flow

```
Agent Execution (in Activity)
    ↓
Streaming Handler
    ↓ (via Signal)
Workflow Event Queue
    ↓ (via Query)
Main Process (polling)
    ↓
Display to User
```

### Dependencies

The [`AgentDependencies`][pydantic_ai_examples.temporal_streaming.datamodels.AgentDependencies] model passes workflow identification from the workflow to activities, enabling the streaming handler to send signals back to the correct workflow instance.

## Prerequisites

1. **Temporal Server**: Install and run Temporal locally

```bash
brew install temporal
temporal server start-dev
```

2. **Python Dependencies**: Install required packages

```bash
pip install pydantic-ai temporalio mcp-run-python pyyaml
```

3. **Configuration File**: Create an `app_conf.yml` file in your project root

```yaml
llm:
  anthropic_api_key: ANTHROPIC_API_KEY  # Will be read from environment variable
```

4. **Environment Variables**: Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

## Running the Example

1. Make sure Temporal server is running:

```bash
temporal server start-dev
```

2. Set the configuration file path (optional, defaults to `./app_conf.yml`):

```bash
export APP_CONFIG_PATH=./app_conf.yml
```

3. Run the example:

```bash
python -m pydantic_ai_examples.temporal_streaming.main
```

## What to Expect

The example will:
1. Connect to Temporal server
2. Start a worker to handle workflows and activities
3. Execute the workflow with a sample financial query
4. Stream events as the agent:
   - Calls tools (Yahoo Finance API, Python sandbox)
   - Receives responses
   - Generates the final result
5. Display all events in real-time
6. Show the final result

## Project Structure

```
temporal_streaming/
├── agents.py              # Agent configuration with MCP toolsets
├── datamodels.py          # Pydantic models for dependencies and events
├── main.py                # Main entry point
├── streaming_handler.py   # Event stream handler
├── utils.py               # Configuration utilities
├── workflow.py            # Temporal workflow definition
└── README.md              # This file
```

## Customization

### Changing the Query

Edit the query in `main.py`:

```python
workflow_handle = await client.start_workflow(
    YahooFinanceSearchWorkflow.run,
    args=['Your custom financial query here'],
    id=workflow_id,
    task_queue=task_queue,
)
```

### Adding More Tools

Add tools to the agent in `agents.py`:

```python
@agent.tool(name='your_tool_name')
async def your_tool(ctx: RunContext[AgentDependencies], param: str) -> str:
    # Your tool implementation
    return result
```

### Modifying Event Handling

Customize what events are captured and displayed in `streaming_handler.py`.

## Key Concepts

### Why Streaming is Different in Temporal

Traditional streaming methods like [`Agent.run_stream()`][pydantic_ai.Agent.run_stream] don't work in Temporal because:
- Activities cannot stream directly to the workflow
- The workflow and activity run in separate processes

### The Solution

This example uses:
- **Event Stream Handler**: Captures events during agent execution
- **Signals**: Push events from activities to the workflow
- **Queries**: Pull events from the workflow to the caller
- **Dependencies**: Pass workflow identification to enable signal routing

## Limitations

- Events are batched per model request/tool call rather than streamed token-by-token
- Query polling introduces a small delay in event delivery
- The workflow waits up to 10 seconds for events to be consumed before completing

## Learn More

- [Temporal Documentation](https://docs.temporal.io/)
- [Pydantic AI Temporal Integration](../../../docs/durable_execution/temporal.md)
- [Streaming with Pydantic AI](../../../docs/agents.md#streaming-all-events)
