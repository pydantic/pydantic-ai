# Deferred Tools

There are a few scenarios where the model should be able to call a tool that should not or cannot be executed during the same agent run inside the same Python process:

- it may need to be approved by the user first
- it may depend on an upstream service, frontend, or user to provide the result
- the result could take longer to generate than it's reasonable to keep the agent process running

To support these use cases, Pydantic AI provides the concept of deferred tools, which come in two flavors documented below:

- tools that [require approval](#human-in-the-loop-tool-approval)
- tools that are [executed externally](#external-tool-execution)

When the model calls a deferred tool, the agent run will end with a [`DeferredToolRequests`][pydantic_ai.output.DeferredToolRequests] output object containing information about the deferred tool calls. Once the approvals and/or results are ready, a new agent run can then be started with the original run's [message history](message-history.md) plus a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] object holding results for each tool call in `DeferredToolRequests`, which will continue the original run where it left off.

Note that handling deferred tool calls requires `DeferredToolRequests` to be in the `Agent`'s [`output_type`](output.md#structured-output) so that the possible types of the agent run output are correctly inferred. If your agent can also be used in a context where no deferred tools are available and you don't want to deal with that type everywhere you use the agent, you can instead pass the `output_type` argument when you run the agent using [`agent.run()`][pydantic_ai.agent.AbstractAgent.run], [`agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync], [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream], or [`agent.iter()`][pydantic_ai.Agent.iter]. Note that the run-time `output_type` overrides the one specified at construction time (for type inference reasons), so you'll need to include the original output type explicitly.

## Human-in-the-Loop Tool Approval

If a tool function always requires approval, you can pass the `requires_approval=True` argument to the [`@agent.tool`][pydantic_ai.Agent.tool] decorator, [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain] decorator, [`Tool`][pydantic_ai.tools.Tool] class, [`FunctionToolset.tool`][pydantic_ai.toolsets.FunctionToolset.tool] decorator, or [`FunctionToolset.add_function()`][pydantic_ai.toolsets.FunctionToolset.add_function] method. Inside the function, you can then assume that the tool call has been approved.

If whether a tool function requires approval depends on the tool call arguments or the agent [run context][pydantic_ai.tools.RunContext] (e.g. [dependencies](dependencies.md) or message history), you can raise the [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] exception from the tool function. The [`RunContext.tool_call_approved`][pydantic_ai.tools.RunContext.tool_call_approved] property will be `True` if the tool call has already been approved.

To require approval for calls to tools provided by a [toolset](toolsets.md) (like an [MCP server](mcp/client.md)), see the [`ApprovalRequiredToolset` documentation](toolsets.md#requiring-tool-approval).

When the model calls a tool that requires approval, the agent run will end with a [`DeferredToolRequests`][pydantic_ai.output.DeferredToolRequests] output object with an `approvals` list holding [`ToolCallPart`s][pydantic_ai.messages.ToolCallPart] containing the tool name, validated arguments, and a unique tool call ID.

Once you've gathered the user's approvals or denials, you can build a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] object with an `approvals` dictionary that maps each tool call ID to a boolean, a [`ToolApproved`][pydantic_ai.tools.ToolApproved] object (with optional `override_args`), or a [`ToolDenied`][pydantic_ai.tools.ToolDenied] object (with an optional custom `message` to provide to the model). This `DeferredToolResults` object can then be provided to one of the agent run methods as `deferred_tool_results`, alongside the original run's [message history](message-history.md).

Here's an example that shows how to require approval for all file deletions, and for updates of specific protected files:

```python {title="tool_requires_approval.py"}
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)

agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests])

PROTECTED_FILES = {'.env'}


@agent.tool
def update_file(ctx: RunContext, path: str, content: str) -> str:
    if path in PROTECTED_FILES and not ctx.tool_call_approved:
        raise ApprovalRequired
    return f'File {path!r} updated: {content!r}'


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'File {path!r} deleted'


result = agent.run_sync('Delete `__init__.py`, write `Hello, world!` to `README.md`, and clear `.env`')
messages = result.all_messages()

assert isinstance(result.output, DeferredToolRequests)
requests = result.output
print(requests)
"""
DeferredToolRequests(
    calls=[],
    approvals=[
        ToolCallPart(
            tool_name='update_file',
            args={'path': '.env', 'content': ''},
            tool_call_id='update_file_dotenv',
        ),
        ToolCallPart(
            tool_name='delete_file',
            args={'path': '__init__.py'},
            tool_call_id='delete_file',
        ),
    ],
    metadata={},
)
"""

results = DeferredToolResults()
for call in requests.approvals:
    result = False
    if call.tool_name == 'update_file':
        # Approve all updates
        result = True
    elif call.tool_name == 'delete_file':
        # deny all deletes
        result = ToolDenied('Deleting files is not allowed')

    results.approvals[call.tool_call_id] = result

result = agent.run_sync(message_history=messages, deferred_tool_results=results)
print(result.output)
"""
I successfully updated `README.md` and cleared `.env`, but was not able to delete `__init__.py`.
"""
print(result.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Delete `__init__.py`, write `Hello, world!` to `README.md`, and clear `.env`',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='delete_file',
                args={'path': '__init__.py'},
                tool_call_id='delete_file',
            ),
            ToolCallPart(
                tool_name='update_file',
                args={'path': 'README.md', 'content': 'Hello, world!'},
                tool_call_id='update_file_readme',
            ),
            ToolCallPart(
                tool_name='update_file',
                args={'path': '.env', 'content': ''},
                tool_call_id='update_file_dotenv',
            ),
        ],
        usage=RequestUsage(input_tokens=63, output_tokens=21),
        model_name='gpt-5',
        timestamp=datetime.datetime(...),
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='update_file',
                content="File 'README.md' updated: 'Hello, world!'",
                tool_call_id='update_file_readme',
                timestamp=datetime.datetime(...),
            )
        ]
    ),
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='update_file',
                content="File '.env' updated: ''",
                tool_call_id='update_file_dotenv',
                timestamp=datetime.datetime(...),
            ),
            ToolReturnPart(
                tool_name='delete_file',
                content='Deleting files is not allowed',
                tool_call_id='delete_file',
                timestamp=datetime.datetime(...),
            ),
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='I successfully updated `README.md` and cleared `.env`, but was not able to delete `__init__.py`.'
            )
        ],
        usage=RequestUsage(input_tokens=79, output_tokens=39),
        model_name='gpt-5',
        timestamp=datetime.datetime(...),
    ),
]
"""
```

_(This example is complete, it can be run "as is")_

## External Tool Execution

When the result of a tool call cannot be generated inside the same agent run in which it was called, the tool is considered to be external.
Examples of external tools are client-side tools implemented by a web or app frontend, and slow tasks that are passed off to a background worker or external service instead of keeping the agent process running.

If whether a tool call should be executed externally depends on the tool call arguments, the agent [run context][pydantic_ai.tools.RunContext] (e.g. [dependencies](dependencies.md) or message history), or how long the task is expected to take, you can define a tool function and conditionally raise the [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] exception. Before raising the exception, the tool function would typically schedule some background task and pass along the [`RunContext.tool_call_id`][pydantic_ai.tools.RunContext.tool_call_id] so that the result can be matched to the deferred tool call later.

If a tool is always executed externally and its definition is provided to your code along with a JSON schema for its arguments, you can use an [`ExternalToolset`](toolsets.md#external-toolset). If the external tools are known up front and you don't have the arguments JSON schema handy, you can also define a tool function with the appropriate signature that does nothing but raise the [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] exception.

When the model calls an external tool, the agent run will end with a [`DeferredToolRequests`][pydantic_ai.output.DeferredToolRequests] output object with a `calls` list holding [`ToolCallPart`s][pydantic_ai.messages.ToolCallPart] containing the tool name, validated arguments, and a unique tool call ID.

Once the tool call results are ready, you can build a [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] object with a `calls` dictionary that maps each tool call ID to an arbitrary value to be returned to the model, a [`ToolReturn`](tools-advanced.md#advanced-tool-returns) object, or a [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] exception in case the tool call failed and the model should [try again](tools-advanced.md#tool-retries). This `DeferredToolResults` object can then be provided to one of the agent run methods as `deferred_tool_results`, alongside the original run's [message history](message-history.md).

Here's an example that shows how to move a task that takes a while to complete to the background and return the result to the model once the task is complete:

```python {title="external_tool.py"}
import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic_ai import (
    Agent,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
)


@dataclass
class TaskResult:
    tool_call_id: str
    result: Any


async def calculate_answer_task(tool_call_id: str, question: str) -> TaskResult:
    await asyncio.sleep(1)
    return TaskResult(tool_call_id=tool_call_id, result=42)


agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests])

tasks: list[asyncio.Task[TaskResult]] = []


@agent.tool
async def calculate_answer(ctx: RunContext, question: str) -> str:
    assert ctx.tool_call_id is not None

    task = asyncio.create_task(calculate_answer_task(ctx.tool_call_id, question))  # (1)!
    tasks.append(task)

    raise CallDeferred


async def main():
    result = await agent.run('Calculate the answer to the ultimate question of life, the universe, and everything')
    messages = result.all_messages()

    assert isinstance(result.output, DeferredToolRequests)
    requests = result.output
    print(requests)
    """
    DeferredToolRequests(
        calls=[
            ToolCallPart(
                tool_name='calculate_answer',
                args={
                    'question': 'the ultimate question of life, the universe, and everything'
                },
                tool_call_id='pyd_ai_tool_call_id',
            )
        ],
        approvals=[],
        metadata={},
    )
    """

    done, _ = await asyncio.wait(tasks)  # (2)!
    task_results = [task.result() for task in done]
    task_results_by_tool_call_id = {result.tool_call_id: result.result for result in task_results}

    results = DeferredToolResults()
    for call in requests.calls:
        try:
            result = task_results_by_tool_call_id[call.tool_call_id]
        except KeyError:
            result = ModelRetry('No result for this tool call was found.')

        results.calls[call.tool_call_id] = result

    result = await agent.run(message_history=messages, deferred_tool_results=results)
    print(result.output)
    #> The answer to the ultimate question of life, the universe, and everything is 42.
    print(result.all_messages())
    """
    [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='Calculate the answer to the ultimate question of life, the universe, and everything',
                    timestamp=datetime.datetime(...),
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='calculate_answer',
                    args={
                        'question': 'the ultimate question of life, the universe, and everything'
                    },
                    tool_call_id='pyd_ai_tool_call_id',
                )
            ],
            usage=RequestUsage(input_tokens=63, output_tokens=13),
            model_name='gpt-5',
            timestamp=datetime.datetime(...),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='calculate_answer',
                    content=42,
                    tool_call_id='pyd_ai_tool_call_id',
                    timestamp=datetime.datetime(...),
                )
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='The answer to the ultimate question of life, the universe, and everything is 42.'
                )
            ],
            usage=RequestUsage(input_tokens=64, output_tokens=28),
            model_name='gpt-5',
            timestamp=datetime.datetime(...),
        ),
    ]
    """
```

1. In reality, you'd likely use Celery or a similar task queue to run the task in the background.
2. In reality, this would typically happen in a separate process that polls for the task status or is notified when all pending tasks are complete.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Attaching Metadata to Deferred Tools

Both [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] and [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] exceptions accept an optional `metadata` parameter that allows you to attach arbitrary context information to deferred tool calls. This metadata is then available in the [`DeferredToolRequests.metadata`][pydantic_ai.tools.DeferredToolRequests.metadata] dictionary, keyed by the tool call ID.

A common pattern is to use [`RunContext`][pydantic_ai.tools.RunContext] to access application dependencies (databases, APIs, calculators) and compute metadata based on the tool arguments and current context. This allows you to provide rich information for approval decisions or external task tracking.

Common use cases for metadata include:

- Computing cost estimates based on tool arguments and dependency services
- Including job IDs or tracking information for external execution systems
- Storing approval context like user permissions or resource availability
- Attaching priority levels computed from current system state

Here's an example showing how to use metadata with deps to make informed approval decisions:

```python {title="deferred_tools_with_metadata.py"}
from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolApproved,
    ToolDenied,
)


@dataclass
class ComputeDeps:
    """Dependencies providing cost estimation and job scheduling."""

    def estimate_cost(self, dataset: str, model_type: str) -> float:
        # In real code, query pricing API or database
        costs = {'gpt-4': 50.0, 'gpt-3.5': 10.0}
        return costs.get(model_type, 25.0)

    def estimate_duration(self, dataset: str) -> int:
        # In real code, estimate based on dataset size
        return 30 if dataset == 'large_dataset' else 5

    def submit_job(self, dataset: str, model_type: str) -> str:
        # In real code, submit to batch processing system
        return f'job_{dataset}_{model_type}'


agent = Agent(
    'openai:gpt-5',
    deps_type=ComputeDeps,
    output_type=[str, DeferredToolRequests],
)


@agent.tool
def train_model(ctx: RunContext[ComputeDeps], dataset: str, model_type: str) -> str:
    """Train ML model - requires approval for expensive models."""
    if not ctx.tool_call_approved:
        # Use deps to compute actual estimates based on args
        cost = ctx.deps.estimate_cost(dataset, model_type)
        duration = ctx.deps.estimate_duration(dataset)

        raise ApprovalRequired(
            metadata={
                'dataset': dataset,
                'model_type': model_type,
                'estimated_cost_usd': cost,
                'estimated_duration_minutes': duration,
            }
        )

    return f'Model {model_type} trained on {dataset}'


@agent.tool
def process_dataset(ctx: RunContext[ComputeDeps], dataset: str, operation: str) -> str:
    """Process dataset in external batch system."""
    # Submit job and defer execution
    job_id = ctx.deps.submit_job(dataset, operation)

    raise CallDeferred(
        metadata={
            'job_id': job_id,
            'dataset': dataset,
            'operation': operation,
        }
    )


deps = ComputeDeps()
result = agent.run_sync(
    'Train gpt-4 on large_dataset and process large_dataset with transform',
    deps=deps,
)
messages = result.all_messages()

assert isinstance(result.output, DeferredToolRequests)
requests = result.output

# Make approval decisions based on metadata
results = DeferredToolResults()
for call in requests.approvals:
    metadata = requests.metadata.get(call.tool_call_id, {})
    cost = metadata.get('estimated_cost_usd', 0)

    print(f'Approval needed: {call.tool_name}')
    #> Approval needed: train_model
    print(f'  Model: {metadata.get("model_type")}, Cost: ${cost}')
    #>   Model: gpt-4, Cost: $50.0

    if cost < 100:
        results.approvals[call.tool_call_id] = ToolApproved()
    else:
        results.approvals[call.tool_call_id] = ToolDenied('Cost exceeds limit')

# Process external jobs using metadata
for call in requests.calls:
    metadata = requests.metadata.get(call.tool_call_id, {})
    job_id = metadata.get('job_id')

    print(f'External job: {job_id}')
    #> External job: job_large_dataset_transform

    # In real code, poll job status and get result
    results.calls[call.tool_call_id] = f'Completed {job_id}'

result = agent.run_sync(message_history=messages, deferred_tool_results=results, deps=deps)
print(result.output)
"""
Model gpt-4 trained on large_dataset and dataset processing job job_large_dataset_transform completed
"""
```

_(This example is complete, it can be run "as is")_

## See Also

- [Function Tools](tools.md) - Basic tool concepts and registration
- [Advanced Tool Features](tools-advanced.md) - Custom schemas, dynamic tools, and execution details
- [Toolsets](toolsets.md) - Managing collections of tools, including `ExternalToolset` for external tools
- [Message History](message-history.md) - Understanding how to work with message history for deferred tools
