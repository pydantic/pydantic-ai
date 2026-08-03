# Durable Execution with Apache Airflow

[Apache Airflow](https://airflow.apache.org) is a workflow orchestrator. Its Pydantic AI integration is provided by the [`apache-airflow-providers-common-ai`](https://airflow.apache.org/docs/apache-airflow-providers-common-ai/stable/index.html) package through `airflow.providers.common.ai`, rather than by `pydantic_ai.durable_exec`.

Unlike the wrapper-object integrations on this page, Airflow's durable unit is an **Airflow task**. You author a normal Pydantic AI agent and run it as a task; Airflow's retry machinery plus a step-level cache resume the agent from its last completed model request or tool call instead of replaying the whole run.

## Durable Execution

When an agent runs as a durable Airflow task, Airflow records each completed **model request** and **tool call** as a cache entry. On a retry, Airflow replays these entries to skip completed work: each entry stores a fingerprint of the request that produced it, and if that fingerprint no longer matches (the conversation diverged since the previous attempt), the step re-runs live instead of returning a stale result. The cache lives in object storage (local, S3, GCS, or Azure) for the lifetime of a single DAG run's task and is deleted when the task succeeds.

For example, imagine an agent calls a model, gets a useful response, starts a tool call, and then the worker crashes. Without durable execution, Airflow's normal retry restarts the task from the top and repeats the model request and any later side effects. With durable execution, the retry replays the run, reuses the cached result for the already-completed model request, and continues from the first operation that has not completed.

This is useful for long-running agents and for runs where a repeated model request or external tool call would cost money, take time, or duplicate a side effect.

## Durable Agent

You make an agent durable by running it through Airflow's `AgentOperator` (or the `@task.agent` decorator) with `durable=True`. Install the provider alongside Airflow:

```bash
uv add "apache-airflow-providers-common-ai"
```

The agent's model and credentials come from an Airflow connection (the examples use `pydanticai_default`). See [Pydantic AI connection](https://airflow.apache.org/docs/apache-airflow-providers-common-ai/stable/connections/pydantic_ai.html) for how to configure one.

Durable execution needs a place to store its step cache. Point `[common.ai] durable_cache_path` at an object-storage location:

```ini title="airflow.cfg"
[common.ai]
durable_cache_path = s3://my-bucket/airflow-agent-cache
```

Here is the smallest durable Pydantic AI agent as an Airflow task:

```python {title="durable_agent_dag.py" test="skip" lint="skip"}
from datetime import timedelta

from airflow.providers.common.ai.operators.agent import AgentOperator
from airflow.sdk import dag


@dag(default_args={"retries": 3, "retry_delay": timedelta(seconds=30)})
def durable_agent_dag():
    AgentOperator(
        task_id="researcher",
        prompt="Summarize quantum error correction.",
        llm_conn_id="pydanticai_default",
        durable=True,
    )


durable_agent_dag()
```

`AgentOperator` does not replace the underlying Pydantic AI agent. It builds the agent from your connection and toolsets, then wraps the model and toolsets so that, while the task runs, it records recoverable operations:

* model requests;
* Pydantic AI tool calls.

The same applies to the `@task.agent` decorator, where the decorated function returns the prompt:

```python {title="durable_agent_decorator.py" test="skip" lint="skip"}
from datetime import timedelta

from airflow.providers.common.ai.toolsets.sql import SQLToolset
from airflow.sdk import dag, task


@dag(default_args={"retries": 3, "retry_delay": timedelta(seconds=30)})
def durable_agent_decorator():
    @task.agent(
        llm_conn_id="pydanticai_default",
        system_prompt="You are a data analyst. Use tools to answer questions.",
        durable=True,
        toolsets=[SQLToolset(db_conn_id="postgres_default", allowed_tables=["orders"])],
    )
    def analyze(question: str) -> str:
        return f"Answer this question about our orders data: {question}"

    analyze("What was our total revenue last month?")


durable_agent_decorator()
```

The agent's retries are Airflow task retries: configure them with the task's `retries` and `retry_delay`. On each retry the cached steps are replayed and the run continues from the first operation that has not completed. For the full reference, see the Airflow [`AgentOperator` durable execution docs](https://airflow.apache.org/docs/apache-airflow-providers-common-ai/stable/operators/agent.html#durable-execution).

## Tools and side effects

Durable execution caches the result of each Pydantic AI tool call, including tools backed by Airflow toolsets (`SQLToolset`, `HookToolset`, `MCPToolset`, and others). On replay the cached result is returned without re-invoking the tool, so a tool that writes to an external system runs at most once per completed step across all retries of a run.

## Human-in-the-loop

Airflow's `AgentOperator` has a separate human-in-the-loop review mode (`enable_hitl_review=True`) that pauses an agent run for human approval, rejection, or change requests through Airflow's HITL UI.

!!! warning

    `durable=True` and `enable_hitl_review=True` cannot be combined today. A durable run replays deterministically from its cache and does not pause for human input; a human-in-the-loop run is interactive and is not yet captured by the step cache. Choose one per task.

## Streaming

Streaming is not yet supported under `durable=True`. The durable model wrapper records complete model requests, not streamed events. Run streaming agents as non-durable tasks.

## Requirements and Constraints

When running a Pydantic AI agent as a durable Airflow task:

* The durable unit is the Airflow task; recovery happens through Airflow task retries, so set `retries` (and a `retry_delay`) on the task.
* Define the agent with a concrete model, for example via a connection that resolves to `Agent('openai:gpt-5-nano', ...)`. The model must be set when `durable=True`.
* Set `[common.ai] durable_cache_path` to an object-storage location the workers can read and write.
* On a retry, a cached model request or tool call is replayed when its stored fingerprint matches the current request, and re-runs live when it diverges. Requests that can't be serialized to a fingerprint fall back to unverified positional replay, so keep runs deterministic across retries.
* `durable=True` and `enable_hitl_review=True` are mutually exclusive.
* Streaming is not supported under `durable=True`.
* The step cache is scoped to one DAG run's task and is deleted when the task succeeds.
