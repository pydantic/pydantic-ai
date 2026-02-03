# Durable Execution Reference

Source: `docs/durable.md`

## Overview

Durable execution frameworks ensure agent runs survive failures, restarts, and deployments.
PydanticAI integrates with three durable execution platforms.

## Temporal

Install: `pydantic-ai-slim[temporal]`

Temporal provides workflow orchestration with automatic retries and state persistence.

```python
from pydantic_ai.agent.temporal import TemporalAgent

# Wrap an agent for Temporal execution
temporal_agent = TemporalAgent(my_agent)
```

### When to use Temporal

- Long-running agent workflows (minutes to hours)
- Workflows that must survive process restarts
- Complex multi-step orchestration with compensation logic

## DBOS

Install: `pydantic-ai-slim[dbos]`

DBOS provides durable execution backed by PostgreSQL.

```python
from pydantic_ai.agent.dbos import DBOSAgent

dbos_agent = DBOSAgent(my_agent)
```

### When to use DBOS

- Database-centric applications
- When PostgreSQL is already your primary datastore
- Simpler deployment requirements than Temporal

## Prefect

Install: `pydantic-ai-slim[prefect]`

Prefect provides workflow orchestration with a focus on data pipelines.

```python
from pydantic_ai.agent.prefect import PrefectAgent

prefect_agent = PrefectAgent(my_agent)
```

### When to use Prefect

- Data pipeline orchestration
- When you already use Prefect for workflow management
- Batch processing of agent tasks

## Choosing a Platform

| Feature | Temporal | DBOS | Prefect |
|---------|----------|------|---------|
| State backend | Custom | PostgreSQL | Custom |
| Complexity | High | Medium | Medium |
| Best for | Complex workflows | DB-centric apps | Data pipelines |

## When NOT to Use Durable Execution

- Simple request-response agents (just use `agent.run()`)
- Short-lived operations that can be retried at the application level
- When the overhead of a durable execution framework is not justified

## Observability for Durable Workflows

For long-running durable workflows, Logfire instrumentation is particularly valuable:

- Trace the full lifecycle of workflows that span hours or days
- Correlate agent runs across workflow restarts
- Monitor workflow health and identify stuck or failing workflows

Combined with the durable execution platform's own observability, this gives complete visibility into your agent systems.
