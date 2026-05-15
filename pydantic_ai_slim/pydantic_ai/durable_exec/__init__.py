"""Durable execution integrations for Pydantic AI.

Each subpackage adds durability for one durable-execution platform via a
capability you attach to an [`Agent`][pydantic_ai.Agent]:

- [`pydantic_ai.durable_exec.temporal`][pydantic_ai.durable_exec.temporal] —
  [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]
- [`pydantic_ai.durable_exec.dbos`][pydantic_ai.durable_exec.dbos] —
  [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability]
- [`pydantic_ai.durable_exec.prefect`][pydantic_ai.durable_exec.prefect] —
  [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability]
"""
