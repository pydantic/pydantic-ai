---
title: Answer analytics questions without exposing arbitrary SQL
description: Give an agent narrow, typed reporting tools over a read-only database instead of letting it execute generated SQL.
---

# Answer analytics questions without exposing arbitrary SQL

Do not give a model a general `run_sql` tool for a production database. Expose the business queries it needs and keep SQL, authorization, and row limits in application code.

```bash
pip/uv-add "pydantic-ai-slim[openai]"
export OPENAI_API_KEY=your-api-key
```

```python {dunder_name="not_main"}
import asyncio
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field

from pydantic_ai import Agent, RunContext


@dataclass
class Analytics:
    database: str


agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=Analytics,
    instructions='Answer sales questions using the reporting tools. Never invent rows.',
)


@agent.tool
def top_products(
    ctx: RunContext[Analytics],
    limit: Annotated[int, Field(ge=1, le=10)] = 5,
) -> list[dict[str, object]]:
    """Return products ranked by revenue."""
    with closing(sqlite3.connect(f'file:{ctx.deps.database}?mode=ro', uri=True)) as db:
        rows = db.execute(
            'SELECT product, SUM(revenue) AS revenue FROM sales GROUP BY product ORDER BY revenue DESC LIMIT ?',
            (limit,),
        ).fetchall()
    return [{'product': product, 'revenue': revenue} for product, revenue in rows]


def create_demo_database(database: str) -> None:
    with closing(sqlite3.connect(database)) as db:
        db.execute('CREATE TABLE sales (product TEXT NOT NULL, revenue INTEGER NOT NULL)')
        db.executemany('INSERT INTO sales VALUES (?, ?)', [('Extract', 1200), ('Logfire', 2100)])
        db.commit()


async def main() -> None:
    database = 'sales.db'
    Path(database).unlink(missing_ok=True)
    create_demo_database(database)
    try:
        result = await agent.run('Which product has the most revenue?', deps=Analytics(database))
    finally:
        Path(database).unlink()
    print(result.output)
    #> Logfire has the most revenue at $2,100.


if __name__ == '__main__':
    asyncio.run(main())
```

In production, open a database account with read-only permissions as a second boundary. Add tenant and date filters inside the tool from authenticated dependencies; do not accept them solely from model-generated arguments.

## Related

See [Function tools](../tools.md) for tool schemas and validation. The [SQL generation example](../examples/sql-gen.md) shows the alternative pattern for a sandboxed database.
