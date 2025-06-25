from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

import logfire
from devtools import debug
from pydantic import BaseModel, Field

from pydantic_ai import Agent, ToolOutput

type QueryColumns = Literal[
    'id', 'first_name', 'last_name', 'dob', 'email', 'address', 'phone'
]
table_name = 'users'
type Param = list[str | int | float | bool]

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()


@dataclass
class ParamsManager:
    params: list[Any] = field(default_factory=list)

    def add_param(self, param: Any) -> str:
        self.params.append(param)
        return f'${len(self.params)}'


class Query(BaseModel, use_attribute_docstrings=True):
    """Structured definition of a database query."""

    select: Literal['*'] | list[SelectColumn] | list[Aggregation] = '*'
    """List of columns to select from the table."""
    filter: FilterClause | FilterAnd | FilterOr | None = None
    """Filter condition for the query."""
    order_by: list[OrderBy] | None = None
    """List of columns to order the results by."""
    limit: int | None = None
    """Maximum number of rows to return."""
    offset: int | None = None
    """Number of rows to skip before returning results."""

    def as_sql(self) -> tuple[str, list[Any]]:
        params_mgr = ParamsManager()
        parts: list[str] = ['SELECT']
        if isinstance(self.select, list):
            parts.append('  ' + ', '.join(c.as_sql() for c in self.select))
        else:
            parts.append(f'  {self.select}')

        parts.append(f'FROM {table_name}')

        if self.filter:
            parts.append(f'WHERE {self.filter.as_sql(params_mgr)}')
        if self.order_by:
            parts.append(
                'ORDER BY ' + ', '.join(order_by.as_sql() for order_by in self.order_by)
            )
        if self.limit:
            parts.append(f'LIMIT {self.limit}')
        if self.offset:
            parts.append(f'OFFSET {self.offset}')
        return '\n'.join(parts), params_mgr.params


class SelectColumn(BaseModel):
    name: QueryColumns
    alias: str | None = Field(pattern=r'^[a-zA-Z_][a-zA-Z0-9_ ]*$')

    def as_sql(self) -> str:
        if self.alias:
            return f'{self.name} AS {f'"{self.alias}"' if " " in self.alias else self.alias}'
        else:
            return self.name


class Aggregation(BaseModel):
    column: QueryColumns | Literal['*']
    function: Literal['SUM', 'AVG', 'MAX', 'MIN', 'COUNT']
    alias: str | None = Field(pattern=r'^[a-zA-Z_][a-zA-Z0-9_ ]*$')

    def as_sql(self) -> str:
        func_call = f'{self.function}({self.column})'
        if self.alias:
            return f'{func_call} AS {f'"{self.alias}"' if " " in self.alias else self.alias}'
        else:
            return func_call


class FilterAnd(BaseModel):
    left: FilterClause | FilterAnd | FilterOr
    right: FilterClause | FilterAnd | FilterOr

    def as_sql(self, params_mgr: ParamsManager) -> str:
        left_sql = self.left.as_sql(params_mgr)
        right_sql = self.right.as_sql(params_mgr)
        return f'{left_sql} AND {right_sql}'


class FilterOr(BaseModel):
    left: FilterClause | FilterAnd | FilterOr
    right: FilterClause | FilterAnd | FilterOr

    def as_sql(self, params_mgr: ParamsManager) -> str:
        left_sql = self.left.as_sql(params_mgr)
        right_sql = self.right.as_sql(params_mgr)
        return f'{left_sql} OR {right_sql}'


class FilterClause(BaseModel):
    column: QueryColumns
    operator: Literal['=', '!=', '<', '>', '<=', '>=']
    value: str | int | float | bool

    def as_sql(self, params_mgr: ParamsManager) -> str:
        value_param = params_mgr.add_param(self.value)
        return f'{self.column} {self.operator} {value_param}'


class OrderBy(BaseModel):
    column: QueryColumns
    direction: Literal['ASC', 'DESC']

    def as_sql(self) -> str:
        return f'{self.column} {self.direction}'


query_agent = Agent(
    'openai:gpt-4o',
    output_type=ToolOutput(type_=Query, name='query'),
    instructions="Generate a query to match the users' preferences.",
)


async def main(user_query: str):
    r = await query_agent.run()
    debug(r.output)
    query, params = r.output.as_sql()
    print(f'SQL:\n-------\n{query}\n-------\n{params=}')


if __name__ == '__main__':
    asyncio.run(
        main(
            # 'Find user IDs for users who were born before 1990 ordered by how old they are.'
            # 'How many users are there.'
            "what is jane's date of birth."
        )
    )
