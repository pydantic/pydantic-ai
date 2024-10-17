from dataclasses import dataclass

from pydantic_ai import Agent
from devtools import debug

system_prompt = """\
Given the following PostgreSQL table of records, your job is to write a SQL query that suits the user's request.

CREATE TABLE records AS (
    start_timestamp timestamp with time zone,
    created_at timestamp with time zone,
    trace_id text,
    span_id text,
    parent_span_id text,
    kind span_kind,
    end_timestamp timestamp with time zone,
    level smallint,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    otel_links jsonb,
    otel_events jsonb,
    is_exception boolean,
    otel_status_code status_code,
    otel_status_message text,
    otel_scope_name text,
    otel_scope_version text,
    otel_scope_attributes jsonb,
    service_namespace text,
    service_name text,
    service_version text,
    service_instance_id text,
    process_pid integer
);

today's date = 2024-10-09

Example
    request: show me records where foobar is false
    response: SELECT * FROM records WHERE attributes->>'foobar' = false'
Example
    request: show me records from yesterday
    response: SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'
Example
    request: show me error records with the tag "foobar"
    response: SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)
"""


@dataclass
class Response:
    sql_query: str


agent = Agent('gemini-1.5-flash', result_type=Response, system_prompt=system_prompt, deps=None)


if __name__ == '__main__':
    with debug.timer('SQL Generation'):
        result = agent.run_sync('show me logs from yesterday, with level "error"')
    debug(result.response.sql_query)
