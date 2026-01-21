# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "logfire",
# ]
# ///
"""Fetch eval data from Logfire and save to JSON files.

Usage:
    LOGFIRE_READ_TOKEN=<token> uv run python demos/code_mode/analysis/fetch_data.py
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from logfire.query_client import AsyncLogfireQueryClient

DATA_DIR = Path(__file__).parent / 'data'
DATA_DIR.mkdir(exist_ok=True)


async def run_query(client: AsyncLogfireQueryClient, sql: str, output_file: str) -> int:
    """Execute query and save results."""
    results = await client.query_json_rows(sql=sql)
    with open(DATA_DIR / output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return len(results.get('rows', []))


async def main():
    token = os.environ.get('LOGFIRE_READ_TOKEN')
    if not token:
        raise ValueError('LOGFIRE_READ_TOKEN environment variable is required')

    print('Fetching data from Logfire...')
    print()

    async with AsyncLogfireQueryClient(read_token=token) as client:
        # Latency data
        print('  Latency data...', end=' ', flush=True)
        count = await run_query(
            client,
            """
            SELECT
                attributes->>'mode' as mode,
                attributes->>'run_number' as run_number,
                duration as duration_seconds
            FROM records
            WHERE service_name = 'code-mode-eval'
                AND span_name = 'eval_run'
                AND attributes->>'mode' IS NOT NULL
            ORDER BY attributes->>'mode', attributes->>'run_number'
            """,
            'latency.json',
        )
        print(f'saved {count} rows')

        # Token data - aggregate by eval_run trace
        print('  Token data...', end=' ', flush=True)
        count = await run_query(
            client,
            """
            SELECT
                r1.attributes->>'mode' as mode,
                r1.attributes->>'run_number' as run_number,
                SUM((r2.attributes->>'gen_ai.usage.details.input_tokens')::int) as input_tokens,
                SUM((r2.attributes->>'gen_ai.usage.details.output_tokens')::int) as output_tokens
            FROM records r1
            JOIN records r2 ON r1.trace_id = r2.trace_id
            WHERE r1.service_name = 'code-mode-eval'
                AND r1.span_name = 'eval_run'
                AND r1.attributes->>'mode' IS NOT NULL
                AND r2.span_name LIKE 'chat%'
            GROUP BY r1.attributes->>'mode', r1.attributes->>'run_number'
            ORDER BY r1.attributes->>'mode', r1.attributes->>'run_number'
            """,
            'tokens.json',
        )
        print(f'saved {count} rows')

        # Request count data
        print('  Request count data...', end=' ', flush=True)
        count = await run_query(
            client,
            """
            SELECT
                r1.attributes->>'mode' as mode,
                r1.attributes->>'run_number' as run_number,
                COUNT(r2.span_id) as request_count
            FROM records r1
            JOIN records r2 ON r1.trace_id = r2.trace_id
            WHERE r1.service_name = 'code-mode-eval'
                AND r1.span_name = 'eval_run'
                AND r1.attributes->>'mode' IS NOT NULL
                AND r2.span_name LIKE 'chat%'
            GROUP BY r1.attributes->>'mode', r1.attributes->>'run_number'
            ORDER BY r1.attributes->>'mode', r1.attributes->>'run_number'
            """,
            'requests.json',
        )
        print(f'saved {count} rows')

        # Cost data
        print('  Cost data...', end=' ', flush=True)
        count = await run_query(
            client,
            """
            SELECT
                r1.attributes->>'mode' as mode,
                r1.attributes->>'run_number' as run_number,
                SUM((r2.attributes->>'operation.cost')::float) as total_cost
            FROM records r1
            JOIN records r2 ON r1.trace_id = r2.trace_id
            WHERE r1.service_name = 'code-mode-eval'
                AND r1.span_name = 'eval_run'
                AND r1.attributes->>'mode' IS NOT NULL
                AND r2.span_name LIKE 'chat%'
            GROUP BY r1.attributes->>'mode', r1.attributes->>'run_number'
            ORDER BY r1.attributes->>'mode', r1.attributes->>'run_number'
            """,
            'cost.json',
        )
        print(f'saved {count} rows')

    print()
    print(f'Data saved to {DATA_DIR}/')


if __name__ == '__main__':
    asyncio.run(main())
