"""Logfire query utilities for the analyze-logfire-data skill.

Usage:
    uv run --with logfire python logfire_query.py

Or import in your scripts:
    from logfire_query import query_sync, load_results
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any


async def query_logfire(sql: str, output_file: str | Path, age_minutes: int = 60) -> list[dict[str, Any]]:
    """Execute a query against Logfire and save results to JSON.

    Args:
        sql: SQL query string (DataFusion SQL syntax)
        output_file: Path to save JSON results
        age_minutes: How far back to look (default 60 minutes, max 43200 = 30 days)

    Returns:
        List of row dictionaries

    Raises:
        RuntimeError: If LOGFIRE_READ_TOKEN is not set
    """
    token = os.environ.get('LOGFIRE_READ_TOKEN')
    if not token:
        raise RuntimeError('LOGFIRE_READ_TOKEN environment variable is not set')

    import logfire

    logfire.configure(token=token)

    rows: list[dict[str, Any]] = []
    async with logfire.query(sql=sql, min_timestamp=f'{age_minutes}m') as response:
        async for row in response:
            rows.append(dict(row))

    output_path = Path(output_file)
    output_path.write_text(json.dumps(rows, indent=2, default=str))
    print(f'Saved {len(rows)} rows to {output_path}')

    return rows


def query_sync(sql: str, output_file: str | Path, age_minutes: int = 60) -> list[dict[str, Any]]:
    """Synchronous wrapper for query_logfire.

    Args:
        sql: SQL query string (DataFusion SQL syntax)
        output_file: Path to save JSON results
        age_minutes: How far back to look (default 60 minutes, max 43200 = 30 days)

    Returns:
        List of row dictionaries
    """
    return asyncio.run(query_logfire(sql, output_file, age_minutes))


def load_results(file_path: str | Path) -> list[dict[str, Any]]:
    """Load query results from a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of row dictionaries
    """
    return json.loads(Path(file_path).read_text())


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print('Usage: python logfire_query.py <sql> <output_file> [age_minutes]')
        print('Example: python logfire_query.py "SELECT * FROM records LIMIT 10" results.json 60')
        sys.exit(1)

    sql = sys.argv[1]
    output = sys.argv[2]
    age = int(sys.argv[3]) if len(sys.argv) > 3 else 60

    query_sync(sql, output, age)
