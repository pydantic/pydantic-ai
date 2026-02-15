#!/usr/bin/env python3
"""Parse VCR cassette files and pretty-print request/response bodies."""

import argparse
import re
import sys
from pathlib import Path

import yaml
from pydantic_core import from_json, to_json


def truncate_base64(obj: object, max_len: int = 100) -> object:
    """Recursively truncate base64-like strings in nested structures."""
    if isinstance(obj, str):
        if len(obj) > max_len and re.match(r'^[A-Za-z0-9+/=]+$', obj[:100]):
            return f'{obj[:50]}...[truncated {len(obj)} chars]...{obj[-20:]}'
        if obj.startswith('data:') and len(obj) > max_len:
            return f'{obj[:80]}...[truncated {len(obj)} chars]'
        return obj
    elif isinstance(obj, dict):
        return {k: truncate_base64(v, max_len) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [truncate_base64(item, max_len) for item in obj]
    return obj


def _extract_body(part: dict[str, object]) -> object | None:
    """Extract body from a request/response, handling both custom and standard VCR formats.

    Custom serializer (json_body_serializer.py) stores parsed JSON as `parsed_body`.
    Standard VCR stores raw strings: `body` (requests) or `body.string` (responses).
    """
    if 'parsed_body' in part:
        return part['parsed_body']
    body = part.get('body')
    if isinstance(body, dict):
        body = body.get('string')
    if isinstance(body, str):
        try:
            return from_json(body)
        except ValueError:
            return body
    return None


def parse_cassette(path: Path, interaction_idx: int | None = None) -> None:
    """Parse and print cassette contents."""
    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        print('Empty or invalid cassette file')
        return

    interactions = data.get('interactions', [])
    if not interactions:
        print('No interactions found in cassette')
        return

    indices = [interaction_idx] if interaction_idx is not None else range(len(interactions))

    for i in indices:
        if i < 0 or i >= len(interactions):
            print(f'Interaction {i} not found (only {len(interactions)} interactions)')
            continue

        interaction = interactions[i]
        req = interaction.get('request', {})
        resp = interaction.get('response', {})

        print(f'\n{"="*60}')
        print(f'INTERACTION {i}')
        print('='*60)

        print(f'\n--- REQUEST ---')
        print(f'Method: {req.get("method", "N/A")}')
        print(f'URI: {req.get("uri", "N/A")}')
        req_body = _extract_body(req)
        if req_body is not None:
            truncated = truncate_base64(req_body)
            print(f'Body:\n{to_json(truncated, indent=2).decode() if not isinstance(truncated, str) else truncated}')

        print(f'\n--- RESPONSE ---')
        status = resp.get('status', {})
        print(f'Status: {status.get("code", "N/A")} {status.get("message", "")}')
        resp_body = _extract_body(resp)
        if resp_body is not None:
            truncated = truncate_base64(resp_body)
            print(f'Body:\n{to_json(truncated, indent=2).decode() if not isinstance(truncated, str) else truncated}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Parse VCR cassette files')
    parser.add_argument('cassette', type=Path, help='Path to cassette YAML file')
    parser.add_argument('--interaction', '-i', type=int, help='Specific interaction index (0-based). Omit to list all interactions.')
    args = parser.parse_args()

    if not args.cassette.exists():
        print(f'File not found: {args.cassette}', file=sys.stderr)
        sys.exit(1)

    parse_cassette(args.cassette, args.interaction)


if __name__ == '__main__':
    main()
