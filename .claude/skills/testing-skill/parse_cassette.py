#!/usr/bin/env python3
"""Parse cassette files and pretty-print request/response bodies."""

import argparse
import json
import re
import sys
from pathlib import Path

from cassetter import Body, Cassette, CassetteLoadError, RecordMode


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


def _extract_body(body: Body) -> object | None:
    """Decode a recorded body: JSON as an object, text as a string (parsed if it is JSON), binary as a size note."""
    if body.body_type == 'json':
        return body.content
    if body.body_type == 'text':
        try:
            return json.loads(body.content)
        except json.JSONDecodeError:
            return body.content
    if body.body_type == 'binary':
        return f'<{len(body.content)} bytes of binary data>'
    return None


def _print_body(body: Body) -> None:
    content = _extract_body(body)
    if content is None:
        return
    truncated = truncate_base64(content)
    print(f'Body:\n{truncated if isinstance(truncated, str) else json.dumps(truncated, indent=2)}')


def parse_cassette(path: Path, interaction_idx: int | None = None) -> None:
    """Parse and print cassette contents, in cassetter's format or the older VCR.py one."""
    cassette = Cassette(path, record_mode=RecordMode.NONE)
    try:
        cassette.load()
    except CassetteLoadError as exc:
        print(f'Not an HTTP cassette cassetter can read: {exc}', file=sys.stderr)
        sys.exit(1)

    interactions = cassette.interactions
    if not interactions:
        print('No HTTP interactions found in cassette')
        return

    indices = [interaction_idx] if interaction_idx is not None else range(len(interactions))

    for i in indices:
        if i < 0 or i >= len(interactions):
            print(f'Interaction {i} not found (only {len(interactions)} interactions)')
            continue

        interaction = interactions[i]
        req = interaction.request
        resp = interaction.response

        print(f'\n{"=" * 60}')
        print(f'INTERACTION {i}')
        print('=' * 60)

        print('\n--- REQUEST ---')
        print(f'Method: {req.method}')
        print(f'URI: {req.uri}')
        _print_body(req.body)

        print('\n--- RESPONSE ---')
        print(f'Status: {resp.status}')
        _print_body(resp.body)


def main() -> None:
    """Print the interactions of the cassette named on the command line."""
    parser = argparse.ArgumentParser(description='Parse cassette files')
    parser.add_argument('cassette', type=Path, help='Path to cassette YAML file')
    parser.add_argument('--interaction', '-i', type=int, help='Specific interaction index (0-based)')
    args = parser.parse_args()

    if not args.cassette.exists():
        print(f'File not found: {args.cassette}', file=sys.stderr)
        sys.exit(1)

    parse_cassette(args.cassette, args.interaction)


if __name__ == '__main__':
    main()
