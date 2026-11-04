"""Helpers shared by the OAuth subscription provider tests."""

from __future__ import annotations

import asyncio
import base64
import json
import socket

import httpx2
from pydantic import JsonValue


def make_jwt(payload: dict[str, JsonValue]) -> str:
    """An unsigned JWT carrying `payload`: the providers read claims without verifying signatures."""

    def encode(part: dict[str, JsonValue]) -> str:
        return base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b'=').decode()

    return f'{encode({"alg": "none"})}.{encode(payload)}.signature'


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def get_callback(url: str, params: dict[str, str]) -> httpx2.Response:
    """GET the callback URL, retrying briefly while the one-shot server binds."""
    async with httpx2.AsyncClient() as client:
        for _ in range(50):
            try:
                return await client.get(url, params=params)
            except httpx2.ConnectError:
                await asyncio.sleep(0.05)
        raise AssertionError('callback server never came up')  # pragma: no cover
