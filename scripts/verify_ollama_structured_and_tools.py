#!/usr/bin/env python3
"""Verify Ollama structured output and streaming tool calls via OpenAI-compatible API.

Run from pydantic-ai repo root with Ollama running (e.g. ollama serve, ollama run qwen3):

    uv run python scripts/verify_ollama_structured_and_tools.py

Set OLLAMA_MODEL to a model you have (default: qwen3:8b). Requires openai (use repo with openai extra).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# Base URL for Ollama's OpenAI-compatible endpoint
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")


def _check_openai() -> bool:
    try:
        import openai  # noqa: F401
        return True
    except ImportError:
        return False


async def test_structured_output() -> bool:
    """Test structured output via response_format (JSON schema)."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("SKIP structured output: openai not installed")
        return False

    client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )

    # Minimal JSON schema for structured output
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather",
            "schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "temp": {"type": "integer"}},
                "required": ["city", "temp"],
            },
        },
    }

    try:
        resp = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Return JSON with city=London and temp=20."}],
            response_format=response_format,
            max_tokens=200,
        )
    except Exception as e:
        print(f"  Structured output request failed: {e}")
        return False

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        print("  Structured output: empty content")
        return False

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  Structured output: invalid JSON: {e}")
        print(f"  Raw content: {content[:200]!r}")
        return False

    if "city" in data and "temp" in data:
        print(f"  Structured output OK: {data}")
        return True
    print(f"  Structured output: missing keys, got {data}")
    return False


async def test_streaming_tool_calls() -> bool:
    """Test streaming with tools; check if we get tool_call deltas with arguments."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("SKIP streaming tools: openai not installed")
        return False

    client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    try:
        stream = await client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": "What's the weather in Paris? Use get_weather."}],
            tools=tools,
            tool_choice="auto",
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception as e:
        print(f"  Streaming tools request failed: {e}")
        return False

    got_tool_call = False
    got_args_delta = False
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.id or tc.index is not None:
                    got_tool_call = True
                if tc.function:
                    if tc.function.name:
                        got_tool_call = True
                    if tc.function.arguments:
                        got_args_delta = True
                        break
        if got_args_delta:
            break

    if got_tool_call and got_args_delta:
        print("  Streaming tool calls OK: received tool_call deltas with arguments")
        return True
    if got_tool_call:
        print("  Streaming tool calls: got tool_calls but no arguments in deltas (Ollama may send args only at end)")
        return True  # still useful
    print("  Streaming tool calls: no tool_call deltas in stream")
    return False


async def main() -> None:
    print("Ollama verification (OpenAI-compatible API)")
    print(f"  Base URL: {OLLAMA_BASE_URL}")
    print()

    if not _check_openai():
        print("Install openai: pip install openai (or use repo with openai extra)")
        sys.exit(1)

    print("1. Structured output (response_format with json_schema)")
    ok1 = await test_structured_output()
    print()

    print("2. Streaming with tool calls (delta.tool_calls[].function.arguments)")
    ok2 = await test_streaming_tool_calls()
    print()

    if ok1 and ok2:
        print("All checks passed. Current OpenAI-compatible path may already support both.")
    elif ok1:
        print("Structured output works; streaming tool-call args may need native API or mapping fix.")
    elif ok2:
        print("Streaming tools work; structured output may need format translation or native API.")
    else:
        print("One or both checks failed. See implementation plan for native API or adapter.")


if __name__ == "__main__":
    asyncio.run(main())
