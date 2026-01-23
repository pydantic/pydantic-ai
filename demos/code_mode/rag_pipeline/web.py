"""RAG Pipeline Web UI - Compare code mode vs traditional mode.

Usage (full mode - requires PINECONE_API_KEY and TAVILY_API_KEY):
    source .env && uv run python demos/code_mode/rag_pipeline/web.py

Usage (zero-setup mode - no API keys needed):
    uv run python demos/code_mode/rag_pipeline/web.py --zero-setup

Then open:
    - Traditional mode: http://localhost:7936
    - Code mode:        http://localhost:7937
"""

from __future__ import annotations

import argparse
import os
import threading

import logfire
import uvicorn

from .demo import (
    DEFAULT_MODEL,
    PROMPT,
    PROMPT_ZERO_SETUP,
    create_code_mode_agent,
    create_code_mode_agent_zero_setup,
    create_pinecone_mcp,
    create_pinecone_mcp_zero_setup,
    create_tavily_mcp,
    create_traditional_agent,
    create_traditional_agent_zero_setup,
)

TRADITIONAL_PORT = 7936
CODE_MODE_PORT = 7937


def main():
    parser = argparse.ArgumentParser(description='RAG Pipeline Demo')
    parser.add_argument(
        '--zero-setup',
        action='store_true',
        help='Run in zero-setup mode (no API keys needed, uses search_docs only)',
    )
    args = parser.parse_args()

    logfire.configure(service_name='rag-pipeline-demo')
    logfire.instrument_pydantic_ai()

    if args.zero_setup:
        # Zero-setup mode: only Pinecone's search_docs, no API keys needed
        pinecone = create_pinecone_mcp_zero_setup()
        traditional_agent = create_traditional_agent_zero_setup(pinecone)
        code_mode_agent = create_code_mode_agent_zero_setup(pinecone)
        prompt = PROMPT_ZERO_SETUP
        mode_label = 'Zero-Setup Mode (search_docs only)'
    else:
        # Full mode: Pinecone + Tavily, requires API keys
        if not os.environ.get('PINECONE_API_KEY'):
            print('ERROR: PINECONE_API_KEY not set. Use --zero-setup for no-key mode.')
            return
        if not os.environ.get('TAVILY_API_KEY'):
            print('ERROR: TAVILY_API_KEY not set. Use --zero-setup for no-key mode.')
            return

        pinecone = create_pinecone_mcp()
        tavily = create_tavily_mcp()
        traditional_agent = create_traditional_agent(pinecone, tavily)
        code_mode_agent = create_code_mode_agent(pinecone, tavily)
        prompt = PROMPT
        mode_label = 'Full Mode (Pinecone + Tavily)'

    traditional_app = traditional_agent.to_web(models=[DEFAULT_MODEL])
    code_mode_app = code_mode_agent.to_web(models=[DEFAULT_MODEL])

    print('=' * 60)
    print('RAG Pipeline: Code Mode vs Traditional Mode')
    print(f'Mode: {mode_label}')
    print('=' * 60)
    print()
    print(f'  Traditional mode: http://localhost:{TRADITIONAL_PORT}')
    print(f'  Code mode:        http://localhost:{CODE_MODE_PORT}')
    print()
    print('=' * 60)
    print('TEST PROMPT:')
    print('=' * 60)
    print(prompt)
    print('=' * 60)
    print()
    print('View traces at https://logfire.pydantic.dev')
    print('=' * 60)

    def run_traditional():
        uvicorn.run(traditional_app, host='127.0.0.1', port=TRADITIONAL_PORT, log_level='warning')

    traditional_thread = threading.Thread(target=run_traditional, daemon=True)
    traditional_thread.start()

    uvicorn.run(code_mode_app, host='127.0.0.1', port=CODE_MODE_PORT, log_level='info')


if __name__ == '__main__':
    main()
