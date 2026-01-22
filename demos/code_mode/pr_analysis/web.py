"""PR Analysis Web UI - Compare code mode vs traditional mode.

Usage:
    source .env && uv run python demos/code_mode/pr_analysis/web.py

Then open:
    - Traditional mode: http://localhost:7934
    - Code mode:        http://localhost:7935
"""

from __future__ import annotations

import threading

import logfire
import uvicorn

from .demo import DEFAULT_MODEL, PROMPT, create_code_mode_agent, create_github_mcp, create_traditional_agent

TRADITIONAL_PORT = 7934
CODE_MODE_PORT = 7935


def scrubbing_callback(match: logfire.ScrubMatch):
    """Preserve 'author' fields that match 'auth' pattern."""
    if match.pattern_match.group(0) == 'auth':
        # Check if the path contains 'author' - don't scrub author fields
        path_str = str(match.path).lower()
        if 'author' in path_str:
            return match.value
    return None


def main():
    logfire.configure(
        service_name='pr-analysis-demo',
        scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback),
    )
    logfire.instrument_pydantic_ai()

    github = create_github_mcp()

    traditional_agent = create_traditional_agent(github)
    code_mode_agent = create_code_mode_agent(github)

    traditional_app = traditional_agent.to_web(models=[DEFAULT_MODEL])
    code_mode_app = code_mode_agent.to_web(models=[DEFAULT_MODEL])

    print('=' * 60)
    print('PR Analysis: Code Mode vs Traditional Mode')
    print('=' * 60)
    print()
    print(f'  Traditional mode: http://localhost:{TRADITIONAL_PORT}')
    print(f'  Code mode:        http://localhost:{CODE_MODE_PORT}')
    print()
    print('=' * 60)
    print('TEST PROMPT:')
    print('=' * 60)
    print(PROMPT)
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
