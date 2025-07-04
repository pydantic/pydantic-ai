"""Basic example of using Agent.to_ag_ui with FastAPI."""

from __future__ import annotations

from pydantic_ai import Agent

agent: Agent[None, str] = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant.',
)
app = agent.to_ag_ui()

if __name__ == '__main__':
    import uvicorn

    from .cli import Args, parse_args

    args: Args = parse_args()

    uvicorn.run(
        'pydantic_ai_ag_ui_examples.dojo_server:app',
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        log_config=args.log_config(),
    )
