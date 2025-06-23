"""Example usage of the AG-UI adapter for PydanticAI.

This provides a FastAPI application that demonstrates how to use the
PydanticAI agent with the AG-UI protocol. It includes examples for
each of the AG-UI dojo features:
- Agentic Chat
- Human in the Loop
- Agentic Generative UI
- Tool Based Generative UI
- Shared State
- Predictive State Updates
"""

from __future__ import annotations

from fastapi import FastAPI

from .api import (
    agentic_chat_router,
    agentic_generative_ui_router,
    human_in_the_loop_router,
    predictive_state_updates_router,
    shared_state_router,
    tool_based_generative_ui_router,
)

app = FastAPI(title='PydanticAI AG-UI server')
app.include_router(agentic_chat_router, tags=['Agentic Chat'])
app.include_router(agentic_generative_ui_router, tags=['Agentic Generative UI'])
app.include_router(human_in_the_loop_router, tags=['Human in the Loop'])
app.include_router(predictive_state_updates_router, tags=['Predictive State Updates'])
app.include_router(shared_state_router, tags=['Shared State'])
app.include_router(tool_based_generative_ui_router, tags=['Tool Based Generative UI'])


if __name__ == '__main__':
    import uvicorn

    from .cli import Args, parse_args

    args: Args = parse_args()

    uvicorn.run(
        'pydantic_ai.ag_ui_examples.dojo_server:app',
        port=args.port,
        reload=args.reload,
        log_config=args.log_config(),
    )
