"""Example of a Pydantic AI agent that understands video using TwelveLabs Pegasus.

In this case the idea is a "video analyst" agent — the user can ask questions about a
video (given its URL), and the agent will use the `analyze_video` tool to call
[TwelveLabs](https://twelvelabs.io) Pegasus, a video-understanding model, to answer.

This shows how to wrap a third-party multimodal API as a Pydantic AI tool: the LLM
decides *what* to ask about the video, and Pegasus does the actual video understanding.

Run with:

    uv run -m pydantic_ai_examples.twelvelabs_video_agent
"""

from __future__ import annotations as _annotations

import asyncio
import os
from dataclasses import dataclass

import logfire
from twelvelabs import AsyncTwelveLabs

from pydantic_ai import Agent, RunContext

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# A public sample video used when the user doesn't provide one. The URL must point at a
# video file TwelveLabs can fetch directly; set VIDEO_URL to use your own.
DEFAULT_VIDEO_URL = 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4'


@dataclass
class Deps:
    twelvelabs: AsyncTwelveLabs
    video_url: str


video_agent = Agent(
    'openai:gpt-5-mini',
    instructions=(
        'You help users understand a video. '
        'Use the `analyze_video` tool to ask the video-understanding model questions, '
        'then answer the user concisely based on what it returns.'
    ),
    deps_type=Deps,
    retries=2,
)


@video_agent.tool
async def analyze_video(ctx: RunContext[Deps], prompt: str) -> str:
    """Analyze the video with TwelveLabs Pegasus and return a text answer.

    Args:
        ctx: The context.
        prompt: What to ask about the video, e.g. "Summarize this video" or
            "What objects appear in the first 10 seconds?".
    """
    response = await ctx.deps.twelvelabs.analyze(
        model_name='pegasus1.5',
        video={'type': 'url', 'url': ctx.deps.video_url},
        prompt=prompt,
        max_tokens=2048,
    )
    return response.data or ''


async def main():
    api_key = os.environ.get('TWELVELABS_API_KEY')
    if not api_key:
        raise RuntimeError(
            'Set TWELVELABS_API_KEY to run this example. '
            'Grab a free key at https://twelvelabs.io.'
        )
    video_url = os.environ.get('VIDEO_URL', DEFAULT_VIDEO_URL)

    async with AsyncTwelveLabs(api_key=api_key) as client:
        deps = Deps(twelvelabs=client, video_url=video_url)
        result = await video_agent.run(
            'Give me a one-sentence summary of this video.', deps=deps
        )
        print('Response:', result.output)


if __name__ == '__main__':
    asyncio.run(main())
