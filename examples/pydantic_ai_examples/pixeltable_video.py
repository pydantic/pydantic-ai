"""Search a training video's transcripts and frame captions, with persistent memory.

Set `OPENAI_API_KEY` and run once with a short expense-policy video containing speech:

    python pixeltable_video.py training.mp4

Insertion calls OpenAI for transcription, captions, and embeddings. Searches embed
their queries and retrieve stored text and timestamps.
"""

import sys

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.audio import audio_splitter
from pixeltable.functions.video import extract_audio, frame_iterator

from pydantic_ai import Agent
from pydantic_ai_harness import Memory, Pixeltable
from pydantic_ai_harness.memory import PixeltableMemoryStore


def main(video_path: str) -> None:
    pxt.create_dir('training', if_exists='ignore')
    videos = pxt.create_table(  # pyright: ignore[reportUnknownMemberType]
        'training.videos', {'title': pxt.String, 'video': pxt.Video}, if_exists='ignore'
    )
    videos.add_computed_column(audio=extract_audio(videos.video), if_exists='ignore')
    embedding = openai.embeddings.using(model='text-embedding-3-small')

    segments = pxt.create_view(
        'training.segments',
        videos,
        iterator=audio_splitter(videos.audio, duration=30.0),
        if_exists='ignore',
    )
    assert segments is not None
    segments.add_computed_column(
        transcript=openai.transcriptions(
            segments.audio_segment, model='whisper-1'
        ).text.astype(pxt.String),
        if_exists='ignore',
    )
    segments.add_embedding_index('transcript', embedding=embedding, if_exists='ignore')

    frames = pxt.create_view(
        'training.frames',
        videos,
        iterator=frame_iterator(videos.video, fps=0.1),
        if_exists='ignore',
    )
    assert frames is not None
    frames.add_computed_column(
        caption=openai.chat_completions(
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': 'Describe this training slide or scene, including any policy numbers.',
                        },
                        {'type': 'image_url', 'image_url': frames.frame},
                    ],
                }
            ],
            model='gpt-5.6-sol',
        )
        .choices[0]
        .message.content.astype(pxt.String),
        if_exists='ignore',
    )
    frames.add_embedding_index('caption', embedding=embedding, if_exists='ignore')

    videos.insert([{'title': 'Expense policy', 'video': video_path}])  # pyright: ignore[reportUnknownMemberType]
    agent = Agent(
        'openai:gpt-5.6-sol',
        instructions=(
            'Answer from retrieved training material. Cite the title and segment_start/segment_end '
            'for transcripts, or frame_attrs.time for frame captions; all times are in seconds. '
            'Say when the retrieved material does not answer the question.'
        ),
        capabilities=[
            Pixeltable(['training.segments', 'training.frames']),
            Memory(PixeltableMemoryStore(table_name='training.memory')),
        ],
    )
    print(
        agent.run_sync(
            'What is the meal reimbursement limit? Cite the audio segment. Remember that I prefer visual examples.'
        ).output
    )
    print(
        agent.run_sync(
            'Using my saved preference, find a frame illustrating that policy and cite its timestamp.'
        ).output
    )


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'training.mp4')
