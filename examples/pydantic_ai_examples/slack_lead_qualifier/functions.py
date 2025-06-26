import logfire

from .agent import agent
from .models import Analysis, Profile
from .slack import send_slack_message
from .store import AnalysisStore

NEW_LEAD_CHANNEL = '#hackathon-sales-qual-agent'
DAILY_SUMMARY_CHANNEL = '#hackathon-sales-qual-agent'


@logfire.instrument('Process Slack member')
async def process_slack_member(profile: Profile):
    result = await agent.run(profile.as_prompt())
    analysis = result.output
    logfire.info('Analysis', analysis=analysis)

    if not isinstance(analysis, Analysis):
        return

    await AnalysisStore().add(analysis)

    await send_slack_message(
        NEW_LEAD_CHANNEL,
        [
            {
                'type': 'header',
                'text': {
                    'type': 'plain_text',
                    'text': f'New Slack member with score {analysis.relevance}/5',
                },
            },
            {
                'type': 'divider',
            },
            *analysis.as_slack_blocks(),
        ],
    )


@logfire.instrument('Send daily summary')
async def send_daily_summary():
    analyses = await AnalysisStore().list()
    logfire.info('Analyses', analyses=analyses)

    if len(analyses) == 0:
        return

    sorted_analyses = sorted(analyses, key=lambda x: x.relevance, reverse=True)
    top_analyses = sorted_analyses[:5]

    blocks = [
        {
            'type': 'header',
            'text': {
                'type': 'plain_text',
                'text': f'Top {len(top_analyses)} new Slack members from the last 24 hours',
            },
        },
    ]

    for analysis in top_analyses:
        blocks.extend(
            [
                {
                    'type': 'divider',
                },
                *analysis.as_slack_blocks(include_relevance=True),
            ]
        )

    await send_slack_message(
        DAILY_SUMMARY_CHANNEL,
        blocks,
    )

    # with logfire.span('Clearing analyses'):
    #     await AnalysisStore().clear()
