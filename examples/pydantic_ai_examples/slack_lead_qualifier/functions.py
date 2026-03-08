import logfire

### [imports]
from .agent import analyze_profile
from .models import Profile

### [imports-daily_summary]
from .slack import send_slack_message
from .store import AnalysisStore  ### [/imports,/imports-daily_summary]

### [constant-new_lead_channel]
NEW_LEAD_CHANNEL = '#new-slack-leads'
### [/constant-new_lead_channel]
### [constant-daily_summary_channel]
DAILY_SUMMARY_CHANNEL = '#daily-slack-leads-summary'
### [/constant-daily_summary_channel]


### [process_slack_member]
@logfire.instrument('Process Slack member')
async def process_slack_member(profile: Profile):
    analysis = await analyze_profile(profile)
    logfire.info('Analysis', analysis=analysis)

    if analysis is None:
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
    )  ### [/process_slack_member]


### [send_daily_summary]
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

    await AnalysisStore().clear()  ### [/send_daily_summary]
