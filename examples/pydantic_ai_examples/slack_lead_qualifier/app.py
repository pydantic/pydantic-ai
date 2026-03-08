from typing import Any

import logfire
from fastapi import FastAPI, HTTPException, status
from logfire.propagate import get_context

from .models import Profile


### [process_slack_member]
def process_slack_member(profile: Profile):
    from .modal import process_slack_member as _process_slack_member

    _process_slack_member.spawn(
        profile.model_dump(), logfire_ctx=get_context()
    )  ### [/process_slack_member]


### [app]
app = FastAPI()
logfire.instrument_fastapi(app, capture_headers=True)


@app.post('/')
async def process_webhook(payload: dict[str, Any]) -> dict[str, Any]:
    if payload['type'] == 'url_verification':
        return {'challenge': payload['challenge']}
    elif (
        payload['type'] == 'event_callback' and payload['event']['type'] == 'team_join'
    ):
        profile = Profile.model_validate(payload['event']['user']['profile'])

        process_slack_member(profile)
        return {'status': 'OK'}

    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)  ### [/app]
