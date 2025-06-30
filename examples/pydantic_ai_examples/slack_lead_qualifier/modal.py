from typing import Any

### [setup_modal]
import modal

image = modal.Image.debian_slim(python_version='3.13').pip_install(
    'pydantic',
    'pydantic_ai_slim[openai,duckduckgo]',
    'logfire[httpx,fastapi]',
    'fastapi[standard]',
    'httpx',
)
app = modal.App(
    name='slack-lead-qualifier',
    image=image,
    secrets=[
        modal.Secret.from_name('logfire'),
        modal.Secret.from_name('openai'),
        modal.Secret.from_name('slack'),
    ],
)  ### [/setup_modal]


### [setup_logfire]
def setup_logfire():
    import logfire

    logfire.configure(service_name=app.name)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)  ### [/setup_logfire]


### [web_app]
@app.function(min_containers=1)
@modal.asgi_app()  # type: ignore
def web_app():
    setup_logfire()

    from .app import app as _app

    return _app  ### [/web_app]


### [process_slack_member]
@app.function()
async def process_slack_member(profile_raw: dict[str, Any], logfire_ctx: Any):
    setup_logfire()

    from logfire.propagate import attach_context

    from .functions import process_slack_member as _process_slack_member
    from .models import Profile

    with attach_context(logfire_ctx):
        profile = Profile.model_validate(profile_raw)
        await _process_slack_member(profile)  ### [/process_slack_member]


### [send_daily_summary]
@app.function(schedule=modal.Cron('0 8 * * *'))  # Every day at 8am UTC
async def send_daily_summary():
    setup_logfire()

    from .functions import send_daily_summary as _send_daily_summary

    await _send_daily_summary()  ### [/send_daily_summary]
