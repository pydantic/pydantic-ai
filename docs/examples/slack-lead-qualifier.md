# Slack Lead Qualifier with Modal

In this example, we're going to build an agentic app that:
- automatically investigates each new member that joins a company's public Slack community to see how good of a sales lead they are for the company's commercial product,
- sends this analysis into a (private) Slack channel, and
- sends a daily summary of the top 5 leads from the previous 24 hours into a (different) Slack channel.

We'll be deploying the app on [Modal](https://modal.com), as it makes it really easy to define an app with web endpoints and scheduled functions in Python and deploy them with a CLI, without needing to set up or manage any infrastructure.

We also add [Pydantic Logfire](https://pydantic.dev/logfire) to get observability into the app and agent as they're running in response to webhooks and on the schedule.

As you may have guessed, Pydantic runs an agent very much like this one to identify leads for Pydantic Logfire among members of the [public Slack](https://logfire.pydantic.dev/docs/join-slack).

## Prerequisites
- A Slack workspace
    - https://docs.slack.dev/apis/events-api
        - https://docs.slack.dev/reference/events/team_join
    - https://docs.slack.dev/quickstart
        - https://docs.slack.dev/reference/scopes/users.read
        - https://docs.slack.dev/reference/scopes/users.read.email
        - https://docs.slack.dev/reference/scopes/users.profile.read
    - Add app to the channels you want it to be able to post in
- An OpenAI API key
- A Logfire account
- A Modal account
    - Store secrets: logfire, openai, slack

## Usage

- Clone the repo
- `cd examples/pydantic_ai_examples`
- Ephemeral: `uv run modal serve -m slack_lead_qualifier.modal`
- Production: `uv run modal deploy -m slack_lead_qualifier.modal`

## Define the models

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/models.py" title="slack_lead_qualifier/models.py" fragment="profile"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/models.py" title="slack_lead_qualifier/models.py" fragment="analysis"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/models.py" title="slack_lead_qualifier/models.py" fragment="unknown"}```

### Define the agent

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/agent.py" title="slack_lead_qualifier/agent.py" fragment="agent"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/agent.py" title="slack_lead_qualifier/agent.py" fragment="analyze_profile"}```

### Define a store for analyses

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/store.py" title="slack_lead_qualifier/store.py" fragment="analysis_store"}```

### Define the functions

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/functions.py" title="slack_lead_qualifier/functions.py" fragment="process_slack_member"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/functions.py" title="slack_lead_qualifier/functions.py" fragment="send_daily_summary"}```

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/slack.py" title="slack_lead_qualifier/slack.py" fragment="send_slack_message"}```

### Define the web app

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/app.py" title="slack_lead_qualifier/app.py" fragment="app"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/app.py" title="slack_lead_qualifier/app.py" fragment="process_slack_member"}```

### Define the Modal app

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" title="slack_lead_qualifier/modal.py" fragment="setup_modal"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" title="slack_lead_qualifier/modal.py" fragment="setup_logfire"}```

```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" title="slack_lead_qualifier/modal.py" fragment="web_app"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" title="slack_lead_qualifier/modal.py" fragment="process_slack_member"}```
```snippet {path="examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" title="slack_lead_qualifier/modal.py" fragment="send_daily_summary"}```
