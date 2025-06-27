# Slack Lead Qualifier with Modal

In this example, we're going to build an agentic app that:

- automatically investigates each new member that joins a company's public Slack community to see how good of a sales lead they are for the company's commercial product,
- sends this analysis into a (private) Slack channel, and
- sends a daily summary of the top 5 leads from the previous 24 hours into a (different) Slack channel.

We'll be deploying the app on [Modal](https://modal.com), as it lets you use Python to define an app with web endpoints, scheduled functions, and background functions, and deploy them with a CLI, without needing to set up or manage any infrastructure.

We also add [Pydantic Logfire](https://pydantic.dev/logfire) to get observability into the app and agent as they're running in response to webhooks and the schedule.

As you may have guessed, Pydantic runs an agent very much like this one to identify leads for Pydantic Logfire among members of our [public Slack](https://logfire.pydantic.dev/docs/join-slack).

<!-- TODO: Screenshot -->

## Prerequisites
<!-- TODO: Improve -->
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
<!-- TODO: Improve -->
With [dependencies installed](./index.md#usage), run:

- To serve ephemerally, until you quit with Ctrl+C:

    ```bash
    python/uv-run -m modal serve -m pydantic_ai_examples.slack_lead_qualifier.modal
    ```

- To deploy to Modal:

    ```bash
    python/uv-run -m modal deploy -m pydantic_ai_examples.slack_lead_qualifier.modal
    ```

## The code

We're going to start with the basics, and then build up into the full app.

### Models

#### `Profile`

First, we define a [Pydantic](https://docs.pydantic.dev) model that represents a Slack user profile. These are the fields we get from the [`team_join`](https://docs.slack.dev/reference/events/team_join) event that's sent to the webhook endpoint that we'll define in a bit.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/models.py" fragment="profile"}```

We also define a `Profile.as_prompt()` helper method that uses [`format_as_xml`][pydantic_ai.format_prompt.format_as_xml] to turn the profile into a string that can be sent to the model.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/models.py" fragment="import-format_as_xml profile-intro profile-as_prompt"}```

#### `Analysis`

The second model we'll need represents the result of the analysis that the agent will perform. We include docstrings to provide additional context to the model on what these fields should contain.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/models.py" fragment="analysis"}```

We also define a `Analysis.as_slack_blocks()` helper method that turns the analysis into some [Slack blocks](https://api.slack.com/reference/block-kit/blocks) that can be sent to the Slack API to post a new message.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/models.py" fragment="analysis-intro analysis-as_slack_blocks"}```

### Agent

Now it's time to get into Pydantic AI and define the agent that will do the actually analysis!

#### `agent`

We specify the model we'll use (`openai:gpt-4o`, provide [instructions](../agents.md#instructions), give the agent access to the [DuckDuckGo search tool](../common-tools.md#duckduckgo-search-tool), and tell it to output either an `Analysis` or `None` using [Native Output](../output.md#native-output).

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/agent.py" fragment="imports agent"}```

#### `analyze_profile`

We also define a `analyze_profile` helper function that takes a `Profile`, runs the agent, and returns an `Analysis` (or `None`), and instrument it using [Logfire](../logfire.md).

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/agent.py" fragment="analyze_profile"}```

### Analysis store

The next building block we'll need is a place to store all the analyses that have been done so that we can look them up when we send the daily summary.

Fortunately, Modal provides us with a convenient way to store some data that can be read back in a subsequent Modal run (webhook or scheduled): [`modal.Dict`](https://modal.com/docs/reference/modal.Dict).

We specify some convenience methods to easily add, list, and clear analyses.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/store.py" fragment="import-modal analysis_store"}```

Note that `# type: ignore` on the last line -- unfortunately `modal` does not fully define its types, so we need this to stop our static type checker `pyright`, which we run over all Pydantic AI code including examples, from complaining.

### Send Slack message

Next, we'll need a way to actually send a Slack message, so we define a simple function that uses Slack's [`chat.postMessage`](https://api.slack.com/methods/chat.postMessage) API.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/slack.py" fragment="send_slack_message"}```

### Features

Now we can start putting these building blocks together to implement the actual features we want!

#### `process_slack_member`

This function takes a [`Profile`](#profile), [analyzes](#analyze_profile) it using the agent, adds it to the [`AnalysisStore`](#analysis-store), and [sends](#send-slack-message) the analysis into the `#new-leads` channel.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/functions.py" fragment="imports constant-new_lead_channel process_slack_member"}```

#### `send_daily_summary`

This function list all of the analyses in the [`AnalysisStore`](#analysis-store), takes the top 5 by relevance, [sends](#send-slack-message) them into the `#daily-leads-summary` channel, and clears the `AnalysisStore` so that the next daily run won't process these analyses again.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/functions.py" fragment="imports-daily_summary constant-daily_summary_channel send_daily_summary"}```

### Web app

As it stands, neither of these functions are actually being called from anywhere.

Let's implement a [FastAPI](https://fastapi.tiangolo.com/) endpoint to handle the `team_join` Slack webhook (also known as the [Slack Events API](https://docs.slack.dev/apis/events-api)) and call the [`process_slack_member`](#process_slack_member) function we just defined. We also instrument FastAPI using Logfire for good measure.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/app.py" fragment="app"}```

#### `process_slack_member` with Modal

I was a little sneaky there -- we're not actually calling the [`process_slack_member`](#process_slack_member) function we defined in `functions.py` directly, as Slack requires webhooks to respond within 3 seconds, and we need a bit more time than that to talk to the LLM, do some web searches, and send the Slack message.

Instead, we're calling the following function defined alongside the app, which uses Modal's [`modal.Function.spawn`](https://modal.com/docs/reference/modal.Function#spawn) feature to run a function in the background. (If you're curious what the Modal side of this function looks like, you can [jump ahead](#backgrounded-process_slack_member).)

Because `modal.py` (which we'll see in the next section) imports `app.py`, we import from `modal.py` inside the function definition because doing so at the top level would have resulted in a circular import error.

We also pass along the current Logfire context to get [Distributed Tracing](https://logfire.pydantic.dev/docs/how-to-guides/distributed-tracing/), meaning that the background function execution will show up nested under the webhook request trace, so that we have everything related to that request in one place.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/app.py" fragment="process_slack_member"}```

### Modal app

Now let's see how easy Modal makes it to deploy all of this.

#### Set up Modal

The first thing we do is define the Modal app, by specifying the base image to use (Debian with Python 3.13), all the Python packages it needs, and all of the secrets defined in the Modal interface that need to be made available during runtime.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" fragment="setup_modal"}```

#### Set up Logfire

Next, we define a function to set up Logfire instrumentation for Pydantic AI and HTTPX.

We cannot do this at the top level of the file, as the requested packages (like `logfire`) will only be available within functions running on Modal (like the ones we'll define next). This file, `modal.py`, runs on your local machine and only has access to the `modal` package.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" fragment="setup_logfire"}```

#### Web app

To deploy a [web endpoint](https://modal.com/docs/guide/webhooks) on Modal, we simply define a function that returns an ASGI app (like FastAPI) and decorate it with `@app.function()` and `@modal.asgi_app()`.

This `web_app` function will be run on Modal, so inside the function we can call the `setup_logfire` function that requires the `logfire` package, and import `app.py` which uses the other requested packages.

By default, Modal spins up a container to handle a function call (like a web request) on-demand, meaning there's a little bit of startup time to each request. However, Slack requires webhooks to respond within 3 seconds, so we specify `min_containers=1` to keep the web endpoint running and ready to answer requests at all times. This is a bit annoying and wasteful, but fortunately [Modal's pricing](https://modal.com/pricing) is pretty reasonable and they offer up to $50k in free credits for startup and academic researchers.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" fragment="web_app"}```

Note that `# type: ignore` on the `@modal.asgi_app()` line -- unfortunately `modal` does not fully define its types, so we need this to stop our static type checker `pyright`, which we run over all Pydantic AI code including examples, from complaining.

#### Scheduled `send_daily_summary`

To define a [scheduled function](https://modal.com/docs/guide/cron), we can use the `@app.function()` decorator with a `schedule` argument. This Modal function will call our imported [`send_daily_summary`](#send_daily_summary) function every day at 8 am UTC.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" fragment="send_daily_summary"}```

#### Backgrounded `process_slack_member`

Finally, we define a Modal function that wraps our [`process_slack_member`](#process_slack_member) function, so that it can run in the background.

As you'll remember from when we [spawned this function from the web app](#process_slack_member-with-modal), we passed along the Logfire context to get [Distributed Tracing](https://logfire.pydantic.dev/docs/how-to-guides/distributed-tracing/), so we need to attach it here.

```snippet {path="/examples/pydantic_ai_examples/slack_lead_qualifier/modal.py" fragment="process_slack_member"}```

## Conclusion

And that's it! Now, assuming you've met the [prerequisites](#prerequisites), you can run or deploy the app using the commands under [usage](#usage).
