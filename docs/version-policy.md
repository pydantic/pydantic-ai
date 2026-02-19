## Version Policy

We will not intentionally make breaking changes in minor releases of V1. V2 will be released in April 2026 at the earliest, 6 months after the release of V1 in September 2025.

Once we release V2, we'll continue to provide security fixes for V1 for another 6 months minimum, so you have time to upgrade your applications.

Functionality marked as deprecated will not be removed until V2.

Of course, some apparently safe changes and bug fixes will inevitably break some users' code &mdash; obligatory link to [xkcd](https://xkcd.com/1172/).

The following changes will **NOT** be considered breaking changes, and may occur in minor releases:

* Bug fixes that may result in existing code breaking, provided that such code was relying on undocumented features/constructs/assumptions.
* Adding new [message parts][pydantic_ai.messages], [stream events][pydantic_ai.messages.AgentStreamEvent], or optional fields (including fields with default values) on existing message (part) and event types. Always code defensively when consuming message parts or event streams, and use the [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter] to (de)serialize message histories.
* Changing OpenTelemetry span attributes. Because different [observability platforms](logfire.md#using-opentelemetry) support different versions of the [OpenTelemetry Semantic Conventions for Generative AI systems](https://opentelemetry.io/docs/specs/semconv/gen-ai/), Pydantic AI lets you configure the [instrumentation version](logfire.md#configuring-data-format), but the default version may change in a minor release. Span attributes for [Pydantic Evals](evals.md) may also change as we iterate on Evals support in [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/).
* Changing how `__repr__` behaves, even of public classes.

In all cases we will aim to minimize churn and do so only when justified by the increase of quality of Pydantic AI for users.

## Beta Features

At Pydantic, we like to move quickly and innovate! To that end, minor releases may introduce beta features (indicated by a `beta` module) that are active works in progress. While in its beta phase, a feature's API and behaviors may not be stable, and it's very possible that changes made to the feature will not be backward-compatible. We aim to move beta features out of beta within a few months after initial release, once users have had a chance to provide feedback and test the feature in production.

## Support for Python versions

Pydantic will drop support for a Python version when the following conditions are met:

* The Python version has reached its [expected end of life](https://devguide.python.org/versions/).
* less than 5% of downloads of the most recent minor release are using that version.
