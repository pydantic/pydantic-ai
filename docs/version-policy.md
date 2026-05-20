## Version Policy

We will not intentionally make breaking changes in minor releases of V1, which was released in September 2025.

Functionality marked as deprecated in a V1 release will not be removed until V2.

Once V2 is released as stable, we'll continue to provide security fixes for V1 for another 6 months minimum, so you have time to upgrade your applications.

## V2 Beta

V2 is now available as a beta pre-release (`pip install pydantic-ai==2.0.0bN` / `uv add pydantic-ai==2.0.0bN`). It collects the breaking changes and behavior changes that we couldn't make under the V1 stability guarantee, alongside a more capable, more coherent foundation for building agents.

During the beta period the V2 API and behaviors are not yet covered by the stability guarantee above: while we don't expect major changes, we may still make adjustments in response to feedback before the stable V2.0 release. We encourage you to try the beta, [report issues](https://github.com/pydantic/pydantic-ai/issues), and pin an exact pre-release version (`==2.0.0bN`) rather than a range.

### Upgrading to V2

To make the upgrade as smooth as possible, we recommend the following path:

1. **Upgrade to the latest V1 release** (`pydantic-ai<2`) first.
2. **Resolve every deprecation warning.** Most of V2's breaking changes were announced in V1 via deprecation warnings that name the new API and, where possible, include a migration snippet. Running your test suite (or app) with warnings visible and addressing each one migrates you across the bulk of V2 ahead of time.
3. **Upgrade to V2** and address the remaining changes that could not be pre-announced via a deprecation — primarily default-behavior changes and a handful of removals that have no V1 deprecation. These are listed explicitly in the [Upgrade Guide](changelog.md), separated from the deprecation-driven changes so you know what still needs your attention.

Jumping straight from an older V1 to V2 without first resolving deprecation warnings is possible but will be considerably more work, since you'll be reconstructing the migration guidance the warnings would have given you.

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
