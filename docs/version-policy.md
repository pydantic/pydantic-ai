## Version Policy

Pydantic AI V1 was released in September 2025. The first V2 beta (`v2.0.0b1`) was released on May 20, 2026, with a stable V2.0 expected within roughly two weeks — see [V2 Beta](#v2-beta) below.

We will not intentionally make breaking changes in minor releases. Functionality marked as deprecated in a release is not removed until the next major version, which we won't release sooner than 3 months after V2.0 ships stable.

We'll continue to provide security fixes for V1 for at least 6 months after V2's stable release, so you have time to upgrade your applications. See [Upgrading to V2](#upgrading-to-v2) for the recommended path.

## V2 Beta

V2 is available as a beta pre-release. It collects the breaking and behavior changes V1's stability guarantee didn't allow, and leans into a harness-first design: [capabilities](capabilities.md) are now a core primitive — a single, composable unit that bundles an agent's tools, [hooks](hooks.md), instructions, and model settings. Pydantic AI stays a small core: some capabilities ship with it, more come from the first-party [Pydantic AI Harness](harness/overview.md), and others are third-party or your own.

To install it, pin the exact pre-release version. Find the current beta on [PyPI](https://pypi.org/project/pydantic-ai/#history) or the [GitHub releases page](https://github.com/pydantic/pydantic-ai/releases), then (replacing `bN` with that version):

```bash
pip/uv-add "pydantic-ai==2.0.0bN"
```

During the beta the V2 API and behaviors aren't yet covered by the stability guarantee above — we don't expect major changes but may still adjust in response to feedback before the stable V2.0 release. Please [try it and report issues](https://github.com/pydantic/pydantic-ai/issues), or reach out in the `#pydantic-ai` channel on [Slack](help.md#slack).

<!-- TODO(v2-launch): once V2.0 is stable, update the beta/GA dates in the Version Policy section above, replace this section with standard install instructions (drop the pre-release pinning), and fold the capabilities framing into the release announcement instead. -->

### Upgrading to V2

To make the upgrade as smooth as possible, we recommend the following path:

1. **Upgrade to the latest V1 release.** Most of what V2 removes is deprecated as of **v1.100.0** (the release V2.0.0b1 is forked from), so any V1 at or above that version surfaces those warnings.
2. **Resolve every deprecation warning.** [Most of V2's breaking changes](changelog.md#changes-covered-by-deprecation-warnings) were announced in V1 via deprecation warnings that name the new API and, where possible, include a migration snippet. Running your test suite (or app) with warnings visible and addressing each one — by hand or by pointing a coding agent at them — migrates you across the bulk of V2 ahead of time.
3. **Upgrade to V2** and make the [changes that couldn't be pre-announced](changelog.md#changes-not-covered-by-deprecation-warnings) via a deprecation — primarily default-behavior changes and a handful of removals with no V1 deprecation.

You can also upgrade straight to V2 and work through the [Upgrade Guide](changelog.md) directly — it's organized so a coding agent can apply the code changes mechanically. Resolving deprecation warnings on the latest V1 first is still the smoother path, since it spreads the work out and leaves you only the behavior changes to reason about consciously at the end.

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
