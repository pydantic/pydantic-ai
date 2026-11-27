# Upgrade notes

Map what changes between the installed Pydantic AI version and the target version (the latest, unless the user names one) to the code you inventoried. The source is the [upgrade guide](https://pydantic.dev/docs/ai/project/changelog/), which is published for the latest release. When the target is an older release, read that release's copy, `https://raw.githubusercontent.com/pydantic/pydantic-ai/v<target>/docs/changelog.md`, or use the published guide and ignore releases newer than the target.

## Installed on V1 (1.x)

Moving to V2 is a migration, not a catch-up. Follow the guide's recommended path in its [v2.0.0b1 entry](https://pydantic.dev/docs/ai/project/changelog/#v200b1-2026-05-20):

1. Upgrade to the latest V1 release and resolve every deprecation warning. The warnings name the new API, so the test suite run with warnings visible finds most of the work.
2. Upgrade to V2 and make the [changes not covered by deprecation warnings](https://pydantic.dev/docs/ai/project/changelog/#changes-not-covered-by-deprecation-warnings).

Report the changes from that list whose APIs the repository uses, then continue below from v2.0.0. The release-notes pass below covers the remaining 1.x releases too.

## Installed on V2

1. Read [Catching up across V2 minor releases](https://pydantic.dev/docs/ai/project/changelog/#catching-up-across-v2-minor-releases). It lists, newest first, every compatibility note since v2.0.0, under the release that shipped it, with a link to its pull request. Take the notes from releases newer than the installed version, up to the target.
2. Releases v2.1.0 through v2.32.2 predate the compatibility-notes section: when the installed version is older than v2.33.0, also read those releases' notes on [GitHub Releases](https://github.com/pydantic/pydantic-ai/releases).
3. Read [Breaking Changes](https://pydantic.dev/docs/ai/project/changelog/#breaking-changes) for the releases in the same range: a few changes, such as the v2.60.0 message-history change, are explained there in more depth.
4. Read [Deprecations](https://pydantic.dev/docs/ai/project/changelog/#deprecations). Deprecated APIs keep working until V3 and warn with `PydanticAIDeprecationWarning`, which names the replacement; report each one the repository uses.

## Map each note to the code

For each note, search the repository for the APIs, settings, message parts, or provider it names (in its code, and in its docs, notebooks and docstring examples, which you report as docs), and confirm at the call site that the code relies on the old behavior:

- **Affects this code:** the note, its release and pull request, the `file:line` it touches, and what to change.
- **Doesn't apply:** group these in one line with the reason (provider not used, API not imported), so the user can see they were checked.
- **Can't tell:** behavior that depends on data or configuration outside the repository, such as stored message histories, durable workflows paused across the upgrade, or a provider's account settings. Put these under blind spots.

Changes that affect stored data come first: message histories saved before the upgrade, and durable workflows that are mid-run when the new version deploys. Say whether the repository stores either.

A change that already applies at the installed version (an API removed or deprecated in a release the repository is already on) goes in this section too, under the release that made it, with what to change.

## Private members

Release notes describe public API only, so they never mention a private method that was removed or changed. For each private attribute or method the repository uses or overrides on a framework class (found by the "reaching into internals" shape), check that it still exists with the same signature at the target version, for example by reading the class at `https://github.com/pydantic/pydantic-ai/tree/v<version>` or by importing it in an isolated environment. When it changed, find the first release where it did by checking releases in between, and report it under that release as a compatibility change, with the public API that replaces it.

## Read the release notes for what the code could use

Each release's notes on [GitHub Releases](https://github.com/pydantic/pydantic-ai/releases) list its changes, one line per pull request, under **Compatibility Notes**, **Features** and **Bug Fixes**. Fetch the releases between the installed version and the target, for example from `https://api.github.com/repos/pydantic/pydantic-ai/releases?per_page=100` (paged; newest first). Pydantic AI Harness releases from before it joined that repository, up to 0.36.0, are on [its own releases page](https://github.com/pydantic/pydantic-ai-harness/releases).

1. **Compatibility Notes:** check every one, as above.
2. **Features and Bug Fixes:** a jump across many releases holds hundreds of lines, most about providers, integrations or packages the repository doesn't use. Keep only the lines that name something on the surface from the inventory (a provider it calls, a module it imports, an integration it has) or that fix a symptom its code works around. Then search the code for what each kept line would replace or change, and confirm at the call site. Each line links its pull request: when the pull request's description has a line starting `Replaces:`, it names the hand-written workaround the change replaces, so search the code for that.
3. Report each finding under "After upgrading to X", grouped by the release that brings it. A bug fix that removes a failure the code hits today is a correctness finding and ranks first. Decide whether it hits today from the repository's locked versions (of Pydantic AI and of the provider SDK the fix is about), not from whether it could: say "fails today with the locked `<sdk>` <version>", or say why it doesn't.

## Find the release that first has an API

For an adoption finding whose replacement the repository can't import yet, the release that brings it is the earliest release whose notes name it. Search the release notes, newest first, for the API's name, and take the oldest match; then confirm it on that release's docs or source at `https://github.com/pydantic/pydantic-ai/tree/v<version>`. A note can name an API after it shipped, for example when it deprecates the old one in its favor: if the API is already in the source of the release before the match, check earlier releases until you find the first that has it. If no note names it, say "in the latest release" rather than guessing an older one.
