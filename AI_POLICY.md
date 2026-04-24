# AI Policy

We support using AI (LLMs, coding assistants, agents) as tools for the work that goes into Pydantic AI — we use them ourselves. But Pydantic AI is foundational infrastructure for a lot of people, and we hold a high bar for every contribution. This page sets expectations for where AI fits and where it doesn't.

## You are responsible for what you submit

Whatever tools you used to produce it, **you** are the author of every issue, PR, and comment under your account. You stand by it, you understand it, and you can answer questions about it. We are responsible for anything we merge and release.

The PR checklist already requires that any AI-generated code has been reviewed line-by-line by the human PR author — that same standard applies to everything else you submit, not just code.

## Write to us in your own words

**Do not use AI to generate issues, PR descriptions, or replies to maintainer questions.**

This isn't about style — it's about the signal we need to do our jobs. We may hide or close contributions we believe were AI-generated at this layer, even if the underlying code or idea is sound.

- **Issues**: we need to understand the problem *you* have — what you're building, what broke, what you tried. A generic LLM summary of a problem space is not the same thing, and doesn't help us prioritize.
- **PR descriptions**: we need to know what *you* decided and why. Your PR body and review replies are how we calibrate whether you're the right [champion](CONTRIBUTING.md#champions) for the change — a champion who can't explain their own PR in their own words isn't one.
- **Review replies**: we need your judgment, not the model's. Copying AI responses back into a review thread means we have to re-ask to find out what you actually think.

If you want to include context from an AI interaction, put it in a quote block (`>`), disclose it, keep it short, and add your own commentary explaining the relevance. Long pasted AI snippets will be ignored.

## No autonomous-agent contributions

**Do not open issues or PRs that an agent produced end-to-end without a human reading and standing by the full submission.**

The distinction isn't "did you use Claude Code to write some of this" — it's whether a human is in the loop who understands the submission and can answer for it. We will close issues and PRs we believe were created autonomously.

## Translation is fine

If English isn't your first language, it's completely OK to use AI to translate or polish your comments. Take the time to make sure the result reflects *your* voice and ideas. If you'd rather, write in your native language and include the AI translation in a quote block — either works.
