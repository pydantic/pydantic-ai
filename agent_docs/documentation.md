# Documentation

> Guidance for documentation, docstrings, comments, examples, and other user-facing text

**When to check**: When writing or reviewing documentation, comments, docstrings, examples, or user-facing text

## Write for reader value

- Minimize the inference required from the reader. Preserve every useful fact, condition, consequence, limitation, and distinction.
- Test each clause, parenthetical, comparison, contrast, and explanatory tail by removing it mentally. Delete it only when the reader loses nothing useful; shorten or rewrite it when the same information can be clearer.
- Prefer direct, concrete statements and observable behavior over generic framing, unsupported promotion, vague referents, empty reassurance, or an obvious inverse.
- Preserve alternatives and negative boundaries when they prevent a plausible misunderstanding. Words such as `rather than`, `only`, `never`, and `without` often carry essential information.
- Preserve useful human voice. Change second person, passive voice, long sentences, parentheses, dashes, or colloquial language only when the specific passage becomes clearer or more accurate.
- Use one precise term for each concept across code, docs, comments, errors, and other user-facing text.

## Documentation and examples

- Help readers decide what to do and then do it correctly. Lead with the reader's action or decision when that improves findability, while retaining prerequisites and consequences.
- Make the recommended approach easy to find. Explain meaningful alternatives, trade-offs, conflicts, and negative boundaries.
- Lead with current APIs. Include deprecated or historical behavior only when readers need migration or compatibility guidance.
- Include implementation details only when they change a user decision or explain observable behavior.
- Document behavior changes in every affected user-facing surface in the same PR. Fix any conflict between documentation and implementation instead of leaving competing contracts.
- Give each maintained fact one canonical home and link to it elsewhere. Link changing provider inventories, feature lists, and setup details to their authoritative source instead of copying them.
- When a section needs a stable explicit anchor, add `{#custom-id}` to its heading and link to that ID. Use generated fragments only when they are clear and stable.
- Put user-facing features where users naturally look for them, not only in API reference docstrings.
- Use current frontier models in reader-facing examples. Verify the latest supported identifiers instead of copying static examples from this guidance.
- Use Markdown headings for real document sections. Register new published pages in `docs/navigation.yml`.
- Link to rendered Pydantic AI Harness documentation when it exists. Use the Harness repository only when no published page covers the capability.

## Docstrings

- Help users choose and correctly use public APIs. State behavior, important conditions, errors, side effects, defaults, precedence, and boundaries without repeating the signature or implementation.
- For configurable features, document the default, fallback and precedence conditions, compatibility consequences, and when users should override it.
- Format code identifiers as Markdown code or API reference links, as appropriate.
- For provider-dependent APIs, identify supported providers and explain differences that affect user choices. Do not claim mechanisms the provider does not document.

## Code comments

- Explain non-obvious intent, invariants, constraints, trade-offs, or why an obvious implementation is wrong. Do not narrate behavior that is clear from the code.
- Describe the current constraint. Keep history only when it explains a live compatibility boundary, workaround, regression risk, or otherwise surprising decision.
- Explain a workaround's intended behavior and the external constraint it compensates for. Mark future cleanup with `TODO:` and link it to a tracking issue.
- Use stable references. Link GitHub issues and PRs with full URLs, and name symbols or behavior instead of line numbers.
