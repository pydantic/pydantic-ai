"""Ready-made checks to plug into a guard.

These are plain functions returning a
[`GuardrailResult`][pydantic_ai_harness.GuardrailResult], not capabilities. That is the
difference that makes them compose: several go into one `InputGuardrail` or
`OutputGuardrail` and run as a chain, next to whatever you write yourself, sharing
one place in the capability list and one set of spans.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import InputGuardrail
from pydantic_ai_harness.guardrails.detectors import blocked_keywords, redact_secrets

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[InputGuardrail(guard=[redact_secrets, blocked_keywords(['internal-only'])])],
)
```

They read text, so they suit a prompt directly. A structured agent output is
not text, and replacing one with a scrubbed string would change its type, so
reach for [`for_text`][pydantic_ai_harness.guardrails.detectors.for_text] to
say what should happen there.

What these cannot do is worth stating. A regex finds a credential because
credentials have a shape; it does not find a prompt injection, which is
ordinary language, and it does not understand context.

That last point has a concrete cost for `email`, which matches anything shaped
like an address because that is all an address is:

```python
redact_personal_data('git clone git@github.com:pydantic/pydantic-ai.git')
# -> replaces `git@github.com`, leaving a command that no longer runs
```

An input guard rewrites the prompt in place, so the model receives the broken
version. On an agent that handles code or paths, reach for
`personal_data(only=['us_ssn', 'credit_card', 'iban'])`, or apply the detector
to the output rather than the prompt. Treat all of these as one cheap layer,
not as the answer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Literal, TypeGuard

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CachePoint, TextContent, ToolReturn, UserContent

from pydantic_ai_harness.guardrails._shared import GuardrailResult
from pydantic_ai_harness.guardrails._tool_guardrail import ToolResultInfo

TextDetector = Callable[[str], GuardrailResult]
"""A check over text. Adapt one with `for_text` or `for_tool_result_text` at non-text guard boundaries."""


def _is_text_tool_return(value: object) -> TypeGuard[ToolReturn[str]]:
    """Whether `value` is a `ToolReturn` with a string payload."""
    return isinstance(value, ToolReturn) and isinstance(value.return_value, str)


def _detect_content_text(detector: TextDetector, text: str) -> tuple[GuardrailResult | None, str | None]:
    """Run `detector` over one text part of `ToolReturn.content`.

    Returns a terminal verdict, or the replacement text -- `None` when the part is left alone.
    """
    verdict = detector(text)
    if verdict.action not in ('allow', 'replace'):
        return verdict, None
    if verdict.action == 'allow':
        return None, None
    replacement = verdict.replacement
    if not isinstance(replacement, str):
        raise UserError('A text detector used with for_tool_result_text() must replace tool content with text.')
    return None, replacement


def _text_of(part: str | TextContent) -> str:
    """The model-visible text of one `ToolReturn.content` part."""
    return part if isinstance(part, str) else part.content


def _carries_no_model_content(part: UserContent) -> bool:
    """Whether `part` puts nothing in front of the model, so the text either side of it is contiguous.

    `CachePoint` is a prompt-caching marker that providers without caching drop, so it separates
    two text parts on the page but not in what the model reads. It is the only member of
    `UserContent` like that: `str` and `TextContent` carry text, and `ImageUrl`, `AudioUrl`,
    `DocumentUrl`, `VideoUrl`, `BinaryContent` and `UploadedFile` each put a payload in the
    prompt, so those do separate the text around them.
    """
    return isinstance(part, CachePoint)


def _rewrite_text_part(part: str | TextContent, text: str) -> str | TextContent:
    """Put `text` back in `part`, keeping the part's shape and any `TextContent.metadata`."""
    return text if isinstance(part, str) else replace(part, content=text)


def _merge_span(
    run: Sequence[UserContent], text_indexes: Sequence[int], first_text: str | TextContent, text: str
) -> list[UserContent]:
    """Collapse a span's text into one part, keeping every marker on a side it still has.

    A marker before the first text part or after the last one keeps its side. The text it
    bounded is still on that side, so moving it would change what the caller asked to be
    cached for nothing. Only a marker between two merged texts has lost the split it marked,
    and that one goes ahead of the merged text, which narrows the cached prefix rather than
    widening it over text the caller kept outside.
    """
    first, last = text_indexes[0], text_indexes[-1]
    interior = [part for part in run[first:last] if not isinstance(part, str | TextContent)]
    return [*run[:first], *interior, _rewrite_text_part(first_text, text), *run[last + 1 :]]


def _detect_text_run(
    detector: TextDetector, run: Sequence[UserContent]
) -> tuple[GuardrailResult | None, Sequence[UserContent], bool]:
    """Run `detector` over one stretch of text the model reads as one span, per part and then joined.

    The model reads that span as one string, so a secret split across two parts of it
    matches nothing in either part on its own. The joined pass catches that, and it scans
    the span's *original* text, not the per-part rewrites. Redacting a fragment can strip
    the very characters a pattern anchors on -- `sk-` at the front of a key -- which would
    leave the rest of the value exposed while the joined pass reported the span clean. It
    can hide a terminal verdict the same way.

    The span keeps its parts when sanitizing them one by one already produced the text that
    sanitizing the whole span produces. Then the boundaries carry nothing, and each part
    keeps its own metadata. When the two differ, something is only reachable across a
    boundary, and the span collapses to the whole-span sanitization -- the text the same
    content would have got as a single string.

    A span holding one text part is scanned once. Its whole-span sanitization is by
    definition its per-part one, so the joined pass could only agree with itself. That
    matters beyond the wasted call: `TextDetector` is a public extension point, so a
    detector that is stateful or bills per call would otherwise see `content='x'` and
    `content=['x']` differently.

    A span can hold parts that carry no model content, which is why the joined pass keys on
    the number of text parts rather than the length of `run`.

    Returns a terminal verdict, the parts to keep, and whether any text changed.
    """
    text_parts: list[str | TextContent] = []
    text_indexes: list[int] = []
    originals: list[str] = []
    texts: list[str] = []
    replaced = False
    for index, part in enumerate(run):
        if not isinstance(part, str | TextContent):
            continue
        original = _text_of(part)
        verdict, replacement = _detect_content_text(detector, original)
        if verdict is not None:
            return verdict, run, False
        replaced |= replacement is not None
        text_parts.append(part)
        text_indexes.append(index)
        originals.append(original)
        texts.append(original if replacement is None else replacement)
    if len(texts) > 1:
        verdict, joined = _detect_content_text(detector, ''.join(originals))
        if verdict is not None:
            return verdict, run, False
        if joined is not None and joined != ''.join(texts):
            return None, _merge_span(run, text_indexes, text_parts[0], joined), True
    rewritten: list[UserContent] = []
    replacements = iter(texts)
    for part in run:
        if isinstance(part, str | TextContent):
            rewritten.append(_rewrite_text_part(part, next(replacements)))
        else:
            rewritten.append(part)
    return None, rewritten, replaced


def _detect_tool_return_content(
    detector: TextDetector, content: str | Sequence[UserContent] | None
) -> tuple[GuardrailResult | None, str | Sequence[UserContent] | None]:
    """Run `detector` over every text-bearing part of `ToolReturn.content`.

    Returns the first terminal verdict a part produced, or the rewritten content --
    `None` when no part was redacted, so the caller keeps the original shape.
    """
    if isinstance(content, str):
        return _detect_content_text(detector, content)
    if content is None:
        return None, None
    rewritten: list[UserContent] = []
    run: list[UserContent] = []
    replaced = False
    # The trailing `None` flushes the last run without repeating the flush after the loop.
    for part in [*content, None]:
        if part is not None and (isinstance(part, str | TextContent) or _carries_no_model_content(part)):
            run.append(part)
            continue
        verdict, parts, run_replaced = _detect_text_run(detector, run)
        if verdict is not None:
            return verdict, None
        rewritten.extend(parts)
        replaced |= run_replaced
        run = []
        if part is not None:
            rewritten.append(part)
    return None, rewritten if replaced else None


_NEWLINE = r'(?:\r?\n|\\+n)'
"""A line break as it reaches a detector.

A key pasted from a JSON service-account file or a `.env` line carries the two characters
a backslash and an `n` where the file had a newline. Requiring a real one left the most common way a
private key arrives in a support chat unredacted.
"""

DEFAULT_SECRET_PATTERNS: Mapping[str, str] = {
    # Vendor key bodies are base64url, whose alphabet includes `_`. A class that
    # stops at `_` redacts half a key and leaves the rest under a label saying it
    # is gone, which is worse than not matching at all.
    'anthropic_key': r'\bsk-ant-[A-Za-z0-9_-]{20,}',
    # Declared after the Anthropic shape it must not claim, and excluding it
    # explicitly so the label does not depend on declaration order.
    'openai_key': r'\bsk-(?!ant-)[A-Za-z0-9_-]{20,}',
    'aws_access_key': r'\b(?:AKIA|ASIA)[0-9A-Z]{16}\b',
    'github_token': r'\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}',
    'slack_token': r'\bxox[bporas]-[A-Za-z0-9-]{10,}',
    'slack_app_token': r'\bxapp-\d-[A-Za-z0-9-]{10,}',
    'stripe_key': r'\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}',
    'stripe_webhook_secret': r'\bwhsec_[A-Za-z0-9]{20,}',
    'google_api_key': r'\bAIza[A-Za-z0-9_-]{35}',
    'google_oauth_secret': r'\bGOCSPX-[A-Za-z0-9_-]{20,}',
    'jwt': r'\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}',
    # A PEM block, terminated or not. The newline after the header is required
    # so prose naming both markers in one sentence is not swallowed as a key.
    # The optional header lines are RFC 1421 fields (`Proc-Type`, `DEK-Info`),
    # which an encrypted key carries and whose `:` and `,` the body class stops
    # at.
    #
    # The two tails share one name rather than sitting under two, because
    # `only` selects by name: a caller narrowing to private keys would pick the
    # terminated shape, and a key pasted without its END marker -- the usual way
    # one reaches a chat window -- would then pass through unredacted under a
    # setting that says it is covered. Alternation is ordered, so a complete
    # block is claimed by the first tail and the second is reached only when no
    # END marker follows. That tail stops at the first line that is not body,
    # since running to the end of the input would delete whatever the user wrote
    # after the key.
    'private_key': (
        rf'-----BEGIN[^-\n]*PRIVATE KEY(?: BLOCK)?-----[ \t]*{_NEWLINE}'
        rf'(?:[A-Za-z-]+: [^\n]*{_NEWLINE})*'
        rf'(?:[A-Za-z0-9+/=\s\\]*?-----END[^-\n]*PRIVATE KEY(?: BLOCK)?-----'
        rf'|\s*(?:[A-Za-z0-9+/=]+(?:{_NEWLINE})?)+)'
    ),
}
"""Credential shapes worth redacting, in the order they are applied.

Prefixed and left-anchored on a word boundary, so false positives are rare: a
prefix is also a substring of ordinary text, and `sk-` without the boundary
takes the rest of `task-management-and-deployment` with it.

An AWS *secret* access key is deliberately absent. It is 40 characters of
base64 with no prefix, so nothing in the value marks it as a key; matching one
means anchoring on the name written beside it, which finds the key only where
it appears as an assignment. That narrower pattern is left to `extra` rather
than shipped as a default.
"""

DEFAULT_PII_PATTERNS: Mapping[str, str] = {
    # Before `credit_card`, whose digit groups would otherwise claim the middle
    # of a spaced IBAN and label it as a card number. The country code is drawn
    # from the IBAN registry rather than any two letters: the set is closed, and
    # anchoring on it is what keeps a build id or a patent number from reading
    # as an account. Case-insensitive, since the printed form is uppercase but
    # the standard is not. Spaces are allowed only where the printed form puts
    # them -- every fourth character -- and the whole match is checked against
    # the ISO 7064 mod-97 digit. Allowing a space before any character let the
    # pattern run across word boundaries and eat whole sentences.
    #
    # External assumption. The 89 codes below are the IBAN-adopting countries
    # and territories in the ISO 13616 IBAN Registry, which SWIFT maintains as
    # registration authority and republishes as members join. Verified
    # 2026-07-29 against
    # <https://www.swift.com/standards/data-standards/iban-international-bank-account-number>.
    # To re-check: download the registry (PDF or .txt) from that page, take its
    # `IBAN prefix country code (ISO 3166)` column, and diff it against this
    # list. A missing code means real IBANs from a new member pass through
    # unredacted, so add it and bump the date; if nothing changed, bump the date
    # alone.
    'iban': (
        r'\b(?i:AD|AE|AL|AT|AZ|BA|BE|BG|BH|BI|BR|BY|CH|CR|CY|CZ|DE|DJ|DK|DO|EE|EG|ES|FI|FK|FO|FR|GB|GE|GI|GL|GR'  # codespell:ignore fo
        r'|GT|HN|HR|HU|IE|IL|IQ|IS|IT|JO|KW|KZ|LB|LC|LI|LT|LU|LV|LY|MA|MC|MD|ME|MK|MN|MR|MT|MU|NI|NL|NO|OM|PK|PL'
        r'|PS|PT|QA|RO|RS|RU|SA|SC|SD|SE|SI|SK|SM|SO|ST|SV|TL|TN|TR|UA|VA|VG|XK)'
        r'\d{2}(?:[A-Za-z0-9]{10,30}|(?: [A-Za-z0-9]{4}){2,7}(?: [A-Za-z0-9]{1,3})?)\b'
    ),
    # 13 to 19 digits, the range ISO/IEC 7812 allows, in any grouping: fixing
    # the groups to 4-4-4-4 and Amex's 4-6-5 left a 13-digit Visa, a 14-digit
    # Diners and a 19-digit PAN unmatched, so a real card passed through.
    # Length alone would then take a millisecond timestamp with it, hence the
    # leading digit: the major industry identifier of a payment card is 2 to 6,
    # which is the same trick the IBAN country code plays above. Every match is
    # checked against the Luhn algorithm, which discards most runs of digits
    # that are not card numbers. Not all of them: roughly one in ten runs of
    # four consecutive years satisfies the checksum by chance.
    'credit_card': r'\b(?=[2-6])(?:\d[ -]?){12,18}\d\b',
    'us_ssn': r'\b\d{3}[- ]\d{2}[- ]\d{4}\b',
    # The local part is bounded and preceded by a negative lookbehind: an
    # unbounded `+` lets a failed match restart at every interior offset, which
    # makes a pasted log quadratic and blocks the event loop for seconds.
    'email': (
        r'(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,24}\b'
    ),
}
"""Personal data shapes, in the order they are applied.

Narrower than the secret set, because these overlap ordinary text. `email`
matches anything shaped like an address, which includes `git@github.com` in a
shell command -- see the module docstring on where that matters.
"""

_REDACTED = '[redacted:{name}]'


def _luhn(digits: str) -> bool:
    """Whether a run of digits satisfies the Luhn checksum every card number does."""
    values = [int(character) for character in digits if character.isdigit()][::-1]
    total = sum(value if index % 2 == 0 else sum(divmod(value * 2, 10)) for index, value in enumerate(values))
    return len(values) >= 12 and total % 10 == 0


def _iban_mod97(candidate: str) -> bool:
    """Whether a candidate satisfies the ISO 7064 mod-97 check every IBAN carries.

    The country code narrows the prefix; nothing narrowed the body, so any `XX##` token
    followed by ten more letters or digits read as an account number -- `RS232 serial cable
    adapter and reboot` among them. A shape this loose needs the checksum the standard
    defines, the same way `credit_card` needs Luhn.
    """
    compact = ''.join(candidate.split()).upper()
    if not 15 <= len(compact) <= 34:
        return False
    rotated = compact[4:] + compact[:4]
    return int(''.join(str(int(character, 36)) for character in rotated)) % 97 == 1


_VALIDATORS: Mapping[str, Callable[[str], bool]] = {'credit_card': _luhn, 'iban': _iban_mod97}
"""Semantic checks that a shape alone cannot make. A name absent here is taken on its shape."""


def _compile(
    patterns: Mapping[str, str], only: Iterable[str] | None, extra: Mapping[str, str] | None
) -> tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...]:
    """The patterns a detector will run, as `(name, compiled, validator)` triples.

    A validator belongs to the built-in pattern it was written for, not to its
    name. A custom pattern supplied under a built-in name -- possible once
    `only` has dropped that built-in -- would otherwise be judged by a check
    written for different text and silently never match.
    """
    if only is None:
        selected = dict(patterns)
    else:
        # `only` filters, it does not reorder. Both mappings document an application order
        # that other patterns depend on -- `iban` before `credit_card` is what stops a spaced
        # account number being labelled a card -- and building the dict by iterating `only`
        # handed that order to the caller's argument order instead.
        requested = set(only)
        for name in requested:
            if name not in patterns:
                raise UserError(f'Unknown pattern {name!r}; available: {sorted(patterns)}.')
        selected = {name: pattern for name, pattern in patterns.items() if name in requested}
    if extra:
        clashes = sorted(set(extra) & set(selected))
        if clashes:
            raise UserError(
                f'`extra` would replace the built-in pattern(s) {clashes} rather than add to them. '
                'Rename them, or pass `only` to drop the built-in first.'
            )
        selected.update(extra)
    if not selected:
        raise UserError('This detector was given no patterns, so it would never match anything.')
    overridden = set(extra or ())
    return tuple(
        (name, re.compile(pattern), None if name in overridden else _VALIDATORS.get(name))
        for name, pattern in selected.items()
    )


def _require_text(value: object) -> str:
    """Return `value` as text, naming the mistake when it is not.

    Every detector takes `object` rather than `str`, because one plugged
    straight into an `OutputGuardrail` is handed the agent output unchanged.
    Checking here says which knob answers that, instead of letting `re` raise a
    TypeError from three frames down.
    """
    if not isinstance(value, str):
        raise UserError(
            f'A text detector received {type(value).__name__}, which it cannot scan. An OutputGuardrail '
            'hands the guard the agent output unchanged, so wrap the detector in for_text() to say '
            'what should happen when that output is not text.'
        )
    return value


def _redactor(
    compiled: tuple[tuple[str, re.Pattern[str], Callable[[str], bool] | None], ...], placeholder: str
) -> TextDetector:
    """A detector that rewrites every match and allows text with none."""

    def detect(text: object) -> GuardrailResult:
        text = _require_text(text)
        cleaned = text
        for name, pattern, valid in compiled:
            substitution = placeholder.replace('{name}', name)

            def replace(match: re.Match[str], valid: Callable[[str], bool] | None = valid, out: str = substitution):
                # A function rather than a template string: `re.sub` reads
                # backreferences in a replacement, so a placeholder containing
                # `\g<0>` would re-emit the very text being redacted.
                return out if valid is None or valid(match.group()) else match.group()

            cleaned = pattern.sub(replace, cleaned)
        return GuardrailResult.replace(cleaned) if cleaned != text else GuardrailResult.allow()

    return detect


def secret_data(
    *,
    only: Iterable[str] | None = None,
    extra: Mapping[str, str] | None = None,
    placeholder: str = _REDACTED,
) -> TextDetector:
    """Build a detector that rewrites credentials out of text.

    Redacting rather than refusing is the useful default here: an agent that
    quotes a key back has still done the work, and blocking the answer loses it
    while leaving the key in the message history either way.

    Args:
        only: Restrict to these names from `DEFAULT_SECRET_PATTERNS`.
        extra: Additional `name -> regex` patterns to include.
        placeholder: What each match is replaced with. `{name}` in it becomes
            the pattern that matched, so a redaction says what it removed.
    """
    return _redactor(_compile(DEFAULT_SECRET_PATTERNS, only, extra), placeholder)


def personal_data(
    *,
    only: Iterable[str] | None = None,
    extra: Mapping[str, str] | None = None,
    placeholder: str = _REDACTED,
) -> TextDetector:
    """Build a detector that rewrites personal data out of text.

    Redaction, not refusal. An email address in a prompt is usually the user
    telling the agent something it needs, so blocking the turn would break
    ordinary use; removing it before it reaches the model does not.

    Args and behaviour match
    [`secret_data`][pydantic_ai_harness.guardrails.detectors.secret_data],
    over `DEFAULT_PII_PATTERNS`.
    """
    return _redactor(_compile(DEFAULT_PII_PATTERNS, only, extra), placeholder)


redact_secrets = secret_data()
"""`secret_data()` with every default pattern, for the common case."""

redact_personal_data = personal_data()
"""`personal_data()` with every default pattern, for the common case."""


def blocked_keywords(
    keywords: Iterable[str],
    *,
    case_sensitive: bool = False,
    whole_words: bool = False,
    message: str | None = None,
) -> TextDetector:
    """Build a detector that refuses text containing any of `keywords`.

    Args:
        keywords: Literal strings to look for. They are escaped, so a keyword
            with regex punctuation matches itself.
        case_sensitive: Match case exactly. Off by default.
        whole_words: Require word boundaries, so `class` stops matching inside
            `classification`.
        message: The refusal text. The default names the keyword that matched,
            which is useful to an operator and visible to the model, so pass
            something neutral when the list itself is sensitive.
    """
    if isinstance(keywords, str):
        raise UserError(
            'blocked_keywords() was given a single string, which would be one keyword per character and '
            f'block nearly every input. Pass a sequence: blocked_keywords([{keywords!r}]).'
        )
    flags = 0 if case_sensitive else re.IGNORECASE
    terms = tuple(keywords)
    if any(not keyword for keyword in terms):
        raise UserError('blocked_keywords() was given an empty keyword, which would match every input.')
    compiled = tuple(
        # `(?<!\w)`/`(?!\w)` rather than `\b`: a boundary needs a word character on
        # one side, so `\bC\+\+\b` never matches `C++` and the keyword would be
        # silently inert.
        re.compile(rf'(?<!\w){re.escape(keyword)}(?!\w)' if whole_words else re.escape(keyword), flags)
        for keyword in terms
    )
    if not compiled:
        raise UserError('blocked_keywords() was given no keywords, so it would never match anything.')

    def detect(text: object) -> GuardrailResult:
        text = _require_text(text)
        for pattern in compiled:
            match = pattern.search(text)
            if match:
                return GuardrailResult.block(message or f'Blocked term: {match.group()!r}.')
        return GuardrailResult.allow()

    return detect


def for_text(
    detector: TextDetector, *, on_other: Literal['raise', 'allow'] = 'raise'
) -> Callable[[object], GuardrailResult]:
    """Adapt a text detector to a guard over a value that may not be text.

    An `OutputGuardrail` receives the agent output unchanged, which for a structured
    output is a model instance rather than a string. Substituting a scrubbed
    string for it would change the output's type, so this refuses to guess:
    `'raise'` fails loudly, telling you to apply the detector to a field
    instead, and `'allow'` lets non-text through when you know the other output
    types carry nothing to check.
    """

    def guard(value: object) -> GuardrailResult:
        if isinstance(value, str):
            return detector(value)
        if on_other == 'allow':
            return GuardrailResult.allow()
        raise UserError(
            f'A text detector received {type(value).__name__}, which it cannot rewrite without changing the '
            "output's type. Apply it to a field of the output, or pass on_other='allow' to skip non-text."
        )

    return guard


def for_tool_result_text(
    detector: TextDetector, *, on_other: Literal['raise', 'allow'] = 'raise'
) -> Callable[[ToolResultInfo], GuardrailResult]:
    """Adapt a text detector to the result object `ToolGuardrail` supplies.

    Plain string results are passed to `detector`. For a `ToolReturn` with a
    string `return_value`, the detector also checks the text in its
    model-directed `content`, both part by part and over each stretch of
    adjacent text parts joined, so a secret split across two of them is caught.
    The `ToolReturn` is rebuilt when either channel is redacted, retaining its
    `metadata`, `kind`, and non-text content.

    A non-text result has no safe text replacement. The default raises with the
    adapter name. `on_other='allow'` skips the whole result rather than part of
    it: neither `return_value` nor `content` is scanned, so a `ToolReturn` with a
    structured `return_value` reaches the model with its `content` untouched.
    Leave `on_other` at its default and apply the detector to a field of the
    output when a skipped result may carry sensitive `content`.
    """

    def guard(info: ToolResultInfo) -> GuardrailResult:
        result = info.result
        if isinstance(result, str):
            verdict = detector(result)
            if verdict.action == 'replace' and not isinstance(verdict.replacement, str):
                raise UserError(
                    'A text detector used with for_tool_result_text() must replace a tool result with text.'
                )
            return verdict
        if _is_text_tool_return(result):
            assert isinstance(result.return_value, str)
            verdict = detector(result.return_value)
            if verdict.action not in ('allow', 'replace'):
                return verdict
            return_value = verdict.replacement if verdict.action == 'replace' else result.return_value
            if not isinstance(return_value, str):
                raise UserError(
                    'A text detector used with for_tool_result_text() must replace a tool result with text.'
                )

            content_verdict, content = _detect_tool_return_content(detector, result.content)
            if content_verdict is not None:
                return content_verdict

            if verdict.action == 'replace' or content is not None:
                return GuardrailResult.replace(
                    replace(
                        result,
                        return_value=return_value,
                        content=result.content if content is None else content,
                    )
                )
            return GuardrailResult.allow()
        if on_other == 'allow':
            return GuardrailResult.allow()
        raise UserError(
            f'for_tool_result_text() received {type(result).__name__}, which has no text payload to rewrite. '
            "Pass on_other='allow' to skip non-text tool results."
        )

    return guard
