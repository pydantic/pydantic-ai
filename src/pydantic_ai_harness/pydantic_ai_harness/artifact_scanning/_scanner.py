"""Pattern scanning for agent-written artifacts that later sessions load into context."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias, get_args

from opentelemetry.trace import Tracer

from pydantic_ai.exceptions import UserError

ArtifactThreat: TypeAlias = Literal[
    'injection', 'role_override', 'exfiltration', 'html_comment', 'invisible_unicode', 'custom'
]
"""The kind of attack a finding points at.

- `'injection'`: text telling the model to drop or replace its instructions.
- `'role_override'`: text giving the model a new identity, or forging a chat-template role marker.
- `'exfiltration'`: a command or URL that sends credentials or credential files somewhere.
- `'html_comment'`: an HTML comment, which Markdown renderers hide from the person reviewing the file.
- `'invisible_unicode'`: zero-width, bidirectional-control, or tag characters a person cannot see.
- `'custom'`: a pattern passed in `ArtifactScanner(extra_patterns=...)`.
"""

_BUILT_IN_THREATS: frozenset[ArtifactThreat] = frozenset(
    threat for threat in get_args(ArtifactThreat) if threat != 'custom'
)

_SECRET_NAME = r'(?-i:[A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?)[A-Z0-9_]*)'
"""An environment variable named like a credential. Case-sensitive, so `$monkey` is not a key."""

_CREDENTIAL_FILE = (
    r'(?:\.env(?!\.(?:example|sample|template|dist)\b)(?:\.[\w-]{1,40}){0,3}(?![\w-])|\.ssh/|\.aws/|\.netrc\b|\.npmrc\b|\.pypirc\b|\.git-credentials\b|\.kube/config\b'
    r'|\.docker/config\.json\b|\bid_rsa\b|\bid_ed25519\b)'
)

DEFAULT_ARTIFACT_PATTERNS: Mapping[str, tuple[ArtifactThreat, str]] = {
    'ignore_instructions': (
        'injection',
        r'\b(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?(?:of\s+)?(?:your\s+|the\s+)?'
        r'(?:previous|prior|above|earlier|preceding|original|system)\s+'
        r'(?:instructions?|prompts?|rules|directives|guidelines|messages)\b',
    ),
    'replace_system_prompt': (
        'injection',
        r'\b(?:new|updated|real|actual)\s+system\s+(?:prompt|instructions?)\b|\bsystem\s+prompt\s+override\b',
    ),
    'new_identity': (
        'role_override',
        r'\byou\s+are\s+(?:now|no\s+longer)\s+(?:a|an|the|in|my)\b|\b(?:adopt|assume)\s+(?:a|the)\s+new\s+persona\b',
    ),
    'chat_template_token': (
        'role_override',
        r'<\|(?:im_start|im_end|system|user|assistant|endoftext|start_header_id|end_header_id|eot_id)\|>'
        r'|\[/?INST\]|<</?SYS>>',
    ),
    'role_label': ('role_override', r'(?m)^[ \t]*(?:system|assistant|developer)[ \t]*:'),
    'secret_in_request': (
        'exfiltration',
        rf'\b(?:curl|wget|nc|ncat|Invoke-WebRequest|iwr)\b[^\n]{{0,300}}\$\{{?{_SECRET_NAME}',
    ),
    'secret_in_url': ('exfiltration', rf'\bhttps?://[^\s\'"<>]{{0,300}}\$\{{?{_SECRET_NAME}'),
    'read_credential_file': (
        'exfiltration',
        rf'\b(?:cat|less|head|tail|base64|xxd|strings|scp)\s+(?:-\S+\s+(?:\d+\s+)?){{0,5}}\S{{0,100}}?{_CREDENTIAL_FILE}',
    ),
    'environment_to_network': ('exfiltration', r'\b(?:env|printenv)\b\s*\|\s*(?:curl|wget|nc|ncat)\b'),
    'ssh_key_injection': ('exfiltration', r'>>?\s*\S{0,100}authorized_keys\b'),
    'html_comment': ('html_comment', r'<!--'),
    'invisible_character': (
        'invisible_unicode',
        '[\u180e\u200b\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\U000e0000-\U000e007f]',
    ),
}
"""The built-in patterns as `name -> (threat, regex)`, matched case-insensitively.

Every pattern except `invisible_character` runs on the text after NFKC normalization and with
invisible and joining characters removed, so full-width letters or a zero-width space inside a
phrase do not hide it. `invisible_character` leaves out the characters ordinary text needs: the
zero-width joiner and non-joiner (emoji sequences, Persian and Indic scripts), the left-to-right
and right-to-left marks, the soft hyphen, and the tag characters of the England, Scotland, and Wales flags.
Repetition is bounded so a long artifact cannot make a pattern backtrack for seconds.

These are a cheap first layer. Prompt injection is ordinary language, so a phrase list misses
paraphrases and can flag notes that quote an attack, such as a memory recording that a web page
said "ignore previous instructions". Narrow `ArtifactScanner(threats=...)` where a category reads
too much ordinary text as a threat.
"""

_INVISIBLE = re.compile(DEFAULT_ARTIFACT_PATTERNS['invisible_character'][1])
_HIDDEN = re.compile('[\u00ad\u200c-\u200f]|' + _INVISIBLE.pattern)
"""What is removed before the phrase patterns run: everything `invisible_character` flags, plus joiners and marks."""
_FLAG_EMOJI_TAGS = re.compile(
    '\U0001f3f4\U000e0067\U000e0062(?:\U000e0065\U000e006e\U000e0067|\U000e0073\U000e0063\U000e0074|\U000e0077\U000e006c\U000e0073)'
    '\U000e007f'
)
"""The flags of England, Scotland, and Wales, the only subdivision flags emoji fonts draw.

Each is a black flag, tag letters naming the region, and a cancel tag. Accepting any tag letters there would let
text be hidden in tag characters inside fake flags.
"""
_MAX_EXCERPT_CHARS = 80


@dataclass(frozen=True)
class ArtifactFinding:
    """One pattern match in a scanned artifact."""

    threat: ArtifactThreat
    """The kind of attack the pattern points at."""

    pattern: str
    """The name of the pattern that matched: a key of `DEFAULT_ARTIFACT_PATTERNS` or of `extra_patterns`."""

    excerpt: str
    """The matched text, at most 80 characters. An invisible character is written as its code point, like `U+200B`."""


@dataclass(frozen=True)
class ArtifactScanResult:
    """What scanning one artifact found."""

    source: str
    """What was scanned, as the caller named it, such as `'memory:MEMORY.md'` or `'skill:code-review'`."""

    findings: tuple[ArtifactFinding, ...] = ()
    """Every match, in pattern order. Empty when nothing matched, which does not mean the artifact is safe."""

    @property
    def flagged(self) -> bool:
        """Whether any pattern matched."""
        return bool(self.findings)

    @property
    def threats(self) -> tuple[ArtifactThreat, ...]:
        """The distinct threats found, sorted."""
        return tuple(sorted({finding.threat for finding in self.findings}))

    def summary(self) -> str:
        """The threats and pattern names found, without quoting the artifact, so it can be shown to the model."""
        by_threat: dict[str, list[str]] = {}
        for finding in self.findings:
            names = by_threat.setdefault(finding.threat, [])
            if finding.pattern not in names:
                names.append(finding.pattern)
        return ', '.join(f'{threat} ({", ".join(names)})' for threat, names in sorted(by_threat.items()))


@dataclass(frozen=True)
class ArtifactScanner:
    """Scan agent-written artifacts, such as memory files and skills, for attack patterns.

    Pass one to `Memory(scanner=...)` or `Skills(scanner=...)`, or call `scan` on any text before an
    application stores it or loads it into a prompt. To use another detector, subclass and override
    `scan`: the capabilities only call `scan` and read `ArtifactScanResult.flagged`.
    """

    threats: Collection[ArtifactThreat] = field(default=_BUILT_IN_THREATS)
    """The built-in threat categories to look for, all of them by default."""

    extra_patterns: Mapping[str, str] = field(default_factory=dict[str, str], hash=False)
    """Additional `name -> regex` patterns, reported under the `'custom'` threat and matched case-insensitively."""

    _compiled: tuple[tuple[str, ArtifactThreat, re.Pattern[str]], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        # Frozen, so the patterns compiled here always match the configuration they were compiled from.
        object.__setattr__(self, 'threats', frozenset(self.threats))
        object.__setattr__(self, 'extra_patterns', MappingProxyType(dict(self.extra_patterns)))
        unknown = sorted(set(self.threats) - _BUILT_IN_THREATS)
        if unknown:
            raise UserError(f'Unknown artifact threat(s) {unknown}; available: {sorted(_BUILT_IN_THREATS)}.')
        clashes = sorted(set(self.extra_patterns) & set(DEFAULT_ARTIFACT_PATTERNS))
        if clashes:
            raise UserError(f'`extra_patterns` reuses built-in pattern name(s) {clashes}; rename them.')
        compiled: list[tuple[str, ArtifactThreat, re.Pattern[str]]] = [
            (name, threat, re.compile(regex, re.IGNORECASE))
            for name, (threat, regex) in DEFAULT_ARTIFACT_PATTERNS.items()
            if threat in self.threats
        ]
        for name, regex in self.extra_patterns.items():
            try:
                compiled.append((name, 'custom', re.compile(regex, re.IGNORECASE)))
            except re.error as error:
                raise UserError(f'`extra_patterns[{name!r}]` is not a valid regular expression: {error}') from error
        if not compiled:
            raise UserError('This scanner was given no threats or patterns, so it would never flag anything.')
        object.__setattr__(self, '_compiled', tuple(compiled))

    def scan(self, content: str, *, source: str = '') -> ArtifactScanResult:
        """Scan `content` and return every finding.

        Args:
            content: The artifact's text.
            source: A label for what is scanned, carried on the result for logs and telemetry.
        """
        # A byte-order mark opening a file is how some editors save UTF-8, not a hidden character.
        raw = content.removeprefix('\ufeff')
        visible = _FLAG_EMOJI_TAGS.sub('', raw)
        normalized = _HIDDEN.sub('', unicodedata.normalize('NFKC', raw))
        findings: list[ArtifactFinding] = []
        for name, threat, pattern in self._compiled:
            text = visible if threat == 'invisible_unicode' else normalized
            findings.extend(
                ArtifactFinding(threat=threat, pattern=name, excerpt=_excerpt(match.group()))
                for match in pattern.finditer(text)
            )
        return ArtifactScanResult(source=source, findings=tuple(findings))


def _excerpt(text: str) -> str:
    visible = ''.join(f'U+{ord(char):04X}' if _HIDDEN.fullmatch(char) else char for char in text)
    return visible[:_MAX_EXCERPT_CHARS]


def record_flagged(
    tracer: Tracer,
    result: ArtifactScanResult,
    *,
    artifact: Literal['memory', 'skill'],
    action: Literal['rejected_write', 'withheld', 'skipped'],
    include_content: bool,
) -> None:
    """Mark the decision taken on a flagged artifact with a zero-duration span.

    The artifact's name and the matched text are only recorded with `include_content`: a memory filename is
    chosen by the model, and the matches are the artifact's content.
    """
    span = tracer.start_span('artifact_scan flagged')
    if span.is_recording():
        attributes: dict[str, str | int | list[str]] = {
            'artifact_scan.artifact': artifact,
            'artifact_scan.action': action,
            'artifact_scan.threats': list(result.threats),
            'artifact_scan.patterns': sorted({finding.pattern for finding in result.findings}),
            'artifact_scan.findings': len(result.findings),
        }
        if include_content:
            attributes['artifact_scan.source'] = result.source
            attributes['artifact_scan.excerpts'] = [finding.excerpt for finding in result.findings]
        span.set_attributes(attributes)
    span.end()
