"""Tests for the ready-made detectors and for chaining several guards."""

from __future__ import annotations

import time

import pytest
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelMessage,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext

from pydantic_ai_harness import GuardrailResult, InputGuardrail, OutputBlocked, OutputGuardrail
from pydantic_ai_harness.guardrails import ToolGuardrail, ToolResultInfo
from pydantic_ai_harness.guardrails.detectors import (
    DEFAULT_PII_PATTERNS,
    DEFAULT_SECRET_PATTERNS,
    TextDetector,
    blocked_keywords,
    for_text,
    for_tool_result_text,
    personal_data,
    redact_personal_data,
    redact_secrets,
    secret_data,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


_OPENAI_KEY = 'sk-abcdefghijklmnopqrstuvwxyz01'


def _echo_prompt() -> FunctionModel:
    """A model that answers with the prompt it received, so a redaction is observable."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts = [part.content for message in messages for part in message.parts if isinstance(part, UserPromptPart)]
        return ModelResponse(parts=[TextPart(content=str(prompts[-1]))])

    return FunctionModel(respond)


class TestSecretRedaction:
    """Credentials are rewritten, not refused."""

    def test_a_key_is_replaced_and_named(self):
        result = redact_secrets(f'the key is {_OPENAI_KEY}')

        assert result.action == 'replace'
        assert result.replacement == 'the key is [redacted:openai_key]'

    def test_clean_text_is_left_alone(self):
        assert redact_secrets('nothing to see').action == 'allow'

    @pytest.mark.parametrize('name', sorted(DEFAULT_SECRET_PATTERNS))
    def test_every_default_pattern_matches_something(self, name: str):
        """A pattern that matches nothing is dead weight, and a typo in one is invisible."""
        samples = {
            'openai_key': _OPENAI_KEY,
            'slack_app_token': 'xapp-1-A01B2C3D4E5-1234567890123-abcdef',
            'stripe_webhook_secret': 'whsec_abcdefghijklmnopqrstuvwx',
            'google_oauth_secret': 'GOCSPX-abcdefghijklmnopqrstuv',
            'anthropic_key': 'sk-ant-abcdefghijklmnopqrstuvwx',
            'aws_access_key': 'AKIAIOSFODNN7EXAMPLE',
            'github_token': 'ghp_abcdefghijklmnopqrstuvwxyz0123',
            'slack_token': 'xoxb-1234567890-abcdef',
            'stripe_key': 'sk_live_abcdefghijklmnopqrstuvwx',
            'google_api_key': 'AIza' + 'a' * 35,
            'jwt': 'eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4',
            'private_key': '-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----',
        }
        assert secret_data(only=[name])(samples[name]).action == 'replace'

    @pytest.mark.parametrize('key', ['sk-abcdefghijklmnopqrstuvwxyz01', 'sk-proj-AbCdEfGhIjKlMnOpQrStUv'])
    def test_both_shapes_of_openai_key_match(self, key: str):
        """Project keys carry hyphens, which an alphanumeric-only class stops at."""
        assert redact_secrets(f'k={key}').replacement == 'k=[redacted:openai_key]'

    def test_an_anthropic_key_keeps_its_own_label(self):
        """Allowing hyphens makes the OpenAI shape a prefix of this one, so it is excluded explicitly."""
        result = redact_secrets('k=sk-ant-abcdefghijklmnopqrstuvwx')

        assert result.replacement == 'k=[redacted:anthropic_key]'

    def test_a_private_key_is_removed_whole(self):
        """Replacing the BEGIN line alone would leave the key material sitting in the text."""
        pem = '-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAx7Vv9mKq\n-----END RSA PRIVATE KEY-----'
        result = redact_secrets(f'here:\n{pem}\ndone')

        assert result.replacement == 'here:\n[redacted:private_key]\ndone'

    def test_an_unterminated_private_key_takes_its_body_and_stops(self):
        """Running to the end of the input instead would delete whatever the user wrote after it."""
        result = redact_secrets('key:\n-----BEGIN PRIVATE KEY-----\nMIIEow\n\nAlso deploy to prod.')

        assert result.replacement == 'key:\n[redacted:private_key]\nAlso deploy to prod.'

    def test_narrowing_to_private_key_still_covers_an_unterminated_one(self):
        """`only` selects by name, so the two PEM shapes have to share one or a narrowed detector misses one."""
        detector = secret_data(only=['aws_access_key', 'private_key'])
        result = detector('-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAxyz\n\nrest of my message')

        assert result.replacement == '[redacted:private_key]\nrest of my message'

    def test_an_encrypted_private_key_is_removed_whole(self):
        """Its RFC 1421 headers carry `:` and `,`, which the base64 body class stops at."""
        pem = (
            '-----BEGIN RSA PRIVATE KEY-----\n'
            'Proc-Type: 4,ENCRYPTED\n'
            'DEK-Info: AES-256-CBC,ABCDEF0123456789\n'
            '\n'
            'MIIEowIBAAKCAQEAx7Vv9mKq\n'
            'abcdEFGH1234+/==\n'
            '-----END RSA PRIVATE KEY-----'
        )
        result = redact_secrets(f'here:\n{pem}\ndone')

        assert result.replacement == 'here:\n[redacted:private_key]\ndone'

    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            (
                'k:\r\n-----BEGIN PRIVATE KEY-----\r\nMIIEow\r\n-----END PRIVATE KEY-----\r\ndone',
                'k:\r\n[redacted:private_key]\r\ndone',
            ),
            (
                'k:\r\n-----BEGIN PRIVATE KEY-----\r\nMIIEowIBAAK\r\nabcdEFGH1234+/==\r\n\r\nAlso deploy.',
                'k:\r\n[redacted:private_key]\r\nAlso deploy.',
            ),
        ],
    )
    def test_a_key_pasted_with_crlf_is_removed_whole(self, text: str, expected: str):
        """A body line ending in CRLF ends the run at the first line, leaving the rest of the key."""
        assert redact_secrets(text).replacement == expected

    def test_prose_naming_both_markers_is_not_a_key(self):
        """The newline after the header is what separates a real block from a sentence about one."""
        text = 'Paste everything between -----BEGIN PRIVATE KEY----- and -----END PRIVATE KEY----- in the field.'

        assert redact_secrets(text).action == 'allow'

    @pytest.mark.parametrize(
        'text',
        [
            'ship the task-management-and-deployment plan',
            'ask the risk-management-and-compliance-team',
            'see the disk-usage-monitoring-report-2024',
        ],
    )
    def test_a_prefix_inside_a_word_is_not_a_key(self, text: str):
        """`sk-` sits inside ordinary hyphenated English, and the class then takes the rest of the phrase."""
        assert redact_secrets(text).action == 'allow'

    @pytest.mark.parametrize('delimiter', [' ', '=', '"', ':'])
    def test_a_key_still_matches_after_the_delimiters_keys_arrive_with(self, delimiter: str):
        """The boundary above must not cost a real key, which arrives after a space, quote or separator."""
        assert redact_secrets(f'key{delimiter}{_OPENAI_KEY}').replacement == f'key{delimiter}[redacted:openai_key]'

    def test_a_subset_leaves_the_rest_alone(self):
        detector = secret_data(only=['aws_access_key'])

        assert detector(f'key {_OPENAI_KEY}').action == 'allow'
        assert detector('key AKIAIOSFODNN7EXAMPLE').action == 'replace'

    def test_an_unknown_pattern_name_is_refused(self):
        with pytest.raises(UserError, match='Unknown pattern'):
            secret_data(only=['nope'])

    def test_a_custom_pattern_joins_the_defaults(self):
        detector = secret_data(extra={'internal': r'INT-\d{4}'})

        assert detector('ticket INT-4321').replacement == 'ticket [redacted:internal]'

    @pytest.mark.parametrize(
        'key',
        ['sk-ant-api03-R2xvYmFs_ZGVmaW5pdGVseVNlY3JldA-abcdEFGH', 'sk-proj-Ab3dEfGhIj_KlMnOpQrStUvWxYz0123456789'],
    )
    def test_a_base64url_key_is_removed_whole(self, key: str):
        """Vendor key bodies contain `_`; a class that stops there leaves most of the key behind."""
        replacement = str(redact_secrets(f'k={key}').replacement)

        assert replacement.startswith('k=[redacted:')
        assert replacement.endswith('_key]')

    def test_scanning_a_large_paste_is_linear(self):
        """An unbounded local part lets a failed match restart at every offset, blocking the event loop."""
        start = time.perf_counter()
        redact_personal_data('9' * 100_000)

        assert time.perf_counter() - start < 1.0

    def test_a_detector_with_no_patterns_is_refused(self):
        """Same rule as an empty keyword list: a check that inspects nothing behaves as absent."""
        with pytest.raises(UserError, match='no patterns'):
            secret_data(only=[])

    def test_a_placeholder_cannot_re_emit_the_secret(self):
        """`re.sub` reads backreferences in a template, so the placeholder is applied literally."""
        result = secret_data(placeholder=r'\g<0>-GONE')(f'k {_OPENAI_KEY}')

        assert result.replacement == r'k \g<0>-GONE'

    def test_a_custom_pattern_is_not_judged_by_a_built_in_validator(self):
        """A validator belongs to the pattern it was written for, not to its name."""
        detector = personal_data(only=['email'], extra={'credit_card': r'CARD-\d{4}'})

        assert detector('ref CARD-1234').replacement == 'ref [redacted:credit_card]'

    def test_extra_may_not_silently_replace_a_built_in(self):
        with pytest.raises(UserError, match='would replace the built-in'):
            secret_data(extra={'openai_key': 'x'})

    def test_a_placeholder_without_the_name_is_used_verbatim(self):
        assert secret_data(placeholder='***')(f'k {_OPENAI_KEY}').replacement == 'k ***'


class TestKeyPastedFromAFile:
    """The most common way a private key reaches a chat is inside JSON or a `.env` line."""

    _BODY = 'MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ'

    def test_a_service_account_json_is_redacted(self):
        """Its newlines are the two characters backslash-n, not a line break."""
        import json

        pem = f'-----BEGIN PRIVATE KEY-----\n{self._BODY}\n-----END PRIVATE KEY-----\n'
        document = json.dumps({'type': 'service_account', 'private_key': pem})

        replacement = str(redact_secrets(document).replacement)

        assert self._BODY not in replacement
        assert '[redacted:private_key]' in replacement

    def test_a_dotenv_line_is_redacted(self):
        line = f'PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\\n{self._BODY}\\n-----END PRIVATE KEY-----"'

        replacement = str(redact_secrets(line).replacement)

        assert self._BODY not in replacement

    def test_an_escaped_key_with_no_end_marker_is_still_taken(self):
        line = f'key=-----BEGIN PRIVATE KEY-----\\n{self._BODY}'

        assert self._BODY not in str(redact_secrets(line).replacement)


class TestPatternOrder:
    """`only` filters; it does not hand the application order to the caller's argument order."""

    def test_only_keeps_the_declared_order(self):
        iban = 'DE89 3704 0044 0532 0130 00'
        detector = personal_data(only=['credit_card', 'iban'])

        assert detector(f'iban {iban}').replacement == 'iban [redacted:iban]'


class TestPersonalData:
    """Personal data is rewritten too, for the same reason."""

    @pytest.mark.parametrize(
        ('name', 'sample', 'expected'),
        [
            ('email', 'write to a.b@example.com', 'write to [redacted:email]'),
            ('us_ssn', 'ssn 123-45-6789', 'ssn [redacted:us_ssn]'),
            ('us_ssn', 'ssn 123 45 6789', 'ssn [redacted:us_ssn]'),
            ('credit_card', 'card 4111 1111 1111 1111', 'card [redacted:credit_card]'),
            ('credit_card', 'amex 3782 822463 10005', 'amex [redacted:credit_card]'),
            ('iban', 'iban GB29NWBK60161331926819', 'iban [redacted:iban]'),
            ('iban', 'iban DE89 3704 0044 0532 0130 00', 'iban [redacted:iban]'),
        ],
    )
    def test_each_default_pattern_replaces_the_whole_match(self, name: str, sample: str, expected: str):
        """Asserting the replacement, not just the action: a partial or mislabelled redaction passes the weaker check."""
        assert name in DEFAULT_PII_PATTERNS
        assert personal_data(only=[name])(sample).replacement == expected

    @pytest.mark.parametrize(
        'number',
        [
            '4222222222222',  # 13, legacy Visa
            '30569309025904',  # 14, Diners Club
            '378282246310005',  # 15, American Express
            '4111111111111111',  # 16, Visa
            '4000000000000000006',  # 19, Visa
            '4000 0000 0000 0000 006',
            '4000-0000-0000-0000-006',
        ],
    )
    def test_every_pan_length_is_redacted(self, number: str):
        """Fixed 4-4-4-4 and 4-6-5 groups let a 13, 14 or 19-digit card through untouched."""
        assert redact_personal_data(f'card {number} on file').replacement == 'card [redacted:credit_card] on file'

    def test_luhn_discards_a_digit_run_that_is_not_a_card(self):
        """It discards most of them. A run that satisfies the checksum by chance is still redacted."""
        assert redact_personal_data('revenue for 2021 2022 2023 2024 was flat').action == 'allow'

    @pytest.mark.parametrize('text', ['ports 8080 8081 8082 8083 are open', 'the job ran at 1721990400000'])
    def test_a_digit_run_that_cannot_be_a_card_is_not_checked_at_all(self, text: str):
        """A payment card's leading digit is 2 to 6, so a timestamp or a port list never reaches Luhn."""
        assert redact_personal_data(text).action == 'allow'

    def test_a_digit_run_longer_than_a_pan_is_left_alone(self):
        """Accepting any length would make an identifier a card whenever the checksum happened to pass."""
        assert redact_personal_data('order 40000000000000000060 shipped').action == 'allow'

    @pytest.mark.parametrize('case', ['DE89 3704 0044 0532 0130 00', 'de89 3704 0044 0532 0130 00'])
    def test_an_iban_matches_in_either_case(self, case: str):
        """The printed form is uppercase; the standard is not."""
        assert redact_personal_data(f'iban {case}').replacement == 'iban [redacted:iban]'

    @pytest.mark.parametrize(
        'text',
        ['see patent US10123456789B2 for details', 'build RC20240115T090000Z finished', 'commit ab12cdef0123456789'],
    )
    def test_an_identifier_is_not_an_account_number(self, text: str):
        """The country code comes from the IBAN registry, not from any two letters."""
        assert redact_personal_data(text).action == 'allow'

    @pytest.mark.parametrize(
        'text',
        [
            'plug in the RS232 serial cable adapter and reboot',
            'the CH340 driver installed correctly on Linux',
            'model MC68000 assembly language reference manual',
            'error code PL99 returned by the payment provider gateway',
        ],
    )
    def test_a_sentence_is_not_an_account_number(self, text: str):
        """A space before every character let the pattern run across words and eat the sentence."""
        assert redact_personal_data(text).action == 'allow'

    @pytest.mark.parametrize(
        'iban',
        [
            'GB29NWBK60161331926819',
            'FR14 2004 1010 0505 0001 3M02 606',
            'NL91ABNA0417164300',
            'ES91 2100 0418 4502 0005 1332',
            'IT60X0542811101000000123456',
        ],
    )
    def test_a_real_iban_still_matches(self, iban: str):
        """Narrowing the shape must not cost the accounts the detector exists for."""
        assert redact_personal_data(f'iban {iban}').replacement == 'iban [redacted:iban]'

    def test_a_token_too_short_to_be_an_iban_is_left_alone(self):
        """The shortest real IBAN is 15 characters; the pattern can match 14."""
        assert redact_personal_data('ref DE891234567890 filed').action == 'allow'

    def test_a_country_code_with_a_wrong_check_digit_is_left_alone(self):
        """The ISO 7064 mod-97 digit is what a shape alone cannot check."""
        assert redact_personal_data('iban DE88 3704 0044 0532 0130 00').action == 'allow'

    def test_a_spaced_iban_keeps_its_own_label(self):
        """Declared before `credit_card`, whose digit groups would otherwise claim the middle of it."""
        assert redact_personal_data('iban DE89 3704 0044 0532 0130 00').replacement == 'iban [redacted:iban]'

    def test_ordinary_text_survives(self):
        assert redact_personal_data('the meeting is at 3pm on the 4th').action == 'allow'


class TestBlockedKeywords:
    """A keyword list refuses rather than rewrites."""

    def test_a_match_blocks_and_names_the_term(self):
        result = blocked_keywords(['classified'])('this is CLASSIFIED')

        assert result.action == 'block'
        assert result.message == "Blocked term: 'CLASSIFIED'."

    def test_case_sensitivity_is_opt_in(self):
        assert blocked_keywords(['SECRET'], case_sensitive=True)('secret').action == 'allow'
        assert blocked_keywords(['SECRET'], case_sensitive=True)('SECRET').action == 'block'

    def test_whole_words_stops_matching_inside_a_longer_word(self):
        detector = blocked_keywords(['class'], whole_words=True)

        assert detector('classification').action == 'allow'
        assert detector('the class').action == 'block'

    def test_a_keyword_with_punctuation_matches_itself(self):
        """Keywords are escaped, so regex characters are not a mini-language."""
        assert blocked_keywords(['a.b'])('axb').action == 'allow'
        assert blocked_keywords(['a.b'])('a.b').action == 'block'

    def test_a_custom_message_hides_the_list(self):
        result = blocked_keywords(['classified'], message='Not available.')('classified')

        assert result.message == 'Not available.'

    @pytest.mark.parametrize('keyword', ['C++', 'a.b!', '#tag'])
    def test_whole_words_still_matches_a_keyword_ending_in_punctuation(self, keyword: str):
        """`\\b` needs a word character beside it, so it would leave these silently inert."""
        detector = blocked_keywords([keyword], whole_words=True)

        assert detector(f'I know {keyword}').action == 'block'
        assert detector(f'{keyword}x').action == 'allow'

    def test_a_bare_string_is_refused(self):
        """`str` is an `Iterable[str]`, so it would be one keyword per character."""
        with pytest.raises(UserError, match='single string'):
            blocked_keywords('internal-only')

    def test_an_empty_keyword_is_refused(self):
        """An empty pattern matches at position 0 of anything, so it would block every input."""
        with pytest.raises(UserError, match='empty keyword'):
            blocked_keywords(['ok', ''])

    def test_no_keywords_is_refused(self):
        with pytest.raises(UserError, match='no keywords'):
            blocked_keywords([])


class TestForText:
    """A text detector meeting a value that is not text."""

    def test_text_reaches_the_detector(self):
        assert for_text(redact_secrets)(f'k {_OPENAI_KEY}').action == 'replace'

    def test_a_non_string_fails_loudly_by_default(self):
        with pytest.raises(UserError, match='cannot rewrite without changing'):
            for_text(redact_secrets)(42)

    async def test_a_structured_output_reaches_the_detector_through_a_guard(self):
        """The scenario `for_text` exists for, driven through the capability rather than by hand."""
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'text': 'ok'})])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            output_type=Answer,
            capabilities=[OutputGuardrail(guard=for_text(redact_secrets))],
        )

        with pytest.raises(UserError, match='cannot rewrite without changing'):
            await agent.run('hi')

    async def test_a_structured_output_can_be_skipped_deliberately(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'text': 'ok'})])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            output_type=Answer,
            capabilities=[OutputGuardrail(guard=for_text(redact_secrets, on_other='allow'))],
        )
        result = await agent.run('hi')

        assert result.output == Answer(text='ok')

    @pytest.mark.parametrize(
        'detector',
        [redact_secrets, redact_personal_data, blocked_keywords(['internal-only'])],
        ids=['secrets', 'personal_data', 'blocked_keywords'],
    )
    def test_a_detector_used_without_for_text_says_so(self, detector: TextDetector):
        """`re` would otherwise raise a TypeError three frames down that names nothing to do about it."""
        with pytest.raises(UserError, match='wrap the detector in for_text'):
            detector(42)  # type: ignore[arg-type]

    def test_a_non_string_can_be_skipped_deliberately(self):
        assert for_text(redact_secrets, on_other='allow')(42).action == 'allow'


class TestForToolResultText:
    """A text detector adapted to the value `ToolGuardrail` hands to its result guard."""

    @staticmethod
    def _info(result: object) -> ToolResultInfo:
        return ToolResultInfo(name='lookup', args={}, tool_call_id='call-1', result=result)

    def test_a_plain_string_reaches_the_detector(self):
        verdict = for_tool_result_text(redact_secrets)(self._info(f'key: {_OPENAI_KEY}'))

        assert verdict == GuardrailResult.replace('key: [redacted:openai_key]')

    @pytest.mark.parametrize('content', [None, 'safe context'])
    def test_an_unmodified_tool_return_is_allowed(self, content: str | None):
        verdict = for_tool_result_text(redact_secrets)(self._info(ToolReturn('no secret', content=content)))

        assert verdict == GuardrailResult.allow()

    def test_tool_return_content_reaches_the_detector(self):
        """Parts that do not run into one another keep their own boundaries and metadata."""
        image = BinaryContent(data=b'image', media_type='image/png')
        result = ToolReturn(
            'safe summary',
            content=[f'key: {_OPENAI_KEY}\n', TextContent(content=f'context: {_OPENAI_KEY}', metadata={'m': 1}), image],
        )

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn(
                'safe summary',
                content=[
                    'key: [redacted:openai_key]\n',
                    TextContent(content='context: [redacted:openai_key]', metadata={'m': 1}),
                    image,
                ],
            )
        )

    def test_a_part_that_runs_into_the_next_is_sanitized_as_one_string(self):
        """`{_OPENAI_KEY}context` is one token to the model, so the span cannot keep its boundaries."""
        result = ToolReturn('safe summary', content=[f'key: {_OPENAI_KEY}', f'context: {_OPENAI_KEY}'])

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn('safe summary', content=['key: [redacted:openai_key]: [redacted:openai_key]'])
        )

    def test_a_redacted_fragment_cannot_hide_the_rest_of_a_match(self):
        """Redacting the fragment holding `sk-` would strip the anchor and leave the tail exposed."""
        head, tail = f'sk-{"a" * 20}', 'a' * 12
        split = for_tool_result_text(redact_secrets)(self._info(ToolReturn('safe summary', content=[head, tail])))
        whole = for_tool_result_text(redact_secrets)(self._info(ToolReturn('safe summary', content=head + tail)))

        assert split == GuardrailResult.replace(ToolReturn('safe summary', content=['[redacted:openai_key]']))
        assert whole == GuardrailResult.replace(ToolReturn('safe summary', content='[redacted:openai_key]'))

    def test_a_redacted_fragment_cannot_hide_a_terminal_verdict(self):
        """The joined pass reads the original span, so a rewrite cannot erase what would block."""

        def redact_then_block(text: str) -> GuardrailResult:
            if 'ZZ' in text:
                return GuardrailResult.block('Blocked term.')
            return GuardrailResult.replace(text.replace('aZ', '-')) if 'aZ' in text else GuardrailResult.allow()

        verdict = for_tool_result_text(redact_then_block)(self._info(ToolReturn('safe summary', content=['aZ', 'Z'])))

        assert verdict.action == 'block'

    @pytest.mark.parametrize(
        'content',
        ['safe context', ['safe context'], ['safe context', BinaryContent(data=b'image', media_type='image/png')]],
        ids=['string', 'one_part', 'one_part_beside_binary'],
    )
    def test_one_stretch_of_text_reaches_the_detector_once(self, content: str | list[UserContent]):
        """A stretch of one part has no boundary to straddle, so a joined pass would only repeat it.

        `TextDetector` is a public extension point, so a stateful or per-call-billed detector
        would otherwise see these three equivalent shapes differently.
        """
        seen: list[str] = []

        def record(text: str) -> GuardrailResult:
            seen.append(text)
            return GuardrailResult.allow()

        for_tool_result_text(record)(self._info(ToolReturn('safe summary', content=content)))

        assert seen == ['safe summary', 'safe context']

    def test_adjacent_text_parts_add_a_single_joined_call(self):
        """The joined pass runs once per stretch, over the per-part rewrites."""
        seen: list[str] = []

        def record(text: str) -> GuardrailResult:
            seen.append(text)
            return GuardrailResult.allow()

        image = BinaryContent(data=b'image', media_type='image/png')
        for_tool_result_text(record)(self._info(ToolReturn('safe summary', content=['a', 'b', image, 'c'])))

        assert seen == ['safe summary', 'a', 'b', 'ab', 'c']

    def test_a_secret_split_across_adjacent_parts_is_redacted(self):
        """A model reads adjacent text parts as one span, so neither half matches on its own."""
        image = BinaryContent(data=b'image', media_type='image/png')
        result = ToolReturn('safe summary', content=[f'key: {_OPENAI_KEY[:20]}', _OPENAI_KEY[20:], image])

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn('safe summary', content=['key: [redacted:openai_key]', image])
        )

    def test_a_secret_split_around_a_cache_point_is_redacted(self):
        """A `CachePoint` splits the parts but not the text, since providers without caching drop it."""
        result = ToolReturn('safe summary', content=[f'key: {_OPENAI_KEY[:20]}', CachePoint(), _OPENAI_KEY[20:]])

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn('safe summary', content=[CachePoint(), 'key: [redacted:openai_key]'])
        )

    def test_a_cache_point_keeps_its_place_when_no_merge_is_needed(self):
        result = ToolReturn('safe summary', content=[f'key: {_OPENAI_KEY}\n', CachePoint(), 'tail'])

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn('safe summary', content=['key: [redacted:openai_key]\n', CachePoint(), 'tail'])
        )

    @pytest.mark.parametrize(
        ('content', 'expected'),
        [
            pytest.param(
                [CachePoint(), f'key: {_OPENAI_KEY[:20]}', _OPENAI_KEY[20:]],
                [CachePoint(), 'key: [redacted:openai_key]'],
                id='leading_marker_stays_in_front',
            ),
            pytest.param(
                [f'key: {_OPENAI_KEY[:20]}', CachePoint(), _OPENAI_KEY[20:]],
                [CachePoint(), 'key: [redacted:openai_key]'],
                id='interior_marker_moves_in_front',
            ),
            pytest.param(
                [f'key: {_OPENAI_KEY[:20]}', _OPENAI_KEY[20:], CachePoint()],
                ['key: [redacted:openai_key]', CachePoint()],
                id='trailing_marker_stays_behind',
            ),
        ],
    )
    def test_a_merge_keeps_each_cache_point_on_a_side_it_still_has(
        self, content: list[UserContent], expected: list[UserContent]
    ):
        """Only a marker between two merged texts lost its split; the outer ones still have a side.

        Moving a trailing marker in front would push text that was already inside the cached
        prefix back out of it, narrowing the caller's boundary for no reason.
        """
        verdict = for_tool_result_text(redact_secrets)(self._info(ToolReturn('safe summary', content=content)))

        assert verdict == GuardrailResult.replace(ToolReturn('safe summary', content=expected))

    def test_text_around_a_binary_part_is_not_joined(self):
        """A `BinaryContent` is real content between the two texts, so joining them would invent adjacency."""
        image = BinaryContent(data=b'image', media_type='image/png')
        result = ToolReturn('safe summary', content=[f'key: {_OPENAI_KEY[:20]}', image, _OPENAI_KEY[20:]])

        assert for_tool_result_text(redact_secrets)(self._info(result)) == GuardrailResult.allow()

    def test_a_split_secret_keeps_the_metadata_of_the_part_it_merges_into(self):
        result = ToolReturn(
            'safe summary',
            content=[
                TextContent(content=f'key: {_OPENAI_KEY[:20]}', metadata={'source': 'vault'}),
                TextContent(content=_OPENAI_KEY[20:]),
            ],
        )

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn(
                'safe summary',
                content=[TextContent(content='key: [redacted:openai_key]', metadata={'source': 'vault'})],
            )
        )

    def test_a_blocked_term_split_across_adjacent_parts_is_blocked(self):
        verdict = for_tool_result_text(blocked_keywords(['blocked']))(
            self._info(ToolReturn('safe summary', content=['blo', 'cked context']))
        )

        assert verdict.action == 'block'

    def test_content_that_starts_with_a_non_text_part_is_scanned(self):
        """The text run opens after the image rather than at index 0."""
        image = BinaryContent(data=b'image', media_type='image/png')
        result = ToolReturn('safe summary', content=[image, f'key: {_OPENAI_KEY}'])

        verdict = for_tool_result_text(redact_secrets)(self._info(result))

        assert verdict == GuardrailResult.replace(
            ToolReturn('safe summary', content=[image, 'key: [redacted:openai_key]'])
        )

    def test_unmodified_multipart_content_is_allowed(self):
        result = ToolReturn('no secret', content=['safe context', TextContent(content='more context')])

        assert for_tool_result_text(redact_secrets)(self._info(result)) == GuardrailResult.allow()

    def test_a_return_value_detector_verdict_is_preserved(self):
        verdict = for_tool_result_text(blocked_keywords(['blocked']))(
            self._info(ToolReturn('blocked summary', content='safe context'))
        )

        assert verdict.action == 'block'

    @pytest.mark.parametrize(
        'content', ['blocked context', ['safe context', 'blocked context']], ids=['string', 'multipart']
    )
    def test_a_content_detector_verdict_is_preserved(self, content: str | list[str]):
        verdict = for_tool_result_text(blocked_keywords(['blocked']))(
            self._info(ToolReturn('safe summary', content=content))
        )

        assert verdict.action == 'block'

    def test_a_content_detector_must_replace_with_text(self):
        def invalid_detector(text: str) -> GuardrailResult:
            return GuardrailResult.replace(42) if text == 'secret' else GuardrailResult.allow()

        with pytest.raises(UserError, match='replace tool content with text'):
            for_tool_result_text(invalid_detector)(self._info(ToolReturn('safe', content=['secret'])))

    @pytest.mark.parametrize('result', ['secret', ToolReturn('secret')])
    def test_a_text_detector_must_replace_with_text(self, result: object):
        def invalid_detector(_: str) -> GuardrailResult:
            return GuardrailResult.replace(42)

        with pytest.raises(UserError, match='must replace a tool result with text'):
            for_tool_result_text(invalid_detector)(self._info(result))

    async def test_a_real_tool_return_is_redacted_without_losing_its_metadata(self):
        metadata = {'request_id': 'r1'}

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='call-1')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[ToolGuardrail(result_guard=for_tool_result_text(redact_secrets))],
        )

        @agent.tool_plain
        def lookup() -> ToolReturn:
            return ToolReturn(f'key: {_OPENAI_KEY}', content=f'context: {_OPENAI_KEY}', metadata=metadata)

        result = await agent.run('hi')
        returns = [
            part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
        ]

        assert [(part.content, part.metadata) for part in returns] == [
            ('key: [redacted:openai_key]', metadata),
        ]
        assert f'context: {_OPENAI_KEY}' not in str(result.all_messages())
        assert 'context: [redacted:openai_key]' in str(result.all_messages())

    def test_allowing_a_non_text_result_skips_its_content_too(self):
        """`on_other='allow'` is an opt-out for the whole result, not a partial scan.

        A `ToolReturn` carrying a structured `return_value` is a non-text result, so its
        `content` reaches the model unscanned. Callers who need that text checked keep
        `on_other` at its default and guard a field of the output instead.
        """
        result = ToolReturn({'rows': 1}, content=f'key: {_OPENAI_KEY}')

        verdict = for_tool_result_text(redact_secrets, on_other='allow')(self._info(result))

        assert verdict == GuardrailResult.allow()

    async def test_a_non_text_tool_result_must_be_allowed_explicitly(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='call-1')])

        blocked_agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[ToolGuardrail(result_guard=for_tool_result_text(redact_secrets))],
        )
        allowed_agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[ToolGuardrail(result_guard=for_tool_result_text(redact_secrets, on_other='allow'))],
        )

        @blocked_agent.tool_plain
        @allowed_agent.tool_plain
        def lookup() -> dict[str, str]:
            return {'status': 'clean'}

        with pytest.raises(UserError, match='for_tool_result_text'):
            await blocked_agent.run('hi')
        result = await allowed_agent.run('hi')

        assert result.output == 'done'


class TestGuardChain:
    """Several guards in one capability, run in order."""

    async def test_a_redaction_is_threaded_into_the_next_guard(self):
        """The second guard sees the cleaned text, which is what makes ordering useful."""
        seen: list[str] = []

        def record(prompt: str) -> GuardrailResult:
            seen.append(prompt)
            return GuardrailResult.allow()

        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[redact_secrets, record])],
        )
        result = await agent.run(f'my key is {_OPENAI_KEY}')

        assert seen == ['my key is [redacted:openai_key]']
        assert result.output == 'my key is [redacted:openai_key]'

    async def test_the_first_block_ends_the_chain(self):
        reached: list[str] = []

        def never(prompt: str) -> GuardrailResult:  # pragma: no cover - the point is that it is not reached
            reached.append(prompt)
            return GuardrailResult.allow()

        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[blocked_keywords(['classified']), never])],
        )
        result = await agent.run('this is classified')

        assert result.output == "Blocked term: 'classified'."
        assert reached == []

    async def test_a_chain_that_only_allows_changes_nothing(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[redact_secrets, blocked_keywords(['nope'])])],
        )

        result = await agent.run('nothing sensitive here')

        assert result.output == 'nothing sensitive here'

    async def test_an_output_chain_blocks_on_the_second_guard(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='here is the classified plan')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[
                OutputGuardrail(guard=[for_text(redact_secrets), for_text(blocked_keywords(['classified']))])
            ],
        )

        with pytest.raises(OutputBlocked, match='Blocked term'):
            await agent.run('hi')

    async def test_an_output_chain_returns_the_accumulated_replacement(self):
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=f'key {_OPENAI_KEY} for a.b@example.com')])

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[OutputGuardrail(guard=[for_text(redact_secrets), for_text(redact_personal_data)])],
        )
        result = await agent.run('hi')

        assert result.output == 'key [redacted:openai_key] for [redacted:email]'

    async def test_an_output_chain_can_accumulate_a_replacement_of_none(self):
        """`None` is a replacement like any other, so the chain has to carry it to the end.

        The chain closes by rebuilding a `replace` verdict from the accumulated
        value. A `None` default on `GuardrailResult.replacement` could not tell
        that verdict apart from one carrying no replacement, and rejected it.
        """

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='here is the classified plan')])

        def drop(output: object) -> GuardrailResult:
            return GuardrailResult.replace(None)

        def allow_anything(output: object) -> bool:
            return True

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[OutputGuardrail(guard=[drop, allow_anything])],
        )
        result = await agent.run('hi')

        assert result.output is None

    async def test_a_callable_that_is_also_a_sequence_stays_one_guard(self):
        """Callability decides, so a guard that happens to be a sequence is not taken apart."""

        class CallableList(list[str]):
            def __call__(self, prompt: str) -> GuardrailResult:
                return GuardrailResult.block('refused by the callable itself')

        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=CallableList(['not', 'guards']))],
        )
        result = await agent.run('hi')

        assert result.output == 'refused by the callable itself'

    @pytest.mark.parametrize(
        ('guard', 'match'),
        [
            ('not-a-guard', 'got str'),
            (b'xy', 'got bytes'),
            (42, 'got int'),
            ([redact_secrets, 'nope'], 'at position 1 of its guard sequence; got str'),
        ],
    )
    async def test_a_shape_that_is_not_a_guard_is_named(self, guard: object, match: str):
        """Left alone these reach `inspect.signature` as a bare TypeError naming nothing useful."""
        agent = Agent(_echo_prompt(), deps_type=type(None), capabilities=[InputGuardrail(guard=guard)])  # type: ignore[arg-type]

        with pytest.raises(UserError, match=match):
            await agent.run('hi')

    async def test_an_empty_chain_is_refused(self):
        """A guardrail that inspects nothing reads as configured and behaves as absent."""
        guards: list[TextDetector] = []
        agent = Agent(_echo_prompt(), deps_type=type(None), capabilities=[InputGuardrail(guard=guards)])

        with pytest.raises(UserError, match='empty sequence of guards'):
            await agent.run('hi')

    async def test_a_set_of_guards_is_refused(self):
        """A set iterates in hash order, so the chain would not run in the order it was written."""
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard={redact_secrets})],  # type: ignore[arg-type]
        )

        with pytest.raises(UserError, match='got set'):
            await agent.run('hi')

    async def test_a_one_shot_iterator_is_refused(self):
        """The chain is rebuilt per request, so an iterator would be spent after the first run."""
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=(guard for guard in [redact_secrets]))],  # type: ignore[arg-type]
        )

        with pytest.raises(UserError, match='got generator'):
            await agent.run('hi')

    async def test_a_bare_false_mid_chain_blocks(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[redact_secrets, lambda prompt: False])],
        )
        result = await agent.run('hi')

        assert result.output == 'Request blocked by input guardrail.'

    async def test_a_guard_taking_run_context_works_beside_one_that_does_not(self):
        """`_takes_ctx` is decided per element, so a mixed chain has to be read per element."""
        seen: list[str] = []

        def with_ctx(ctx: RunContext[None], prompt: str) -> GuardrailResult:
            seen.append(f'ctx:{prompt}')
            return GuardrailResult.allow()

        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[redact_secrets, with_ctx])],
        )
        await agent.run(f'k {_OPENAI_KEY}')

        assert seen == ['ctx:k [redacted:openai_key]']

    async def test_a_retry_in_a_chain_names_the_guard_that_returned_it(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[redact_secrets, lambda prompt: GuardrailResult.retry('again')])],
        )

        with pytest.raises(UserError, match='guard at position 1 did'):
            await agent.run('hi')

    async def test_an_output_chain_retries_and_re_runs_on_the_new_output(self):
        outputs = iter(['first with a secret', 'second is clean'])
        seen: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=next(outputs))])

        def once(output: object) -> GuardrailResult:
            seen.append(str(output))
            return GuardrailResult.retry('drop the secret') if 'secret' in str(output) else GuardrailResult.allow()

        agent = Agent(
            FunctionModel(respond),
            deps_type=type(None),
            capabilities=[OutputGuardrail(guard=[for_text(redact_secrets), once])],
        )
        result = await agent.run('hi')

        assert seen == ['first with a secret', 'second is clean']
        assert result.output == 'second is clean'

    async def test_a_parallel_input_guard_runs_a_chain_and_blocks_from_any_position(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[redact_secrets, blocked_keywords(['classified'])], parallel=True)],
        )
        result = await agent.run('this is classified')

        assert result.output == "Blocked term: 'classified'."

    async def test_a_parallel_input_guard_refuses_a_chain_that_redacts(self):
        """The model call already started with the original prompt, so a redaction is too late."""
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[blocked_keywords(['nope']), redact_secrets], parallel=True)],
        )

        with pytest.raises(UserError, match='incompatible with GuardrailResult.replace'):
            await agent.run(f'k {_OPENAI_KEY}')

    async def test_a_mid_chain_replacement_must_still_be_prompt_text(self):
        agent = Agent(
            _echo_prompt(),
            deps_type=type(None),
            capabilities=[InputGuardrail(guard=[lambda prompt: GuardrailResult.replace(123), redact_secrets])],
        )

        with pytest.raises(UserError, match='guard at position 0 returned int'):
            await agent.run('hi')
