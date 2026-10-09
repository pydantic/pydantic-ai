"""Pull requests run a subset of the CI test matrix; the merge queue, `main` and tags run all of it.

Each test job's matrix carries a single-value `pull_request` dimension whose `exclude` entries drop the
legs a pull request skips. Those legs may only repeat what another leg already checks: a leg that collects
coverage, detects blocking calls or enforces the per-test budget has to run on the pull request too. For
coverage that is a correctness requirement, not a preference: if the merge queue collected coverage the pull
request did not, a `pragma: no cover` line only it reaches would fail `strict-no-cover` in the queue, and a
line only it covers would fail the pull request's gate.

These tests expand every matrix for both events from `ci.yml` itself and evaluate the job-level `env`
expressions per leg, so a new coverage leg on a skipped version fails here rather than in the merge queue.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'ci.yml'

# Job-level env vars that put a check on specific legs; each such leg must run on a pull request.
LANE_VARS = ('COLLECT_COVERAGE', 'BLOCKBUSTER_ENABLED', 'PYTEST_TEST_BUDGET_SECONDS')
REDUCED_JOBS = ('test', 'test-all-extras', 'test-durable-exec', 'test-lowest-versions', 'test-examples')

Leg = dict[str, object]


_MAPPING: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
_TOP_LEVEL: TypeAdapter[dict[str | bool, object]] = TypeAdapter(dict[str | bool, object])
_LIST: TypeAdapter[list[object]] = TypeAdapter(list[object])


def _maybe_mapping(value: object) -> dict[str, object] | None:
    try:
        return _MAPPING.validate_python(value)
    except ValidationError:
        return None


def _mapping(value: object) -> dict[str, object]:
    return _MAPPING.validate_python(value)


def _list(value: object) -> list[object]:
    return _LIST.validate_python(value)


def _workflow() -> dict[str, object]:
    loaded = _TOP_LEVEL.validate_python(yaml.safe_load(WORKFLOW.read_text(encoding='utf-8')))
    # PyYAML reads the bare `on` key as the boolean `True`.
    return {'on' if key is True else str(key): value for key, value in loaded.items()}


def _job(name: str) -> dict[str, object]:
    return _mapping(_mapping(_workflow()['jobs'])[name])


# --- A small evaluator for the GitHub Actions expressions these jobs use ----------------------------------------------

_TOKEN = re.compile(
    r"\s*(?:(?P<str>'(?:[^']|'')*')|(?P<num>\d+(?:\.\d+)?)|(?P<op>==|!=|&&|\|\||[()!,])|(?P<name>[A-Za-z_][\w.-]*))"
)


class _Expression:
    def __init__(self, source: str, context: Mapping[str, object]):
        self.tokens: list[tuple[str, str]] = []
        position = 0
        source = source.strip()
        while position < len(source):
            match = _TOKEN.match(source, position)
            assert match is not None and match.lastgroup is not None, f'cannot parse {source[position:]!r}'
            self.tokens.append((match.lastgroup, match.group(match.lastgroup)))
            position = match.end()
        self.position = 0
        self.context = context

    def evaluate(self) -> object:
        value = self._or()
        assert self.position == len(self.tokens), self.tokens[self.position :]
        return value

    def _peek(self) -> str | None:
        return self.tokens[self.position][1] if self.position < len(self.tokens) else None

    def _take(self) -> tuple[str, str]:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _or(self) -> object:
        value = self._and()
        while self._peek() == '||':
            self._take()
            right = self._and()
            value = value if _truthy(value) else right
        return value

    def _and(self) -> object:
        value = self._comparison()
        while self._peek() == '&&':
            self._take()
            right = self._comparison()
            value = right if _truthy(value) else value
        return value

    def _comparison(self) -> object:
        value = self._unary()
        while (operator := self._peek()) in ('==', '!='):
            self._take()
            equal = _equal(value, self._unary())
            value = equal if operator == '==' else not equal
        return value

    def _unary(self) -> object:
        if self._peek() == '!':
            self._take()
            return not _truthy(self._unary())
        return self._primary()

    def _primary(self) -> object:
        kind, text = self._take()
        if text == '(':
            value = self._or()
            assert self._take()[1] == ')'
            return value
        if kind == 'str':
            return text[1:-1].replace("''", "'")
        if kind == 'num':
            return float(text)
        assert kind == 'name', text
        if text in ('true', 'false'):
            return text == 'true'
        if text == 'null':
            return None
        if self._peek() == '(':
            self._take()
            arguments: list[object] = []
            while self._peek() != ')':
                arguments.append(self._or())
                if self._peek() == ',':
                    self._take()
            self._take()
            return _call(text, arguments)
        return _lookup(self.context, text)


def _truthy(value: object) -> bool:
    return value not in (False, None, 0, '')


def _equal(left: object, right: object) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.lower() == right.lower()
    if isinstance(left, (int, float)) and isinstance(right, str):
        return float(left) == float(right)
    if isinstance(left, str) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _call(name: str, arguments: list[object]) -> object:
    if name == 'fromJSON':
        (argument,) = arguments
        assert isinstance(argument, str)
        return json.loads(argument)
    raise AssertionError(f'unsupported function {name}()')


def _lookup(context: Mapping[str, object], path: str) -> object:
    value: object = context
    for part in path.split('.'):
        mapping = _maybe_mapping(value)
        if mapping is None:
            return None
        value = mapping.get(part)
    return value


def _render(value: object, context: Mapping[str, object]) -> object:
    """Evaluate a YAML value as Actions would: a lone `${{ }}` keeps its type, embedded ones become text."""
    if not isinstance(value, str) or '${{' not in value:
        return value
    whole = re.fullmatch(r'\s*\$\{\{(.*)\}\}\s*', value, re.DOTALL)
    if whole and '${{' not in whole.group(1):
        return _Expression(whole.group(1), context).evaluate()
    return re.sub(r'\$\{\{(.*?)\}\}', lambda m: _text(_Expression(m.group(1), context).evaluate()), value)


def _text(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return '' if value is None else str(value)


# --- Matrix expansion -------------------------------------------------------------------------------------------------


def _matches(leg: Leg, pattern: Mapping[str, object]) -> bool:
    for key, expected in pattern.items():
        actual = leg.get(key)
        if (expected_mapping := _maybe_mapping(expected)) is not None:
            actual_mapping = _maybe_mapping(actual)
            if actual_mapping is None or not _matches(actual_mapping, expected_mapping):
                return False
        elif _text(actual) != _text(expected):
            return False
    return True


def _expand(job_name: str, event_name: str) -> list[Leg]:
    """The legs of a job's matrix, following GitHub's product, then `exclude`, then `include` order."""
    context: dict[str, object] = {
        'github': {'event_name': event_name},
        # The full path: a `pydantic-clai2`-only change is a separate, narrower route.
        'needs': {'classify': {'outputs': {'clai2_only': 'false'}}},
    }
    matrix = _mapping(_mapping(_job(job_name)['strategy'])['matrix'])
    excludes = [_mapping(entry) for entry in _list(matrix.pop('exclude', []))]
    includes = [_mapping(entry) for entry in _list(matrix.pop('include', []))]
    dimensions = {key: [_render(value, context) for value in _list(values)] for key, values in matrix.items()}
    legs: list[Leg] = [dict(zip(dimensions, values)) for values in itertools.product(*dimensions.values())]
    legs = [leg for leg in legs if not any(_matches(leg, entry) for entry in excludes)]
    for entry in includes:
        extendable = [
            leg for leg in legs if all(key not in dimensions or _matches(leg, {key: entry[key]}) for key in entry)
        ]
        if extendable:
            for leg in extendable:
                leg.update(entry)
        else:
            legs.append(dict(entry))
    return legs


def _identity(leg: Leg) -> str:
    return json.dumps(
        {
            key: value if _maybe_mapping(value) is not None else _text(value)
            for key, value in leg.items()
            if key != 'pull_request'
        },
        sort_keys=True,
        default=str,
    )


def _lane_values(job_name: str, leg: Leg) -> dict[str, str]:
    env = _mapping(_job(job_name).get('env', {}))
    context = {'matrix': leg, 'github': {'event_name': 'merge_group'}}
    return {name: _text(_render(env[name], context)) for name in LANE_VARS if name in env}


# --- Tests ------------------------------------------------------------------------------------------------------------


def test_ci_runs_on_the_merge_queue_without_cancelling_it():
    workflow = _workflow()
    triggers = _mapping(workflow['on'])
    assert _mapping(triggers['merge_group'])['types'] == ['checks_requested']

    cancel_in_progress = _mapping(workflow['concurrency'])['cancel-in-progress']
    for event_name, cancels in (('pull_request', True), ('merge_group', False), ('push', False)):
        assert _render(cancel_in_progress, {'github': {'event_name': event_name}}) is cancels


@pytest.mark.parametrize('job_name', REDUCED_JOBS)
def test_pull_requests_run_a_subset_of_the_full_matrix(job_name: str):
    full = {_identity(leg) for leg in _expand(job_name, 'merge_group')}
    reduced = {_identity(leg) for leg in _expand(job_name, 'pull_request')}

    assert reduced < full
    assert {_identity(leg) for leg in _expand(job_name, 'push')} == full


@pytest.mark.parametrize('job_name', REDUCED_JOBS)
def test_pull_requests_keep_every_leg_that_carries_a_check(job_name: str):
    reduced = {_identity(leg) for leg in _expand(job_name, 'pull_request')}
    for leg in _expand(job_name, 'merge_group'):
        lanes = {name: value for name, value in _lane_values(job_name, leg).items() if value not in ('', 'false')}
        if lanes:
            assert _identity(leg) in reduced, f'{job_name} {leg} sets {lanes} but a pull request skips it'


@pytest.mark.parametrize('job_name', ('test', 'test-all-extras', 'test-examples'))
def test_pull_requests_test_the_oldest_and_newest_python(job_name: str):
    versions = sorted(
        {_text(leg['python-version']) for leg in _expand(job_name, 'merge_group')},
        key=lambda v: tuple(map(int, v.split('.'))),
    )
    reduced = {_text(leg['python-version']) for leg in _expand(job_name, 'pull_request')}

    assert {versions[0], versions[-1]} <= reduced


def test_every_dependency_resolution_runs_on_a_pull_request():
    resolutions = {_text(leg['resolution']) for leg in _expand('test-durable-exec', 'pull_request')}
    assert resolutions == {'locked', 'lowest'}
    assert _expand('test-lowest-versions', 'pull_request')


# Steps that add coverage, or report to the author of the pull request.
PULL_REQUEST_STEP_COMMANDS = ('coverage run', 'shard_durations.py report')


@pytest.mark.parametrize('job_name', REDUCED_JOBS)
def test_pull_requests_keep_every_leg_a_coverage_or_report_step_runs_on(job_name: str):
    reduced = {_identity(leg) for leg in _expand(job_name, 'pull_request')}
    for step in map(_mapping, _list(_job(job_name)['steps'])):
        condition, run = step.get('if'), step.get('run')
        if not isinstance(condition, str) or not isinstance(run, str):
            continue
        if 'matrix.' not in condition or not any(command in run for command in PULL_REQUEST_STEP_COMMANDS):
            continue
        condition = re.sub(r'^\s*\$\{\{(.*)\}\}\s*$', r'\1', condition, flags=re.DOTALL)
        condition = re.sub(r'^\s*!cancelled\(\)\s*&&', '', condition)
        for leg in _expand(job_name, 'merge_group'):
            if _truthy(_Expression(condition, {'matrix': leg}).evaluate()):
                assert _identity(leg) in reduced, f'{job_name}: {step.get("name")!r} runs on {leg}, which a PR skips'
