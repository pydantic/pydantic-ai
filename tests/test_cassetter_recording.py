from __future__ import annotations

from pathlib import Path

import pytest
from cassetter import Cassette, NoMatchError, RecordMode

from tests.cassetter_recording import install_cassetter_recording_workaround


@pytest.mark.parametrize('normalize_uri', [False, True])
def test_recording_does_not_replay_new_interactions(tmp_path: Path, normalize_uri: bool) -> None:
    install_cassetter_recording_workaround()
    cassette = Cassette(
        tmp_path / 'new.yaml', record_mode=RecordMode.ALL, uri_normalizer=(lambda uri: uri) if normalize_uri else None
    )
    cassette.load()
    uri = 'https://example.com/v1/chat/completions'

    cassette.record('POST', uri, {}, b'first request', 200, {}, b'first response')
    with pytest.raises(NoMatchError):
        cassette.play('POST', uri, {}, b'second request')
    cassette.record('POST', uri, {}, b'second request', 200, {}, b'second response')

    assert [item.response.body.content for item in cassette.interactions] == ['first response', 'second response']
    assert cassette.play_count == 0


def test_new_episodes_replays_existing_interaction_once(tmp_path: Path) -> None:
    install_cassetter_recording_workaround()
    path = tmp_path / 'existing.yaml'
    uri = 'https://example.com/v1/chat/completions'
    original = Cassette(path, record_mode=RecordMode.ALL)
    original.load()
    original.record('POST', uri, {}, b'first request', 200, {}, b'first response')
    original.save()

    cassette = Cassette(path, record_mode=RecordMode.NEW_EPISODES)
    cassette.load()
    assert cassette.play('POST', uri, {}, b'first request').body.content == 'first response'
    with pytest.raises(NoMatchError):
        cassette.play('POST', uri, {}, b'second request')
    cassette.record('POST', uri, {}, b'second request', 200, {}, b'second response')
    assert len(cassette.interactions) == 2


def test_new_episodes_replays_each_existing_interaction_once(tmp_path: Path) -> None:
    install_cassetter_recording_workaround()
    path = tmp_path / 'existing-turns.yaml'
    uri = 'https://example.com/v1/chat/completions'
    original = Cassette(path, record_mode=RecordMode.ALL)
    original.load()
    original.record('POST', uri, {}, b'first request', 200, {}, b'first response')
    original.record('POST', uri, {}, b'second request', 200, {}, b'second response')
    original.save()

    cassette = Cassette(path, record_mode=RecordMode.NEW_EPISODES)
    cassette.load()
    assert cassette.play('POST', uri, {}, b'first request').body.content == 'first response'
    assert cassette.play('POST', uri, {}, b'second request').body.content == 'second response'
    with pytest.raises(NoMatchError):
        cassette.play('POST', uri, {}, b'third request')
    assert cassette.play_count == 2


def test_playback_mode_can_repeat_an_interaction(tmp_path: Path) -> None:
    install_cassetter_recording_workaround()
    cassette = Cassette(tmp_path / 'playback.yaml', record_mode=RecordMode.NONE)
    uri = 'https://example.com/v1/chat/completions'
    cassette.record('POST', uri, {}, b'request', 200, {}, b'response')

    assert cassette.play('POST', uri, {}, b'request').body.content == 'response'
    assert cassette.play('POST', uri, {}, b'request').body.content == 'response'
