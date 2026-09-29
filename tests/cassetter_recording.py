"""Keep cassetter recordings from replaying earlier requests in the same run."""

from __future__ import annotations

from functools import wraps
from threading import RLock

from cassetter import Cassette, HttpResponse, NoMatchError

_original_play = Cassette.play
_original_record = Cassette.record
_recording_lock = RLock()


@wraps(_original_play)
def _play_once_while_recording(
    self: Cassette, method: str, uri: str, headers: dict[str, list[str]], body: bytes | None
) -> HttpResponse:
    if not self.can_record:
        return _original_play(self, method, uri, headers, body)

    with _recording_lock:
        played_before = self.played_indices
        counts_before = self.play_counts
        response = _original_play(self, method, uri, headers, body)
        index = next(index for index, count in self.play_counts.items() if count > counts_before[index])
        if played_before[index]:
            self._play_counter = counts_before
            raise NoMatchError(f'no unplayed interaction for {method} {uri}')
        return response


@wraps(_original_record)
def _record_as_played(
    self: Cassette,
    method: str,
    uri: str,
    request_headers: dict[str, list[str]],
    request_body: bytes | None,
    status: int,
    response_headers: dict[str, list[str]],
    response_body: bytes | None,
    order: int | None = None,
) -> HttpResponse:
    with _recording_lock:
        index = len(self.interactions)
        response = _original_record(
            self, method, uri, request_headers, request_body, status, response_headers, response_body, order
        )
        if self.can_record and len(self.interactions) > index:
            # cassetter#136: a newly recorded interaction must not satisfy the next request.
            assert self._inner is not None
            self._inner.mark_played(index)
            if self._match_inner is not None:
                self._match_inner.mark_played(index)
        return response


def install_cassetter_recording_workaround() -> None:
    Cassette.play = _play_once_while_recording
    Cassette.record = _record_as_played
