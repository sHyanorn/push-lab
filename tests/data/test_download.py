import hashlib
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from push_lab.data.download import ChecksumError, fetch
from push_lab.data.sources import Source

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class _State:
    ignore_range = False
    reject_range = False  # answer Range requests with 416
    status: int | None = None  # answer every request with this error status
    cut_after: int | None = None  # drop the connection after this many bytes, once
    requests: list[str | None]

    def __init__(self) -> None:
        self.requests = []


class _Handler(BaseHTTPRequestHandler):
    state: _State

    def do_GET(self) -> None:
        range_header = self.headers.get("Range")
        self.state.requests.append(range_header)
        if self.state.status is not None:
            self.send_error(self.state.status)
            return
        if range_header and self.state.reject_range:
            self.send_error(416)
            return
        start = 0
        if range_header and not self.state.ignore_range:
            start = int(range_header.removeprefix("bytes=").rstrip("-"))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            self.send_response(200)
        body = PAYLOAD[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.state.cut_after is not None:
            self.wfile.write(body[: self.state.cut_after])
            self.state.cut_after = None
            self.close_connection = True
            return
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, _State]]:
    state = _State()
    handler = type("Handler", (_Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.bin", state
    httpd.shutdown()
    httpd.server_close()


def _source(url: str) -> Source:
    return Source(
        dataset="test",
        url=url,
        filename="file.bin",
        size=len(PAYLOAD),
        algorithm="sha256",
        checksum=hashlib.sha256(PAYLOAD).hexdigest(),
    )


def _fetch(source: Source, dest: Path) -> Path:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return fetch(source, dest, opener=opener, backoff=(0.0, 0.0), sleep=lambda _: None)


def test_downloads_and_verifies(server: tuple[str, _State], tmp_path: Path) -> None:
    url, _ = server
    path = _fetch(_source(url), tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert not (tmp_path / "file.bin.part").exists()


def test_resumes_partial_file_with_range(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:1000])
    path = _fetch(_source(url), tmp_path)
    assert path.read_bytes() == PAYLOAD
    assert state.requests == ["bytes=1000-"]


def test_restarts_when_server_ignores_range(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    state.ignore_range = True
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:1000])
    assert _fetch(_source(url), tmp_path).read_bytes() == PAYLOAD


def test_restarts_after_416(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    state.reject_range = True
    (tmp_path / "file.bin.part").write_bytes(PAYLOAD[:1000])
    assert _fetch(_source(url), tmp_path).read_bytes() == PAYLOAD
    assert state.requests == ["bytes=1000-", None]


def test_permanent_client_error_is_not_retried(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    state.status = 404
    with pytest.raises(urllib.error.HTTPError):
        _fetch(_source(url), tmp_path)
    assert len(state.requests) == 1


def test_bad_resumed_file_is_downloaded_again(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    (tmp_path / "file.bin.part").write_bytes(b"x" * len(PAYLOAD))
    assert _fetch(_source(url), tmp_path).read_bytes() == PAYLOAD
    assert state.requests == [None]


def test_retries_after_dropped_connection(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    state.cut_after = 300_000
    assert _fetch(_source(url), tmp_path).read_bytes() == PAYLOAD
    assert state.requests == [None, "bytes=300000-"]


def test_checksum_mismatch_removes_file(server: tuple[str, _State], tmp_path: Path) -> None:
    url, _ = server
    source = replace(_source(url), checksum="0" * 64)
    with pytest.raises(ChecksumError):
        _fetch(source, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_skips_valid_existing_file(server: tuple[str, _State], tmp_path: Path) -> None:
    url, state = server
    (tmp_path / "file.bin").write_bytes(PAYLOAD)
    _fetch(_source(url), tmp_path)
    assert state.requests == []


def test_replaces_corrupt_existing_file(server: tuple[str, _State], tmp_path: Path) -> None:
    url, _ = server
    (tmp_path / "file.bin").write_bytes(b"x" * len(PAYLOAD))
    assert _fetch(_source(url), tmp_path).read_bytes() == PAYLOAD
