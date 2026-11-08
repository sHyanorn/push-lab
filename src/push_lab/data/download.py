"""Resumable, checksum-verified downloads using only the standard library.

A file is written to ``<name>.part``, resumed with an HTTP Range request when a partial
file exists, and renamed to its final name only after its size and checksum match.
"""

import argparse
import hashlib
import http.client
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from push_lab.data import paths
from push_lab.data.sources import SOURCES, Source

CHUNK = 1 << 20
BACKOFF = (2.0, 4.0, 8.0, 16.0)
RETRYABLE_CLIENT_ERRORS = frozenset({408, 429})


class ChecksumError(Exception):
    pass


def file_digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def is_valid(path: Path, source: Source) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == source.size
        and file_digest(path, source.algorithm) == source.checksum
    )


def _transfer(source: Source, part: Path, opener: urllib.request.OpenerDirector) -> None:
    offset = part.stat().st_size if part.exists() else 0
    if offset > source.size:
        part.unlink()
        offset = 0
    if offset == source.size:
        return
    request = urllib.request.Request(source.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with opener.open(request, timeout=60) as response:
        mode = "ab"
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise OSError(f"unexpected Content-Range {content_range!r} for offset {offset}")
        else:
            # The server ignored the Range header and sent the whole file
            mode = "wb"
        with part.open(mode) as f:
            while block := response.read(CHUNK):
                f.write(block)
    if part.stat().st_size != source.size:
        raise OSError(f"incomplete download: {part.stat().st_size} of {source.size} bytes")


def _download_with_retries(
    source: Source,
    part: Path,
    opener: urllib.request.OpenerDirector,
    backoff: Sequence[float],
    sleep: Callable[[float], None],
) -> None:
    for attempt in range(len(backoff) + 1):
        try:
            _transfer(source, part, opener)
            return
        except urllib.error.HTTPError as error:
            if error.code == 416:
                # The server rejects the resume offset: start the file again
                part.unlink(missing_ok=True)
            elif 400 <= error.code < 500 and error.code not in RETRYABLE_CLIENT_ERRORS:
                raise
            if attempt == len(backoff):
                raise
        except urllib.error.URLError, http.client.HTTPException, OSError:
            if attempt == len(backoff):
                raise
        sleep(backoff[attempt])


def fetch(
    source: Source,
    dest_dir: Path,
    *,
    opener: urllib.request.OpenerDirector | None = None,
    backoff: Sequence[float] = BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download ``source`` into ``dest_dir`` unless a valid copy is already there."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / source.filename
    if is_valid(final, source):
        return final
    final.unlink(missing_ok=True)
    part = dest_dir / (source.filename + ".part")
    opener = opener or urllib.request.build_opener()
    # A partial file left by an earlier run may hold bad bytes; if the resumed result fails
    # its checksum, download once more from scratch
    resumed = part.exists()
    while True:
        _download_with_retries(source, part, opener, backoff, sleep)
        digest = file_digest(part, source.algorithm)
        if digest == source.checksum:
            break
        part.unlink()
        if not resumed:
            raise ChecksumError(
                f"{source.filename}: {source.algorithm} {digest} != expected {source.checksum}"
            )
        resumed = False
    os.replace(part, final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the project's datasets.")
    parser.add_argument("--only", nargs="+", choices=sorted(SOURCES), help="datasets to fetch")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root")
    args = parser.parse_args(argv)
    for dataset in args.only or sorted(SOURCES):
        for source in SOURCES[dataset]:
            path = fetch(source, paths.raw_dir(args.root, dataset))
            print(f"{dataset}: {path} ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
