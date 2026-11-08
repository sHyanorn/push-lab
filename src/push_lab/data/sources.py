"""Download sources with their expected sizes and checksums."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Source:
    dataset: str
    url: str
    filename: str
    size: int
    algorithm: Literal["md5", "sha256"]
    checksum: str


KUAIRAND_PURE = Source(
    dataset="kuairand",
    url="https://zenodo.org/api/records/10439422/files/KuaiRand-Pure.tar.gz/content",
    filename="KuaiRand-Pure.tar.gz",
    size=47_432_272,
    # Published by Zenodo for record 10439422
    algorithm="md5",
    checksum="0820331067a3784d9691136f772b35a7",
)

SOURCES: dict[str, tuple[Source, ...]] = {
    "kuairand": (KUAIRAND_PURE,),
}
