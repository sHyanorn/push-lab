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

HILLSTROM = Source(
    dataset="hillstrom",
    # The host serves no valid certificate for HTTPS; integrity rests on the checksum (D015)
    url=(
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    filename="hillstrom.csv",
    size=3_964_977,
    # Computed on 2026-11-08; the source publishes no checksum
    algorithm="sha256",
    checksum="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
)

CRITEO_UPLIFT = Source(
    dataset="criteo",
    url=(
        "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
        "criteo-research-uplift-v2.1.csv.gz"
    ),
    filename="criteo-research-uplift-v2.1.csv.gz",
    size=311_422_618,
    # Git LFS object id published by Hugging Face
    algorithm="sha256",
    checksum="2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc",
)

SOURCES: dict[str, tuple[Source, ...]] = {
    "kuairand": (KUAIRAND_PURE,),
    "hillstrom": (HILLSTROM,),
    "criteo": (CRITEO_UPLIFT,),
}
