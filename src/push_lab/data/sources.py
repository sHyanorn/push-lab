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

# Small sample shipped in the Open Bandit Pipeline repository, pinned to one commit
OBD_COMMIT = "8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4"
_OBD_BASE = f"https://raw.githubusercontent.com/st-tech/zr-obp/{OBD_COMMIT}/obd"


def _obd(policy: str, kind: str, size: int, sha256: str) -> Source:
    remote = "men.csv" if kind == "log" else "item_context.csv"
    local = f"{policy}_men.csv" if kind == "log" else f"{policy}_men_item_context.csv"
    return Source(
        dataset="obd",
        url=f"{_OBD_BASE}/{policy}/men/{remote}",
        filename=local,
        size=size,
        # Computed on 2026-11-08 from the pinned commit
        algorithm="sha256",
        checksum=sha256,
    )


OBD_MEN = (
    _obd(
        "random",
        "log",
        3_336_811,
        "ede6d08c6877f99cd3770f7d09e26c738e6aa6de4445ccccfb1356ba5d06b1eb",
    ),
    _obd(
        "random", "items", 4_287, "6d855c2b7e917d0448d71bbe62c023147495882c0fdc8c41e86fef86abf9fe09"
    ),
    _obd(
        "bts", "log", 3_240_432, "db6d1f7ed2d591ba521533e45286d4c1fba7fe57a8ad2e80342210335f43c527"
    ),
    _obd("bts", "items", 4_287, "3b411ed6b0471316321a6b93f4f756bb25888eac6218cd71d0b603d41d85732d"),
)

SOURCES: dict[str, tuple[Source, ...]] = {
    "kuairand": (KUAIRAND_PURE,),
    "hillstrom": (HILLSTROM,),
    "criteo": (CRITEO_UPLIFT,),
    "obd": OBD_MEN,
}
