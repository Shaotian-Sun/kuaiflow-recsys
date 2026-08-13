"""Download KuaiRand-Pure from the dataset's official Zenodo record."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile
from urllib.request import urlretrieve


DATA_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
EXPECTED_MD5 = "0820331067a3784d9691136f772b35a7"


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_kuairand_pure(raw_dir: str | Path) -> Path:
    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "KuaiRand-Pure.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".part")
        urlretrieve(DATA_URL, partial)
        partial.replace(archive)
    checksum = _md5(archive)
    if checksum != EXPECTED_MD5:
        raise RuntimeError(
            f"Checksum mismatch for {archive}: expected {EXPECTED_MD5}, got {checksum}"
        )
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(destination, filter="data")
    return destination
