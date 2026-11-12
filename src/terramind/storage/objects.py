"""Object storage for map layers and run artifacts: MinIO (S3) or the local disk.

The explorer's PNG layers and each run's report.json are published to a bucket
so the API can serve them without the training machine's file system.

    TERRAMIND_S3_ENDPOINT=127.0.0.1:9000
    TERRAMIND_S3_ACCESS_KEY=<MINIO_ROOT_USER from .env>
    TERRAMIND_S3_SECRET_KEY=<MINIO_ROOT_PASSWORD from .env>
"""

from __future__ import annotations

import io
import os
import shutil
from pathlib import Path

from terramind.config import data_dir, repo_root

BUCKET = "terramind"


class LocalStore:
    def __init__(self, root: Path | None = None):
        self.root = root or data_dir() / "derived" / "objects"

    def put(self, key: str, path: Path) -> None:
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)

    def get(self, key: str) -> bytes:
        return (self.root / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.root / key).exists()


class S3Store:
    def __init__(self):
        from minio import Minio

        self.client = Minio(
            os.environ["TERRAMIND_S3_ENDPOINT"],
            access_key=os.environ["TERRAMIND_S3_ACCESS_KEY"],
            secret_key=os.environ["TERRAMIND_S3_SECRET_KEY"],
            secure=os.environ.get("TERRAMIND_S3_SECURE", "0") == "1",
        )
        if not self.client.bucket_exists(BUCKET):
            self.client.make_bucket(BUCKET)

    def put(self, key: str, path: Path) -> None:
        self.client.fput_object(BUCKET, key, str(path))

    def get(self, key: str) -> bytes:
        res = self.client.get_object(BUCKET, key)
        try:
            return io.BytesIO(res.read()).getvalue()
        finally:
            res.close()
            res.release_conn()

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self.client.stat_object(BUCKET, key)
            return True
        except S3Error:
            return False


def open_objects():
    return S3Store() if os.environ.get("TERRAMIND_S3_ENDPOINT") else LocalStore()


def publish() -> dict[str, int]:
    """Upload explorer layers and run reports."""
    from terramind.explorer.build import out_dir as explorer_dir

    store = open_objects()
    counts = {"layers": 0, "reports": 0}
    for png in explorer_dir().glob("*/*.png"):
        store.put(f"layers/{png.parent.name}/{png.name}", png)
        counts["layers"] += 1
    store.put("layers/index.json", explorer_dir() / "index.json")
    for report in (repo_root() / "experiments").glob("**/runs/*/report.json"):
        store.put(
            f"runs/{report.parent.parent.parent.name}/{report.parent.name}/report.json", report
        )
        counts["reports"] += 1
    return counts
