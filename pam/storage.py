"""Result storage: local directory (dev) or S3 (AWS). Chosen by PAM_S3_BUCKET."""
from __future__ import annotations

import json
import os
from pathlib import Path


class LocalStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def get(self, key: str) -> bytes | None:
        p = self.root / key
        return p.read_bytes() if p.exists() else None

    def describe(self) -> str:
        return f"local:{self.root}"


class S3Store:
    def __init__(self, bucket: str, prefix: str = "pam/"):
        import boto3
        self.s3 = boto3.client("s3")
        self.bucket, self.prefix = bucket, prefix

    def put(self, key, data, content_type="application/octet-stream"):
        self.s3.put_object(Bucket=self.bucket, Key=self.prefix + key, Body=data, ContentType=content_type,
                           ServerSideEncryption="AES256")

    def get(self, key):
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=self.prefix + key)["Body"].read()
        except self.s3.exceptions.NoSuchKey:
            return None

    def describe(self):
        return f"s3://{self.bucket}/{self.prefix}"


def make_store():
    bucket = os.environ.get("PAM_S3_BUCKET")
    if bucket:
        return S3Store(bucket, os.environ.get("PAM_S3_PREFIX", "pam/"))
    return LocalStore(os.environ.get("PAM_DATA_DIR", "/tmp/pam-results"))


def put_json(store, key: str, obj) -> None:
    store.put(key, json.dumps(obj, indent=1).encode(), "application/json")
