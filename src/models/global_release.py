"""Immutable release packaging and retrieval for the global candidate."""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import re
import shutil
import tarfile
from typing import Any

import mlflow.pyfunc
import yaml

from src.models.global_artifact import GLOBAL_REGISTERED_MODEL_NAME
from src.optimization.global_onnx import validate_global_onnx_bundle
from src.serving.artifacts import download_verified_model
from src.serving.global_api import validate_global_forecaster
from src.serving.global_onnx_forecaster import GlobalONNXVolatilityForecaster

GLOBAL_RELEASE_SCHEMA_VERSION = 1
GLOBAL_RELEASE_ROLE = "global_candidate"
GLOBAL_ONNX_NAME = "global_volatility_lstm_fp32.onnx"
GLOBAL_ONNX_MANIFEST_NAME = "global_volatility_lstm_fp32.json"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GlobalReleaseConfig:
    """One checksum-pinned global candidate release identity."""

    version: str
    tag: str
    asset_name: str
    url: str
    sha256: str
    registered_model_name: str
    registered_model_version: int


@dataclass(frozen=True)
class GlobalReleaseArchive:
    """Metadata emitted after deterministic archive construction."""

    path: Path
    sha256: str
    size_bytes: int
    files: int
    registered_model_version: int


def load_global_release_config(path: Path) -> GlobalReleaseConfig:
    """Load and strictly validate the immutable candidate release pin."""
    with path.open() as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Global release config schema_version must be 1")
    release = raw.get("release")
    if not isinstance(release, dict):
        raise ValueError("Global release config must contain a release mapping")
    if release.get("role") != GLOBAL_RELEASE_ROLE:
        raise ValueError("Global release role must be 'global_candidate'")
    string_fields = ("version", "tag", "asset_name", "url", "sha256")
    if any(
        not isinstance(release.get(field), str) or not release[field]
        for field in string_fields
    ):
        raise ValueError("Global release string fields must be non-empty")
    if release.get("registered_model_name") != GLOBAL_REGISTERED_MODEL_NAME:
        raise ValueError("Global release registered-model name is incompatible")
    model_version = release.get("registered_model_version")
    if not isinstance(model_version, int) or model_version <= 0:
        raise ValueError("Global release registered-model version must be positive")
    if release["version"] != release["tag"]:
        raise ValueError("Global release version and tag must match")
    if release["asset_name"] != f"{release['tag']}.tar.gz":
        raise ValueError("Global release asset name must be derived from its tag")
    expected_suffix = (
        f"/releases/download/{release['tag']}/{release['asset_name']}"
    )
    if not release["url"].startswith("https://github.com/") or not release[
        "url"
    ].endswith(expected_suffix):
        raise ValueError("Global release URL must identify the immutable GitHub asset")
    if not SHA256_PATTERN.fullmatch(release["sha256"]):
        raise ValueError("Global release SHA-256 must contain 64 lowercase hex digits")
    return GlobalReleaseConfig(
        version=release["version"],
        tag=release["tag"],
        asset_name=release["asset_name"],
        url=release["url"],
        sha256=release["sha256"],
        registered_model_name=release["registered_model_name"],
        registered_model_version=model_version,
    )


def _release_paths(directory: Path) -> dict[str, Path]:
    return {
        "artifact_manifest": directory / "artifacts" / "manifest.json",
        "model_state": directory / "artifacts" / "model_state.pt",
        "scaler": directory / "artifacts" / "scaler.pkl",
        "onnx": directory / "optimized" / GLOBAL_ONNX_NAME,
        "onnx_manifest": directory / "optimized" / GLOBAL_ONNX_MANIFEST_NAME,
        "registered_model_meta": directory / "registered_model_meta",
        "mlmodel": directory / "MLmodel",
        "python_model": directory / "python_model.pkl",
    }


def validate_global_release_directory(
    directory: Path,
    *,
    registered_model_version: int,
) -> dict[str, Any]:
    """Require one extracted snapshot to satisfy both candidate runtimes."""
    if registered_model_version <= 0:
        raise ValueError("Registered model version must be positive")
    if (directory / "CI_MODEL_NOTICE.txt").exists():
        raise ValueError("CI fixtures cannot be published as global candidates")
    paths = _release_paths(directory)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Global release directory is incomplete; missing: " + ", ".join(missing)
        )

    with paths["registered_model_meta"].open() as metadata_file:
        registry_metadata = yaml.safe_load(metadata_file)
    if not isinstance(registry_metadata, dict):
        raise ValueError("Global release registry metadata must be a mapping")
    if registry_metadata.get("model_name") != GLOBAL_REGISTERED_MODEL_NAME:
        raise ValueError("Global release registry model name is incompatible")
    if str(registry_metadata.get("model_version")) != str(
        registered_model_version
    ):
        raise ValueError("Global release registry model version is incompatible")

    artifact_manifest, export_manifest = validate_global_onnx_bundle(
        model_path=paths["onnx"],
        scaler_path=paths["scaler"],
        artifact_manifest_path=paths["artifact_manifest"],
        export_manifest_path=paths["onnx_manifest"],
    )
    pyfunc = mlflow.pyfunc.load_model(str(directory))
    validate_global_forecaster(pyfunc)
    onnx = GlobalONNXVolatilityForecaster(
        paths["onnx"],
        paths["scaler"],
        paths["artifact_manifest"],
        paths["onnx_manifest"],
        threads=1,
    )
    validate_global_forecaster(onnx)
    return {
        "registered_model_name": GLOBAL_REGISTERED_MODEL_NAME,
        "registered_model_version": registered_model_version,
        "artifact_manifest": artifact_manifest,
        "onnx_manifest": export_manifest,
    }


def _archive_files(source: Path) -> list[Path]:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Global release source contains no files")
    non_files = [
        path
        for path in source.rglob("*")
        if not path.is_file() and not path.is_dir()
    ]
    if non_files:
        raise ValueError("Global release source may contain only files and directories")
    return files


def create_global_release_archive(
    source: Path,
    output: Path,
    *,
    registered_model_version: int,
) -> GlobalReleaseArchive:
    """Validate and write a byte-reproducible candidate tarball."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")
    validate_global_release_directory(
        source,
        registered_model_version=registered_model_version,
    )
    files = _archive_files(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as raw_archive:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_archive,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for path in files:
                    info = archive.gettarinfo(
                        str(path),
                        arcname=path.relative_to(source).as_posix(),
                    )
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    info.pax_headers = {}
                    with path.open("rb") as source_file:
                        archive.addfile(info, source_file)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return GlobalReleaseArchive(
        path=output,
        sha256=digest,
        size_bytes=output.stat().st_size,
        files=len(files),
        registered_model_version=registered_model_version,
    )


def download_global_release(config_path: Path, output: Path) -> Path:
    """Download, checksum, extract, and runtime-validate the configured release."""
    config = load_global_release_config(config_path)
    downloaded = download_verified_model(config.url, config.sha256, output)
    try:
        validate_global_release_directory(
            downloaded,
            registered_model_version=config.registered_model_version,
        )
    except Exception:
        shutil.rmtree(downloaded)
        raise
    return downloaded
