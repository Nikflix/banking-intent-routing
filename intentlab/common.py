"""Artifact integrity, acquisition, and statistical utilities."""

import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import sklearn


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire(url, path, expected):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "portfolio-ml/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Dataset checksum mismatch; review upstream changes")
        path.write_bytes(data)
    if sha256(path) != expected:
        raise ValueError("Cached dataset checksum mismatch")
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def environment():
    return {
        "python": sys.version.split()[0],
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seed": 42,
    }


def save_bundle(directory, payload):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, directory / "model.joblib", compress=3)
    manifest = {
        "sha256": sha256(directory / "model.joblib"),
        "environment": environment(),
    }
    write_json(directory / "manifest.json", manifest)
    return manifest


def load_bundle(directory):
    # Only load trusted local artifacts. A matching hash is integrity, not authentication.
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if sha256(directory / "model.joblib") != manifest["sha256"]:
        raise ValueError("Model checksum mismatch")
    if manifest["environment"]["sklearn"] != sklearn.__version__:
        raise ValueError("scikit-learn version mismatch; recreate the training environment")
    return joblib.load(directory / "model.joblib")


def wilson(successes, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p = successes / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(center - margin), float(center + margin)]
