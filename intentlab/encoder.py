"""Pinned MiniLM encoder: masked mean pooling and L2 normalization on CPU."""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def download_encoder(directory="data/encoder"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for remote, local in [
        ("tokenizer.json", "tokenizer.json"),
        ("onnx/model.onnx", "model.onnx"),
    ]:
        dest = directory / local
        if not dest.exists():
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{remote}"
            partial = dest.with_suffix(".part")
            with (
                urllib.request.urlopen(url, timeout=120) as source,
                partial.open("wb") as target,
            ):
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
            partial.replace(dest)
    return directory


class Encoder:
    def __init__(self, directory="data/encoder"):
        directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        settings = ort.SessionOptions()
        settings.intra_op_num_threads = 2
        settings.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"), settings, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts, batch_size=32):
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer.encode_batch(list(texts[start : start + batch_size]))
            inputs = {
                "input_ids": np.array([t.ids for t in tokens], dtype=np.int64),
                "attention_mask": np.array([t.attention_mask for t in tokens], dtype=np.int64),
                "token_type_ids": np.array([t.type_ids for t in tokens], dtype=np.int64),
            }
            expected = {item.name for item in self.session.get_inputs()}
            hidden = self.session.run(None, {k: v for k, v in inputs.items() if k in expected})[0]
            mask = inputs["attention_mask"][..., None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
            batches.append(pooled.astype(np.float32))
        return np.concatenate(batches)

    def cached(self, texts, directory="data/embeddings"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(json.dumps([REVISION, 256, list(texts)]).encode()).hexdigest()
        path = directory / f"{digest}.npy"
        if path.exists():
            return np.load(path, allow_pickle=False)
        values = self.encode(texts)
        np.save(path, values, allow_pickle=False)
        return values
