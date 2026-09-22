"""Inference shared by FastAPI and the demo. Never download at request time."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intentlab.common import load_bundle, sha256
from intentlab.encoder import REVISION, Encoder
from intentlab.evaluation import scale_probabilities


class RoutingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Text must contain non-whitespace characters")
        return value


class RoutingService:
    def __init__(self, directory="artifacts", encoder_directory="data/encoder"):
        self.bundle = load_bundle(directory)
        self.version = sha256(Path(directory) / "model.joblib")[:12]
        self.encoder = None
        if self.bundle["name"].startswith("minilm"):
            if self.bundle["encoder_revision"] != REVISION:
                raise ValueError("Encoder revision does not match classifier")
            for name, expected in self.bundle["encoder_hashes"].items():
                if sha256(Path(encoder_directory) / name) != expected:
                    raise ValueError("Encoder checksum mismatch")
            self.encoder = Encoder(encoder_directory)

    def predict(self, request):
        values = self.encoder.encode([request.text]) if self.encoder else [request.text]
        model = self.bundle["model"]
        p = scale_probabilities(model.predict_proba(values), self.bundle["temperature"])[0]
        indices = p.argsort()[-3:][::-1]
        review = bool(p[indices[0]] < self.bundle["threshold"])
        return {
            "intent": None if review else str(model.classes_[indices[0]]),
            "action": "human_review" if review else "route",
            "confidence": float(p[indices[0]]),
            "threshold": self.bundle["threshold"],
            "candidates": [
                {"intent": str(model.classes_[i]), "probability": float(p[i])} for i in indices
            ],
            "model": self.bundle["name"],
            "model_version": self.version,
        }
