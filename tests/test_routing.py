import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from intentlab.api import create_app
from intentlab.common import load_bundle, save_bundle
from intentlab.data import normalize
from intentlab.evaluation import (
    add_typo,
    fit_temperature,
    routing_metrics,
    scale_probabilities,
    select_threshold,
)
from intentlab.service import RoutingRequest, RoutingService


@pytest.mark.parametrize("value", ["", "   ", "x" * 2001])
def test_invalid_text_is_rejected(value):
    with pytest.raises(ValidationError):
        RoutingRequest(text=value)


def test_normalization_and_typo_are_deterministic():
    assert normalize("  Lost   CARD ") == "lost card"
    assert add_typo("please help me") == "pelase help me"
    assert RoutingRequest(text=" help ").text == "help"


def test_temperature_keeps_class_order_and_probability_sum():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    calibrated = scale_probabilities(p, 0.5)
    assert np.allclose(calibrated.sum(axis=1), 1)
    assert np.array_equal(calibrated.argmax(axis=1), p.argmax(axis=1))
    assert calibrated[0, 0] > p[0, 0]
    temperature = fit_temperature(np.array(["a", "c"]), p, np.array(["a", "b", "c"]))
    assert 0.2 <= temperature <= 5


def test_policy_abstains_when_validation_evidence_is_insufficient():
    classes = np.array(["a", "b"])
    p = np.tile([0.9, 0.1], (100, 1))
    y = np.array(["a", "b"] * 50)
    threshold, _ = select_threshold(y, p, classes)
    assert threshold > 1
    result = routing_metrics(y, p, classes, threshold)
    assert result["accepted"] == 0 and result["accuracy"] is None


def test_policy_accepts_high_confidence_correct_predictions():
    p = np.tile([0.98, 0.02], (200, 1))
    y, classes = np.array(["a"] * 200), np.array(["a", "b"])
    threshold, _ = select_threshold(y, p, classes)
    assert routing_metrics(y, p, classes, threshold)["coverage"] == 1


@pytest.fixture
def model_directory(tmp_path):
    text = [
        "lost bank card",
        "my card is missing",
        "transfer failed",
        "bank transfer missing",
        "refund pending",
        "where is my refund",
    ]
    labels = ["card", "card", "transfer", "transfer", "refund", "refund"]
    model = make_pipeline(TfidfVectorizer(), LogisticRegression()).fit(text, labels)
    save_bundle(
        tmp_path,
        {"model": model, "threshold": 1.01, "temperature": 1.0, "name": "tfidf_test"},
    )
    return tmp_path


def test_api_review_does_not_assign_an_intent(model_directory):
    service = RoutingService(model_directory)
    with TestClient(create_app(service)) as client:
        assert client.get("/ready").status_code == 200
        response = client.post("/predict", json={"text": "I lost my card"})
        result = response.json()
        assert response.status_code == 200
        assert result["action"] == "human_review" and result["intent"] is None
        assert len(result["candidates"]) == 3
        assert "text" not in result
        assert client.post("/predict", json={"text": "  "}).status_code == 422


def test_model_corruption_is_rejected(model_directory):
    with (model_directory / "model.joblib").open("ab") as target:
        target.write(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(model_directory)


def test_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"text": "hello"}).status_code == 503


@pytest.mark.skipif(
    not Path("artifacts/model.joblib").exists(),
    reason="Run training for full encoder integration",
)
def test_trained_service_and_report_agree():
    service = RoutingService()
    report = json.loads(Path("reports/metrics.json").read_text())
    with TestClient(create_app(service)) as client:
        result = client.post(
            "/predict", json={"text": "I lost my card and need to block it"}
        ).json()
    assert result["model"] == report["selected_model"]
    assert result["threshold"] == report["routing_test"]["threshold"]
    assert result["model_version"] == report["artifact"]["sha256"][:12]
    assert np.isfinite([r["probability"] for r in result["candidates"]]).all()
