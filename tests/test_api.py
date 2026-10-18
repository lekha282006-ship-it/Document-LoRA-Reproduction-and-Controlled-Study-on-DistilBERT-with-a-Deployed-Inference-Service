"""Tests for FastAPI inference service (/health and /predict)."""

import pytest
from fastapi.testclient import TestClient
from src.serve import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """GET /health must return 200 and report health status and model."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "status" in data
    assert "model" in data
    assert "device" in data
    assert "adapter_loaded" in data


def test_predict_positive_sentiment(client):
    """POST /predict must return valid label and confidence score for positive sentence."""
    payload = {"text": "An astonishing, poignant, and uplifting masterpiece."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "class_id" in data
    assert data["label"] in ["positive", "negative"]
    assert 0.0 <= data["score"] <= 1.0
    assert data["class_id"] in [0, 1]


def test_predict_negative_sentiment(client):
    """POST /predict must return valid label and confidence score for negative sentence."""
    payload = {"text": "A horribly dull, wretched, and thoroughly unwatchable disaster."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

    data = response.json()
    assert "label" in data
    assert "score" in data
    assert "class_id" in data
    assert data["label"] in ["positive", "negative"]
    assert 0.0 <= data["score"] <= 1.0


def test_predict_empty_text_validation(client):
    """POST /predict must return 422 unprocessable entity or 400 bad request for empty string."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code in [400, 422]
