import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pour "import app"
os.environ.pop("MLFLOW_TRACKING_URI", None)
os.environ["MODEL_PATH"] = "/chemin/inexistant/model.onnx"  # aucun vrai modèle en CI

from app.main import app  # noqa: E402


class FakeDetector:
    """Remplace le vrai modèle ONNX : les tests de l'API ne dépendent pas des poids."""

    imgsz = (640, 640)
    names = ["person", "head", "helmet"]
    providers = ["CPUExecutionProvider"]

    def predict(self, img, conf=None):
        return {
            "detections": [
                {
                    "label": "head",
                    "class_id": 12,
                    "confidence": 0.9,
                    "box": {"x1": 10, "y1": 10, "x2": 50, "y2": 50},
                },
                {
                    "label": "head",
                    "class_id": 12,
                    "confidence": 0.8,
                    "box": {"x1": 100, "y1": 10, "x2": 140, "y2": 50},
                },
                {
                    "label": "helmet",
                    "class_id": 10,
                    "confidence": 0.7,
                    "box": {"x1": 12, "y1": 0, "x2": 48, "y2": 20},
                },
            ],
            "latency_ms": {"preprocess": 1.0, "inference": 5.0, "postprocess": 0.5, "total": 6.5},
        }


@pytest.fixture
def client():
    with TestClient(app) as c:  # déclenche le lifespan (aucun modèle trouvé)
        yield c


@pytest.fixture
def client_with_model(client):
    app.state.detector = FakeDetector()
    app.state.model_info = {"source": "test", "name": "fake", "version": "0", "alias": None}
    yield client
    app.state.detector = None


@pytest.fixture
def jpeg_bytes():
    img = np.full((200, 300, 3), 127, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()
