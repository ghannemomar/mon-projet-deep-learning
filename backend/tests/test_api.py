"""Tests de l'API (lancés par GitHub Actions à chaque push)."""

import numpy as np

from app.inference import decode_output, letterbox, nms
from app.rules import check_compliance

# ---------- endpoints ----------


def test_health_sans_modele(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": False}


def test_predict_503_sans_modele(client, jpeg_bytes):
    r = client.post("/predict", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 503


def test_metadata_503_sans_modele(client):
    assert client.get("/model/metadata").status_code == 503


def test_predict_400_fichier_pas_image(client_with_model):
    r = client_with_model.post("/predict", files={"file": ("a.txt", b"bonjour", "text/plain")})
    assert r.status_code == 400


def test_predict_400_fichier_vide(client_with_model):
    r = client_with_model.post("/predict", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_predict_ok(client_with_model, jpeg_bytes):
    r = client_with_model.post("/predict", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert body["image_width"] == 300 and body["image_height"] == 200
    assert body["compliance"]["alert"] is True
    assert body["compliance"]["heads_without_helmet"] == 1
    assert body["latency_ms"]["total"] > 0


def test_predict_conf_invalide(client_with_model, jpeg_bytes):
    r = client_with_model.post("/predict?conf=5", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 422


def test_visualize_renvoie_jpeg(client_with_model, jpeg_bytes):
    r = client_with_model.post("/predict/visualize", files={"file": ("a.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"


def test_metadata_ok(client_with_model):
    r = client_with_model.get("/model/metadata")
    assert r.status_code == 200
    assert "helmet" in r.json()["classes"]


def test_swagger_disponible(client):
    assert client.get("/docs").status_code == 200
    assert "/predict" in client.get("/openapi.json").json()["paths"]


# ---------- règle d'alerte ----------


def _det(label, x1, y1, x2, y2):
    return {"label": label, "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def test_regle_tete_avec_casque_conforme():
    res = check_compliance([_det("head", 0, 20, 40, 60), _det("helmet", 2, 5, 38, 30)])
    assert res["alert"] is False


def test_regle_tete_sans_casque_alerte():
    res = check_compliance([_det("head", 0, 20, 40, 60)])
    assert res["alert"] is True and res["heads_without_helmet"] == 1


def test_regle_casque_loin_de_la_tete():
    res = check_compliance([_det("head", 0, 20, 40, 60), _det("helmet", 300, 300, 340, 330)])
    assert res["alert"] is True


def test_regle_aucune_tete():
    assert check_compliance([])["alert"] is False


# ---------- post-traitement ----------


def test_letterbox_taille_et_echelle():
    img = np.zeros((480, 960, 3), np.uint8)
    canvas, r, left, top = letterbox(img, (640, 640))
    assert canvas.shape == (640, 640, 3)
    assert r == 640 / 960 and left == 0 and top == 160


def test_nms_supprime_doublon():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)
    assert sorted(keep.tolist()) == [0, 2]


def test_decode_format_yolov8():
    nc, n = 17, 8400
    out = np.zeros((1, 4 + nc, n), np.float32)
    out[0, :4, 0] = [320, 320, 100, 100]  # une boîte centrée
    out[0, 4 + 10, 0] = 0.9  # classe 10 = helmet
    boxes, scores, cls = decode_output(out, nc, conf=0.25, iou=0.45)
    assert len(boxes) == 1 and cls[0] == 10
    assert np.allclose(boxes[0], [270, 270, 370, 370])


def test_decode_format_yolo26_end2end():
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = [10, 10, 50, 50, 0.8, 12]
    boxes, scores, cls = decode_output(out, 17, conf=0.25, iou=0.45)
    assert len(boxes) == 1 and cls[0] == 12
