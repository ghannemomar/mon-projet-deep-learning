"""Inférence ONNX Runtime pour les modèles YOLO exportés (YOLOv8 / YOLO11 / YOLO26).

Le modèle est chargé UNE seule fois au démarrage de l'API (voir main.py, lifespan).
Deux formats de sortie sont gérés :
- YOLOv8 / YOLO11 : (1, 4 + nc, N)  -> il faut faire la NMS nous-mêmes ;
- YOLO26 (end-to-end, sans NMS) : (1, 300, 6) = x1, y1, x2, y2, score, classe.
"""

from __future__ import annotations

import ast
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Ordre de sh17.yaml (alphabétique). Sert de secours si le .onnx n'a pas de métadonnées.
SH17_NAMES = [
    "person",
    "ear",
    "ear-mufs",
    "face",
    "face-guard",
    "face-mask",
    "foot",
    "tool",
    "glasses",
    "gloves",
    "helmet",
    "hands",
    "head",
    "medical-suit",
    "shoes",
    "safety-suit",
    "safety-vest",
]


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    """Redimensionne en gardant les proportions et complète avec du gris (comme Ultralytics)."""
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size[0] - nh) // 2, (size[1] - nw) // 2
    canvas = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, r, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Non-Maximum Suppression en numpy. boxes au format x1, y1, x2, y2."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=int)


def decode_output(
    out: np.ndarray, nc: int, conf: float, iou: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transforme la sortie brute du réseau en (boîtes xyxy, scores, classes) dans le repère 640."""
    pred = out[0]
    if pred.ndim == 2 and pred.shape[1] == 6 and nc + 4 != 6:
        # YOLO26 end-to-end : déjà filtré, pas de NMS
        boxes, scores, cls = pred[:, :4], pred[:, 4], pred[:, 5].astype(int)
        keep = scores >= conf
        return boxes[keep], scores[keep], cls[keep]

    if pred.shape[0] == 4 + nc:  # (4 + nc, N) -> (N, 4 + nc)
        pred = pred.T
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(axis=1)
    scores = cls_scores.max(axis=1)
    keep = scores >= conf
    xywh, scores, cls = pred[keep, :4], scores[keep], cls[keep]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    if len(boxes) == 0:
        return boxes, scores, cls
    # NMS par classe : on décale les boîtes de chaque classe pour qu'elles ne se chevauchent pas
    offset = cls[:, None] * 10000.0
    idx = nms(boxes + offset, scores, iou)
    return boxes[idx], scores[idx], cls[idx]


class Detector:
    """Enveloppe une session ONNX Runtime et expose predict(image_bgr)."""

    def __init__(self, model_path: str | Path, conf: float = 0.25, iou: float = 0.45):
        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.imgsz = (h if isinstance(h, int) else 640, w if isinstance(w, int) else 640)
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = self._parse_names(meta.get("names"))
        self.conf, self.iou = conf, iou
        logger.info("Modèle chargé : %s (%d classes, entrée %s)", model_path, len(self.names), self.imgsz)

    @staticmethod
    def _parse_names(raw: str | None) -> list[str]:
        if not raw:
            return SH17_NAMES
        try:
            parsed = ast.literal_eval(raw)  # Ultralytics stocke "{0: 'person', 1: 'ear', ...}"
            return [parsed[i] for i in sorted(parsed)] if isinstance(parsed, dict) else list(parsed)
        except (ValueError, SyntaxError):
            return SH17_NAMES

    @property
    def providers(self) -> list[str]:
        return self.session.get_providers()

    def predict(self, img_bgr: np.ndarray, conf: float | None = None) -> dict:
        conf = self.conf if conf is None else conf
        t0 = time.perf_counter()
        canvas, r, left, top = letterbox(img_bgr, self.imgsz)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        t1 = time.perf_counter()
        out = self.session.run(None, {self.input_name: blob})[0]
        t2 = time.perf_counter()
        boxes, scores, cls = decode_output(out, len(self.names), conf, self.iou)
        # retour au repère de l'image d'origine
        h, w = img_bgr.shape[:2]
        boxes = boxes.copy()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - left) / r).clip(0, w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - top) / r).clip(0, h)
        t3 = time.perf_counter()
        detections = [
            {
                "label": self.names[int(c)] if int(c) < len(self.names) else str(int(c)),
                "class_id": int(c),
                "confidence": round(float(s), 4),
                "box": {
                    "x1": round(float(b[0]), 1),
                    "y1": round(float(b[1]), 1),
                    "x2": round(float(b[2]), 1),
                    "y2": round(float(b[3]), 1),
                },
            }
            for b, s, c in zip(boxes, scores, cls)
        ]
        return {
            "detections": detections,
            "latency_ms": {
                "preprocess": round((t1 - t0) * 1000, 2),
                "inference": round((t2 - t1) * 1000, 2),
                "postprocess": round((t3 - t2) * 1000, 2),
                "total": round((t3 - t0) * 1000, 2),
            },
        }


def load_detector() -> tuple[Detector | None, dict]:
    """Charge le modèle : d'abord depuis le MLflow Model Registry (alias production),
    sinon depuis un fichier local (MODEL_PATH). Renvoie (detector, infos)."""
    name = os.getenv("MODEL_NAME", "sh17-ppe-detector")
    alias = os.getenv("MODEL_ALIAS", "production")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        for attempt in range(int(os.getenv("MLFLOW_RETRIES", "5"))):
            try:
                import mlflow
                from mlflow import MlflowClient

                mlflow.set_tracking_uri(tracking_uri)
                version = MlflowClient().get_model_version_by_alias(name, alias)
                local = mlflow.artifacts.download_artifacts(
                    f"models:/{name}@{alias}", dst_path="/tmp/registry_model"
                )
                onnx_file = next(Path(local).rglob("*.onnx"))
                info = {
                    "source": "mlflow-registry",
                    "name": name,
                    "version": str(version.version),
                    "alias": alias,
                    "path": str(onnx_file),
                }
                return Detector(onnx_file), info
            except StopIteration:
                logger.warning("Aucun fichier .onnx dans la version %s@%s", name, alias)
                break
            except Exception as exc:  # serveur pas prêt, modèle absent...
                logger.warning("MLflow indisponible (essai %d) : %s", attempt + 1, exc)
                time.sleep(3)

    path = Path(os.getenv("MODEL_PATH", "models/model.onnx"))
    if path.exists():
        return Detector(path), {
            "source": "local-file",
            "name": name,
            "version": "local",
            "alias": None,
            "path": str(path),
        }
    logger.error("Aucun modèle trouvé (registry %s@%s ni fichier %s)", name, alias, path)
    return None, {"source": "none", "name": name, "version": None, "alias": None, "path": None}
