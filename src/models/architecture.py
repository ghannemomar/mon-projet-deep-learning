"""Les architectures comparées. Toutes partent de poids pré-entraînés sur COCO (transfer learning)."""

from ultralytics import YOLO

MODELS = {
    "yolov8n": "yolov8n.pt",  # baseline : même modèle que le benchmark SH17 (58,0 mAP50)
    "yolo11n": "yolo11n.pt",
    "yolo11s": "yolo11s.pt",
    "yolo26n": "yolo26n.pt",  # end-to-end, sans NMS : sortie ONNX (1, 300, 6)
    "yolo26s": "yolo26s.pt",
}


def build_model(name: str) -> YOLO:
    return YOLO(MODELS.get(name, name))
