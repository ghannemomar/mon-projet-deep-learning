"""API FastAPI de détection d'EPI.

Endpoints :
- GET  /health            : l'API tourne-t-elle, le modèle est-il chargé ?
- GET  /model/metadata    : version du modèle (MLflow Registry), classes, taille d'entrée
- POST /predict           : image -> détections JSON + conformité + latence
- POST /predict/visualize : image -> image JPEG annotée (boîtes colorées + bandeau d'alerte)
Swagger : http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.inference import Detector, load_detector
from app.rules import check_compliance
from app.schemas import ErrorResponse, HealthResponse, ModelMetadata, PredictResponse

logging.basicConfig(level=logging.INFO)
MAX_BYTES = 10 * 1024 * 1024  # 10 Mo


@asynccontextmanager
async def lifespan(app: FastAPI):
    # chargé une seule fois au démarrage, réutilisé par toutes les requêtes
    app.state.detector, app.state.model_info = load_detector()
    yield


app = FastAPI(
    title="API Détection d'EPI (SH17 + YOLO)",
    description="Détection de personnes et d'équipements de protection, alerte si tête sans casque.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

ERRORS = {400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


def get_detector(request: Request) -> Detector:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return detector


async def read_image(file: UploadFile) -> np.ndarray:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide")
    if len(content) > MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image trop lourde (max 10 Mo)")
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Le fichier n'est pas une image valide")
    return img


async def run_prediction(request: Request, file: UploadFile, conf: float):
    detector = get_detector(request)
    img = await read_image(file)
    result = await run_in_threadpool(detector.predict, img, conf)  # n'empêche pas les autres requêtes
    return img, result, check_compliance(result["detections"])


@app.get("/health", response_model=HealthResponse, tags=["système"])
async def health(request: Request):
    loaded = getattr(request.app.state, "detector", None) is not None
    return {"status": "ok", "model_loaded": loaded}


@app.get("/model/metadata", response_model=ModelMetadata, responses=ERRORS, tags=["modèle"])
async def model_metadata(request: Request):
    detector = get_detector(request)
    info = request.app.state.model_info
    return {
        "name": info["name"],
        "version": info["version"],
        "alias": info["alias"],
        "source": info["source"],
        "input_size": list(detector.imgsz),
        "classes": detector.names,
        "providers": detector.providers,
    }


@app.post("/predict", response_model=PredictResponse, responses=ERRORS, tags=["détection"])
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Image JPEG ou PNG"),
    conf: float = Query(0.25, ge=0.01, le=1.0, description="Seuil de confiance"),
):
    img, result, compliance = await run_prediction(request, file, conf)
    return {
        "detections": result["detections"],
        "count": len(result["detections"]),
        "image_width": img.shape[1],
        "image_height": img.shape[0],
        "compliance": compliance,
        "latency_ms": result["latency_ms"],
    }


def color_for(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id * 7 + 3)  # couleur stable par classe
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


@app.post(
    "/predict/visualize",
    responses={**ERRORS, 200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
    tags=["détection"],
)
async def predict_visualize(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=1.0),
):
    img, result, compliance = await run_prediction(request, file, conf)
    thick = max(2, round(max(img.shape[:2]) / 500))
    for d in result["detections"]:
        b = d["box"]
        p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
        color = color_for(d["class_id"])
        cv2.rectangle(img, p1, p2, color, thick)
        cv2.putText(
            img,
            f"{d['label']} {d['confidence']:.2f}",
            (p1[0], max(p1[1] - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * thick / 2,
            color,
            max(1, thick // 2),
        )
    banner = (0, 0, 220) if compliance["alert"] else (0, 160, 0)
    cv2.rectangle(img, (0, 0), (img.shape[1], 40 * thick // 2), banner, -1)
    cv2.putText(
        img,
        compliance["message"].encode("ascii", "ignore").decode(),
        (10, 28 * thick // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * thick / 2,
        (255, 255, 255),
        max(1, thick // 2),
    )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"X-Latency-Ms": str(result["latency_ms"]["total"])},
    )
