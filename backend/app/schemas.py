"""Schémas Pydantic : ils valident les réponses et génèrent la doc Swagger (/docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Box(BaseModel):
    x1: float = Field(..., description="Coin haut-gauche x (pixels)")
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    label: str = Field(..., examples=["helmet"])
    class_id: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0, le=1)
    box: Box


class Compliance(BaseModel):
    alert: bool = Field(..., description="True si au moins une tête est sans casque")
    heads: int
    helmets: int
    heads_without_helmet: int
    violations: list[Box]
    message: str


class Latency(BaseModel):
    preprocess: float
    inference: float
    postprocess: float
    total: float = Field(..., description="Temps serveur total en millisecondes")


class PredictResponse(BaseModel):
    detections: list[Detection]
    count: int
    image_width: int
    image_height: int
    compliance: Compliance
    latency_ms: Latency


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool


class ModelMetadata(BaseModel):
    name: str
    version: str | None
    alias: str | None
    source: str = Field(..., description="mlflow-registry, local-file ou none")
    input_size: list[int]
    classes: list[str]
    providers: list[str]


class ErrorResponse(BaseModel):
    detail: str
