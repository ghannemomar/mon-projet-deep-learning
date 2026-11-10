"""Petites aides MLflow partagées par train / evaluate / export."""

from __future__ import annotations

import os

import mlflow

EXPERIMENT = "sh17-ppe"
MODEL_NAME = "sh17-ppe-detector"


def setup_experiment() -> str:
    """Crée (ou réutilise) l'expérience. MLFLOW_ARTIFACT_ROOT fixe où vont les artefacts
    (sur Kaggle : /mlflow/mlruns, le même chemin que dans le conteneur docker-compose)."""
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        exp_id = mlflow.create_experiment(EXPERIMENT, artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT"))
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def clean(name: str) -> str:
    """Nom de métrique Ultralytics -> nom MLflow lisible ("metrics/mAP50(B)" -> "val_mAP50")."""
    name = name.strip().replace("(B)", "")
    return name.replace("metrics/", "val_").replace("/", "_")
