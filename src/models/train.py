"""Entraînement d'un modèle YOLO sur SH17 avec suivi MLflow.

Exemples :
    python -m src.models.train                                  # réglages de params.yaml (baseline)
    python -m src.models.train --model yolo26n --name yolo26n-640
    python -m src.models.train --model yolov8n --imgsz 960 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from ultralytics import YOLO, settings

from src.models.architecture import build_model
from src.utils.mlflow_utils import clean, setup_experiment

settings.update({"mlflow": False})  # on logge nous-mêmes, pour tout contrôler


def data_yaml() -> str:
    """Écrit un yaml avec chemin absolu (marche en local, sur Kaggle et sur Colab)."""
    data = yaml.safe_load(open("data/processed/sh17.yaml"))
    data["path"] = str(Path("data/processed").resolve())
    Path("runs").mkdir(exist_ok=True)
    yaml.safe_dump(data, open("runs/data.yaml", "w"), sort_keys=False)
    return "runs/data.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    for key, typ in [("model", str), ("imgsz", int), ("epochs", int), ("batch", int), ("fraction", float)]:
        ap.add_argument(f"--{key}", type=typ)
    ap.add_argument("--name")
    args = ap.parse_args()

    p = yaml.safe_load(open("params.yaml"))["train"]
    p.update({k: v for k, v in vars(args).items() if v is not None and k != "name"})
    p["model"] = Path(p["model"]).stem
    run_name = args.name or f"{p['model']}-{p['imgsz']}"
    data = data_yaml()

    setup_experiment()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(p)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        mlflow.set_tags({"git_commit": commit or "inconnu", "dataset": "SH17 (data/processed, dvc.lock)"})
        if Path("dvc.lock").exists():
            mlflow.log_artifact("dvc.lock", "data_version")

        model = build_model(p["model"])
        model.train(
            data=data,
            imgsz=p["imgsz"],
            epochs=p["epochs"],
            batch=p["batch"],
            patience=p["patience"],
            seed=p["seed"],
            deterministic=True,
            cache=p.get("cache", False),
            fraction=p.get("fraction", 1.0),
            workers=p.get("workers", 4),
            project="runs",
            name=run_name,
            exist_ok=True,
            plots=True,
        )
        save_dir = Path(model.trainer.save_dir)

        # 1) courbes d'apprentissage, une valeur par epoch (split val = 10 % du train)
        df = pd.read_csv(save_dir / "results.csv")
        df.columns = df.columns.str.strip()
        for _, row in df.iterrows():
            mlflow.log_metrics(
                {clean(c): float(row[c]) for c in df.columns if c not in ("epoch", "time")},
                step=int(row["epoch"]),
            )
        mlflow.log_metric("val_best_mAP50_95", float(df["metrics/mAP50-95(B)"].max()))
        mlflow.log_metric("val_best_mAP50", float(df["metrics/mAP50(B)"].max()))

        # 2) évaluation finale sur le TEST (val officiel SH17, jamais vu pendant l'entraînement)
        best_pt = save_dir / "weights" / "best.pt"
        best = YOLO(best_pt)
        m = best.val(
            data=data,
            split="test",
            imgsz=p["imgsz"],
            batch=16,
            plots=True,
            project="runs",
            name=f"{run_name}-test",
            exist_ok=True,
        )
        mlflow.log_metrics(
            {
                "test_mAP50": float(m.box.map50),
                "test_mAP50_95": float(m.box.map),
                "test_precision": float(m.box.mp),
                "test_recall": float(m.box.mr),
                "gpu_inference_ms": float(m.speed["inference"]),
            }
        )
        per_class = {}
        for i, idx in enumerate(m.box.ap_class_index):
            label = best.names[int(idx)]
            per_class[label] = {"AP50": float(m.box.ap50[i]), "AP50_95": float(m.box.ap[i])}
            mlflow.log_metric(f"test_AP50_{label}", float(m.box.ap50[i]))
        json.dump(per_class, open(save_dir / "test_per_class.json", "w"), indent=2)

        # 3) artefacts : poids, courbes, matrice de confusion
        mlflow.log_artifact(str(best_pt), "weights")
        for f in [
            *save_dir.glob("*.png"),
            *save_dir.glob("*.jpg"),
            save_dir / "results.csv",
            save_dir / "test_per_class.json",
        ]:
            mlflow.log_artifact(str(f), "plots")
        for f in Path(m.save_dir).glob("*.png"):
            mlflow.log_artifact(str(f), "plots_test")
        mlflow.set_tag("weights_path", str(best_pt.resolve()))
        print(f"Run {run.info.run_id} : test mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    main()
