"""Analyse d'erreurs du meilleur modèle + tableau comparatif de tous les runs.

1) Rappel par taille d'objet sur le test (justifie le test d'un imgsz plus grand :
   35,7 % des boîtes font moins de 32 px à 640 px d'entrée).
2) Tableau comparatif des runs (mAP, latence) -> comparison.csv, logué dans MLflow.

Lancement : python -m src.models.evaluate            (choisit le meilleur run sur la VALIDATION)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.utils.mlflow_utils import setup_experiment  # noqa: E402

BUCKETS = [("<16 px", 0, 16), ("16-32 px", 16, 32), ("32-96 px", 32, 96), (">96 px", 96, 1e9)]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def best_run(exp_id: str) -> pd.Series:
    runs = mlflow.search_runs([exp_id], order_by=["metrics.val_best_mAP50_95 DESC"])
    runs = runs[runs["metrics.val_best_mAP50_95"].notna()]
    return runs.iloc[0]


def size_analysis(weights: str, imgsz: int, conf: float = 0.25) -> pd.DataFrame:
    cfg = yaml.safe_load(open("runs/data.yaml"))
    img_dir = Path(cfg["path"]) / cfg["test"]
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/"))
    model = YOLO(weights)
    found = {b[0]: 0 for b in BUCKETS}
    total = {b[0]: 0 for b in BUCKETS}
    for res in model.predict(source=str(img_dir), imgsz=imgsz, conf=conf, stream=True, verbose=False):
        h, w = res.orig_shape
        scale = imgsz / max(h, w)  # taille de l'objet telle que le réseau la voit
        lbl = lbl_dir / f"{Path(res.path).stem}.txt"
        rows = [list(map(float, x.split())) for x in lbl.read_text().splitlines() if x.strip()]
        if not rows:
            continue
        g = np.array(rows)
        gcls = g[:, 0].astype(int)
        gt = np.stack(
            [
                (g[:, 1] - g[:, 3] / 2) * w,
                (g[:, 2] - g[:, 4] / 2) * h,
                (g[:, 1] + g[:, 3] / 2) * w,
                (g[:, 2] + g[:, 4] / 2) * h,
            ],
            1,
        )
        matched = np.zeros(len(gt), bool)
        pb = res.boxes.xyxy.cpu().numpy()
        pc = res.boxes.cls.cpu().numpy().astype(int)
        order = res.boxes.conf.cpu().numpy().argsort()[::-1]
        if len(pb):
            ious = iou_matrix(pb, gt)
            for i in order:  # appariement glouton, une vérité terrain par prédiction
                cand = np.where((gcls == pc[i]) & ~matched & (ious[i] >= 0.5))[0]
                if len(cand):
                    matched[cand[ious[i, cand].argmax()]] = True
        sizes = np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) * scale
        for name, lo, hi in BUCKETS:
            sel = (sizes >= lo) & (sizes < hi)
            total[name] += int(sel.sum())
            found[name] += int((sel & matched).sum())
    return pd.DataFrame(
        [
            {
                "taille": n,
                "objets": total[n],
                "detectes": found[n],
                "rappel": found[n] / total[n] if total[n] else 0.0,
            }
            for n, _, _ in BUCKETS
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    args = ap.parse_args()
    exp_id = setup_experiment()

    # tableau comparatif de tous les runs terminés
    runs = mlflow.search_runs([exp_id])
    cols = {
        "tags.mlflow.runName": "run",
        "params.model": "modele",
        "params.imgsz": "imgsz",
        "params.epochs": "epochs",
        "metrics.val_best_mAP50_95": "val_mAP50_95",
        "metrics.test_mAP50": "test_mAP50",
        "metrics.test_mAP50_95": "test_mAP50_95",
        "metrics.test_AP50_helmet": "AP50_helmet",
        "metrics.test_AP50_safety-vest": "AP50_vest",
        "metrics.gpu_inference_ms": "gpu_ms",
    }
    comp = runs[[c for c in cols if c in runs]].rename(columns=cols)
    comp = comp[comp["test_mAP50"].notna()].sort_values("val_mAP50_95", ascending=False)
    Path("reports").mkdir(exist_ok=True)
    comp.to_csv("reports/comparison.csv", index=False)
    print(comp.to_string(index=False))

    run = mlflow.get_run(args.run_id) if args.run_id else None
    row = best_run(exp_id) if run is None else None
    run_id = args.run_id or row["run_id"]
    run = mlflow.get_run(run_id)
    weights = run.data.tags.get("weights_path")
    if not weights or not Path(weights).exists():
        weights = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights/best.pt")
    imgsz = int(run.data.params["imgsz"])
    print(f"Meilleur run (sur la validation) : {run.info.run_name} ({run_id})")

    df = size_analysis(weights, imgsz)
    df.to_csv("reports/recall_by_size.csv", index=False)
    print(df.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(df["taille"], df["rappel"] * 100, color="#3b6ea5")
    ax.set_ylabel("Rappel (%) à IoU 0,5")
    ax.set_title(f"Rappel par taille d'objet — {run.info.run_name}")
    for i, v in enumerate(df["rappel"] * 100):
        ax.text(i, v + 1, f"{v:.0f} %", ha="center")
    fig.tight_layout()
    fig.savefig("reports/recall_by_size.png", dpi=150)

    with mlflow.start_run(run_id=run_id):
        for _, r in df.iterrows():
            mlflow.log_metric(
                f"test_recall_{r['taille'].replace(' ', '').replace('<', 'lt').replace('>', 'gt')}",
                float(r["rappel"]),
            )
        for f in ["reports/recall_by_size.csv", "reports/recall_by_size.png", "reports/comparison.csv"]:
            mlflow.log_artifact(f, "analysis")


if __name__ == "__main__":
    main()
