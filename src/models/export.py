"""Export ONNX du meilleur modèle + mesure de latence + MLflow Model Registry.

1) choisit le meilleur run sur la VALIDATION (pas sur le test, pour ne pas tricher) ;
2) exporte best.pt en ONNX ;
3) mesure la latence CPU PyTorch vs ONNX Runtime (même image, 50 passages après 10 d'échauffement) ;
4) vérifie que le mAP de l'ONNX sur le test est proche de celui de PyTorch ;
5) enregistre le modèle dans le Registry : alias "staging", puis "production" si la vérification passe ;
6) copie model.onnx + metadata.json dans models/ (utilisé par l'API).

Lancement : python -m src.models.export
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from mlflow import MlflowClient
from ultralytics import YOLO

from src.models.evaluate import best_run
from src.utils.mlflow_utils import MODEL_NAME, setup_experiment


def bench(fn, n: int = 50, warmup: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return {"p50": float(np.percentile(times, 50)), "p95": float(np.percentile(times, 95))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id")
    ap.add_argument("--skip-accuracy-check", action="store_true")
    args = ap.parse_args()
    exp_id = setup_experiment()
    run_id = args.run_id or best_run(exp_id)["run_id"]
    run = mlflow.get_run(run_id)
    imgsz = int(run.data.params["imgsz"])
    weights = run.data.tags.get("weights_path")
    if not weights or not Path(weights).exists():
        weights = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="weights/best.pt")
    print(f"Export du run {run.info.run_name} ({run_id}), imgsz={imgsz}")

    model = YOLO(weights)
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False))

    # latence CPU du réseau seul (même tenseur d'entrée pour les deux moteurs)
    cfg_path = "runs/data.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    sample = sorted((Path(cfg["path"]) / cfg["test"]).glob("*.jpg"))[0]
    img = cv2.imread(str(sample))
    blob = cv2.resize(img, (imgsz, imgsz))[:, :, ::-1].transpose(2, 0, 1)[None]
    blob = np.ascontiguousarray(blob, dtype=np.float32) / 255
    net = YOLO(weights).model.float().eval().to("cpu")
    tensor = torch.from_numpy(blob)
    with torch.no_grad():
        torch_cpu = bench(lambda: net(tensor))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_cpu = bench(lambda: sess.run(None, {sess.get_inputs()[0].name: blob}))
    print(f"PyTorch CPU p50={torch_cpu['p50']:.1f} ms | ONNX CPU p50={onnx_cpu['p50']:.1f} ms")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "torch_cpu_ms_p50": torch_cpu["p50"],
                "torch_cpu_ms_p95": torch_cpu["p95"],
                "onnx_cpu_ms_p50": onnx_cpu["p50"],
                "onnx_cpu_ms_p95": onnx_cpu["p95"],
                "onnx_speedup": torch_cpu["p50"] / onnx_cpu["p50"],
                "onnx_size_mb": onnx_path.stat().st_size / 1e6,
            }
        )
        ok = True
        if not args.skip_accuracy_check:
            m = YOLO(str(onnx_path), task="detect").val(
                data=cfg_path, split="test", imgsz=imgsz, batch=1, device="cpu", plots=False
            )
            mlflow.log_metric("onnx_test_mAP50", float(m.box.map50))
            ref = run.data.metrics.get("test_mAP50", m.box.map50)
            ok = abs(m.box.map50 - ref) < 0.02  # au plus 2 points d'écart
            print(f"mAP50 ONNX={m.box.map50:.3f} vs PyTorch={ref:.3f} -> {'OK' if ok else 'ÉCART'}")
        info = mlflow.onnx.log_model(
            onnx.load(str(onnx_path)), "onnx_model", registered_model_name=MODEL_NAME
        )

    client = MlflowClient()
    version = getattr(info, "registered_model_version", None)
    if version is None:
        version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    client.set_model_version_tag(MODEL_NAME, version, "imgsz", str(imgsz))
    client.set_model_version_tag(MODEL_NAME, version, "run_name", run.info.run_name)
    if ok:
        client.set_registered_model_alias(MODEL_NAME, "production", version)
        try:  # vocabulaire "Staging / Production" des anciennes versions de MLflow
            client.transition_model_version_stage(
                MODEL_NAME, version, "Production", archive_existing_versions=True
            )
        except Exception as exc:  # noqa: BLE001
            print(f"(stages non disponibles : {exc})")
    print(f"Registry : {MODEL_NAME} v{version} -> staging{' + production' if ok else ''}")

    Path("models").mkdir(exist_ok=True)
    shutil.copy(onnx_path, "models/model.onnx")
    json.dump(
        {
            "name": MODEL_NAME,
            "version": str(version),
            "run_id": run_id,
            "run_name": run.info.run_name,
            "imgsz": imgsz,
            "onnx_cpu_ms_p50": onnx_cpu["p50"],
            "torch_cpu_ms_p50": torch_cpu["p50"],
        },
        open("models/metadata.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
