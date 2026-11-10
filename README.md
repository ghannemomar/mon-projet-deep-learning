# Détection d'EPI en milieu de travail (YOLO + MLOps)

Projet du module **Deep Learning Avancé** — Cycle ingénieur IA & Data Science, 3e année, TEK-UP.
Axe 3 : détection d'objets en temps réel.

Une application web détecte, à partir d'une image ou de la webcam, les personnes et leurs équipements
de protection (17 classes du dataset SH17) et affiche une **alerte rouge** quand une tête est détectée sans casque.

## Lancer la démo

```bash
docker-compose up --build
```

| Service | Adresse | Rôle |
| --- | --- | --- |
| Frontend React | http://localhost:3000 | Upload, galerie, webcam, boîtes, alerte, latence |
| API FastAPI | http://localhost:8000/docs | Swagger : `/health`, `/model/metadata`, `/predict`, `/predict/visualize` |
| MLflow | http://localhost:5000 | Runs d'entraînement + Model Registry (alias `staging` / `production`) |

Le backend charge le modèle depuis le **MLflow Model Registry** (`models:/sh17-ppe-detector@production`).
S'il n'y en a pas, il prend `models/model.onnx`. Sans modèle, l'API répond `503`.

Avant le premier lancement, placer à la racine le contenu de `outputs.zip` produit par le notebook Kaggle :
`mlflow/` (dézipper `mlflow.zip` dedans), `models/model.onnx`, et les images de `examples/` dans `frontend/public/examples/`.

## Architecture

```
Webcam / image ──► Frontend React ──HTTP──► FastAPI ──► ONNX Runtime (modèle chargé une fois)
                                              ▲
                     MLflow Model Registry ───┘ (alias production)

Cycle ML :  SH17 brut ─(DVC: prepare)─► data/processed ─► train YOLO ─► MLflow (runs)
            ─► evaluate (par classe, par taille) ─► export ONNX + latence ─► Registry staging → production
Cycle CI :  push / pull request ─► black + flake8 + pytest + eslint + build ─► docker compose build
```

## Arborescence

```
.github/workflows/   ci-tests.yml (lint + pytest + build front), docker-build.yml
data/                raw/ et processed/ : versionnés par DVC, jamais dans Git
src/data/dataset.py  split train/val/test + redimensionnement (stage DVC "prepare")
src/models/          architecture.py, train.py (MLflow), evaluate.py, export.py (ONNX + Registry)
backend/             FastAPI : app/{main,schemas,inference,rules}.py, tests/test_api.py, Dockerfile
frontend/            React (Vite) : src/{components,services}, Dockerfile (build + nginx)
notebooks/           kaggle_training.ipynb (entraînement GPU de bout en bout)
docker-compose.yml   mlflow + backend + frontend
dvc.yaml, params.yaml, dvc.lock, .dvcignore
```

`data/raw` est la dépendance du stage DVC `prepare` ; `dvc.lock` contient les empreintes exactes des
données brutes et produites (il joue le rôle de `data.dvc`).

## Données : SH17

- 8 099 images (Pexels), 75 994 instances, 17 classes, format YOLO, licence **CC BY-NC-SA 4.0**.
- Source : https://github.com/ahmadmughees/SH17dataset — article arXiv 2407.04590 — copie Kaggle.
- Ordre des classes = `sh17.yaml` (alphabétique), différent du README officiel.

| Split | Origine | Images |
| --- | --- | --- |
| train | 90 % du train officiel | 5 831 |
| val | 10 % du train officiel, seed 42 | 648 |
| test | val officiel (comparable au benchmark publié) | 1 620 |

Le split est fait **avant** toute augmentation (augmentations Ultralytics à la volée, sur train uniquement).
Les images sont réduites une fois à 1 280 px de grand côté ; les labels normalisés restent valides.

## Reproduire l'entraînement

Sur Kaggle (GPU) : ouvrir `notebooks/kaggle_training.ipynb`, ajouter le dataset SH17 en Input, mettre son
nom GitHub, puis *Save & Run All*. Le notebook fait : `dvc repro` → 3 entraînements suivis dans MLflow
→ analyse → export ONNX → Registry → `outputs.zip`.

En local (avec GPU) :

```bash
pip install -r requirements-train.txt
dvc repro                                   # prepare
python -m src.models.train                  # baseline (params.yaml)
python -m src.models.train --model yolo26n --name yolo26n-640
python -m src.models.evaluate               # meilleur run choisi sur la validation
python -m src.models.export                 # ONNX + latence + Registry
mlflow ui                                   # ou docker-compose
```

## Règle d'alerte

Une tête (`head`) est conforme si le centre d'un casque (`helmet`) se trouve dans sa boîte agrandie de 50 %
vers le haut. Sinon : alerte. On part de `head` (11 985 instances) car `helmet` est rare (927).
Limite : un casque non détecté produit une fausse alerte.

## Qualité du code

```bash
cd backend && pip install -r requirements-dev.txt && pytest -v     # 18 tests
black --check backend src && flake8 backend src
cd frontend && npm ci && npm run lint && npm run build
```

Les tests de l'API n'ont pas besoin des poids : le modèle est remplacé par un faux détecteur
(`tests/conftest.py`), et le post-traitement ONNX (letterbox, NMS, formats YOLOv8 et YOLO26) est testé sur des sorties synthétiques.

## Résultats

Voir le tableau `reports/comparison.csv` (généré par `evaluate.py`) et l'expérience `sh17-ppe` dans MLflow.
Repères publiés sur SH17 : YOLOv8n 58,0 mAP50 / 36,6 mAP50-95 ; YOLOv9-e 70,9 / 48,7.
