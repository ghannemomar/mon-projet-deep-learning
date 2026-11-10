"""Règle de conformité : une tête (head) sans casque (helmet) déclenche une alerte.

Un casque "couvre" une tête si son centre se trouve dans la boîte de la tête,
agrandie vers le haut de 50 % de sa hauteur (le casque dépasse souvent du haut de la tête).
On part de head (classe très fréquente) plutôt que de person, car helmet est rare :
c'est une limite connue du projet (fausses alertes possibles si le casque n'est pas détecté).
"""

from __future__ import annotations


def helmet_covers_head(head: dict, helmet: dict) -> bool:
    hx1, hy1, hx2, hy2 = (head["box"][k] for k in ("x1", "y1", "x2", "y2"))
    top = hy1 - 0.5 * (hy2 - hy1)
    cx = (helmet["box"]["x1"] + helmet["box"]["x2"]) / 2
    cy = (helmet["box"]["y1"] + helmet["box"]["y2"]) / 2
    return hx1 <= cx <= hx2 and top <= cy <= hy2


def check_compliance(detections: list[dict]) -> dict:
    heads = [d for d in detections if d["label"] == "head"]
    helmets = [d for d in detections if d["label"] == "helmet"]
    without = [h for h in heads if not any(helmet_covers_head(h, m) for m in helmets)]
    alert = len(without) > 0
    if not heads:
        message = "Aucune tête détectée"
    elif alert:
        message = f"ALERTE : {len(without)} personne(s) sans casque sur {len(heads)}"
    else:
        message = f"Conforme : {len(heads)} tête(s), toutes avec casque"
    return {
        "alert": alert,
        "heads": len(heads),
        "helmets": len(helmets),
        "heads_without_helmet": len(without),
        "violations": [h["box"] for h in without],
        "message": message,
    }
