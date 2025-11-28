"""
Outil Roboflow pour télécharger automatiquement les jeux de données nécessaires.

Deux méthodes sont disponibles :
    1. API Roboflow (clé requise)  -> `--method api`
    2. Liens directs (curl/zip)    -> `--method direct`

Usage rapide :
    python scripts/download_datasets.py --dataset all --target ./datasets
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional

try:  # Import différé : uniquement nécessaire en mode API
    from roboflow import Roboflow
except ImportError:  # pragma: no cover
    Roboflow = None  # type: ignore


DATASETS: Dict[str, Dict[str, str]] = {
    "basketball": {
        "workspace": "cricket-qnb5l",
        "project": "basketball-xil7x",
        "version": 1,
        "format": "yolov11",
        "description": "Détection basket/joueurs/cercle (shot.pt)",
        "direct_url": "https://universe.roboflow.com/ds/908y19b1ug?key=xL1VbGFi1r",
    },
    "shotanalysis": {
        "workspace": "copyme-3cenq",
        "project": "shotanalysis",
        "version": 21,
        "format": "yolov11",
        "description": "Détection des phases de tir (copyme.pt)",
        "direct_url": "https://universe.roboflow.com/ds/lV02Spm6qY?key=6mnRZhiPHE",
    },
}

DownloadMethod = Literal["api", "direct", "auto"]


def _ensure_empty(dir_path: Path) -> None:
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)


def _flatten_single_child(root: Path) -> None:
    entries = list(root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for child in nested.iterdir():
            shutil.move(str(child), root / child.name)
        nested.rmdir()


def download_dataset_direct(dataset_key: str, target_dir: Path) -> Path:
    meta = DATASETS[dataset_key]
    url = meta.get("direct_url")
    if not url:
        raise SystemExit(f"Aucune URL directe configurée pour '{dataset_key}'.")

    dataset_dir = target_dir / dataset_key
    _ensure_empty(dataset_dir)

    print(f"Téléchargement direct '{dataset_key}' → {dataset_dir}")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    curl_cmd = [
        "curl",
        "-L",
        url,
        "-o",
        str(tmp_path),
        "-H",
        "User-Agent: curl/8.0 (ShotPrediction downloader)",
    ]
    try:
        subprocess.run(curl_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        tmp_path.unlink(missing_ok=True)
        raise SystemExit(f"Échec du téléchargement direct ({dataset_key}): {exc}") from exc

    with zipfile.ZipFile(tmp_path) as archive:
        archive.extractall(dataset_dir)
    tmp_path.unlink(missing_ok=True)

    _flatten_single_child(dataset_dir)
    print(f"✅ Dataset '{dataset_key}' prêt dans : {dataset_dir}")
    return dataset_dir


def download_dataset_api(
    api_key: str,
    dataset_key: str,
    target_dir: Path,
    fmt: Optional[str] = None,
) -> Path:
    if Roboflow is None:
        raise SystemExit(
            "Le package 'roboflow' est requis pour la méthode API. "
            "Installe-le avec 'pip install roboflow'."
        )

    meta = DATASETS[dataset_key]
    fmt = fmt or meta["format"]

    target_dir = target_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(meta["workspace"]).project(meta["project"])
    version = project.version(meta["version"])

    print(
        f"Téléchargement API '{dataset_key}' (workspace={meta['workspace']}, "
        f"project={meta['project']}, version={meta['version']}) → {target_dir}"
    )

    # Roboflow télécharge dans le répertoire courant, pas dans target_dir
    # On télécharge d'abord, puis on déplace
    dataset = version.download(fmt)
    
    actual_location = Path(dataset.location)
    dataset_dir = target_dir / dataset_key
    _ensure_empty(dataset_dir)
    
    # Déplacer tous les fichiers du répertoire téléchargé vers dataset_dir
    print(f"Déplacement de {actual_location} → {dataset_dir}")
    for item in actual_location.iterdir():
        shutil.move(str(item), dataset_dir / item.name)
    
    # Supprimer le répertoire vide
    if actual_location.exists() and not any(actual_location.iterdir()):
        actual_location.rmdir()
    elif actual_location.exists():
        # Si le répertoire n'est pas vide, le supprimer récursivement
        shutil.rmtree(actual_location)

    print(f"✅ Dataset '{dataset_key}' disponible dans : {dataset_dir}")
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Télécharge les datasets Roboflow du projet ShotPrediction."
    )
    parser.add_argument(
        "--dataset",
        choices=["basketball", "shotanalysis", "all"],
        default="basketball",
        help="Dataset à télécharger (par défaut : basketball).",
    )
    parser.add_argument(
        "--target",
        default="./datasets",
        help="Dossier de destination (par défaut ./datasets).",
    )
    parser.add_argument(
        "--format",
        default=None,
        help="Format Roboflow (API uniquement).",
    )
    parser.add_argument(
        "--method",
        choices=["api", "direct", "auto"],
        default="auto",
        help="Méthode de téléchargement (auto : API si clé dispo, sinon direct).",
    )
    return parser.parse_args()


def resolve_targets(option: str) -> Iterable[str]:
    return DATASETS.keys() if option == "all" else [option]


def main() -> None:
    args = parse_args()
    target_dir = Path(args.target).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    if args.method == "api" and not api_key:
        raise SystemExit(
            "Méthode API demandée mais ROBOFLOW_API_KEY est absent. "
            "Définis-le ou utilise --method direct."
        )

    method: DownloadMethod
    if args.method == "auto":
        method = "api" if api_key else "direct"
    else:
        method = args.method

    print(f"Mode de téléchargement sélectionné : {method.upper()}")

    for key in resolve_targets(args.dataset):
        if method == "api":
            download_dataset_api(
                api_key=api_key or "",
                dataset_key=key,
                target_dir=target_dir,
                fmt=args.format,
            )
        else:
            download_dataset_direct(
                dataset_key=key,
                target_dir=target_dir,
            )


if __name__ == "__main__":
    main()