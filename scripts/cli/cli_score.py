import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.models.predictor import score_image

DEFAULT_MODEL = str(ROOT / "models" / "random_forest_model.joblib")
DEFAULT_IMAGE_DIR = ROOT / "tests" / "assets"  # <-- dossier par défaut

def main():
    parser = argparse.ArgumentParser(description="Évaluer une image d'item Dofus avec un modèle ML.")
    parser.add_argument("image_path", type=str, help="Nom ou chemin de l'image de l'item")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL,
                        help=f"Chemin vers le modèle (défaut: {DEFAULT_MODEL})")
    parser.add_argument("-o", "--output", type=str, help="Fichier JSON pour sauvegarder les résultats")
    args = parser.parse_args()

    # Résoudre l'image
    image_path = Path(args.image_path)
    if not image_path.exists():
        candidate = DEFAULT_IMAGE_DIR / args.image_path
        if candidate.exists():
            image_path = candidate
        else:
            raise FileNotFoundError(f"Image {args.image_path} introuvable (ni chemin direct, ni dans {DEFAULT_IMAGE_DIR})")

    result = score_image(args.model, str(image_path))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Résultat sauvegardé dans {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
