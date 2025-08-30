# src/models/predictor.py
from typing import List, Dict, Any
import joblib
import numpy as np
import sys

from src.features.features_extraction import build_feature_dataframe
from src.ocr.tests import reader_enhanced
from src.ocr.tests.reader_enhanced import extract_stat_lines, extract_stats_with_bounds


def load_model(model_path: str):
    return joblib.load(model_path)


def get_expected_features(model) -> List[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        raise ValueError("Le modèle ne contient pas feature_names_in_. Entraînez/sauvegardez avec scikit-learn>=1.0.")
    return list(names)


def score_image(model_path: str, image_path: str) -> Dict[str, Any]:
    model = load_model(model_path)
    feature_names = get_expected_features(model)

    # --- OCR Stats avec reader_enhanced ---
    lines, crops = reader_enhanced.extract_stat_lines(image_path)
    stats = reader_enhanced.extract_stats_with_bounds(lines, crops)

    # --- Features ---
    X = build_feature_dataframe(stats, feature_names)
    y_pred = model.predict(X)

    # --- OCR Metadata ---
    try:
        meta_lines = reader_enhanced.extract_metadata_lines(image_path)
        meta = reader_enhanced.parse_metadata(meta_lines)
        item_name = meta.name
    except Exception:
        item_name = None

    return {
        "prediction": float(y_pred[0]) if hasattr(y_pred, "__getitem__") else float(y_pred),
        "lines": lines,
        "parsed_stats": stats,
        "features": {name: val for name, val in zip(feature_names, X.iloc[0].tolist())},
        "item_name": item_name,
        "current_stats": {s["stat"]: s["value"] for s in stats},
    }

def predict_instance(instance, model_path: str):
    """
    Prend un ItemInstance, calcule ses features et renvoie la prédiction du modèle ML.
    """
    import pandas as pd
    import numpy as np

    model = load_model(model_path)
    feature_names = get_expected_features(model)

    features_dict = instance.get_features()

    X = pd.DataFrame([[features_dict.get(f, np.nan) for f in feature_names]],
                     columns=feature_names)

    y_pred = model.predict(X)

    return {
        "prediction": float(y_pred[0]) if hasattr(y_pred, "__getitem__") else float(y_pred),
        "features": features_dict,
        "used_features": {f: features_dict.get(f, np.nan) for f in feature_names}
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m src.models.predictor <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    result = score_image(model_path, image_path)
    print("\n[RESULT] --- PREDICTION ---")
    print("Score:", result["prediction"])
    print("\n[RESULT] --- FEATURES ---")
    for k, v in result["features"].items():
        print(f"{k}: {v}")
    print("\n[RESULT] --- RAW STATS ---")
    for line in result["lines"]:
        print(" •", line)
    if result.get("item_name"):
        print("\n[RESULT] --- ITEM NAME ---")
        print(result["item_name"])