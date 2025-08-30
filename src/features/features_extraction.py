# src/features/features_extraction.py
from typing import Dict, Any, List
import numpy as np
import pandas as pd

def normalize_stat_name(ocr_name: str) -> str:
    """
    Normalize OCR stat names to match stat_pool keys.
    Examples: "dommages eau" -> "dommage_eau", "résistance air" -> "résistance_air"
    """
    # Common OCR variations to stat_pool mappings
    ocr_to_pool = {
        "dommages": "dommage",
        "résistance": "résistance",
        "dommage": "dommage",
        "résistance": "résistance",
        "critique": "critique",
        "soins": "soin",
        "pa": "pa",
        "pm": "pm",
        "po": "po",
        "vitalité": "vitalité",
        "sagesse": "sagesse",
        "force": "force",
        "chance": "chance",
        "agilité": "agilité",
        "intelligence": "intelligence",
        "puissance": "puissance",
        "coups_critiques": "coups_critiques",
        "invocation": "invocation",
        "tacle": "tacle",
        "fuite": "fuite",
        "prospection": "prospection",
        "initiative": "initiative",
        "pods": "pods"
    }
    
    # Clean and normalize the OCR name
    clean_name = ocr_name.lower().strip()
    
    # Handle multi-word stats
    if " " in clean_name:
        words = clean_name.split()
        if words[0] in ocr_to_pool:
            base = ocr_to_pool[words[0]]
            if len(words) > 1:
                # Handle element types
                if words[1] in ["eau", "feu", "terre", "air", "neutre"]:
                    return f"{base}_{words[1]}"
                elif words[1] in ["critiques", "critique"]:
                    return f"{base}_critique"
                elif words[1] in ["poussée", "poussee"]:
                    return f"{base}_poussée"
                elif words[1] in ["dist", "distance"]:
                    return f"{base}_dist_%"
                elif words[1] in ["sort", "sorts"]:
                    return f"{base}_sort_%"
                elif words[1] in ["mêlée", "melee", "mêlee"]:
                    return f"{base}_mêlée_%"
                else:
                    return f"{base}_{words[1]}"
            return base
    
    # Single word stats
    if clean_name in ocr_to_pool:
        return ocr_to_pool[clean_name]
    
    # Handle special cases
    if clean_name == "critique":
        return "critique"
    elif clean_name == "soins":
        return "soin"
    
    return clean_name


def calculate_features_from_stats(stats: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate features from parsed stats for the RandomForest model.
    """
    if not stats:
        return {
            "nb_stats": 0.0, "nb_perfect_lines": 0.0, "nb_high_ratio": 0.0,
            "total_weight": 0.0, "exo_weight": 0.0, "over_weight": 0.0,
            "avg_ratio": 0.0, "nb_essential_stats_ratio": 0.0,
            "nb_secondary_stats_ratio": 0.0, "nb_heavy_stats_ratio": 0.0,
            "nb_resistance_pourcent_ratio": 0.0, "nb_dommage_pourcent_ratio": 0.0
        }
    
    # Normalize stat names and match with stat_pool
    from src.config.stat_pool import stat_pool
    
    normalized_stats = []
    for stat in stats:
        normalized_name = normalize_stat_name(stat["stat"])
        if normalized_name in stat_pool:
            stat_copy = stat.copy()
            stat_copy["stat"] = normalized_name
            stat_copy["pool_stat"] = stat_pool[normalized_name]
            normalized_stats.append(stat_copy)
        else:
            # Keep original for exo stats
            normalized_stats.append(stat)
    
    # Count stats by type
    nb_stats = len(normalized_stats)
    nb_exo = sum(1 for s in normalized_stats if s.get("is_exo", False))
    nb_with_bounds = sum(1 for s in normalized_stats if s.get("bounds_min") is not None)
    
    # Calculate ratios for stats with bounds
    ratios = []
    for stat in normalized_stats:
        if stat.get("bounds_min") is not None and stat.get("bounds_max") is not None:
            min_val = stat["bounds_min"]
            max_val = stat["bounds_max"]
            current_val = stat["value"]
            if max_val > min_val:
                ratio = (current_val - min_val) / (max_val - min_val)
                ratios.append(ratio)
    
    avg_ratio = np.mean(ratios) if ratios else 0.0
    nb_perfect_lines = sum(1 for r in ratios if abs(r - 1.0) < 0.01)
    nb_high_ratio = sum(1 for r in ratios if 0.85 <= r < 0.99)
    
    # Calculate weights using stat_pool weights
    total_weight = 0.0
    exo_weight = 0.0
    for stat in normalized_stats:
        if "pool_stat" in stat:
            weight = stat["pool_stat"].weight
            value = abs(stat.get("value", 0))
            total_weight += value * weight
            if stat.get("is_exo", False):
                exo_weight += value * weight
    
    # Over stats (values above max)
    over_weight = 0.0
    for stat in normalized_stats:
        if stat.get("bounds_max") is not None:
            if stat["value"] > stat["bounds_max"]:
                over_weight += stat["value"] - stat["bounds_max"]
    
    # Stat type ratios using stat_pool categories
    from src.config.stat_pool import résistance_pourcent_stats, dommage_pourcent_stats, essential_stats, secondary_stats, heavy_stats
    
    nb_resistance_pourcent = sum(1 for s in normalized_stats if s["stat"] in résistance_pourcent_stats)
    nb_dommage_pourcent = sum(1 for s in normalized_stats if s["stat"] in dommage_pourcent_stats)
    nb_essential = sum(1 for s in normalized_stats if s["stat"] in essential_stats)
    nb_secondary = sum(1 for s in normalized_stats if s["stat"] in secondary_stats)
    nb_heavy = sum(1 for s in normalized_stats if s["stat"] in heavy_stats)
    
    nb_resistance_pourcent_ratio = nb_resistance_pourcent / nb_stats if nb_stats > 0 else 0.0
    nb_dommage_pourcent_ratio = nb_dommage_pourcent / nb_stats if nb_stats > 0 else 0.0
    nb_essential_stats_ratio = nb_essential / nb_stats if nb_stats > 0 else 0.0
    nb_secondary_stats_ratio = nb_secondary / nb_stats if nb_stats > 0 else 0.0
    nb_heavy_stats_ratio = nb_heavy / nb_stats if nb_stats > 0 else 0.0
    
    return {
        "nb_stats": float(nb_stats),
        "nb_perfect_lines": float(nb_perfect_lines),
        "nb_high_ratio": float(nb_high_ratio),
        "total_weight": float(total_weight),
        "exo_weight": float(exo_weight),
        "over_weight": float(over_weight),
        "avg_ratio": float(avg_ratio),
        "nb_essential_stats_ratio": float(nb_essential_stats_ratio),
        "nb_secondary_stats_ratio": float(nb_secondary_stats_ratio),
        "nb_heavy_stats_ratio": float(nb_heavy_stats_ratio),
        "nb_resistance_pourcent_ratio": float(nb_resistance_pourcent_ratio),
        "nb_dommage_pourcent_ratio": float(nb_dommage_pourcent_ratio)
    }


def build_feature_dataframe(stats: List[Dict[str, Any]], feature_names: List[str]) -> "pd.DataFrame":
    """
    Build a pandas DataFrame with proper feature names for sklearn models.
    """
    features = calculate_features_from_stats(stats)
    
    # Ensure all expected features are present
    X = [features.get(name, 0.0) for name in feature_names]
    return pd.DataFrame([X], columns=feature_names)