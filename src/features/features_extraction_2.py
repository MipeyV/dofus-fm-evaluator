# scr/features/features_extraction_2.py
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from src.config.stat_pool import stat_pool, résistance_pourcent_stats, dommage_pourcent_stats, essential_stats, secondary_stats, heavy_stats

def calculate_features_from_stats(stats: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    À partir de stats normalisées → calcule les features globales.
    """
    if not stats:
        return {name: 0.0 for name in [
            "nb_stats","nb_perfect_lines","nb_high_ratio","total_weight","exo_weight","over_weight",
            "avg_ratio","nb_essential_stats_ratio","nb_secondary_stats_ratio","nb_heavy_stats_ratio",
            "nb_resistance_pourcent_ratio","nb_dommage_pourcent_ratio"
        ]}

    nb_stats = len(stats)
    ratios = []
    total_weight = exo_weight = over_weight = 0.0

    for stat in stats:
        val = stat.get("value", 0)
        min_b, max_b = stat.get("bounds_min"), stat.get("bounds_max")

        # Ratio
        if min_b is not None and max_b is not None and max_b > min_b:
            ratio = (val - min_b) / (max_b - min_b)
            ratios.append(ratio)

        # Weights
        if stat["stat"] in stat_pool:
            w = stat_pool[stat["stat"]].weight
            total_weight += val * w
            if stat.get("is_exo", False):
                exo_weight += val * w

        # Over
        if max_b is not None and val > max_b:
            over_weight += val - max_b

    avg_ratio = float(np.mean(ratios)) if ratios else 0.0
    nb_perfect_lines = sum(1 for r in ratios if abs(r-1.0)<0.01)
    nb_high_ratio = sum(1 for r in ratios if 0.85<=r<0.99)

    # Catégories
    nb_res_p = sum(1 for s in stats if s["stat"] in résistance_pourcent_stats)
    nb_dom_p = sum(1 for s in stats if s["stat"] in dommage_pourcent_stats)
    nb_ess   = sum(1 for s in stats if s["stat"] in essential_stats)
    nb_sec   = sum(1 for s in stats if s["stat"] in secondary_stats)
    nb_heavy = sum(1 for s in stats if s["stat"] in heavy_stats)

    return {
        "nb_stats": float(nb_stats),
        "nb_perfect_lines": float(nb_perfect_lines),
        "nb_high_ratio": float(nb_high_ratio),
        "total_weight": float(total_weight),
        "exo_weight": float(exo_weight),
        "over_weight": float(over_weight),
        "avg_ratio": float(avg_ratio),
        "nb_essential_stats_ratio": nb_ess/nb_stats,
        "nb_secondary_stats_ratio": nb_sec/nb_stats,
        "nb_heavy_stats_ratio": nb_heavy/nb_stats,
        "nb_resistance_pourcent_ratio": nb_res_p/nb_stats,
        "nb_dommage_pourcent_ratio": nb_dom_p/nb_stats,
    }

def build_feature_dataframe(stats: List[Dict[str, Any]], feature_names: List[str]) -> pd.DataFrame:
    features = calculate_features_from_stats(stats)
    row = [features.get(f, 0.0) for f in feature_names]
    return pd.DataFrame([row], columns=feature_names)
