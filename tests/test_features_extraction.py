# tests/test_features_extraction.py
import pytest
from src.features.features_extraction_2 import calculate_features_from_stats, build_feature_dataframe

def test_empty_stats_returns_zeros():
    features = calculate_features_from_stats([])
    for val in features.values():
        assert val == 0.0

def test_simple_stat_with_bounds():
    stats = [{
        "stat": "force",
        "value": 50,
        "bounds_min": 0,
        "bounds_max": 100,
        "is_exo": False
    }]
    features = calculate_features_from_stats(stats)
    assert features["nb_stats"] == 1
    assert 0.4 <= features["avg_ratio"] <= 0.6  # 50/100 ~ 0.5

def test_dataframe_building():
    stats = [{
        "stat": "vitalité",
        "value": 100,
        "bounds_min": 50,
        "bounds_max": 150,
        "is_exo": False
    }]
    feature_names = [
        "nb_stats","nb_perfect_lines","nb_high_ratio","total_weight","exo_weight",
        "over_weight","avg_ratio","nb_essential_stats_ratio"
    ]
    df = build_feature_dataframe(stats, feature_names)
    assert set(df.columns) == set(feature_names)
    assert df.shape == (1, len(feature_names))
