# tests/test_match_stats.py
import pytest
from src.core.match_stats import parse_stat_line

def test_parse_value_name_bounds():
    line = "60 force [50 à 80]"
    stat = parse_stat_line(line)
    assert stat is not None
    assert stat["stat"] == "force"
    assert stat["value"] == 60
    assert stat["bounds_min"] == 50
    assert stat["bounds_max"] == 80
    assert stat["is_exo"] is False

def test_parse_percent_stat():
    line = "12% résistance feu [10 à 12]"
    stat = parse_stat_line(line)
    assert stat["stat"] == "résistance_feu_%"
    assert stat["value"] == 12
    assert stat["bounds_min"] == 10
    assert stat["bounds_max"] == 12

def test_parse_compact_pa():
    line = "1PA"
    stat = parse_stat_line(line)
    assert stat["stat"] == "PA"
    assert stat["value"] == 1
    assert stat["is_exo"] is True
