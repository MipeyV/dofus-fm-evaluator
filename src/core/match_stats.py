# src/core/match_stats.py
import re
import unicodedata
from typing import Optional, TypedDict
from src.config.stat_pool import stat_pool

class ExtractedStat(TypedDict):
    stat: str
    value: int
    bounds: Optional[str]
    bounds_min: Optional[int]
    bounds_max: Optional[int]
    is_exo: bool

# --- dictionnaire d’alias comme avant
ALIASES = {
    "vitalite": "vitalité",
    "sagesse": "sagesse",
    "force": "force",
    "chance": "chance",
    "agilite": "agilité",
    "intelligence": "intelligence",
    "puissance": "puissance",
    "pa": "PA", "pm": "PM", "po": "PO",
    "critique": "coups_critiques", "critiques": "coups_critiques",
    "cc": "coups_critiques",
    "dommages": "dommage", "dommage": "dommage",
    "dommages neutre": "dommage_neutre", "dommage neutre": "dommage_neutre",
    "dommages terre": "dommage_terre", "dommage terre": "dommage_terre",
    "dommages feu": "dommage_feu", "dommage feu": "dommage_feu",
    "dommages eau": "dommage_eau", "dommage eau": "dommage_eau",
    "dommages air": "dommage_air", "dommage air": "dommage_air",
    "dommages critique": "dommage_critique", "dommage critique": "dommage_critique",
    "resistance neutre": "résistance_neutre",
    "resistance terre": "résistance_terre",
    "resistance feu": "résistance_feu",
    "resistance eau": "résistance_eau",
    "resistance air": "résistance_air",
    "resistance poussee": "résistance_poussée",
    "resistance critique": "résistance_critique",
    "resistance neutre %": "résistance_neutre_%",
    "resistance terre %": "résistance_terre_%",
    "resistance feu %": "résistance_feu_%",
    "resistance eau %": "résistance_eau_%",
    "resistance air %": "résistance_air_%",
    "dommages melee %": "dommage_mêlee_%",
    "dommages distance %": "dommage_dist_%",
    "dommages sort %": "dommage_sort_%",
    "invocation": "invocation",
    "tacle": "tacle", "fuite": "fuite",
    "soins": "soin", "soin": "soin",
    "prospection": "prospection",
    "initiative": "initiative", "pods": "pods",
    "retrait pa": "retrait_pa", "retraits pa": "retrait_pa",
    "retrait pm": "retrait_pm", "retraits pm": "retrait_pm",
    "esquive pa": "esquive_pa", "esquives pa": "esquive_pa",
    "esquive pm": "esquive_pm", "esquives pm": "esquive_pm",
}

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_stat_name(raw_name: str, had_percent: bool) -> str:
    s = raw_name.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\[(].*$", "", s).strip()
    s = s.replace("%", "").strip()
    s_ascii = _strip_accents(s)

    if had_percent and ("resistance" in s_ascii or "dommage" in s_ascii):
        s_ascii = s_ascii + " %"

    mapped = ALIASES.get(s_ascii)
    if mapped:
        return mapped

    s_underscored = s.replace(" ", "_")
    if s_underscored in stat_pool:
        return s_underscored

    s_u_ascii = _strip_accents(s_underscored).replace("melee", "mêlee")
    if s_u_ascii in stat_pool:
        return s_u_ascii

    return s_underscored

def parse_bounds(bounds_str: Optional[str]):
    if not bounds_str:
        return None, None, None
    cleaned = bounds_str.strip()
    cleaned = re.sub(r'^[\[(]\s*', '', cleaned)
    cleaned = re.sub(r'[\])]\s*$', '', cleaned)
    cleaned = cleaned.replace('à', ' ').replace('-', ' ')
    nums = re.findall(r"-?\d+", cleaned)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1]), f"{nums[0]} à {nums[1]}"
    if len(nums) == 1:
        return int(nums[0]), int(nums[0]), f"{nums[0]} à {nums[0]}"
    return None, None, None

def parse_stat_line(line: str) -> Optional[ExtractedStat]:
    """Parse une ligne OCR de stat → ExtractedStat normalisé."""
    s = line.strip()
    had_percent = "%" in s

    # value% name [bounds]
    m = re.match(r"^(-?\d+)\s*%\s+(.+?)(\s*[\[(].*)?$", s)
    if m:
        value = int(m.group(1))
        raw_name = m.group(2).strip()
        name = normalize_stat_name(raw_name, had_percent=True)
        bmin, bmax, bstr = parse_bounds(m.group(3))
        return {"stat": name, "value": value, "bounds": bstr, "bounds_min": bmin,
                "bounds_max": bmax, "is_exo": bmin is None}

    # value name [bounds]
    m = re.match(r"^(-?\d+)\s+(.+?)(\s*[\[(].*)?$", s)
    if m:
        value = int(m.group(1))
        raw_name = m.group(2).strip()
        name = normalize_stat_name(raw_name, had_percent=had_percent)
        bmin, bmax, bstr = parse_bounds(m.group(3))
        return {"stat": name, "value": value, "bounds": bstr, "bounds_min": bmin,
                "bounds_max": bmax, "is_exo": bmin is None}

    # name value [bounds]
    m = re.match(r"^(.+?)\s+(-?\d+)\s*%?(\s*[\[(].*)?$", s)
    if m:
        raw_name = m.group(1).strip()
        value = int(m.group(2))
        name = normalize_stat_name(raw_name, had_percent=had_percent)
        bmin, bmax, bstr = parse_bounds(m.group(3))
        return {"stat": name, "value": value, "bounds": bstr, "bounds_min": bmin,
                "bounds_max": bmax, "is_exo": bmin is None}

    # compact "1PA"
    m = re.match(r"^(-?\d+)\s*([A-Za-zÀ-ÿ]+)\s*$", s)
    if m:
        value = int(m.group(1))
        raw_name = m.group(2).strip()
        name = normalize_stat_name(raw_name, had_percent=had_percent)
        return {"stat": name, "value": value, "bounds": None,
                "bounds_min": None, "bounds_max": None, "is_exo": True}

    return None