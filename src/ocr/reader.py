import cv2
import pytesseract
import re
import numpy as np
import tempfile
from typing import Optional, TypedDict

from regex import F

from src.config.stat_pool import stat_pool  # utilisé pour vérifier/mapper les clés

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class ExtractedStat(TypedDict):
    stat: str
    value: int
    bounds: Optional[str]
    bounds_min: Optional[int]
    bounds_max: Optional[int]
    is_exo: bool
    is_over: bool


def read_image_stats(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='fra')
    return text

# -------------------------
# Détection couleur bleue = EXO
# -------------------------
def is_line_exo(color_crop: np.ndarray) -> bool:
    hsv = cv2.cvtColor(color_crop, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 120])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    ratio = cv2.countNonZero(mask) / (color_crop.shape[0] * color_crop.shape[1])
    return ratio > 0.02

def extract_stat_lines(image_path: str) -> tuple[list[str], list[np.ndarray]]:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

    # coordonnées ROI
    PADDING_LEFT = 50
    x_start = effets_loc[0] + PADDING_LEFT
    x_end = poids_loc[0] - 10
    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

    # --- Fallback OCR complet si mauvaise bbox ---
    if effets_val < 0.6 or poids_val < 0.6 or x_end <= x_start or y_end <= y_start:
        print("[WARN] Fallback OCR complet (anchors non fiables)")
        config_full = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
        up_full = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        up_full = cv2.medianBlur(up_full, 3)
        raw_text = pytesseract.image_to_string(up_full, lang='fra', config=config_full)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return lines, []

    # --- sinon, OCR ciblé ---
    cropped_color = image[y_start:y_end, x_start:x_end]
    cropped_gray = gray[y_start:y_end, x_start:x_end]

    blurred = cv2.GaussianBlur(cropped_gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    up = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.medianBlur(up, 3)
    config = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
    text = pytesseract.image_to_string(up, lang="fra", config=config)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # Découpe approximative par ligne
    crops = []
    h = cropped_color.shape[0] // max(1, len(lines))
    for i in range(len(lines)):
        y1, y2 = i*h, min((i+1)*h, cropped_color.shape[0])
        crops.append(cropped_color[y1:y2, :])

    return lines, crops

def extract_stats_with_bounds(lines: list[str], crops: Optional[list[np.ndarray]] = None) -> list[ExtractedStat]:
    import unicodedata

    extracted: list[ExtractedStat] = []

    # --- 1) parsing bornes robuste : accepte [..], (..), ou fermeture manquante
    def parse_bounds(bounds_str: Optional[str]) -> tuple[Optional[int], Optional[int], Optional[str]]:
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

    # --- 2) détection couleur bleu (exo)
    def is_exo_color(crop: np.ndarray) -> bool:
        if crop is None:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        ratio = cv2.countNonZero(mask) / (crop.shape[0] * crop.shape[1])
        return ratio > 0.05  # seuil 5% de pixels bleus

    # --- 3) dictionnaire d’alias connu ---
    ALIASES = {
        # basiques
        "vitalite": "vitalité",
        "sagesse": "sagesse",
        "force": "force",
        "chance": "chance",
        "agilite": "agilité",
        "intelligence": "intelligence",
        "puissance": "puissance",

        # PA/PM/PO
        "pa": "PA",
        "pm": "PM",
        "po": "PO",

        # critiques
        "critique": "coups_critiques",
        "critiques": "coups_critiques",
        "coups critiques": "coups_critiques",
        "cc": "coups_critiques",

        # dommages fixes
        "dommages": "dommage",
        "dommage": "dommage",
        "dommages neutre": "dommage_neutre",
        "dommage neutre": "dommage_neutre",
        "dommages terre": "dommage_terre",
        "dommage terre": "dommage_terre",
        "dommages feu": "dommage_feu",
        "dommage feu": "dommage_feu",
        "dommages eau": "dommage_eau",
        "dommage eau": "dommage_eau",
        "dommages air": "dommage_air",
        "dommage air": "dommage_air",
        "dommages critique": "dommage_critique",
        "dommage critique": "dommage_critique",

        # résistances fixes
        "resistance neutre": "résistance_neutre",
        "resistances neutre": "résistance_neutre",
        "resistance terre": "résistance_terre",
        "resistances terre": "résistance_terre",
        "resistance feu": "résistance_feu",
        "resistances feu": "résistance_feu",
        "resistance eau": "résistance_eau",
        "resistances eau": "résistance_eau",
        "resistance air": "résistance_air",
        "resistances air": "résistance_air",
        "resistance poussee": "résistance_poussée",
        "resistances poussee": "résistance_poussée",
        "resistance critique": "résistance_critique",
        "resistances critique": "résistance_critique",

        # résistances %
        "resistance neutre %": "résistance_neutre_%",
        "resistance terre %": "résistance_terre_%",
        "resistance feu %": "résistance_feu_%",
        "resistance eau %": "résistance_eau_%",
        "resistance air %": "résistance_air_%",

        # dommages %
        "dommages melee %": "dommage_mêlee_%",
        "dommage melee %": "dommage_mêlee_%",
        "dommages distance %": "dommage_dist_%",
        "dommage distance %": "dommage_dist_%",
        "dommages sort %": "dommage_sort_%",

        # divers
        "invocation": "invocation",
        "invocations": "invocation",
        "tacle": "tacle",
        "tacles": "tacle",
        "fuite": "fuite",
        "fuites": "fuite",
        "soin": "soin",
        "soins": "soin",
        "prospection": "prospection",
        "prospections": "prospection",
        "initiative": "initiative",
        "initiatives": "initiative",
        "pods": "pods",
        "retrait pa": "retrait_pa",
        "retraits pa": "retrait_pa",
        "retrait pm": "retrait_pm",
        "retraits pm": "retrait_pm",
        "esquive pa": "esquive_pa",
        "esquives pa": "esquive_pa",
        "esquive pm": "esquive_pm",
        "esquives pm": "esquive_pm",
    }

    # --- 4) normalisation nom ---
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

    # --- 5) parse d’une ligne ---
    def try_parse(line: str, idx: int) -> Optional[ExtractedStat]:
        s = line.strip()
        had_percent = "%" in s

        patterns = [
            (r"^(-?\d+)\s*%\s+(.+?)(\s*[\[(].*)?$", True),
            (r"^(-?\d+)\s+(.+?)(\s*[\[(].*)?$", False),
            (r"^(.+?)\s+(-?\d+)\s*%?(\s*[\[(].*)?$", False),
            (r"^(-?\d+)\s*([A-Za-zÀ-ÿ]+)\s*$", False),
        ]

        for pat, force_percent in patterns:
            m = re.match(pat, s)
            if not m:
                continue

            # Parsing des valeurs
            if pat == patterns[0][0]:
                value, raw_name, bounds = int(m.group(1)), m.group(2).strip(), m.group(3)
                name = normalize_stat_name(raw_name, had_percent=True)
            elif pat == patterns[1][0]:
                value, raw_name, bounds = int(m.group(1)), m.group(2).strip(), m.group(3)
                name = normalize_stat_name(raw_name, had_percent=had_percent)
            elif pat == patterns[2][0]:
                raw_name, value, bounds = m.group(1).strip(), int(m.group(2)), m.group(3)
                name = normalize_stat_name(raw_name, had_percent=had_percent)
            else:
                value, raw_name = int(m.group(1)), m.group(2).strip()
                name, bounds = normalize_stat_name(raw_name, had_percent=had_percent), None

        bmin, bmax, bstr = parse_bounds(bounds)
        is_exo = False

        # 1) Détection par couleur
        if crops is not None and idx < len(crops):
            if is_exo_color(crops[idx]):
                is_exo = True

        # 2) Détection par position : premières lignes sans bounds
        if not bmin and not bmax and idx < 2:
            is_exo = True

        return {
            "stat": name,
            "value": value,
            "bounds": bstr,
            "bounds_min": bmin,
            "bounds_max": bmax,
            "is_exo": is_exo,
            "is_over": False
        }
        return None

    # --- Boucle lignes ---
    for idx, line in enumerate(lines):
        parsed = try_parse(line, idx)
        if parsed:
            # exo = couleur bleue détectée
            if crops is not None and idx < len(crops) and is_exo_color(crops[idx]):
                parsed["is_exo"] = True
            else:
                # exo si pas de bornes ET ligne tout en haut (0 ou 1)
                if parsed["bounds_min"] is None and parsed["bounds_max"] is None and idx <= 1:
                    parsed["is_exo"] = True
                else:
                    parsed["is_exo"] = False
            extracted.append(parsed)

    return extracted

def crop_item_from_screenshot(image_path: str) -> str:
    """Rogne automatiquement la zone item depuis un screen brut et renvoie le chemin du PNG temporaire"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, _, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, _, _, poids_loc = cv2.minMaxLoc(poids_res)

    # coordonnées de la zone
    PADDING_LEFT = 50
    x_start = effets_loc[0] + PADDING_LEFT
    x_end = poids_loc[0] - 10
    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

    x_start = max(0, min(x_start, gray.shape[1]-1))
    x_end   = max(0, min(x_end, gray.shape[1]-1))
    y_start = max(0, min(y_start, gray.shape[0]-1))
    y_end   = max(0, min(y_end, gray.shape[0]-1))

    if x_end <= x_start or y_end <= y_start:
        raise ValueError("Impossible de détecter correctement la zone item")

    cropped = image[y_start:y_end, x_start:x_end]

    # sauvegarde temporaire
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmpfile.name, cropped)
    return tmpfile.name
