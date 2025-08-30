# src/ocr/reader_enhanced.py
import cv2
from networkx import is_edge_cover
import pytesseract
import re
import numpy as np
import tempfile
from typing import Optional, TypedDict
from src.ocr.reader import extract_stats_with_bounds
from src.core.item_models import ItemTemplate, ItemInstance, ItemMetadata
from src.config.stat_pool import stat_pool, StatDefinition  # utilisé pour vérifier/mapper les clés

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ExtractedStat(TypedDict):
    stat: str
    value: int
    bounds: Optional[str]
    bounds_min: Optional[int]
    bounds_max: Optional[int]
    is_exo: bool

# -------------------------
# OCR brut
# -------------------------
def read_image_stats(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='fra')
    return text


# -------------------------
# Extraction zone des lignes de stats
# -------------------------
def extract_stat_lines(image_path: str) -> tuple[list[str], list[np.ndarray]]:
    """
    OCR sur la zone stats.
    Retourne (liste des lignes OCR, liste de crops couleur par ligne).
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

    # fallback complet
    if effets_val < 0.6 or poids_val < 0.6:
        blurred_full = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_full = clahe.apply(blurred_full)
        up_full = cv2.resize(enhanced_full, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        up_full = cv2.medianBlur(up_full, 3)
        config_full = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
        raw_text = pytesseract.image_to_string(up_full, lang='fra', config=config_full)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        return lines, []

    # coordonnées
    PADDING_LEFT = 50
    x_start = effets_loc[0] + PADDING_LEFT
    ROI_MAX_WIDTH_PX = 330
    x_end = min(x_start + ROI_MAX_WIDTH_PX, gray.shape[1] - 10)
    right_limit_by_anchor = poids_loc[0] - 10
    if right_limit_by_anchor > x_start + 40:
        x_end = min(x_end, right_limit_by_anchor)

    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

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

    # découpe approximative
    crops = []
    h = cropped_color.shape[0] // max(1, len(lines))
    for i in range(len(lines)):
        y1, y2 = i*h, min((i+1)*h, cropped_color.shape[0])
        crops.append(cropped_color[y1:y2, :])

    return lines, crops    

def _is_signature_line(t: str) -> bool:
    tl = t.lower()
    return ("par" in tl) and ("modif" in tl or "difi" in tl or "fabriq" in tl)


# -------------------------
# Crop zone stats
# -------------------------
def crop_item_from_screenshot(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

    PADDING_LEFT = 50
    x_start = effets_loc[0] + PADDING_LEFT
    x_end = poids_loc[0] - 10
    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

    if x_end <= x_start or y_end <= y_start or effets_val < 0.6 or poids_val < 0.6:
        print("[WARN] Bbox incohérente, fallback utilisé")
        x_start = max(0, effets_loc[0] + PADDING_LEFT)
        x_end = min(x_start + 350, gray.shape[1] - 1)
        y_start = effets_loc[1] + effets_anchor.shape[0] + 5
        y_end = min(y_start + 400, gray.shape[0] - 1)

    cropped = image[y_start:y_end, x_start:x_end]
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmpfile.name, cropped)
    return tmpfile.name


# -------------------------
# Crop zone METADATA
# -------------------------
def crop_metadata_from_screenshot(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    _, _, _, effets_loc = cv2.minMaxLoc(effets_res)

    PADDING_LEFT = 50
    x_start = max(0, effets_loc[0] - PADDING_LEFT)
    x_end = min(x_start + 400, gray.shape[1] - 1)
    y_end = effets_loc[1] - 5
    y_start = max(0, y_end - 120)

    if x_end <= x_start or y_end <= y_start:
        raise ValueError("Impossible de détecter correctement la zone metadata")

    cropped = image[y_start:y_end, x_start:x_end]
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmpfile.name, cropped)
    return tmpfile.name


def extract_metadata_lines(image_path: str) -> list[str]:
    cropped_path = crop_metadata_from_screenshot(image_path)
    gray = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
    config = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
    text = pytesseract.image_to_string(gray, lang='fra', config=config)
    return [line.strip() for line in text.splitlines() if line.strip()]


# -------------------------
# Parse metadata (robuste avec regex + types connus)
# -------------------------
ITEM_TYPES = [
    "anneau", "amulette", 
    "bottes", "ceinture", 
    "cape", "coiffe", 
    "dagues", "épée", "faux", "hache", "lance", "marteau", "pelle", 
    "bâton", "arc", "baguette",
    "bouclier"
]

def parse_metadata(lines: list[str]) -> ItemMetadata:
    name = "?"
    level = None
    type_name = "?"
    set_name = None

    for line in lines:
        clean = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s\-]", "", line)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            continue

        # --- Ligne nom (première ligne rencontrée)
        if name == "?":
            # supprime un chiffre parasite au début ex "1 Anneau Crustique"
            name = re.sub(r"^\d+\s*", "", clean).strip().title()
            continue

        # --- Ligne niveau + type
        m = re.search(r"Niveau\s+(\d+)", clean, re.IGNORECASE)
        if m:
            level = int(m.group(1))
            for t in ITEM_TYPES:
                if t in clean.lower():
                    type_name = t.capitalize()
                    break
            continue

        # --- Ligne panoplie
        if "panoplie" in clean.lower():
            # extrait juste "Panoplie ..." sans les parasites avant
            m_set = re.search(r"(Panoplie.+)$", clean, re.IGNORECASE)
            set_name = m_set.group(1).strip().title() if m_set else clean
            continue

    return ItemMetadata(name=name, level=level, type_name=type_name, set_name=set_name)

# -------------------------
# Assemblage
# -------------------------
def read_item(image_path: str):
    stat_lines, crops = extract_stat_lines(image_path)
    extracted = extract_stats_with_bounds(stat_lines, crops)  # on passe les crops !

    stats_defs = {}
    current_stats = {}
    for stat in extracted:
        name = stat["stat"]
        min_v = stat["bounds_min"] or stat["value"]
        max_v = stat["bounds_max"] or stat["value"]
        weight = stat_pool[name].weight if name in stat_pool else 1
        stats_defs[name] = StatDefinition(name, min_v, max_v, weight)
        current_stats[name] = stat["value"]

    meta_lines = extract_metadata_lines(image_path)
    metadata = parse_metadata(meta_lines)

    template = ItemTemplate(name=metadata.name, stats=stats_defs)
    instance = ItemInstance(template, current_stats)

    class Item:
        def __init__(self, metadata, template, instance):
            self.metadata = metadata
            self.template = template
            self.instance = instance

    return Item(metadata, template, instance)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr.reader_enhanced <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        item = read_item(image_path)
        print("[RESULT] --- METADATA ---")
        print(item.metadata.__dict__)
        print("\n[RESULT] --- TEMPLATE ---")
        print(item.template.stats)
        print("\n[RESULT] --- FEATURES ---")
        print(item.instance.get_features())
        print("\n[RESULT] --- EVALUATION ---")
        print(item.instance.evaluate_quality_algo())
    except Exception as e:
        print(f"[ERROR] {e}")
