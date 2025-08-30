# src/ocr/reader_simplified.py
import cv2
import pytesseract
import numpy as np
import tempfile

from typing import List
from src.core.match_stats import parse_stat_line, ExtractedStat  # réutilise la logique commune

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------
# OCR brut
# -------------------------
def read_image_stats(image_path: str) -> str:
    """OCR brut de l’image entière (debug/diagnostic)."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, lang='fra')


# -------------------------
# Extraction zone des lignes de stats
# -------------------------
def extract_stat_lines(image_path: str) -> List[str]:
    """Extrait uniquement la zone stats (entre EFFETS et POIDS) et retourne les lignes OCRisées."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

    # fallback si ancrages faibles
    if effets_val < 0.6 or poids_val < 0.6:
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        config = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
        raw_text = pytesseract.image_to_string(up, lang='fra', config=config)
        return [line.strip() for line in raw_text.splitlines() if line.strip()]

    # coordonnées ROI
    PADDING_LEFT = 50
    x_start = effets_loc[0] + PADDING_LEFT
    x_end = poids_loc[0] - 10
    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

    # sécurités
    x_start = max(0, min(x_start, gray.shape[1]-1))
    x_end = max(0, min(x_end, gray.shape[1]-1))
    y_start = max(0, min(y_start, gray.shape[0]-1))
    y_end = max(0, min(y_end, gray.shape[0]-1))

    if x_end <= x_start or y_end <= y_start:
        # fallback OCR sur toute l'image si crop vide
        up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        config = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
        raw_text = pytesseract.image_to_string(up, lang='fra', config=config)
        return [line.strip() for line in raw_text.splitlines() if line.strip()]
    
    cropped = gray[y_start:y_end, x_start:x_end]

    # OCR amélioré
    blurred = cv2.GaussianBlur(cropped, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    up = cv2.resize(enhanced, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up = cv2.medianBlur(up, 3)

    config = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
    text = pytesseract.image_to_string(up, lang="fra", config=config)

    return [line.strip() for line in text.splitlines() if line.strip()]


# -------------------------
# Extraction + parsing complet
# -------------------------
def extract_stats_with_bounds(image_path: str) -> List[ExtractedStat]:
    """Pipeline complet : OCR → normalisation → parsing → stats exploitables."""
    lines = extract_stat_lines(image_path)
    extracted = []
    for line in lines:
        stat = parse_stat_line(line)
        if stat:
            extracted.append(stat)
    return extracted


# -------------------------
# Crop zone brute (debug)
# -------------------------
def crop_item_from_screenshot(image_path: str) -> str:
    """Rogne automatiquement la zone stats depuis un screen brut et renvoie le chemin du PNG temporaire."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, _, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, _, _, poids_loc = cv2.minMaxLoc(poids_res)

    PADDING_LEFT = 50
    x_start = effets_loc[0] + PADDING_LEFT
    x_end = poids_loc[0] - 10
    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

    cropped = image[y_start:y_end, x_start:x_end]

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cv2.imwrite(tmpfile.name, cropped)
    return tmpfile.name
