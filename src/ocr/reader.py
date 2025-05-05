import cv2
import pytesseract
import re
from typing import List, Dict, Optional

from src.config.stat_pool import stat_pool  # Utilise les définitions centralisées

# Définir le chemin Tesseract si nécessaire (pour Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def read_image_stats(image_path: str) -> str:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='fra')
    return text

def extract_stat_lines(image_path: str) -> List[str]:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ancrage par template matching
    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    poids_anchor = cv2.imread("tests/assets/poids.png", 0)

    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    poids_res = cv2.matchTemplate(gray, poids_anchor, cv2.TM_CCOEFF_NORMED)

    _, effets_val, _, effets_loc = cv2.minMaxLoc(effets_res)
    _, poids_val, _, poids_loc = cv2.minMaxLoc(poids_res)

    if effets_val < 0.6 or poids_val < 0.6:
        raw_text = pytesseract.image_to_string(gray, lang='fra')
        return [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Largeur plus forte pour supprimer les pictos
    PADDING_LEFT = 50  # Augmenté
    x_start = effets_loc[0] + PADDING_LEFT
    x_end = effets_loc[0] + effets_anchor.shape[1]
    y_start = effets_loc[1] + effets_anchor.shape[0]
    y_end = poids_loc[1]

    cropped = gray[y_start:y_end, x_start:x_end]

    # Meilleur contraste
    _, binary = cv2.threshold(cropped, 150, 255, cv2.THRESH_BINARY)

    # Config OCR plus stable
    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(binary, lang="fra", config=config)

    # Debug
    cv2.imwrite("tests/debug_zone_analyse.png", binary)

    return [line.strip() for line in text.splitlines() if line.strip()]

def extract_stats_with_bounds(lines: List[str]) -> List[Dict[str, Optional[str]]]:
    extracted = []
    for line in lines:
        # Gestion des lignes signature
        if "modifié par" in line.lower() or "fabriqué par" in line.lower():
            extracted.append({
                "stat": "signature",
                "value": line.strip(),
                "bounds": None,
                "is_exo": None
            })
            continue

        # Match format: nom valeur [min à max], [val], ou (val)
        match = re.match(r"(.+?)\s+(-?\d+)\s*(\[(\-?\d+)\s+à\s+(\-?\d+)\]|\[(\-?\d+)\]|\((.*?)\))", line)
        if match:
            name = match.group(1).strip().lower()
            value = int(match.group(2))
            if match.group(4) and match.group(5):
                bounds = f"{match.group(4)} à {match.group(5)}"
            elif match.group(6):
                bounds = f"{match.group(6)} à {match.group(6)}"
            else:
                bounds = match.group(7)
            extracted.append({
                "stat": name,
                "value": value,
                "bounds": bounds,
                "is_exo": False
            })
        else:
            # Match exo simple : nom valeur
            try:
                name, value = line.rsplit(" ", 1)
                extracted.append({
                    "stat": name.strip().lower(),
                    "value": int(value),
                    "bounds": None,
                    "is_exo": True
                })
            except Exception:
                continue
    return extracted