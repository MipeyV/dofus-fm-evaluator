import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_stat_lines(image_path: str):
    # 1. Lire et convertir l'image en niveaux de gris
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. OCR avec Tesseract
    raw_text = pytesseract.image_to_string(gray, lang='fra')

    # 3. Lignes nettoyées
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # 4. Délimitation de la zone d'effets (entre EFFETS et POIDS)
    effets_idx = next((i for i, line in enumerate(lines) if "EFFETS" in line.upper()), None)
    poids_idx = next((i for i, line in enumerate(lines) if "POIDS" in line.upper()), None)

    if effets_idx is None or poids_idx is None or poids_idx <= effets_idx:
        return []

    stat_lines = lines[effets_idx + 1:poids_idx]

    # 5. Parsing ligne par ligne
    parsed_stats = []
    for line in stat_lines:
        match = re.match(r"([+-]?\d+)\s+(.+?)\s+\[(\d+)\s+à\s+(\d+)\]", line)
        if match:
            current = int(match.group(1))
            name = match.group(2).strip()
            min_val = int(match.group(3))
            max_val = int(match.group(4))
            parsed_stats.append({
                "stat": name,
                "current": current,
                "min": min_val,
                "max": max_val
            })
        else:
            # fallback si pas de stats théoriques encadrées
            match_simple = re.match(r"([+-]?\d+)\s+(.+)", line)
            if match_simple:
                current = int(match_simple.group(1))
                name = match_simple.group(2).strip()
                parsed_stats.append({
                    "stat": name,
                    "current": current,
                    "min": None,
                    "max": None
                })

    return parsed_stats

def extract_stats_with_bounds(lines):
    stats = []

    for line in lines:
        # Exemple : "341 Vitalité [301 à 350]"
        match = re.match(r"([\+\-]?\d+)\s+(.+?)\s+\[(\d+)\s*[à\-]\s*(\d+)\]", line)
        if match:
            value, stat, low, high = match.groups()
            stats.append({
                "stat": stat.strip(),
                "value": int(value),
                "bounds": (int(low), int(high)),
                "is_exo": False
            })
        else:
            # Cas sans borne → considéré comme exo
            match_exo = re.match(r"([\+\-]?\d+)\s+(.+)", line)
            if match_exo:
                value, stat = match_exo.groups()
                stats.append({
                    "stat": stat.strip(),
                    "value": int(value),
                    "bounds": None,
                    "is_exo": True
                })

    return stats

def parse_effect_line(line):
    # Exemple : "+336 Vitalité [301 à 350]"
    match = re.match(r"([+-]?\d+)\s+(.+?)\s+\[(\d+)\s+à\s+(\d+)\]", line)
    if match:
        current_val = int(match.group(1))
        stat_name = match.group(2).strip()
        min_val = int(match.group(3))
        max_val = int(match.group(4))
        return {
            "stat": stat_name,
            "current": current_val,
            "min": min_val,
            "max": max_val
        }
    return None