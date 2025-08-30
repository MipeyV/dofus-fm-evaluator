# src/ocr/reader_metadata.py
import cv2
import pytesseract
import re
import tempfile
from src.core.item_models import ItemMetadata

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------
# Crop zone METADATA (au-dessus de "EFFETS")
# -------------------------
def crop_metadata_from_screenshot(image_path: str) -> str:
    """Rogne la zone metadata (nom, niveau, panoplie) située au-dessus de l'ancre 'EFFETS'."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    effets_anchor = cv2.imread("tests/assets/effets.png", 0)
    effets_res = cv2.matchTemplate(gray, effets_anchor, cv2.TM_CCOEFF_NORMED)
    _, _, _, effets_loc = cv2.minMaxLoc(effets_res)

    # Zone : largeur fixe 400px, hauteur 120px au-dessus de "effets"
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
    """OCRise la zone metadata et renvoie les lignes brutes."""
    cropped_path = crop_metadata_from_screenshot(image_path)
    gray = cv2.imread(cropped_path, cv2.IMREAD_GRAYSCALE)
    config = "--psm 6 --oem 3 -c preserve_interword_spaces=1 -c user_defined_dpi=300"
    text = pytesseract.image_to_string(gray, lang="fra", config=config)
    return [line.strip() for line in text.splitlines() if line.strip()]


# -------------------------
# Parsing des lignes metadata
# -------------------------
def parse_metadata(lines: list[str]) -> ItemMetadata:
    """
    Nettoie et parse les lignes OCR pour extraire :
    - nom de l’item
    - niveau
    - type (anneau, coiffe, etc.)
    - nom de la panoplie
    """
    if not lines:
        return ItemMetadata()

    # -------------------
    # Nom de l’item (ligne 0 nettoyée)
    # -------------------
    raw_name = lines[0]
    # supprime chiffres parasites et caractères spéciaux
    clean_name = re.sub(r"^\d+\s*", "", raw_name)       # supprime un chiffre au début
    clean_name = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s\-]", "", clean_name)  # enlève les symboles
    clean_name = re.sub(r"\s+", " ", clean_name).strip()
    name = clean_name.title()

    # -------------------
    # Niveau et type (ligne 1)
    # -------------------
    level = None
    type_name = "?"
    if len(lines) > 1:
        match = re.search(r"Niveau\s*(\d+)", lines[1], re.IGNORECASE)
        if match:
            level = int(match.group(1))

        TYPES = [
            "Anneau", "Coiffe", "Ceinture", "Bottes", "Amulette",
            "Hache", "Épée", "Marteau", "Dagues", "Faux",
            "Bâton", "Arc", "Baguette", "Pelle", "Cape", "Bouclier"
        ]
        for t in TYPES:
            if t.lower() in lines[1].lower():
                type_name = t
                break

    # -------------------
    # Panoplie (ligne 2 et suivantes)
    # -------------------
    set_name = None
    for line in lines[2:]:
        if "Panoplie" in line:
            # nettoyage des caractères parasites AVANT "Panoplie"
            m_set = re.search(r"(Panoplie.+)$", line, re.IGNORECASE)
            if m_set:
                clean_set = m_set.group(1)   # garde uniquement à partir de "Panoplie"
            else:
                clean_set = line

            # supprime symboles spéciaux
            clean_set = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9\s\-]", "", clean_set)
            clean_set = re.sub(r"\s+", " ", clean_set).strip().title()
            set_name = clean_set
            break

    return ItemMetadata(
        name=name,
        level=level,
        type_name=type_name,
        set_name=set_name,
    )