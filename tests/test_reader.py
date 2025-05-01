import pytest
from src.ocr.reader import extract_stat_lines

def test_extract_stat_lines_simple():
    text = """
    Nom de l'item
    EFFETS
    +50 Vitalité
    +10 Sagesse
    +2 Portée
    POIDS : 345
    """

    expected = ["+50 Vitalité", "+10 Sagesse", "+2 Portée"]
    result = extract_stat_lines(text)
    assert result == expected

import cv2
import pytesseract

def test_detect_effets_anchor():
    image = cv2.imread("tests/assets/effets.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="fra")
    
    assert "EFFETS" in text.upper()

def test_detect_poids_anchor():
    image = cv2.imread("tests/assets/poids.png")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="fra")
    
    assert "POIDS" in text.upper()