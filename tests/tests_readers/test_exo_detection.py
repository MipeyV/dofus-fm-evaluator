import numpy as np
import cv2
import pytest
from src.ocr.reader import extract_stats_with_bounds

def test_exo_detection_top_line():
    # Cas simulé : une stat sans bornes (donc exo)
    lines = [
        "1 PA",            # pas de bornes → exo attendu car c'est la 1ère ligne
        "350 Vitalité [301 à 350]"  # bornes présentes → pas exo
    ]
    crops = [np.zeros((10, 50, 3), dtype=np.uint8),  # crop factice (pas bleu)
             np.zeros((10, 50, 3), dtype=np.uint8)]

    stats = extract_stats_with_bounds(lines, crops)

    assert stats[0]["is_exo"] is True, "La 1ère ligne sans bornes doit être détectée comme EXO"
    assert stats[1]["is_exo"] is False, "La vitalité avec bornes ne doit pas être EXO"

def test_exo_detection_color_blue():
    # Cas simulé : stat avec bornes mais crop bleu → exo attendu
    lines = ["80 Force [61 à 80]"]
    crop = np.zeros((10, 50, 3), dtype=np.uint8)
    # Remplissage bleu (dans la plage HSV de l'exo)
    crop[:] = (255, 0, 0)  # BGR pur bleu
    crops = [crop]

    stats = extract_stats_with_bounds(lines, crops)
    assert stats[0]["is_exo"] is True, "La stat doit être EXO car le crop est bleu"

def test_no_exo_with_bounds_and_no_color():
    # Cas normal : bornes présentes et pas de bleu → pas exo
    lines = ["12% Dommages [9 à 12]"]
    crops = [np.zeros((10, 50, 3), dtype=np.uint8)]  # pas bleu

    stats = extract_stats_with_bounds(lines, crops)
    assert stats[0]["is_exo"] is False, "Une stat bornée sans bleu ne doit pas être EXO"