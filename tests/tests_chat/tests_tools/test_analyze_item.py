# tests/tests_chat/tests_tools/test_analyze_item.py
import pytest
from scripts.tools import _analyze_item_impl

def test_analyze_item_on_sample():
    # Image de test connue (ex: Anneau Crustique lvl 200 dans assets)
    image_path = "tests/assets/item_0006.png"

    result = _analyze_item_impl(image_path)

    # Le résultat doit être un dict
    assert isinstance(result, dict)

    # Pas d'erreur
    assert "error" not in result

    # Vérifie présence des champs attendus
    assert "item" in result
    assert "level" in result
    assert "stats_detected" in result
    assert "evaluation" in result
    assert "commentaire" in result

    # Vérifie contenu minimal
    assert isinstance(result["stats_detected"], dict)
    assert isinstance(result["evaluation"], dict)
    assert "quality" in result["evaluation"]
    assert "note" in result["evaluation"]

    # L'item doit ressembler à un "Anneau Crustique"
    assert "anneau" in result["item"].lower()
    assert result["level"] == 200
