import pytest
from src.ocr.reader_metadata import parse_metadata
from src.core.item_models import ItemMetadata

def test_parse_metadata_clean():
    lines = [
        "1 Anneau Crustique",
        "Niveau 200 - & Anneau",
        "Je Panoplie des Tréfonds",
    ]
    meta = parse_metadata(lines)
    assert isinstance(meta, ItemMetadata)
    assert meta.name == "Anneau Crustique"
    assert meta.level == 200
    assert meta.type_name == "Anneau"
    assert meta.set_name == "Panoplie Des Tréfonds"

def test_parse_metadata_with_noise():
    lines = [
        "12345  Anneau Crustique!!!",
        "Niveau 200 -  $$ Anneau",
        "; Je Panoplie des Tréfonds ;;",
    ]
    meta = parse_metadata(lines)
    assert meta.name == "Anneau Crustique"
    assert meta.level == 200
    assert meta.type_name == "Anneau"
    assert meta.set_name == "Panoplie Des Tréfonds"

def test_parse_metadata_missing_set():
    lines = [
        "Amulette Magique",
        "Niveau 150 Amulette",
    ]
    meta = parse_metadata(lines)
    assert meta.name == "Amulette Magique"
    assert meta.level == 150
    assert meta.type_name == "Amulette"
    assert meta.set_name is None
