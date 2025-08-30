# tests/tests_readers/test_reader_combined.py
import pytest
from src.ocr.reader_combined import read_item
from src.core.item_models import ItemMetadata, ItemTemplate, ItemInstance


def test_read_item_integration():
    image_path = "tests/assets/item_0006.png"
    item = read_item(image_path)

    assert item.metadata.name == "Anneau Crustique"
    assert item.metadata.level == 200
    assert item.metadata.type_name == "Anneau"
    assert "Tréfonds" in (item.metadata.set_name or "")

    assert len(item.template.stats) > 0              # définitions
    assert len(item.instance.current_stats) > 0      # valeurs réelles
