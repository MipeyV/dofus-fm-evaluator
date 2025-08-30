# tests/test_reader_simplified.py
import src.ocr.tests.reader_simplified as reader

ASSET_PATH = "tests/assets/item_0006.png"

def test_extract_stats_with_bounds():
    stats = reader.extract_stats_with_bounds(ASSET_PATH)
    assert isinstance(stats, list)
    assert all("stat" in s for s in stats)
    assert len(stats) > 5  # au moins 5 lignes reconnues
