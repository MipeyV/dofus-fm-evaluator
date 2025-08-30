import pytest
from src.ocr.tests import reader_enhanced

def test_reader_enhanced_exo_detection():
    image_path = "tests/assets/item_0006.png"

    lines, crops = reader_enhanced.extract_stat_lines(image_path)
    assert len(lines) > 0
    assert isinstance(crops, list)

    stats = reader_enhanced.extract_stats_with_bounds(lines, crops)
    assert isinstance(stats, list)
    assert len(stats) > 0

    # Vérifie que les flags sont présents
    for s in stats:
        assert "is_exo" in s
        assert "is_over" in s

    # Au moins une stat est mappée dans le stat_pool
    assert any(s["stat"] in reader_enhanced.stat_pool for s in stats)
