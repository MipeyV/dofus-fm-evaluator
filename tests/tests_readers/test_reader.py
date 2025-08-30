from src.ocr.reader import extract_stat_lines, extract_stats_with_bounds

def test_extract_stats_with_bounds_on_asset():
    image_path = "tests/assets/item_0001.png"
    raw_lines = extract_stat_lines(image_path)
    stats = extract_stats_with_bounds(raw_lines)

    print("\nLignes OCR extraites :")
    for l in raw_lines:
        print("  >", l)

    print("\nRésultat du parsing :")
    for s in stats:
        print("  >", s)

    assert isinstance(stats, list)
    assert all("stat" in s and "value" in s for s in stats)

    assert any(s["bounds"] is not None for s in stats if s["stat"] != "signature")
