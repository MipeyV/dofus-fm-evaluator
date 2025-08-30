# src/ocr/reader_combined.py
import sys

from src.ocr.reader import (
    crop_item_from_screenshot,
    extract_stat_lines,
    extract_stats_with_bounds,
)
from src.ocr.reader_metadata import (
    crop_metadata_from_screenshot,
    extract_metadata_lines,
    parse_metadata,
)
from src.core.item_models import ItemTemplate, ItemInstance, ItemMetadata
from src.config.stat_pool import stat_pool, StatDefinition


def read_item(image_path: str):
    # 1) Stats
    stat_lines, crops = extract_stat_lines(image_path)   # <<< déstructuration
    extracted = extract_stats_with_bounds(stat_lines)

    stats_defs = {}
    current_stats = {}
    for stat in extracted:
        name = stat["stat"]
        min_v = stat["bounds_min"] or stat["value"]
        max_v = stat["bounds_max"] or stat["value"]
        weight = stat_pool[name].weight if name in stat_pool else 1
        stats_defs[name] = StatDefinition(name, min_v, max_v, weight)
        current_stats[name] = stat["value"]

    # 2) Metadata
    meta_lines = extract_metadata_lines(image_path)
    metadata = parse_metadata(meta_lines)

    template = ItemTemplate(name=metadata.name, stats=stats_defs)
    instance = ItemInstance(template, current_stats)

    # 3) Assemblage
    class Item:
        def __init__(self, metadata, template, instance, crops):
            self.metadata = metadata
            self.template = template
            self.instance = instance
            self.crops = crops   # on garde aussi les crops pour debug ou exo detection

    return Item(metadata, template, instance, crops)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.ocr.reader_combined <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    item = read_item(image_path)

    print("\n[RESULT] --- METADATA ---")
    print(item.metadata.__dict__)

    print("\n[RESULT] --- TEMPLATE (stats defs) ---")
    for k, v in item.template.stats.items():
        print(f"{k}: {v}")

    print("\n[RESULT] --- INSTANCE (current stats) ---")
    for k, v in item.instance.current_stats.items():
        print(f"{k}: {v}")

    print("\n[RESULT] --- FEATURES ---")
    for k, v in item.instance.get_features().items():
        print(f"{k}: {v}")

    print("\n[RESULT] --- EVALUATION ---")
    print(item.instance.evaluate_quality_algo())