from src.ocr.reader import extract_stat_lines, extract_stats_with_bounds
from src.features.features_extraction import calculate_features_from_stats

# Extract stats
lines = extract_stat_lines('tests/assets/item_0001.png')
stats = extract_stats_with_bounds(lines)

print("Stats avec ratios:")
for s in stats:
    if s.get('bounds_min') is not None and s.get('bounds_max') is not None:
        min_val = s['bounds_min']
        max_val = s['bounds_max']
        current_val = s['value']
        if max_val > min_val:
            ratio = (current_val - min_val) / (max_val - min_val)
            print(f"{s['stat']}: {current_val} [{min_val}-{max_val}] -> ratio: {ratio:.3f}")
            if 0.9 <= ratio < 1.0:
                print(f"  -> HIGH RATIO (0.9-1.0)")
            elif abs(ratio - 1.0) < 0.01:
                print(f"  -> PERFECT (1.0)")

print("\nFeatures calculées:")
features = calculate_features_from_stats(stats)
for k, v in features.items():
    print(f"  {k}: {v}")
