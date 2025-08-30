# src/core/utils.py
import json
from src.core.item_models import ItemTemplate, StatDefinition, ItemInstance
from src.config.stat_pool import stat_pool

def document_to_itemtemplate(doc):
    raw_stats = doc.metadata.get("stats", {})
    
    # parse JSON si string
    if isinstance(raw_stats, str):
        try:
            raw_stats = json.loads(raw_stats)
        except Exception:
            raw_stats = {}

    stats_defs = {}
    for stat_name, values in raw_stats.items():
        # matching avec stat_pool
        if stat_name in stat_pool:
            base = stat_pool[stat_name]
            weight = base.weight
        else:
            # fallback si stat pas connue → au moins 1
            weight = values.get("weight", 1)

        stats_defs[stat_name] = StatDefinition(
            name=stat_name,
            min_value=values.get("min", 0),
            max_value=values.get("max", 0),
            weight=weight
        )

    return ItemTemplate(
        name=doc.metadata.get("name", "Inconnu"),
        stats=stats_defs
    )

def build_item_from_ocr(item_name: str, current_stats: dict) -> ItemInstance:
    """
    Construit un ItemInstance uniquement avec l'OCR
    si FAISS ne trouve pas l'item.
    """
    stats_defs = {}
    for stat, val in current_stats.items():
        if stat in stat_pool:
            sd = stat_pool[stat]
            # ici : on ne connaît pas les bornes min/max exactes -> on prend val comme borne
            stats_defs[stat] = StatDefinition(
                name=stat,
                min_value=val,   # fallback minimal
                max_value=val,   # fallback minimal
                weight=sd.weight
            )
        else:
            stats_defs[stat] = StatDefinition(
                name=stat,
                min_value=val,
                max_value=val,
                weight=1
            )

    template = ItemTemplate(name=item_name, stats=stats_defs)
    return ItemInstance(template, current_stats)