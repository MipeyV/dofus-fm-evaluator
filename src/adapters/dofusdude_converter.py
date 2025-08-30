# src/adapters/dofusdude_converter.py
from src.core.item_models import ItemTemplate
from src.config.stat_pool import stat_pool, StatDefinition


def api_item_to_template(item_json: dict) -> ItemTemplate:
    """
    Convertit un item provenant de l’API dofusdude (dict) en ItemTemplate.
    """
    stats = {}

    # Effets peut être None → on force []
    for effect in item_json.get("effects") or []:
        name = effect.get("type", {}).get("name")
        min_val = effect.get("int_minimum", 0) or 0
        max_val = effect.get("int_maximum", 0) or 0

        if name:
            stats[name.lower()] = StatDefinition(
                name=name.lower(),
                min_value=min_val,
                max_value=max_val,
                weight=1  # ⚡ TODO : ajuster avec ton stat_pool
            )

    return ItemTemplate(
        name=item_json.get("name", "Inconnu"),
        stats=stats
    )