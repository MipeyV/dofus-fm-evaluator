import numpy as np
from src.config.stat_pool import (
    StatDefinition,  # déjà défini dans stat_pool.py
    stat_pool, essential_stats, basic_stats, secondary_stats, heavy_stats,
    résistance_elem_stats, résistance_pourcent_stats, dommage_elem_stats, dommage_pourcent_stats
)


class ItemTemplate:
    def __init__(self, name, stats, pui_category=None):
        self.name = name
        self.stats = stats
        # si non fourni, on calcule automatiquement
        self.pui_category = pui_category or self.compute_pui_category()

    def get_stat_max(self, stat_name):
        return self.stats.get(stat_name).max_value if stat_name in self.stats else 0

    def get_stat_weight(self, stat_name):
        return self.stats.get(stat_name).weight if stat_name in self.stats else 0

    def has_stat(self, stat_name):
        return stat_name in self.stats

    def compute_pui_category(self):
        """Catégorise l’item en fonction de son poids total max"""
        total_weight = sum(sd.max_value * sd.weight for sd in self.stats.values())
        if total_weight > 1000:
            return "grand"
        elif total_weight > 400:
            return "moyen"
        return "faible"


class ItemInstance:
    def __init__(self, template, current_stats, label=None):
        self.template = template
        self.current_stats = current_stats
        self.label = label

    def get_ratio(self, stat):
        base = self.template.get_stat_max(stat)
        val = self.current_stats.get(stat, 0)
        if base <= 0:
            return np.nan
        return val / base

    def is_over(self, stat):
        ratio = self.get_ratio(stat)
        return ratio > 1 if not np.isnan(ratio) else False

    def is_exo(self, stat):
        return stat not in self.template.stats and self.current_stats.get(stat, 0) > 0

    def get_total_weight(self):
        weight = 0
        for stat in self.template.stats:
            value = min(self.current_stats.get(stat, 0), self.template.get_stat_max(stat))
            weight += value * self.template.get_stat_weight(stat)
        return weight

    def get_exo_weight(self):
        weight = 0
        for stat in self.current_stats:
            if self.is_exo(stat) and stat in stat_pool:
                weight += self.current_stats[stat] * stat_pool[stat].weight
        return weight

    def get_stat_tolerance(stat_name, base_value, current_value, weight_unit):
        stat_weight = current_value * weight_unit
        if stat_weight <= 30 * weight_unit:
            return base_value
        elif stat_weight <= 60 * weight_unit:
            return 3 * base_value
        else:
            return 10 * base_value

    def get_features(self):
        nb_stats = len(self.template.stats)
        perfect_lines = sum(
            1 for stat in self.template.stats
            if self.current_stats.get(stat, 0) == self.template.get_stat_max(stat)
        )
        high_ratios = sum(
            1 for stat in self.template.stats
            if 0.9 < self.get_ratio(stat) < 1
        )
        nb_overs = sum(1 for stat in self.template.stats if self.is_over(stat))
        total_weight = self.get_total_weight()
        exo_weight = self.get_exo_weight()
        ratios = [self.get_ratio(stat) for stat in self.template.stats]
        avg_ratio = np.nanmean(ratios) if ratios else 0.0
        over_weight = sum(
            (self.current_stats[stat] - self.template.get_stat_max(stat)) * self.template.get_stat_weight(stat)
            for stat in self.template.stats
            if self.is_over(stat)
        )

        nb_basic_stats_ratio = sum(1 for stat in self.template.stats if stat in basic_stats) / nb_stats if nb_stats else 0
        nb_essential_stats_ratio = sum(1 for stat in self.template.stats if stat in essential_stats) / nb_stats if nb_stats else 0
        nb_secondary_stats_ratio = sum(1 for stat in self.template.stats if stat in secondary_stats) / nb_stats if nb_stats else 0
        nb_heavy_stats_ratio = sum(1 for stat in self.template.stats if stat in heavy_stats) / nb_stats if nb_stats else 0
        nb_résistance_elem_ratio = sum(1 for stat in self.template.stats if stat in résistance_elem_stats) / nb_stats if nb_stats else 0
        nb_resistance_pourcent_ratio = sum(1 for stat in self.template.stats if stat in résistance_pourcent_stats) / nb_stats if nb_stats else 0
        nb_dommage_elem_ratio = sum(1 for stat in self.template.stats if stat in dommage_elem_stats) / nb_stats if nb_stats else 0
        nb_dommage_pourcent_ratio = sum(1 for stat in self.template.stats if stat in dommage_pourcent_stats) / nb_stats if nb_stats else 0

        return {
            "item_name": self.template.name,
            "pui_category": self.template.pui_category,
            "nb_stats": nb_stats,
            "nb_perfect_lines": perfect_lines,
            "nb_high_ratio": high_ratios,
            "nb_overs": nb_overs,
            "total_weight": total_weight,
            "exo_weight": exo_weight,
            "over_weight": over_weight,
            "avg_ratio": avg_ratio,
            "is_exo": exo_weight > 0,
            "is_over": nb_overs > 0,
            "nb_basic_stats_ratio": nb_basic_stats_ratio,
            "nb_essential_stats_ratio": nb_essential_stats_ratio,
            "nb_secondary_stats_ratio": nb_secondary_stats_ratio,
            "nb_heavy_stats_ratio": nb_heavy_stats_ratio,
            "nb_résistance_elem_ratio": nb_résistance_elem_ratio,
            "nb_resistance_pourcent_ratio": nb_resistance_pourcent_ratio,
            "nb_dommage_elem_ratio": nb_dommage_elem_ratio,
            "nb_dommage_pourcent_ratio": nb_dommage_pourcent_ratio
        }

    def evaluate_quality_algo(self):
        stats = self.template.stats
        n_stats = len(stats)

        perfect_lines = sum(self.current_stats.get(stat, 0) == self.template.get_stat_max(stat) for stat in stats)
        ratio_perfect = perfect_lines / n_stats if n_stats else 0

        exo = self.get_exo_weight() > 0
        over = any(self.is_over(stat) for stat in stats)

        if ratio_perfect > 0.9:
            quality = "parfait"
        elif ratio_perfect > 0.7:
            quality = "très bon jet"
        elif all(self.get_ratio(stat) >= 0.6 for stat in stats if stat in stat_pool):
            quality = "bon"
        else:
            quality = "nul"

        return {
            "quality": quality,
            "exo": exo,
            "over": over,
            "puit": self.template.pui_category
        }

class ItemMetadata:
    def __init__(self, name="Inconnu", level="?", type_name="?", set_name=None):
        self.name = name
        self.level = level
        self.type_name = type_name
        self.set_name = set_name

    def __repr__(self):
        return f"<ItemMetadata name={self.name}, lvl={self.level}, type={self.type_name}, set={self.set_name}>"
    

class Item:
    def __init__(self, metadata: ItemMetadata, instance: ItemInstance):
        self.metadata = metadata
        self.instance = instance

    @property
    def template(self) -> ItemTemplate:
        return self.instance.template

    def summary(self) -> dict:
        """
        Renvoie un résumé pratique de l’item :
        - Métadonnées
        - Features calculées
        - Évaluation de qualité
        """
        return {
            "metadata": {
                "name": self.metadata.name,
                "level": self.metadata.level,
                "type": self.metadata.type_name,
                "set": self.metadata.set_name,
            },
            "features": self.instance.get_features(),
            "evaluation": self.instance.evaluate_quality_algo(),
        }

    def __repr__(self):
        return f"<Item {self.metadata.name} (lvl {self.metadata.level}) | {self.instance.evaluate_quality_algo()}>"