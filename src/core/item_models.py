import numpy as np
from src.config.stat_pool import stat_pool

class ItemTemplate:
    def __init__(self, name, stats, pui_category="moyen"):
        self.name = name
        self.stats = stats
        self.pui_category = pui_category

    def get_stat_max(self, stat_name):
        return self.stats.get(stat_name).max_value if stat_name in self.stats else 0

    def get_stat_weight(self, stat_name):
        return self.stats.get(stat_name).weight if stat_name in self.stats else 0

    def has_stat(self, stat_name):
        return stat_name in self.stats
    
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
        avg_ratio = np.nanmean([self.get_ratio(stat) for stat in self.template.stats])
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
        def stat_is_perfect(stat):
            return self.current_stats.get(stat, 0) == self.template.get_stat_max(stat)

        def stat_is_within_tolerance(stat):
            if stat not in stat_pool:
                return True
            max_val = self.template.get_stat_max(stat)
            val = self.current_stats.get(stat, 0)
            base_weight = stat_pool[stat].weight
            if max_val == 0:
                return True
            diff = abs(max_val - val)
            line_weight = max_val * base_weight
            if line_weight <= 30:
                return diff <= 1
            elif line_weight <= 60:
                return diff <= 3
            else:
                return diff <= 10

        stats = self.template.stats
        n_stats = len(stats)

        perfect_lines = sum(stat_is_perfect(stat) for stat in stats)
        high_ratios = sum(1 for stat in stats if 0.9 <= self.get_ratio(stat) < 1)
        ratio_perfect = perfect_lines / n_stats if n_stats else 0

        total_weight = self.get_total_weight()
        exo_weight = self.get_exo_weight()
        over = any(self.is_over(stat) for stat in stats)
        exo = exo_weight > 0

        pui = self.template.pui_category
        note = 100
        malus = 0

        # Malus sur les essential stats non parfaites
        for stat in essential_stats:
            if stat in stats and self.get_ratio(stat) < 1:
                malus += 20

        # Malus si résistances pas parfaites
        for stat in résistance_elem_stats.union(résistance_pourcent_stats):
            if stat in stats and self.get_ratio(stat) < 1:
                malus += 10

        # Tolérances pour les autres lignes
        for stat in stats:
            if stat in essential_stats or stat not in stat_pool:
                continue
            if not stat_is_within_tolerance(stat):
                malus += 5

        # Bonus si très bon jet
        if malus < 10 and ratio_perfect > 0.8:
            note = 100
            quality = "parfait"
        elif malus < 25 and all(
            (stat not in stats or self.get_ratio(stat) == 1)
            for stat in essential_stats.union(résistance_pourcent_stats, résistance_elem_stats)
        ):
            note = max(85, 100 - malus)
            quality = "très bon jet"
        elif all(self.get_ratio(stat) >= 0.6 for stat in stats if stat in stat_pool):
            note = max(60, 100 - malus)
            quality = "jet craft"
        else:
            note = max(30, 100 - malus)
            quality = "jet nul"

        return {
            "note": note,
            "exo": exo,
            "over": over,
            "puit": pui,
            "quality": quality
        }